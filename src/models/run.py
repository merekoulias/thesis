"""CLI: train one credit-scoring model with nested cross-validation."""

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.models.io import load_training_data
from src.models.nested import run_nested_cv
from src.models.runs import (
    METRICS_FILE,
    TUNING_DIR,
    run_dir_for,
    save_nested_run,
)
from src.models.settings import (
    default_cv_config_path,
    default_model_config_path,
    load_cv_settings,
    load_model_config,
)
from src.models.types import CVSettings, ModelConfig, TrainingData

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_repo_root(),
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def _format_duration(seconds: float) -> str:
    total_minutes = int(seconds // 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _subsample_training_data(
    data: TrainingData, max_rows: int, seed: int
) -> TrainingData:
    n = len(data.y)
    y = data.y.to_numpy()
    idx = np.arange(n)
    train_idx, _ = train_test_split(
        idx,
        train_size=max_rows,
        stratify=y,
        random_state=seed,
    )
    train_idx = np.sort(train_idx)
    X = data.X.iloc[train_idx].reset_index(drop=True)
    y_s = data.y.iloc[train_idx].reset_index(drop=True)
    ids = data.ids.iloc[train_idx].reset_index(drop=True)
    return TrainingData(
        X=X,
        y=y_s,
        ids=ids,
        feature_names=data.feature_names,
    )


def _model_config_dict(model_config: ModelConfig) -> Dict[str, Any]:
    return {
        "name": model_config.name,
        "fixed_params": model_config.fixed_params,
        "default_params": model_config.default_params,
        "search_space": model_config.search_space,
        "early_stopping_rounds": model_config.early_stopping_rounds,
    }


def _cv_settings_dict(cv_settings: CVSettings) -> Dict[str, Any]:
    return asdict(cv_settings)


def main(argv: Optional[List[str]] = None) -> None:
    """Run nested CV for one model and write a run directory.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="Train a credit-scoring model with nested cross-validation."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["lgbm", "rf", "logreg", "mlp"],
        help="Registry model name.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("artifacts/features"),
        help="Directory with X_train, y_train, ids_train parquet files.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--cv-config",
        type=Path,
        default=None,
        help="Nested-CV settings YAML (default: configs/models/cv.yaml).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Per-model YAML (default: configs/models/<model>.yaml).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stratified subsample size for smoke runs.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint in an existing run directory.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run directory before starting.",
    )
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir
    if args.max_rows is not None:
        default_runs = (_repo_root() / "artifacts" / "runs").resolve()
        if Path(runs_dir).resolve() == default_runs:
            parser.error("--max-rows requires a --runs-dir other than artifacts/runs")

    cv_path = args.cv_config or default_cv_config_path()
    model_path = args.model_config or default_model_config_path(args.model)
    cv_settings = load_cv_settings(cv_path)
    model_config = load_model_config(model_path)
    if model_config.name != args.model:
        parser.error(
            f"--model {args.model} does not match model config name {model_config.name}"
        )

    data = load_training_data(args.features_dir)
    max_rows: Optional[int] = args.max_rows
    if max_rows is not None:
        if max_rows < 2 or max_rows > len(data.y):
            parser.error(f"--max-rows must be between 2 and {len(data.y)}")
        data = _subsample_training_data(data, max_rows, cv_settings.seed)

    run_dir = run_dir_for(runs_dir, model_config.name)
    if (run_dir / METRICS_FILE).exists() or (run_dir / TUNING_DIR).exists():
        if args.overwrite:
            shutil.rmtree(run_dir)
        elif not args.resume:
            print(
                f"Run directory exists: {run_dir}. Use --resume or --overwrite.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    root = logging.getLogger()
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)

    started_at = datetime.now(timezone.utc).isoformat()
    n_samples = len(data.y)
    n_features = data.X.shape[1]

    result = run_nested_cv(
        data,
        model_config,
        cv_settings,
        checkpoint_dir=run_dir / TUNING_DIR,
        resume=args.resume,
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    run_config: Dict[str, Any] = {
        "model": model_config.name,
        "variant": "baseline",
        "cv_settings": _cv_settings_dict(cv_settings),
        "model_config": _model_config_dict(model_config),
        "features_dir": str(args.features_dir),
        "n_samples": n_samples,
        "n_features": n_features,
        "max_rows": max_rows,
        "git_commit": _git_commit(),
        "versions": _package_versions(),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    save_nested_run(result, run_dir, run_config)

    rank1 = result.runs[1]
    fold_str = ", ".join(f"{auc:.4f}" for auc in rank1.fold_roc_aucs)
    rel = run_dir
    try:
        rel = run_dir.relative_to(_repo_root())
    except ValueError:
        rel = run_dir
    duration = _format_duration(result.runtime_seconds)
    print(
        f"Wrote {rel} | rank-1 OOF AUC={rank1.oof_roc_auc:.4f} | "
        f"fold AUCs=[{fold_str}] | {duration}"
    )


if __name__ == "__main__":
    main()
