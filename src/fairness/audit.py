"""Run the baseline fairness audit for one model and write artifacts."""

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.fairness.calibration import calibration_table, decile_bins, sufficiency_tests
from src.fairness.comparison import (
    expected_signs,
    kendall_cells,
    oracle_benchmark,
    proxy_importance,
)
from src.fairness.conditional import conditional_parity, conditional_parity_cells
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.gaps import cell_summary, pair_gaps
from src.fairness.group_metrics import group_counts, group_metrics, utility
from src.fairness.io import PathLike, assign_pti_quintiles, load_audit_frame
from src.fairness.plots import plot_calibration, plot_cells_error_rates
from src.fairness.report import audit_config_payload, write_model_report, write_outputs
from src.fairness.robustness import robustness_table
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import (
    PTI_COL,
    PTI_Q_COL,
    SCORE_COL,
    Y_COL,
    YHAT_COL,
    AuditConfig,
)
from src.models.io import load_yaml_config
from src.models.runs import IMPORTANCE_FILE, load_metrics, run_dir_for

PathLikeArg = Union[str, Path]

logger = logging.getLogger(__name__)

FAIRNESS_GROUPS_FILE = "fairness_groups_train.parquet"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths() -> Dict[str, Path]:
    """Load default path settings from ``configs/paths.yaml``."""
    raw = load_yaml_config(_repo_root() / "configs" / "paths.yaml")
    root = _repo_root()
    return {
        "runs_dir": root / raw["runs_dir"],
        "features_dir": root / raw["features_dir"],
        "data_dir": root / raw["data_dir"],
        "fairness_dir": root / raw["fairness_dir"],
    }


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_versions() -> Dict[str, str]:
    import matplotlib
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "audit.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _timed_step(name: str, func):
    start = time.perf_counter()
    result = func()
    logger.info("%s completed in %.1fs", name, time.perf_counter() - start)
    return result


def _model_out_dir(out_dir: PathLike, model: str, variant: str) -> Path:
    return Path(out_dir) / model / variant


def _assert_all_auc_matches(
    metrics_frame: pd.DataFrame, run_auc: float, tol: float = 1e-12
) -> None:
    row = metrics_frame[
        (metrics_frame["grouping"] == "ALL")
        & (metrics_frame["group"] == "all")
        & (metrics_frame["metric"] == "auc")
    ]
    if row.empty:
        raise ValueError("ALL / all / auc row missing from group_metrics")
    auc = float(row.iloc[0]["value"])
    if not np.isclose(auc, run_auc, rtol=0, atol=tol):
        raise ValueError(f"ALL AUC {auc} != run oof_roc_auc {run_auc}")


def run_audit(
    model: str,
    config: AuditConfig,
    runs_dir: PathLike,
    features_dir: PathLike,
    data_dir: PathLike,
    out_dir: PathLike,
    overwrite: bool = False,
) -> Path:
    """Audit rank 1 plus robustness ranks; write plan 5.6 outputs.

    Args:
        model: Registry model name.
        config: Audit configuration (``n_boot`` may already be overridden).
        runs_dir: Root of model run directories.
        features_dir: Feature artifacts directory.
        data_dir: Raw CSV directory for PTI.
        out_dir: Fairness output root.
        overwrite: Replace an existing model output directory.

    Returns:
        Path to ``<out_dir>/<model>/<variant>/``.
    """
    out_path = _model_out_dir(out_dir, model, config.variant)
    if out_path.exists() and any(out_path.iterdir()) and not overwrite:
        raise SystemExit(
            1,
            f"Output directory {out_path} already exists; pass --overwrite to replace it.",
        )
    _setup_logging(out_path)
    started_at = datetime.now(timezone.utc)
    run_dir = run_dir_for(runs_dir, model, config.variant)
    groups_path = Path(features_dir) / FAIRNESS_GROUPS_FILE

    input_paths = {
        "run_dir": str(run_dir.resolve()),
        "groups_path": str(groups_path.resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "oof_rank_1": str((run_dir / "oof_predictions.parquet").resolve()),
        "metrics_rank_1": str((run_dir / "metrics.json").resolve()),
    }

    frame = _timed_step(
        "load_audit_frame",
        lambda: load_audit_frame(run_dir, 1, groups_path, data_dir, config),
    )
    run_metrics = load_metrics(run_dir, 1)

    thr = _timed_step(
        "threshold",
        lambda: base_rate_threshold(
            frame[SCORE_COL].to_numpy(), frame[Y_COL].to_numpy()
        ),
    )
    frame[YHAT_COL] = apply_threshold(frame[SCORE_COL].to_numpy(), thr.threshold)

    counts = _timed_step("group_counts", lambda: group_counts(frame, config))
    metrics_df = _timed_step(
        "group_metrics",
        lambda: group_metrics(frame, config, model, 1),
    )
    _assert_all_auc_matches(metrics_df, float(run_metrics["oof_roc_auc"]))
    util = _timed_step("utility", lambda: utility(frame, thr, run_metrics))
    gaps = _timed_step("pair_gaps", lambda: pair_gaps(frame, config))
    cells = _timed_step("cell_summary", lambda: cell_summary(frame, config, model, 1))
    calibration = _timed_step(
        "calibration_table", lambda: calibration_table(frame, config)
    )
    cal_tests = _timed_step(
        "calibration_tests", lambda: sufficiency_tests(frame, config)
    )
    edges, _ = decile_bins(frame[SCORE_COL].to_numpy(), config.calibration_bins)
    bin_edges = {"edges": edges.tolist()}

    cond_parity, cond_tests = _timed_step(
        "conditional_parity",
        lambda: conditional_parity(frame, config, gaps),
    )
    cond_cells = _timed_step(
        "conditional_parity_cells", lambda: conditional_parity_cells(frame, config)
    )
    _, cut_points = assign_pti_quintiles(frame[PTI_COL], config.cdp_n_quantiles)
    pti_meta = {
        "cut_points": [float(x) for x in cut_points],
        "n_missing": int(frame[PTI_Q_COL].isna().sum()),
    }

    oracle = _timed_step(
        "oracle_benchmark",
        lambda: oracle_benchmark(frame, config, gaps),
    )
    signs = _timed_step("expected_signs", lambda: expected_signs(frame, config, gaps))
    kendall = _timed_step("kendall_cells", lambda: kendall_cells(frame, config))

    importance_path = run_dir / IMPORTANCE_FILE
    importance = pd.read_parquet(importance_path) if importance_path.exists() else None
    proxy = _timed_step(
        "proxy_importance",
        lambda: proxy_importance(importance, config, model),
    )

    robust = _timed_step(
        "robustness_table",
        lambda: robustness_table(run_dir, config, groups_path, data_dir),
    )

    finished_at = datetime.now(timezone.utc)
    audit_cfg = audit_config_payload(
        config,
        model,
        input_paths,
        _git_commit(),
        _package_versions(),
        started_at,
        finished_at,
    )

    results: Dict[str, Any] = {
        "model": model,
        "threshold": thr,
        "utility": util,
        "group_counts": counts,
        "group_metrics": metrics_df,
        "gaps": gaps,
        "cell_summary": cells,
        "calibration": calibration,
        "calibration_tests": cal_tests,
        "bin_edges": bin_edges,
        "conditional_parity": cond_parity,
        "conditional_parity_tests": cond_tests,
        "conditional_parity_cells": cond_cells,
        "pti_quintiles": pti_meta,
        "oracle_benchmark": oracle,
        "expected_signs": signs,
        "kendall": kendall,
        "proxy_importance": proxy,
        "robustness": robust,
        "audit_config": audit_cfg,
    }

    _timed_step("write_outputs", lambda: write_outputs(results, out_path))

    fig_dir = out_path / "figures"
    _timed_step(
        "figures",
        lambda: (
            plot_calibration(
                calibration,
                "CODE_GENDER",
                model,
                fig_dir / "calibration_CODE_GENDER.png",
            ),
            plot_calibration(
                calibration, "AGE_BAND", model, fig_dir / "calibration_AGE_BAND.png"
            ),
            plot_cells_error_rates(
                metrics_df, config, model, fig_dir / "cells_error_rates.png"
            ),
        ),
    )

    _timed_step("report", lambda: write_model_report(results, out_path, config))
    return out_path


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Run the baseline fairness audit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["logreg", "lgbm", "mlp", "rf"])
    group.add_argument("--all", action="store_true", help="Audit every model in config")
    parser.add_argument("--config", type=Path, default=default_fairness_config_path())
    parser.add_argument("--runs-dir", type=Path, default=paths["runs_dir"])
    parser.add_argument("--features-dir", type=Path, default=paths["features_dir"])
    parser.add_argument("--data-dir", type=Path, default=paths["data_dir"])
    parser.add_argument("--out-dir", type=Path, default=paths["fairness_dir"])
    parser.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help="Override bootstrap replicates (smoke runs)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for ``python -m src.fairness.audit``."""
    args = _parse_args(argv)
    config = load_audit_config(args.config)
    if args.n_boot is not None:
        config = replace(config, n_boot=args.n_boot)

    models = list(config.models) if args.all else [args.model]
    for model in models:
        run_audit(
            model,
            config,
            args.runs_dir,
            args.features_dir,
            args.data_dir,
            args.out_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
