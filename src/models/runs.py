"""Save and load nested-CV run directories."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.constants import TARGET_COL
from src.models.metrics import reference_metrics, summarize_folds
from src.models.types import (
    CANDIDATES_FILE,
    INNER_SCORES_FILE,
    NestedCVResult,
    RankedRun,
    candidates_payload,
)

PathLike = Union[str, Path]

DEFAULT_VARIANT = "baseline"
OOF_FILE = "oof_predictions.parquet"
METRICS_FILE = "metrics.json"
SELECTED_PARAMS_FILE = "selected_params.json"
RUN_CONFIG_FILE = "run_config.json"
IMPORTANCE_FILE = "feature_importance.parquet"
LOG_FILE = "run.log"
TUNING_DIR = "tuning"
ROBUSTNESS_DIR = "robustness"
RANKINGS_FILE = "rankings.json"

METRICS_JSON_KEYS: List[str] = [
    "model",
    "rank",
    "n_samples",
    "n_features",
    "n_folds",
    "oof_roc_auc",
    "oof_brier",
    "oof_log_loss",
    "base_rate",
    "brier_reference",
    "log_loss_reference",
    "fold_roc_aucs",
    "mean_fold_roc_auc",
    "std_fold_roc_auc",
    "folds",
    "mean_roc_auc_gap",
    "mean_brier_gap",
    "mean_log_loss_gap",
    "mean_selection_optimism",
    "runtime_seconds",
]


def run_dir_for(
    runs_dir: PathLike, model_name: str, variant: str = DEFAULT_VARIANT
) -> Path:
    """Return runs_dir / model_name / variant.

    Args:
        runs_dir: Root directory for all model runs.
        model_name: Registry model name.
        variant: Run variant subdirectory name.

    Returns:
        Path to the run directory.
    """
    return Path(runs_dir) / model_name / variant


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, allow_nan=False)
        f.write("\n")


def _rankings_payload(result: NestedCVResult) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sel in result.selections:
        out.append(
            {
                "outer_fold": sel.outer_fold,
                "ranking": sel.ranking,
                "mean_inner_auc": {str(k): v for k, v in sel.mean_inner_auc.items()},
                "mean_best_iteration": {
                    str(k): v for k, v in sel.mean_best_iteration.items()
                },
            }
        )
    return out


def _selected_params_payload(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_to_jsonable(entry) for entry in selected]


def build_metrics_dict(
    result: NestedCVResult,
    ranked: RankedRun,
    n_features: int,
) -> Dict[str, Any]:
    """Build metrics.json content for one ranked run (plan 5.8 key order).

    Args:
        result: Full nested-CV result.
        ranked: One rank's OOF run.
        n_features: Feature count from training data.

    Returns:
        Metrics dictionary with keys in ``METRICS_JSON_KEYS`` order.
    """
    oof = ranked.oof_predictions
    y = oof[TARGET_COL].to_numpy()
    refs = reference_metrics(y)
    fold_aucs = ranked.fold_roc_aucs
    summary = summarize_folds(ranked.fold_metrics)
    ordered: Dict[str, Any] = {}
    values = {
        "model": result.model_name,
        "rank": ranked.rank,
        "n_samples": len(oof),
        "n_features": n_features,
        "n_folds": len(result.selections),
        "oof_roc_auc": ranked.oof_roc_auc,
        "oof_brier": ranked.oof_brier,
        "oof_log_loss": ranked.oof_log_loss,
        "base_rate": refs["base_rate"],
        "brier_reference": refs["brier_reference"],
        "log_loss_reference": refs["log_loss_reference"],
        "fold_roc_aucs": fold_aucs,
        "mean_fold_roc_auc": float(np.mean(fold_aucs)),
        "std_fold_roc_auc": float(np.std(fold_aucs, ddof=0)),
        "folds": ranked.fold_metrics,
        "mean_roc_auc_gap": summary["mean_roc_auc_gap"],
        "mean_brier_gap": summary["mean_brier_gap"],
        "mean_log_loss_gap": summary["mean_log_loss_gap"],
        "mean_selection_optimism": summary["mean_selection_optimism"],
        "runtime_seconds": result.runtime_seconds,
    }
    for key in METRICS_JSON_KEYS:
        ordered[key] = values[key]
    return ordered


def _write_rank_outputs(
    run_dir: Path,
    result: NestedCVResult,
    ranked: RankedRun,
    n_features: int,
    rank_subdir: Optional[Path] = None,
) -> None:
    base = rank_subdir if rank_subdir is not None else run_dir
    base.mkdir(parents=True, exist_ok=True)
    ranked.oof_predictions.to_parquet(base / OOF_FILE, index=False)
    _write_json(base / METRICS_FILE, build_metrics_dict(result, ranked, n_features))
    _write_json(
        base / SELECTED_PARAMS_FILE,
        _selected_params_payload(ranked.selected),
    )


def save_nested_run(
    result: NestedCVResult,
    run_dir: PathLike,
    run_config: Dict[str, Any],
) -> Path:
    """Write the run directory (plan 5.8). Rank 1 at the top level; ranks >= 2 under
    robustness/rank_<r>/. Writes tuning/rankings.json from result.selections.
    tuning/candidates.json and tuning/inner_scores.csv are (re)written from the result
    so a run without checkpointing is still complete (tuning/fingerprint.json is only
    written by run_nested_cv when it checkpoints).

    Args:
        result: Nested-CV outputs to persist.
        run_dir: Target run directory.
        run_config: Metadata written to run_config.json (must include ``n_features``).

    Returns:
        Path to the run directory.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_features = int(run_config["n_features"])

    _write_json(out / RUN_CONFIG_FILE, run_config)

    tuning = out / TUNING_DIR
    tuning.mkdir(parents=True, exist_ok=True)
    _write_json(tuning / CANDIDATES_FILE, candidates_payload(result.candidates))
    result.inner_scores.to_csv(tuning / INNER_SCORES_FILE, index=False)
    _write_json(tuning / RANKINGS_FILE, _rankings_payload(result))

    _write_rank_outputs(out, result, result.runs[1], n_features)

    if result.feature_importance is not None:
        result.feature_importance.to_parquet(out / IMPORTANCE_FILE, index=False)

    for rank in range(2, max(result.runs) + 1):
        rank_dir = out / ROBUSTNESS_DIR / f"rank_{rank}"
        _write_rank_outputs(
            out,
            result,
            result.runs[rank],
            n_features,
            rank_subdir=rank_dir,
        )

    return out


def _oof_path(run_dir: Path, rank: int) -> Path:
    if rank == 1:
        return run_dir / OOF_FILE
    return run_dir / ROBUSTNESS_DIR / f"rank_{rank}" / OOF_FILE


def load_oof(run_dir: PathLike, rank: int = 1) -> pd.DataFrame:
    """Load OOF predictions of rank 1 (top level) or rank r (robustness/rank_r).

    Args:
        run_dir: Run directory path.
        rank: Rank index (1 = main run).

    Returns:
        OOF predictions frame (plan 5.7 schema).
    """
    path = _oof_path(Path(run_dir), rank)
    if not path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {path}")
    return pd.read_parquet(path)


def load_metrics(run_dir: PathLike, rank: int = 1) -> Dict[str, Any]:
    """Load metrics.json for the given rank.

    Args:
        run_dir: Run directory path.
        rank: Rank index (1 = main run).

    Returns:
        Parsed metrics dictionary.
    """
    base = Path(run_dir)
    if rank == 1:
        path = base / METRICS_FILE
    else:
        path = base / ROBUSTNESS_DIR / f"rank_{rank}" / METRICS_FILE
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_selected_params(run_dir: PathLike, rank: int = 1) -> Dict[int, Dict[str, Any]]:
    """Return {outer_fold: resolved params} — the frozen hyperparameters for later runs.

    Args:
        run_dir: Run directory path.
        rank: Rank index (1 = main run).

    Returns:
        Mapping from outer fold index to resolved parameter dict.
    """
    base = Path(run_dir)
    if rank == 1:
        path = base / SELECTED_PARAMS_FILE
    else:
        path = base / ROBUSTNESS_DIR / f"rank_{rank}" / SELECTED_PARAMS_FILE
    with path.open(encoding="utf-8") as f:
        entries = json.load(f)
    return {int(entry["outer_fold"]): entry["params"] for entry in entries}
