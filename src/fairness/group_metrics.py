"""Group counts, rates, and pooled utility at the audit threshold."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.fairness.stats import bootstrap_auc_ci, mean_ci, proportion_ci, seed_for
from src.fairness.types import (
    SCORE_COL,
    Y_COL,
    YHAT_COL,
    AuditConfig,
    ThresholdResult,
)

_SCORE_CLIP_LO = 1e-15
_SCORE_CLIP_HI = 1.0 - 1e-15

ALL_GROUPING = "ALL"
ALL_GROUP = "all"

SCORE_METRICS: Tuple[str, ...] = ("auc", "brier", "log_loss")

GROUP_METRICS_COLUMNS: Tuple[str, ...] = (
    "grouping",
    "group",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "denominator",
)


@dataclass(frozen=True)
class MetricSpec:
    """Subpopulation, event, and denominator name for one rate metric."""

    name: str
    subpopulation: Callable[[np.ndarray, np.ndarray], np.ndarray]
    event: Callable[[np.ndarray, np.ndarray], np.ndarray]
    denominator: str


def _subpop_all(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.ones_like(y, dtype=bool)


def _event_approved(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return yhat == 0


def _subpop_negatives(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return y == 0


def _event_flagged(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return yhat == 1


def _subpop_positives(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return y == 1


def _event_not_flagged(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return yhat == 0


def _subpop_flagged(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return yhat == 1


def _event_positive(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return y == 1


def _subpop_approved(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return yhat == 0


METRIC_SPECS: Dict[str, MetricSpec] = {
    "approval_rate": MetricSpec("approval_rate", _subpop_all, _event_approved, "n"),
    "fpr": MetricSpec("fpr", _subpop_negatives, _event_flagged, "negatives"),
    "fnr": MetricSpec("fnr", _subpop_positives, _event_not_flagged, "positives"),
    "tpr": MetricSpec("tpr", _subpop_positives, _event_flagged, "positives"),
    "ppv": MetricSpec("ppv", _subpop_flagged, _event_positive, "flagged"),
    "default_among_approved": MetricSpec(
        "default_among_approved",
        _subpop_approved,
        _event_positive,
        "approved",
    ),
}


def group_masks(
    frame: pd.DataFrame, config: AuditConfig
) -> List[Tuple[str, str, np.ndarray]]:
    """Return ``[(grouping, group, bool mask)]`` for ALL/all and each configured group.

    Order: ``ALL / all``, then each ``groupings`` column in config order with labels
    in the order listed in ``fairness.yaml``.
    """
    n = len(frame)
    masks: List[Tuple[str, str, np.ndarray]] = [
        (ALL_GROUPING, ALL_GROUP, np.ones(n, dtype=bool))
    ]
    for grouping_col, labels in config.groupings.items():
        col = frame[grouping_col].to_numpy()
        for label in labels:
            mask = col == label
            masks.append((grouping_col, label, mask))
    return masks


def rate(y: np.ndarray, yhat: np.ndarray, metric: str) -> Tuple[int, int]:
    """Return ``(events, denominator)`` for ``metric`` on the given rows."""
    spec = METRIC_SPECS[metric]
    sub = spec.subpopulation(y, yhat)
    denom = int(sub.sum())
    events = int((sub & spec.event(y, yhat)).sum())
    return events, denom


def _count_row(
    grouping: str, group: str, y: np.ndarray, yhat: np.ndarray
) -> Dict[str, Any]:
    n = y.size
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    flagged = int((yhat == 1).sum())
    approved = int((yhat == 0).sum())
    base_rate = float(positives / n) if n else float("nan")
    return {
        "grouping": grouping,
        "group": group,
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "flagged": flagged,
        "approved": approved,
        "base_rate": base_rate,
    }


def group_counts(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Build the ``group_counts.csv`` table (ALL row plus every configured group)."""
    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    rows = []
    for grouping, group, mask in group_masks(frame, config):
        rows.append(_count_row(grouping, group, y[mask], yhat[mask]))
    return pd.DataFrame(rows)


def _brier_values(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    return (score - y) ** 2


def _log_loss_values(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    s = np.clip(score, _SCORE_CLIP_LO, _SCORE_CLIP_HI)
    return -(y * np.log(s) + (1.0 - y) * np.log(1.0 - s))


def _metric_rows_for_group(
    grouping: str,
    group: str,
    y: np.ndarray,
    yhat: np.ndarray,
    score: np.ndarray,
    config: AuditConfig,
    model: str,
    rank: int,
) -> List[Dict[str, Any]]:
    z = config.z
    alpha = 1.0 - config.confidence
    rows: List[Dict[str, Any]] = []

    for metric_name in METRIC_SPECS:
        events, denom = rate(y, yhat, metric_name)
        value, ci_low, ci_high = proportion_ci(events, denom, z)
        rows.append(
            {
                "grouping": grouping,
                "group": group,
                "metric": metric_name,
                "value": value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "denominator": denom,
            }
        )

    auc_seed = seed_for(config.seed, f"auc|{model}|{rank}|{grouping}|{group}")
    auc_val, auc_lo, auc_hi = bootstrap_auc_ci(y, score, config.n_boot, auc_seed, alpha)
    rows.append(
        {
            "grouping": grouping,
            "group": group,
            "metric": "auc",
            "value": auc_val,
            "ci_low": auc_lo,
            "ci_high": auc_hi,
            "denominator": int(y.size),
        }
    )

    brier_val, brier_lo, brier_hi = mean_ci(_brier_values(y, score), z)
    rows.append(
        {
            "grouping": grouping,
            "group": group,
            "metric": "brier",
            "value": brier_val,
            "ci_low": brier_lo,
            "ci_high": brier_hi,
            "denominator": int(y.size),
        }
    )

    ll_val, ll_lo, ll_hi = mean_ci(_log_loss_values(y, score), z)
    rows.append(
        {
            "grouping": grouping,
            "group": group,
            "metric": "log_loss",
            "value": ll_val,
            "ci_low": ll_lo,
            "ci_high": ll_hi,
            "denominator": int(y.size),
        }
    )
    return rows


def group_metrics(
    frame: pd.DataFrame, config: AuditConfig, model: str, rank: int
) -> pd.DataFrame:
    """Build the long ``group_metrics.csv`` table for every group and metric."""
    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    score = frame[SCORE_COL].to_numpy(dtype=float)
    rows: List[Dict[str, Any]] = []
    for grouping, group, mask in group_masks(frame, config):
        rows.extend(
            _metric_rows_for_group(
                grouping,
                group,
                y[mask],
                yhat[mask],
                score[mask],
                config,
                model,
                rank,
            )
        )
    return pd.DataFrame(rows)


def utility(
    frame: pd.DataFrame,
    threshold: ThresholdResult,
    run_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble ``utility.json`` fields from run metrics and thresholded predictions."""
    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    fold_aucs = run_metrics["fold_roc_aucs"]
    tpr_events, tpr_denom = rate(y, yhat, "tpr")
    ppv_events, ppv_denom = rate(y, yhat, "ppv")
    tpr = tpr_events / tpr_denom if tpr_denom else float("nan")
    ppv = ppv_events / ppv_denom if ppv_denom else float("nan")
    if tpr + ppv > 0:
        f1 = 2.0 * ppv * tpr / (ppv + tpr)
    else:
        f1 = float("nan")
    return {
        "roc_auc": float(run_metrics["oof_roc_auc"]),
        "fold_roc_auc_mean": float(run_metrics["mean_fold_roc_auc"]),
        "fold_roc_auc_std": float(run_metrics["std_fold_roc_auc"]),
        "fold_roc_auc_min": float(min(fold_aucs)),
        "fold_roc_auc_max": float(max(fold_aucs)),
        "brier": float(run_metrics["oof_brier"]),
        "brier_reference": float(run_metrics["brier_reference"]),
        "log_loss": float(run_metrics["oof_log_loss"]),
        "log_loss_reference": float(run_metrics["log_loss_reference"]),
        "tpr": float(tpr),
        "ppv": float(ppv),
        "f1": float(f1),
        "n_flagged": int(threshold.n_flagged),
    }
