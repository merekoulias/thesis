"""Primary-pair gaps across baseline rank 1 and robustness ranks 2–5."""

from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.fairness.gaps import pair_mask
from src.fairness.group_metrics import rate
from src.fairness.io import PathLike, load_audit_frame
from src.fairness.stats import diff_ci
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import SCORE_COL, Y_COL, YHAT_COL, AuditConfig

ROBUSTNESS_COLUMNS: Tuple[str, ...] = (
    "rank",
    "threshold",
    "n_flagged",
    "roc_auc",
    "metric",
    "diff",
    "diff_ci_low",
    "diff_ci_high",
)


def _primary_pair_gaps(
    frame: pd.DataFrame, config: AuditConfig
) -> Dict[str, Tuple[float, float, float]]:
    """Return ``(diff, ci_low, ci_high)`` per robustness metric for the primary pair."""
    pair = next(p for p in config.pairs if p.pair_id == config.primary_pair)
    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    mask_a = pair_mask(frame, pair.a)
    mask_b = pair_mask(frame, pair.b)
    y_a, h_a = y[mask_a], yhat[mask_a]
    y_b, h_b = y[mask_b], yhat[mask_b]
    out: Dict[str, Tuple[float, float, float]] = {}

    for metric in config.robustness_metrics:
        events_a, den_a = rate(y_a, h_a, metric)
        events_b, den_b = rate(y_b, h_b, metric)
        rate_a = float(events_a / den_a) if den_a else float("nan")
        rate_b = float(events_b / den_b) if den_b else float("nan")
        out[metric] = diff_ci(rate_a, den_a, rate_b, den_b, config.z)
    return out


def robustness_table(
    run_dir: PathLike,
    config: AuditConfig,
    groups_path: PathLike,
    data_dir: PathLike,
) -> pd.DataFrame:
    """Build the ``robustness.csv`` table for ranks 1 and ``robustness_ranks``.

    For each rank, loads the audit frame, applies that rank's base-rate threshold,
    and records primary-pair gaps (Wald CI, no bootstrap) and pooled ROC AUC.

    Args:
        run_dir: Model run directory (``artifacts/runs/<model>/<variant>``).
        config: Audit settings.
        groups_path: Fairness groups parquet path.
        data_dir: Raw CSV directory for PTI.

    Returns:
        DataFrame with ``ROBUSTNESS_COLUMNS`` (one row per rank × metric).
    """
    ranks = [1] + list(config.robustness_ranks)
    rows: List[Dict[str, Any]] = []

    for rank in ranks:
        frame = load_audit_frame(run_dir, rank, groups_path, data_dir, config)
        score = frame[SCORE_COL].to_numpy()
        y = frame[Y_COL].to_numpy()
        thr = base_rate_threshold(score, y)
        frame[YHAT_COL] = apply_threshold(score, thr.threshold)
        roc_auc = float(roc_auc_score(y, score))
        metric_gaps = _primary_pair_gaps(frame, config)

        for metric in config.robustness_metrics:
            diff, diff_lo, diff_hi = metric_gaps[metric]
            rows.append(
                {
                    "rank": rank,
                    "threshold": thr.threshold,
                    "n_flagged": thr.n_flagged,
                    "roc_auc": roc_auc,
                    "metric": metric,
                    "diff": diff,
                    "diff_ci_low": diff_lo,
                    "diff_ci_high": diff_hi,
                }
            )

    return pd.DataFrame(rows, columns=list(ROBUSTNESS_COLUMNS))
