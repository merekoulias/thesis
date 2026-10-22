"""Primary-pair gaps across baseline rank 1 and robustness ranks 2–5."""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.fairness.gaps import pair_gaps, pair_mask
from src.fairness.group_metrics import rate
from src.fairness.io import load_audit_frame
from src.fairness.stats import diff_ci
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import SCORE_COL, Y_COL, YHAT_COL, AuditConfig

PathLike = Union[str, Path]

ROBUSTNESS_COLUMNS: tuple = (
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
) -> Dict[str, Dict[str, float]]:
    """Return diff and Wald CI per robustness metric for the primary pair."""
    pair = next(p for p in config.pairs if p.pair_id == config.primary_pair)
    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    mask_a = pair_mask(frame, pair.a)
    mask_b = pair_mask(frame, pair.b)
    y_a, h_a = y[mask_a], yhat[mask_a]
    y_b, h_b = y[mask_b], yhat[mask_b]
    z = config.z
    out: Dict[str, Dict[str, float]] = {}

    for metric in config.robustness_metrics:
        events_a, den_a = rate(y_a, h_a, metric)
        events_b, den_b = rate(y_b, h_b, metric)
        rate_a = float(events_a / den_a) if den_a else float("nan")
        rate_b = float(events_b / den_b) if den_b else float("nan")
        diff, diff_lo, diff_hi = diff_ci(rate_a, den_a, rate_b, den_b, z)
        out[metric] = {
            "diff": diff,
            "diff_ci_low": diff_lo,
            "diff_ci_high": diff_hi,
        }
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
        thr = base_rate_threshold(frame[SCORE_COL].to_numpy(), frame[Y_COL].to_numpy())
        frame = frame.copy()
        frame[YHAT_COL] = apply_threshold(frame[SCORE_COL].to_numpy(), thr.threshold)
        roc_auc = float(roc_auc_score(frame[Y_COL], frame[SCORE_COL]))

        if rank == 1:
            gaps = pair_gaps(frame, config)
            primary = gaps[
                (gaps["pair_id"] == config.primary_pair)
                & (gaps["metric"].isin(config.robustness_metrics))
            ]
            gap_by_metric = {row["metric"]: row for _, row in primary.iterrows()}
        else:
            gap_by_metric = None

        metric_gaps = _primary_pair_gaps(frame, config)

        for metric in config.robustness_metrics:
            if gap_by_metric is not None:
                gap_row = gap_by_metric[metric]
                diff = float(gap_row["diff"])
                diff_lo = float(gap_row["diff_ci_low"])
                diff_hi = float(gap_row["diff_ci_high"])
            else:
                diff = metric_gaps[metric]["diff"]
                diff_lo = metric_gaps[metric]["diff_ci_low"]
                diff_hi = metric_gaps[metric]["diff_ci_high"]

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
