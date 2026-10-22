"""Decile calibration tables and sufficiency (G) tests by grouping."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.fairness.group_metrics import group_masks
from src.fairness.stats import proportion_ci, stratified_g_test
from src.fairness.types import SCORE_COL, Y_COL, AuditConfig

CALIBRATION_COLUMNS: Tuple[str, ...] = (
    "grouping",
    "group",
    "bin",
    "n",
    "events",
    "observed_rate",
    "ci_low",
    "ci_high",
    "mean_score",
    "calibration_gap",
)

CALIBRATION_TESTS_COLUMNS: Tuple[str, ...] = (
    "grouping",
    "g_stat",
    "df",
    "p_value",
)


def decile_bins(score: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign equal-frequency score deciles from pooled quantile edges.

    Args:
        score: Pooled model scores (one value per audited row).
        n_bins: Number of bins (e.g. 10 deciles).

    Returns:
        ``(edges, bin)`` where ``edges`` has length ``n_bins + 1`` and ``bin`` is
        in ``0 .. n_bins - 1``.
    """
    edges = np.quantile(score, np.linspace(0.0, 1.0, n_bins + 1))
    bin_idx = np.searchsorted(edges[1:-1], score, side="right")
    return edges, bin_idx


def calibration_table(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Build the long calibration table (plan 5.6 ``calibration.csv``).

    Bins come from the pooled score; each (grouping, group, bin) reports counts,
    observed default rate with a plug-in interval, mean score, and calibration gap.

    Args:
        frame: Audit frame with ``y`` and ``score``.
        config: Audit settings (``calibration_bins``, groupings).

    Returns:
        DataFrame with ``CALIBRATION_COLUMNS``; includes ``ALL / all``.
    """
    score = frame[SCORE_COL].to_numpy()
    y = frame[Y_COL].to_numpy()
    z = config.z
    _, bins = decile_bins(score, config.calibration_bins)
    rows: List[Dict[str, Any]] = []

    for grouping, group, mask in group_masks(frame, config):
        g_score = score[mask]
        g_y = y[mask]
        g_bins = bins[mask]
        for bin_idx in range(config.calibration_bins):
            b_mask = g_bins == bin_idx
            n = int(b_mask.sum())
            events = int(g_y[b_mask].sum())
            obs, ci_lo, ci_hi = proportion_ci(events, n, z)
            mean_score = float(g_score[b_mask].mean()) if n > 0 else float("nan")
            gap = obs - mean_score if n > 0 else float("nan")
            rows.append(
                {
                    "grouping": grouping,
                    "group": group,
                    "bin": bin_idx,
                    "n": n,
                    "events": events,
                    "observed_rate": obs,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "mean_score": mean_score,
                    "calibration_gap": gap,
                }
            )

    return pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))


def sufficiency_tests(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """G-test of calibration sufficiency across score bins for each grouping.

    For each bin, builds a ``k × 2`` table (groups × [``y=1``, ``y=0``]) and
    combines bins with ``stratified_g_test``.

    Args:
        frame: Audit frame with ``y``, ``score``, and grouping columns.
        config: Audit settings.

    Returns:
        DataFrame with ``CALIBRATION_TESTS_COLUMNS`` (one row per configured grouping).
    """
    score = frame[SCORE_COL].to_numpy()
    y = frame[Y_COL].to_numpy()
    _, bins = decile_bins(score, config.calibration_bins)
    rows: List[Dict[str, Any]] = []

    for grouping_col, labels in config.groupings.items():
        col = frame[grouping_col].to_numpy()
        tables: List[np.ndarray] = []
        for bin_idx in range(config.calibration_bins):
            bin_mask = bins == bin_idx
            table_rows: List[List[float]] = []
            for label in labels:
                g_mask = col == label
                mask = bin_mask & g_mask
                y_bin = y[mask]
                n_events = float(y_bin.sum())
                n_neg = float(len(y_bin) - n_events)
                table_rows.append([n_events, n_neg])
            tables.append(np.asarray(table_rows, dtype=float))
        g_stat, df, p_value = stratified_g_test(tables)
        rows.append(
            {
                "grouping": grouping_col,
                "g_stat": g_stat,
                "df": df,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows, columns=list(CALIBRATION_TESTS_COLUMNS))
