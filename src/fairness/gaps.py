"""Pairwise rate gaps and worst-cell summaries over gender × age cells."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.fairness.group_metrics import rate
from src.fairness.stats import diff_ci, g_test, ratio_ci, seed_for, worst_group_summary
from src.fairness.types import Y_COL, YHAT_COL, AuditConfig

GAPS_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "metric",
    "rate_a",
    "rate_b",
    "den_a",
    "den_b",
    "diff",
    "diff_ci_low",
    "diff_ci_high",
    "g_stat",
    "df",
    "p_value",
    "ratio",
    "ratio_ci_low",
    "ratio_ci_high",
    "ratio_lower_over_higher",
    "primary",
)

CELL_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "metric",
    "k_groups",
    "max_min_gap",
    "max_min_ci_low",
    "max_min_ci_high",
    "max_cell",
    "min_cell",
    "var_uncorrected",
    "var_single_corrected",
    "var_double_corrected_ci_low",
    "var_double_corrected_ci_high",
    "sd_uncorrected",
    "sd_single_corrected",
    "sd_double_corrected_ci_low",
    "sd_double_corrected_ci_high",
    "g_stat",
    "df",
    "p_value",
)


def pair_mask(frame: pd.DataFrame, side: Dict[str, str]) -> np.ndarray:
    """Boolean mask of rows matching every column filter in ``side``.

    Args:
        frame: Audit frame with the grouping columns used in pair filters.
        side: Map of column name to exact label value (e.g. ``CODE_GENDER`` → ``F``).

    Returns:
        Length-``len(frame)`` mask; all-False if no row matches.
    """
    mask = np.ones(len(frame), dtype=bool)
    for col, val in side.items():
        mask &= frame[col].to_numpy() == val
    return mask


def _rate_value(events: int, denom: int) -> float:
    if denom == 0:
        return float("nan")
    return float(events / denom)


def pair_gaps(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Build the long ``gaps.csv`` table for every pair × ``config.gap_metrics``.

    Signed gaps are rate(a) − rate(b). G-tests use the 2 × 2 table of events vs
    non-events within each side's metric denominator. Ratio columns are filled only
    for ``approval_rate``.

    Args:
        frame: Audit frame with ``y`` and ``yhat``.
        config: Audit settings (pairs, gap metrics, primary pair/metrics).

    Returns:
        DataFrame with ``GAPS_COLUMNS``.
    """
    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    z = config.z
    rows: List[Dict[str, Any]] = []

    for pair in config.pairs:
        mask_a = pair_mask(frame, pair.a)
        mask_b = pair_mask(frame, pair.b)
        y_a, h_a = y[mask_a], yhat[mask_a]
        y_b, h_b = y[mask_b], yhat[mask_b]

        for metric in config.gap_metrics:
            events_a, den_a = rate(y_a, h_a, metric)
            events_b, den_b = rate(y_b, h_b, metric)
            rate_a = _rate_value(events_a, den_a)
            rate_b = _rate_value(events_b, den_b)
            diff, diff_lo, diff_hi = diff_ci(rate_a, den_a, rate_b, den_b, z)
            table = np.array(
                [
                    [events_a, den_a - events_a],
                    [events_b, den_b - events_b],
                ],
                dtype=float,
            )
            g_stat, df, p_value = g_test(table)

            ratio = ratio_lo = ratio_hi = ratio_lower_over_higher = float("nan")
            if metric == "approval_rate":
                ratio, ratio_lo, ratio_hi = ratio_ci(rate_a, den_a, rate_b, den_b, z)
                if not np.isnan(ratio):
                    ratio_lower_over_higher = float(min(ratio, 1.0 / ratio))

            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "metric": metric,
                    "rate_a": rate_a,
                    "rate_b": rate_b,
                    "den_a": den_a,
                    "den_b": den_b,
                    "diff": diff,
                    "diff_ci_low": diff_lo,
                    "diff_ci_high": diff_hi,
                    "g_stat": g_stat,
                    "df": df,
                    "p_value": p_value,
                    "ratio": ratio,
                    "ratio_ci_low": ratio_lo,
                    "ratio_ci_high": ratio_hi,
                    "ratio_lower_over_higher": ratio_lower_over_higher,
                    "primary": (
                        pair.pair_id == config.primary_pair
                        and metric in config.primary_metrics
                    ),
                }
            )

    return pd.DataFrame(rows, columns=list(GAPS_COLUMNS))


def cell_summary(
    frame: pd.DataFrame, config: AuditConfig, model: str, rank: int
) -> pd.DataFrame:
    """Build ``cell_summary.csv`` for ``config.cell_summary_metrics`` over ``GENDER_AGE``.

    Per metric, rates are computed in each configured gender × age cell; Lum et al.
    worst-group summary and a G-test on the 10 × 2 event tables follow.

    Args:
        frame: Audit frame with ``y``, ``yhat``, and ``GENDER_AGE``.
        config: Audit settings (cell summary metrics, groupings, seed, ``n_boot``).
        model: Model name (bootstrap seed key).
        rank: Run rank (bootstrap seed key).

    Returns:
        DataFrame with ``CELL_SUMMARY_COLUMNS`` (one row per cell-summary metric).
    """
    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    alpha = 1.0 - config.confidence
    cell_labels = config.groupings["GENDER_AGE"]
    gender_age = frame["GENDER_AGE"].to_numpy()
    rows: List[Dict[str, Any]] = []

    for metric in config.cell_summary_metrics:
        events_list: List[int] = []
        den_list: List[int] = []
        rate_list: List[float] = []

        for label in cell_labels:
            mask = gender_age == label
            ev, den = rate(y[mask], yhat[mask], metric)
            events_list.append(ev)
            den_list.append(den)
            rate_list.append(_rate_value(ev, den))

        w_seed = seed_for(config.seed, f"worst|{model}|{rank}|{metric}")
        active_idx = [i for i, d in enumerate(den_list) if d > 0]
        if active_idx:
            w_den = [den_list[i] for i in active_idx]
            w_rate = [rate_list[i] for i in active_idx]
            summary = worst_group_summary(w_den, w_rate, config.n_boot, w_seed, alpha)
            arr_rates = np.asarray(rate_list, dtype=float)
            max_idx = int(np.nanargmax(arr_rates))
            min_idx = int(np.nanargmin(arr_rates))
            max_cell = cell_labels[max_idx]
            min_cell = cell_labels[min_idx]
        else:
            summary = worst_group_summary([0], [0.0], config.n_boot, w_seed, alpha)
            max_cell = min_cell = ""

        g_table = np.column_stack(
            [np.asarray(events_list, dtype=float), np.asarray(den_list) - events_list]
        )
        g_stat, df, p_value = g_test(g_table)

        rows.append(
            {
                "metric": metric,
                "k_groups": summary["k_groups"],
                "max_min_gap": summary["max_min_gap"],
                "max_min_ci_low": summary["max_min_ci_low"],
                "max_min_ci_high": summary["max_min_ci_high"],
                "max_cell": max_cell,
                "min_cell": min_cell,
                "var_uncorrected": summary["var_uncorrected"],
                "var_single_corrected": summary["var_single_corrected"],
                "var_double_corrected_ci_low": summary["var_double_corrected_ci_low"],
                "var_double_corrected_ci_high": summary["var_double_corrected_ci_high"],
                "sd_uncorrected": summary["sd_uncorrected"],
                "sd_single_corrected": summary["sd_single_corrected"],
                "sd_double_corrected_ci_low": summary["sd_double_corrected_ci_low"],
                "sd_double_corrected_ci_high": summary["sd_double_corrected_ci_high"],
                "g_stat": g_stat,
                "df": df,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows, columns=list(CELL_SUMMARY_COLUMNS))
