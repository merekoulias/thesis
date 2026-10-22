"""Conditional demographic parity on PTI quintiles."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.fairness.gaps import pair_mask
from src.fairness.group_metrics import rate
from src.fairness.io import assign_pti_quintiles
from src.fairness.stats import diff_ci, proportion_ci, stratified_g_test
from src.fairness.types import (
    GENDER_AGE_COL,
    PTI_COL,
    PTI_Q_COL,
    Y_COL,
    YHAT_COL,
    AuditConfig,
    PairSpec,
)

logger = logging.getLogger(__name__)

RATIO_METRIC = "approval_rate"

CONDITIONAL_PARITY_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "quintile",
    "n_a",
    "n_b",
    "approval_rate_a",
    "approval_rate_b",
    "diff",
    "diff_ci_low",
    "diff_ci_high",
)

CONDITIONAL_PARITY_TESTS_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "g_stat",
    "df",
    "p_value",
    "unconditional_diff",
    "unconditional_diff_ci_low",
    "unconditional_diff_ci_high",
    "n_excluded_missing_pti",
)

CONDITIONAL_PARITY_CELLS_COLUMNS: Tuple[str, ...] = (
    "group",
    "quintile",
    "n",
    "approved",
    "approval_rate",
    "ci_low",
    "ci_high",
)


def _pair_for_id(config: AuditConfig, pair_id: str) -> PairSpec:
    for pair in config.pairs:
        if pair.pair_id == pair_id:
            return pair
    raise ValueError(f"Unknown pair_id: {pair_id}")


def _approval_rate(y: np.ndarray, yhat: np.ndarray) -> Tuple[float, int, int]:
    events, denom = rate(y, yhat, RATIO_METRIC)
    if denom == 0:
        return float("nan"), events, denom
    return float(events / denom), events, denom


def _check_pti_metadata(frame: pd.DataFrame, config: AuditConfig) -> int:
    """Log warnings when PTI quintiles or missing count differ from config checks."""
    n_missing = int(frame[PTI_COL].isna().sum())
    if n_missing != config.cdp_expected_missing:
        logger.warning(
            "PTI missing count %s != expected %s",
            n_missing,
            config.cdp_expected_missing,
        )
    _, cut_points = assign_pti_quintiles(frame[PTI_COL], config.cdp_n_quantiles)
    rounded = [round(float(x), 3) for x in cut_points]
    expected = config.cdp_expected_cut_points
    if rounded != expected:
        logger.warning(
            "PTI cut points %s != expected %s",
            rounded,
            expected,
        )
    return n_missing


def _unconditional_approval_gap(
    gaps: pd.DataFrame, pair_id: str
) -> Tuple[float, float, float]:
    row = gaps[(gaps["pair_id"] == pair_id) & (gaps["metric"] == RATIO_METRIC)]
    if len(row) != 1:
        raise ValueError(f"Missing approval_rate gap for pair {pair_id}")
    r = row.iloc[0]
    return (
        float(r["diff"]),
        float(r["diff_ci_low"]),
        float(r["diff_ci_high"]),
    )


def conditional_parity(
    frame: pd.DataFrame, config: AuditConfig, gaps: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-quintile approval gaps and stratified G-tests (plan 5.6 CDP tables).

    Rows with unknown ``PTI_Q`` are excluded from quintile tables; the unconditional
    approval gap is copied from ``gaps`` on all rows.

    Args:
        frame: Audit frame with ``y``, ``yhat``, ``PTI``, and ``PTI_Q``.
        config: Audit settings (``cdp_pairs``, quintile count).
        gaps: Output of ``pair_gaps`` (for unconditional approval gaps).

    Returns:
        ``(conditional_parity, conditional_parity_tests)`` DataFrames.
    """
    n_excluded = _check_pti_metadata(frame, config)
    work = frame[frame[PTI_Q_COL].notna()].copy()
    y = work[Y_COL].to_numpy()
    yhat = work[YHAT_COL].to_numpy()
    pti_q = work[PTI_Q_COL].to_numpy()
    z = config.z
    quintiles = [f"Q{i}" for i in range(1, config.cdp_n_quantiles + 1)]

    parity_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for pair_id in config.cdp_pairs:
        pair = _pair_for_id(config, pair_id)
        mask_a_full = pair_mask(work, pair.a)
        mask_b_full = pair_mask(work, pair.b)
        tables: List[np.ndarray] = []

        for q_label in quintiles:
            q_mask = pti_q == q_label
            mask_a = mask_a_full & q_mask
            mask_b = mask_b_full & q_mask
            y_a, h_a = y[mask_a], yhat[mask_a]
            y_b, h_b = y[mask_b], yhat[mask_b]
            n_a = int(mask_a.sum())
            n_b = int(mask_b.sum())
            rate_a, ev_a, den_a = _approval_rate(y_a, h_a)
            rate_b, ev_b, den_b = _approval_rate(y_b, h_b)
            diff, diff_lo, diff_hi = diff_ci(rate_a, den_a, rate_b, den_b, z)
            parity_rows.append(
                {
                    "pair_id": pair_id,
                    "quintile": q_label,
                    "n_a": n_a,
                    "n_b": n_b,
                    "approval_rate_a": rate_a,
                    "approval_rate_b": rate_b,
                    "diff": diff,
                    "diff_ci_low": diff_lo,
                    "diff_ci_high": diff_hi,
                }
            )
            tables.append(
                np.array(
                    [
                        [ev_a, den_a - ev_a],
                        [ev_b, den_b - ev_b],
                    ],
                    dtype=float,
                )
            )

        g_stat, df, p_value = stratified_g_test(tables)
        u_diff, u_lo, u_hi = _unconditional_approval_gap(gaps, pair_id)
        test_rows.append(
            {
                "pair_id": pair_id,
                "g_stat": g_stat,
                "df": df,
                "p_value": p_value,
                "unconditional_diff": u_diff,
                "unconditional_diff_ci_low": u_lo,
                "unconditional_diff_ci_high": u_hi,
                "n_excluded_missing_pti": n_excluded,
            }
        )

    parity_df = pd.DataFrame(parity_rows, columns=list(CONDITIONAL_PARITY_COLUMNS))
    tests_df = pd.DataFrame(test_rows, columns=list(CONDITIONAL_PARITY_TESTS_COLUMNS))
    return parity_df, tests_df


def conditional_parity_cells(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Appendix table: approval rate per gender × age cell and PTI quintile.

    Args:
        frame: Audit frame with ``y``, ``yhat``, ``GENDER_AGE``, and ``PTI_Q``.
        config: Audit settings.

    Returns:
        DataFrame with ``CONDITIONAL_PARITY_CELLS_COLUMNS`` (10 cells × quintiles).
    """
    work = frame[frame[PTI_Q_COL].notna()]
    y = work[Y_COL].to_numpy()
    yhat = work[YHAT_COL].to_numpy()
    cells = work[GENDER_AGE_COL].to_numpy()
    pti_q = work[PTI_Q_COL].to_numpy()
    z = config.z
    quintiles = [f"Q{i}" for i in range(1, config.cdp_n_quantiles + 1)]
    cell_labels = config.groupings[GENDER_AGE_COL]
    rows: List[Dict[str, Any]] = []

    for cell in cell_labels:
        cell_mask = cells == cell
        for q_label in quintiles:
            mask = cell_mask & (pti_q == q_label)
            y_sub = y[mask]
            h_sub = yhat[mask]
            n = int(mask.sum())
            events, denom = rate(y_sub, h_sub, RATIO_METRIC)
            appr, ci_lo, ci_hi = (
                proportion_ci(events, denom, z)
                if denom > 0
                else (float("nan"), float("nan"), float("nan"))
            )
            rows.append(
                {
                    "group": cell,
                    "quintile": q_label,
                    "n": n,
                    "approved": events,
                    "approval_rate": appr,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                }
            )

    return pd.DataFrame(rows, columns=list(CONDITIONAL_PARITY_CELLS_COLUMNS))
