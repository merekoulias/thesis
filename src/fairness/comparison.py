"""Label-implied benchmark, expected signs, Kendall tau, and proxy importance."""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.fairness.gaps import pair_mask
from src.fairness.types import GENDER_AGE_COL, Y_COL, YHAT_COL, AuditConfig

ORACLE_BENCHMARK_COLUMNS: tuple = (
    "pair_id",
    "oracle_approval_a",
    "oracle_approval_b",
    "oracle_diff",
    "oracle_dir",
    "model_diff",
    "model_dir",
    "model_dir_ci_low",
    "model_dir_ci_high",
    "direction",
    "oracle_in_model_ci",
)

EXPECTED_SIGNS_COLUMNS: tuple = (
    "pair_id",
    "base_rate_a",
    "base_rate_b",
    "expected_sign_eod",
    "observed_eod",
    "eod_ci_excludes_0",
    "eod_matches",
    "expected_sign_dfnr",
    "observed_dfnr",
    "dfnr_ci_excludes_0",
    "dfnr_matches",
)

PROXY_IMPORTANCE_COLUMNS: tuple = (
    "feature_group",
    "n_features",
    "importance_share",
    "best_feature",
    "best_feature_rank",
    "importance_type",
)

OTHER_FEATURE_GROUP = "other"
MLP_MODEL = "mlp"


def _base_rate_side(frame: pd.DataFrame, side: Dict[str, str]) -> float:
    mask = pair_mask(frame, side)
    y = frame.loc[mask, Y_COL].to_numpy()
    if y.size == 0:
        return float("nan")
    return float(y.mean())


def _direction_label(model_dir: float, oracle_dir: float) -> str:
    if np.isnan(model_dir) or np.isnan(oracle_dir):
        return "equal"
    dm = model_dir - 1.0
    do = oracle_dir - 1.0
    if dm * do < 0:
        return "reversed"
    if abs(dm) > abs(do):
        return "wider"
    if np.isclose(abs(dm), abs(do)):
        return "equal"
    return "narrower"


def oracle_benchmark(
    frame: pd.DataFrame, config: AuditConfig, gaps: pd.DataFrame
) -> pd.DataFrame:
    """Compare model approval DIR gaps to label-implied oracle approval rates.

    Args:
        frame: Audit frame with ``y``.
        config: Audit settings (pair list).
        gaps: Output of ``pair_gaps`` (uses ``approval_rate`` rows).

    Returns:
        DataFrame with ``ORACLE_BENCHMARK_COLUMNS``.
    """
    approval = gaps[gaps["metric"] == "approval_rate"].set_index("pair_id")
    rows: List[Dict[str, Any]] = []

    for pair in config.pairs:
        br_a = _base_rate_side(frame, pair.a)
        br_b = _base_rate_side(frame, pair.b)
        oracle_a = 1.0 - br_a
        oracle_b = 1.0 - br_b
        oracle_diff = oracle_a - oracle_b
        oracle_dir = (
            oracle_a / oracle_b
            if oracle_b != 0 and not np.isnan(oracle_b)
            else float("nan")
        )

        if pair.pair_id in approval.index:
            gap_row = approval.loc[pair.pair_id]
            model_diff = float(gap_row["diff"])
            model_dir = float(gap_row["ratio"])
            model_dir_ci_low = float(gap_row["ratio_ci_low"])
            model_dir_ci_high = float(gap_row["ratio_ci_high"])
        else:
            model_diff = model_dir = model_dir_ci_low = model_dir_ci_high = float("nan")

        if (
            np.isnan(model_dir_ci_low)
            or np.isnan(model_dir_ci_high)
            or np.isnan(oracle_dir)
        ):
            oracle_in_ci = False
        else:
            oracle_in_ci = bool(model_dir_ci_low <= oracle_dir <= model_dir_ci_high)

        rows.append(
            {
                "pair_id": pair.pair_id,
                "oracle_approval_a": oracle_a,
                "oracle_approval_b": oracle_b,
                "oracle_diff": oracle_diff,
                "oracle_dir": oracle_dir,
                "model_diff": model_diff,
                "model_dir": model_dir,
                "model_dir_ci_low": model_dir_ci_low,
                "model_dir_ci_high": model_dir_ci_high,
                "direction": _direction_label(model_dir, oracle_dir),
                "oracle_in_model_ci": oracle_in_ci,
            }
        )

    return pd.DataFrame(rows, columns=list(ORACLE_BENCHMARK_COLUMNS))


def _ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return False
    return bool(ci_low > 0.0 or ci_high < 0.0)


def expected_signs(
    frame: pd.DataFrame, config: AuditConfig, gaps: pd.DataFrame
) -> pd.DataFrame:
    """Chouldechova expected signs for FPR and FNR gaps vs base-rate ordering.

    Args:
        frame: Audit frame with ``y``.
        config: Audit settings (pair list).
        gaps: Output of ``pair_gaps``.

    Returns:
        DataFrame with ``EXPECTED_SIGNS_COLUMNS``.
    """
    fpr_gaps = gaps[gaps["metric"] == "fpr"].set_index("pair_id")
    fnr_gaps = gaps[gaps["metric"] == "fnr"].set_index("pair_id")
    rows: List[Dict[str, Any]] = []

    for pair in config.pairs:
        br_a = _base_rate_side(frame, pair.a)
        br_b = _base_rate_side(frame, pair.b)
        expected_eod = int(np.sign(br_a - br_b))
        expected_dfnr = int(-expected_eod)

        if pair.pair_id in fpr_gaps.index:
            eod_row = fpr_gaps.loc[pair.pair_id]
            observed_eod = float(eod_row["diff"])
            eod_excludes = _ci_excludes_zero(
                float(eod_row["diff_ci_low"]), float(eod_row["diff_ci_high"])
            )
        else:
            observed_eod = float("nan")
            eod_excludes = False

        if pair.pair_id in fnr_gaps.index:
            dfnr_row = fnr_gaps.loc[pair.pair_id]
            observed_dfnr = float(dfnr_row["diff"])
            dfnr_excludes = _ci_excludes_zero(
                float(dfnr_row["diff_ci_low"]), float(dfnr_row["diff_ci_high"])
            )
        else:
            observed_dfnr = float("nan")
            dfnr_excludes = False

        rows.append(
            {
                "pair_id": pair.pair_id,
                "base_rate_a": br_a,
                "base_rate_b": br_b,
                "expected_sign_eod": expected_eod,
                "observed_eod": observed_eod,
                "eod_ci_excludes_0": eod_excludes,
                "eod_matches": bool(np.sign(observed_eod) == expected_eod),
                "expected_sign_dfnr": expected_dfnr,
                "observed_dfnr": observed_dfnr,
                "dfnr_ci_excludes_0": dfnr_excludes,
                "dfnr_matches": bool(np.sign(observed_dfnr) == expected_dfnr),
            }
        )

    return pd.DataFrame(rows, columns=list(EXPECTED_SIGNS_COLUMNS))


def kendall_cells(frame: pd.DataFrame, config: AuditConfig) -> Dict[str, Any]:
    """Kendall tau-b between cell base rates and FPR over the 10 gender × age cells.

    Args:
        frame: Audit frame with ``y``, ``yhat``, and ``GENDER_AGE``.
        config: Audit settings (cell label order in ``groupings``).

    Returns:
        Dict with ``tau``, ``p_value``, ``n_cells``, and ``cells`` (per-cell rates).
    """
    from src.fairness.group_metrics import rate

    y = frame[Y_COL].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    cell_labels = config.groupings[GENDER_AGE_COL]
    gender_age = frame[GENDER_AGE_COL].to_numpy()

    base_rates: List[float] = []
    fprs: List[float] = []
    cells: List[Dict[str, Any]] = []

    for label in cell_labels:
        mask = gender_age == label
        y_g = y[mask]
        h_g = yhat[mask]
        n = int(y_g.size)
        base_rate = float(y_g.mean()) if n else float("nan")
        events, den = rate(y_g, h_g, "fpr")
        fpr = float(events / den) if den else float("nan")
        base_rates.append(base_rate)
        fprs.append(fpr)
        cells.append({"group": label, "base_rate": base_rate, "fpr": fpr})

    br_arr = np.asarray(base_rates, dtype=float)
    fpr_arr = np.asarray(fprs, dtype=float)
    valid = ~(np.isnan(br_arr) | np.isnan(fpr_arr))
    if valid.sum() < 2:
        tau = p_value = float("nan")
    else:
        result = kendalltau(br_arr[valid], fpr_arr[valid])
        tau = float(result.statistic)
        p_value = float(result.pvalue)

    return {
        "tau": tau,
        "p_value": p_value,
        "n_cells": len(cell_labels),
        "cells": cells,
    }


def _assign_watch_group(feature: str, watch_list: Dict[str, List[str]]) -> str:
    for group_name, patterns in watch_list.items():
        for pattern in patterns:
            if re.search(pattern, feature):
                return group_name
    return OTHER_FEATURE_GROUP


def proxy_importance(
    importance: Optional[pd.DataFrame],
    config: AuditConfig,
    model: str,
) -> Optional[pd.DataFrame]:
    """Summarise feature importances by the configured proxy watch list.

    Args:
        importance: Saved importance table, or ``None`` when unavailable.
        config: Audit settings (watch list and per-model importance type).
        model: Registry model name.

    Returns:
        Summary table with ``PROXY_IMPORTANCE_COLUMNS``, or ``None`` for ``mlp``.
    """
    if model == MLP_MODEL:
        return None
    if importance is None:
        return None

    imp_type = config.importance_type[model]
    subset = importance[importance["importance_type"] == imp_type].copy()
    if imp_type == "coef":
        subset["importance"] = subset["importance"].abs()
    avg = (
        subset.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    avg["feature_group"] = avg["feature"].map(
        lambda f: _assign_watch_group(f, config.proxy_watch_list)
    )
    total = float(avg["importance"].sum())
    if total == 0.0:
        total = float("nan")

    avg["rank_overall"] = np.arange(1, len(avg) + 1)

    group_order = list(config.proxy_watch_list.keys()) + [OTHER_FEATURE_GROUP]
    rows: List[Dict[str, Any]] = []
    for group_name in group_order:
        grp = avg[avg["feature_group"] == group_name]
        if grp.empty:
            continue
        best = grp.iloc[0]
        share = (
            float(grp["importance"].sum() / total)
            if not np.isnan(total)
            else float("nan")
        )
        rows.append(
            {
                "feature_group": group_name,
                "n_features": int(len(grp)),
                "importance_share": share,
                "best_feature": best["feature"],
                "best_feature_rank": int(best["rank_overall"]),
                "importance_type": imp_type,
            }
        )

    return pd.DataFrame(rows, columns=list(PROXY_IMPORTANCE_COLUMNS))
