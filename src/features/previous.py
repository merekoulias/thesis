"""Previous-application aggregates to one row per applicant."""

from typing import List

import numpy as np
import pandas as pd

from src.features.types import DAYS_SENTINEL, SK_ID_CURR, SK_ID_PREV


_PREV_DAY_COLS = (
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
)


def _replace_prev_day_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 365243 date sentinels with NaN and add indicators."""
    out = df.copy()
    for col in _PREV_DAY_COLS:
        if col not in out.columns:
            continue
        anom = out[col] == DAYS_SENTINEL
        out[f"{col}_ANOM"] = anom.astype(np.int8)
        out.loc[anom, col] = np.nan
    if "SELLERPLACE_AREA" in out.columns:
        out["SELLERPLACE_AREA_MISSING"] = (
            out["SELLERPLACE_AREA"] == -1
        ).astype(np.int8)
        out.loc[out["SELLERPLACE_AREA"] == -1, "SELLERPLACE_AREA"] = np.nan
    return out


def _onehot_counts(
    df: pd.DataFrame, col: str, prefix: str
) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col], prefix=prefix, dtype=np.int8)
    tmp = pd.concat([df[[SK_ID_CURR]], dummies], axis=1)
    return tmp.groupby(SK_ID_CURR).sum()


def aggregate_previous(previous: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous Home Credit applications to SK_ID_CURR.

    Args:
        previous: previous_application table.

    Returns:
        One row per SK_ID_CURR with outcome, amount, and recency features.
    """
    prev = _replace_prev_day_sentinels(previous)

    if "AMT_APPLICATION" in prev.columns and "AMT_CREDIT" in prev.columns:
        prev["PREV_CREDIT_APP_RATIO"] = prev["AMT_CREDIT"] / prev[
            "AMT_APPLICATION"
        ].replace(0, np.nan)

    # High-missing interest fields: use missing indicators only
    for col in ("RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED"):
        if col in prev.columns:
            prev[f"{col}_MISSING"] = prev[col].isna().astype(np.int8)

    for col in ("AMT_DOWN_PAYMENT", "RATE_DOWN_PAYMENT"):
        if col in prev.columns:
            prev[f"{col}_MISSING"] = prev[col].isna().astype(np.int8)

    numeric_aggs = {
        SK_ID_PREV: "count",
        "DAYS_DECISION": ["min", "max", "mean"],
        "AMT_ANNUITY": ["sum", "mean", "max"],
        "AMT_APPLICATION": ["sum", "mean", "max"],
        "AMT_CREDIT": ["sum", "mean", "max"],
        "AMT_DOWN_PAYMENT": ["sum", "mean"],
        "AMT_GOODS_PRICE": ["sum", "mean", "max"],
        "CNT_PAYMENT": ["mean", "max", "sum"],
        "PREV_CREDIT_APP_RATIO": ["mean", "min", "max"],
        "RATE_DOWN_PAYMENT": ["mean"],
        "SELLERPLACE_AREA": ["mean", "max"],
        "SELLERPLACE_AREA_MISSING": ["mean"],
        "RATE_INTEREST_PRIMARY_MISSING": ["mean"],
        "RATE_INTEREST_PRIVILEGED_MISSING": ["mean"],
        "AMT_DOWN_PAYMENT_MISSING": ["mean"],
        "RATE_DOWN_PAYMENT_MISSING": ["mean"],
    }
    for col in _PREV_DAY_COLS:
        if col in prev.columns:
            numeric_aggs[col] = ["mean", "min"]
            anom = f"{col}_ANOM"
            if anom in prev.columns:
                numeric_aggs[anom] = ["mean"]

    use_aggs = {
        k: v
        for k, v in numeric_aggs.items()
        if k == SK_ID_PREV or k in prev.columns
    }
    num = prev.groupby(SK_ID_CURR).agg(use_aggs)
    num.columns = [
        f"PREV_{a}_{b}".upper()
        for a, b in num.columns.to_flat_index()
    ]
    count_cols = [c for c in num.columns if "SK_ID_PREV" in c]
    if count_cols:
        num = num.rename(columns={count_cols[0]: "PREV_APP_COUNT"})

    cat_parts: List[pd.DataFrame] = [num]
    for col, prefix in (
        ("NAME_CONTRACT_STATUS", "PREV_STATUS"),
        ("NAME_CONTRACT_TYPE", "PREV_CONTRACT"),
        ("NAME_CLIENT_TYPE", "PREV_CLIENT"),
        ("NAME_PRODUCT_TYPE", "PREV_PRODUCT"),
        ("CHANNEL_TYPE", "PREV_CHANNEL"),
    ):
        if col in prev.columns:
            cat_parts.append(_onehot_counts(prev, col, prefix))

    out = cat_parts[0]
    for part in cat_parts[1:]:
        out = out.join(part, how="outer")

    out = out.reset_index()
    out["HAS_PREV_APP_HISTORY"] = 1
    return out
