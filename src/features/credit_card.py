"""Credit-card monthly balance aggregates to one row per applicant."""

import numpy as np
import pandas as pd

from src.features.types import SK_ID_CURR, SK_ID_PREV


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance to SK_ID_CURR.

    Sparse coverage is surfaced via HAS_CREDIT_CARD_HISTORY.
    Keeps one receivable field (AMT_TOTAL_RECEIVABLE).

    Args:
        cc: credit_card_balance table.

    Returns:
        One row per SK_ID_CURR with utilization and DPD features.
    """
    work = cc.copy()
    work["CC_DPD_FLAG"] = (work["SK_DPD"] > 0).astype(np.int8)
    work["CC_DPD_DEF_FLAG"] = (work["SK_DPD_DEF"] > 0).astype(np.int8)

    # Utilization where limit > 0
    limit = work["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    work["CC_UTILIZATION"] = work["AMT_BALANCE"] / limit
    work["CC_OVER_LIMIT"] = (work["CC_UTILIZATION"] > 1).astype(np.int8)

    # Payment below minimum due
    if (
        "AMT_PAYMENT_CURRENT" in work.columns
        and "AMT_INST_MIN_REGULARITY" in work.columns
    ):
        comparable = work["AMT_INST_MIN_REGULARITY"].notna() & work[
            "AMT_PAYMENT_CURRENT"
        ].notna()
        below = comparable & (
            work["AMT_PAYMENT_CURRENT"] < work["AMT_INST_MIN_REGULARITY"]
        )
        work["CC_BELOW_MIN"] = below.astype(np.int8)
    else:
        work["CC_BELOW_MIN"] = 0

    work["CC_HAS_DRAWING"] = (
        work.get("AMT_DRAWINGS_CURRENT", 0).fillna(0) > 0
    ).astype(np.int8)

    # Clip negative balances / receivables for stable aggregates
    for col in ("AMT_BALANCE", "AMT_TOTAL_RECEIVABLE"):
        if col in work.columns:
            work[f"{col}_NEG"] = (work[col] < 0).astype(np.int8)
            work[col] = work[col].clip(lower=0)

    prev = work.groupby([SK_ID_PREV, SK_ID_CURR], as_index=False).agg(
        CC_HISTORY_LEN=("MONTHS_BALANCE", "count"),
        CC_MONTH_MIN=("MONTHS_BALANCE", "min"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_UTIL_MEAN=("CC_UTILIZATION", "mean"),
        CC_UTIL_MAX=("CC_UTILIZATION", "max"),
        CC_OVER_LIMIT_RATE=("CC_OVER_LIMIT", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
        CC_DPD_MEAN=("SK_DPD", "mean"),
        CC_DPD_RATE=("CC_DPD_FLAG", "mean"),
        CC_DPD_DEF_RATE=("CC_DPD_DEF_FLAG", "mean"),
        CC_DRAWING_RATE=("CC_HAS_DRAWING", "mean"),
        CC_BELOW_MIN_RATE=("CC_BELOW_MIN", "mean"),
        CC_RECEIVABLE_MEAN=("AMT_TOTAL_RECEIVABLE", "mean"),
        CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
    )

    client = prev.groupby(SK_ID_CURR).agg(
        {
            "CC_HISTORY_LEN": ["sum", "mean", "max"],
            "CC_BALANCE_MEAN": ["mean"],
            "CC_BALANCE_MAX": ["max"],
            "CC_LIMIT_MEAN": ["mean"],
            "CC_UTIL_MEAN": ["mean"],
            "CC_UTIL_MAX": ["max"],
            "CC_OVER_LIMIT_RATE": ["mean"],
            "CC_DPD_MAX": ["max"],
            "CC_DPD_MEAN": ["mean"],
            "CC_DPD_RATE": ["mean"],
            "CC_DPD_DEF_RATE": ["mean"],
            "CC_DRAWING_RATE": ["mean"],
            "CC_BELOW_MIN_RATE": ["mean"],
            "CC_RECEIVABLE_MEAN": ["mean"],
            "CC_DRAWINGS_MEAN": ["mean"],
            SK_ID_PREV: ["nunique"],
        }
    )
    client.columns = [
        f"{a}_{b}".upper() for a, b in client.columns.to_flat_index()
    ]
    rename = {
        c: "CC_PREV_COUNT"
        for c in client.columns
        if "SK_ID_PREV" in c
    }
    client = client.rename(columns=rename)
    out = client.reset_index()
    out["HAS_CREDIT_CARD_HISTORY"] = 1
    return out
