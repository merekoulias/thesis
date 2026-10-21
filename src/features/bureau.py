"""Bureau and bureau_balance aggregates to one row per applicant."""

from typing import List

import numpy as np
import pandas as pd

from src.features.types import SK_ID_BUREAU, SK_ID_CURR


def _aggregate_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly bureau balances to SK_ID_BUREAU."""
    work = bb.copy()
    work["BB_MONTHS"] = 1
    work["BB_STATUS_C"] = (work["STATUS"] == "C").astype(np.int8)
    work["BB_STATUS_X"] = (work["STATUS"] == "X").astype(np.int8)
    work["BB_STATUS_0"] = (work["STATUS"] == "0").astype(np.int8)
    work["BB_STATUS_DPD"] = (
        work["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.int8)
    )

    # Freshest month: MONTHS_BALANCE closest to 0
    idx = work.groupby(SK_ID_BUREAU)["MONTHS_BALANCE"].idxmax()
    latest = work.loc[idx, [SK_ID_BUREAU, "MONTHS_BALANCE", "STATUS"]]
    latest = latest.rename(
        columns={
            "MONTHS_BALANCE": "BB_LATEST_MONTH",
            "STATUS": "BB_LATEST_STATUS",
        }
    )

    agg = work.groupby(SK_ID_BUREAU).agg(
        BB_HISTORY_LEN=("BB_MONTHS", "sum"),
        BB_STATUS_C_RATE=("BB_STATUS_C", "mean"),
        BB_STATUS_X_RATE=("BB_STATUS_X", "mean"),
        BB_STATUS_0_RATE=("BB_STATUS_0", "mean"),
        BB_STATUS_DPD_RATE=("BB_STATUS_DPD", "mean"),
        BB_MIN_MONTH=("MONTHS_BALANCE", "min"),
    )
    out = agg.reset_index().merge(latest, on=SK_ID_BUREAU, how="left")
    return out


def _onehot_counts(
    df: pd.DataFrame, col: str, prefix: str, id_col: str
) -> pd.DataFrame:
    """One-hot encode a column then sum counts per id_col."""
    dummies = pd.get_dummies(df[col], prefix=prefix, dtype=np.int8)
    tmp = pd.concat([df[[id_col]], dummies], axis=1)
    return tmp.groupby(id_col).sum()


def aggregate_bureau(
    bureau: pd.DataFrame, bureau_balance: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate bureau (+ balance) features to SK_ID_CURR.

    Args:
        bureau: External credit bureau credits.
        bureau_balance: Monthly status history for bureau credits.

    Returns:
        One row per SK_ID_CURR with bureau aggregates and history flags.
    """
    bb_feat = _aggregate_bureau_balance(bureau_balance)
    bur = bureau.merge(bb_feat, on=SK_ID_BUREAU, how="left")
    bur["HAS_BUREAU_BALANCE"] = bur["BB_HISTORY_LEN"].notna().astype(np.int8)

    # Flag / clip anomalous negative debt and limit values
    for col in ("AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT"):
        if col in bur.columns:
            bur[f"{col}_NEG"] = (bur[col] < 0).astype(np.int8)
            bur[col] = bur[col].clip(lower=0)

    active = (bur["CREDIT_ACTIVE"] == "Active").astype(np.int8)
    closed = (bur["CREDIT_ACTIVE"] == "Closed").astype(np.int8)
    bur["BUREAU_ACTIVE"] = active
    bur["BUREAU_CLOSED"] = closed
    bur["BUREAU_OVERDUE"] = (bur["CREDIT_DAY_OVERDUE"] > 0).astype(np.int8)

    numeric_aggs = {
        SK_ID_BUREAU: "count",
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_LIMIT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "AMT_ANNUITY": ["sum", "mean"],
        "CNT_CREDIT_PROLONG": ["sum", "mean"],
        "BUREAU_ACTIVE": ["sum", "mean"],
        "BUREAU_CLOSED": ["sum", "mean"],
        "BUREAU_OVERDUE": ["sum", "mean"],
        "HAS_BUREAU_BALANCE": ["mean", "sum"],
        "BB_HISTORY_LEN": ["mean", "sum"],
        "BB_STATUS_DPD_RATE": ["mean", "max"],
        "BB_STATUS_0_RATE": ["mean"],
        "AMT_CREDIT_SUM_DEBT_NEG": ["sum"],
        "AMT_CREDIT_SUM_LIMIT_NEG": ["sum"],
    }
    # Keep only available columns
    use_aggs = {
        k: v for k, v in numeric_aggs.items() if k == SK_ID_BUREAU or k in bur.columns
    }
    num = bur.groupby(SK_ID_CURR).agg(use_aggs)
    num.columns = [
        f"BUREAU_{a}_{b}".upper() if isinstance(b, str) else f"BUREAU_{a}"
        for a, b in num.columns.to_flat_index()
    ]
    # Fix count column name
    count_cols = [c for c in num.columns if c.startswith("BUREAU_SK_ID")]
    if count_cols:
        num = num.rename(columns={count_cols[0]: "BUREAU_CREDIT_COUNT"})

    cat_parts: List[pd.DataFrame] = [num]
    if "CREDIT_ACTIVE" in bur.columns:
        cat_parts.append(
            _onehot_counts(bur, "CREDIT_ACTIVE", "BUREAU_CREDIT_ACTIVE", SK_ID_CURR)
        )
    if "CREDIT_TYPE" in bur.columns:
        # Limit rare types explosion: top categories only via get_dummies
        cat_parts.append(
            _onehot_counts(bur, "CREDIT_TYPE", "BUREAU_CREDIT_TYPE", SK_ID_CURR)
        )

    out = cat_parts[0]
    for part in cat_parts[1:]:
        out = out.join(part, how="outer")

    out = out.reset_index()
    out["HAS_BUREAU_HISTORY"] = 1
    return out
