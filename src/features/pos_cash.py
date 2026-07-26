"""POS / cash monthly balance aggregates to one row per applicant."""

import numpy as np
import pandas as pd

from src.features.types import SK_ID_CURR, SK_ID_PREV


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance to SK_ID_CURR via SK_ID_PREV.

    Args:
        pos: POS_CASH_balance table.

    Returns:
        One row per SK_ID_CURR with DPD and term-progress features.
    """
    work = pos.copy()
    work["POS_DPD_FLAG"] = (work["SK_DPD"] > 0).astype(np.int8)
    work["POS_DPD_DEF_FLAG"] = (work["SK_DPD_DEF"] > 0).astype(np.int8)
    work["POS_COMPLETED"] = (
        work["NAME_CONTRACT_STATUS"] == "Completed"
    ).astype(np.int8)
    work["POS_ACTIVE"] = (
        work["NAME_CONTRACT_STATUS"] == "Active"
    ).astype(np.int8)

    # Previous-application level
    prev = work.groupby([SK_ID_PREV, SK_ID_CURR], as_index=False).agg(
        POS_HISTORY_LEN=("MONTHS_BALANCE", "count"),
        POS_MONTH_MIN=("MONTHS_BALANCE", "min"),
        POS_MONTH_MAX=("MONTHS_BALANCE", "max"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_DPD_RATE=("POS_DPD_FLAG", "mean"),
        POS_DPD_DEF_RATE=("POS_DPD_DEF_FLAG", "mean"),
        POS_COMPLETED_RATE=("POS_COMPLETED", "mean"),
        POS_ACTIVE_RATE=("POS_ACTIVE", "mean"),
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_CNT_INSTALMENT_FUTURE_MIN=("CNT_INSTALMENT_FUTURE", "min"),
    )

    # Freshest status per previous application
    idx = work.groupby(SK_ID_PREV)["MONTHS_BALANCE"].idxmax()
    latest = work.loc[
        idx, [SK_ID_PREV, "NAME_CONTRACT_STATUS", "SK_DPD"]
    ].rename(
        columns={
            "NAME_CONTRACT_STATUS": "POS_LATEST_STATUS",
            "SK_DPD": "POS_LATEST_DPD",
        }
    )
    prev = prev.merge(latest, on=SK_ID_PREV, how="left")

    status_dummies = pd.get_dummies(
        prev["POS_LATEST_STATUS"], prefix="POS_LATEST", dtype=np.int8
    )
    prev = pd.concat(
        [prev.drop(columns=["POS_LATEST_STATUS"]), status_dummies],
        axis=1,
    )

    # Client level
    num_cols = [
        c
        for c in prev.columns
        if c not in (SK_ID_PREV, SK_ID_CURR)
        and pd.api.types.is_numeric_dtype(prev[c])
    ]
    client = prev.groupby(SK_ID_CURR)[num_cols].agg(
        ["mean", "max", "sum"]
    )
    client.columns = [
        f"{a}_{b}".upper() for a, b in client.columns.to_flat_index()
    ]
    client["POS_PREV_COUNT"] = prev.groupby(SK_ID_CURR)[SK_ID_PREV].nunique()
    out = client.reset_index()
    out["HAS_POS_CASH_HISTORY"] = 1
    return out
