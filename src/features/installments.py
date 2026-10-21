"""Installment payment aggregates to one row per applicant."""

import numpy as np
import pandas as pd

from src.features.types import SK_ID_CURR, SK_ID_PREV


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installment payments to SK_ID_CURR.

    Collapses partial payments at
    (SK_ID_PREV, NUM_INSTALMENT_NUMBER) first, then to the client.

    Args:
        installments: installments_payments table.

    Returns:
        One row per SK_ID_CURR with payment-behavior features.
    """
    inst = installments.copy()
    inst["PAYMENT_MISSING"] = (
        inst["DAYS_ENTRY_PAYMENT"].isna() & inst["AMT_PAYMENT"].isna()
    ).astype(np.int8)

    # Delay: entry - due (positive => late). Missing entry => unpaid.
    inst["PAYMENT_DELAY"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["IS_LATE"] = (inst["PAYMENT_DELAY"] > 0).astype(np.int8)

    # Collapse partial payments for the same scheduled installment
    sched_key = [SK_ID_PREV, "NUM_INSTALMENT_NUMBER"]
    # SK_ID_CURR is constant per SK_ID_PREV in this dataset
    per_inst = inst.groupby(sched_key + [SK_ID_CURR], as_index=False).agg(
        AMT_INSTALMENT=("AMT_INSTALMENT", "max"),
        AMT_PAYMENT=("AMT_PAYMENT", "sum"),
        PAYMENT_DELAY=("PAYMENT_DELAY", "max"),
        PAYMENT_MISSING=("PAYMENT_MISSING", "max"),
        N_PARTIALS=("AMT_PAYMENT", "size"),
    )
    per_inst["IS_LATE"] = (per_inst["PAYMENT_DELAY"] > 0).astype(np.int8)
    per_inst["IS_UNDERPAY"] = (
        per_inst["AMT_PAYMENT"] < per_inst["AMT_INSTALMENT"]
    ).astype(np.int8)
    per_inst["PAYMENT_RATIO"] = per_inst["AMT_PAYMENT"] / per_inst[
        "AMT_INSTALMENT"
    ].replace(0, np.nan)

    client = per_inst.groupby(SK_ID_CURR).agg(
        INST_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
        INST_LATE_RATE=("IS_LATE", "mean"),
        INST_UNDERPAY_RATE=("IS_UNDERPAY", "mean"),
        INST_DELAY_MEAN=("PAYMENT_DELAY", "mean"),
        INST_DELAY_MAX=("PAYMENT_DELAY", "max"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INST_PAYMENT_RATIO_MIN=("PAYMENT_RATIO", "min"),
        INST_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        INST_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
        INST_PAYMENT_MISSING_RATE=("PAYMENT_MISSING", "mean"),
        INST_PREV_NUNIQUE=(SK_ID_PREV, "nunique"),
    )
    out = client.reset_index()
    out["HAS_INSTALLMENT_HISTORY"] = 1
    return out
