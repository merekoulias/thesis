"""Tests for feature aggregations and application cleaning."""

import numpy as np
import pandas as pd

from src.data.loader import HomeCreditData
from src.features.application import clean_application
from src.features.bureau import aggregate_bureau
from src.features.installments import aggregate_installments
from src.features.types import DAYS_SENTINEL, SK_ID_CURR, TARGET_COL


def test_clean_application_sentinel_and_building_dedup(
    tiny_home_credit_data: HomeCreditData,
) -> None:
    data = tiny_home_credit_data
    # Simulate XNA already dropped
    train = data.application_train[
        data.application_train["CODE_GENDER"] != "XNA"
    ].copy()
    test = data.application_test.copy()

    train_x, test_x, y, report = clean_application(train, test, xna_dropped=1)

    assert TARGET_COL not in train_x.columns
    assert TARGET_COL not in test_x.columns
    assert len(y) == len(train_x)
    assert "DAYS_EMPLOYED_ANOM" in train_x.columns
    assert int(train_x["DAYS_EMPLOYED_ANOM"].sum()) == 1
    assert train_x["DAYS_EMPLOYED"].isna().sum() == 1
    assert "APARTMENTS_AVG" in train_x.columns
    assert "APARTMENTS_MEDI" not in train_x.columns
    assert "APARTMENTS_MODE" not in train_x.columns
    assert "HOUSETYPE_MODE" in train_x.columns
    assert report.xna_gender_rows_dropped == 1
    assert "APARTMENTS_MEDI" in report.building_columns_dropped


def test_bureau_aggregates_one_row_per_client(
    tiny_home_credit_data: HomeCreditData,
) -> None:
    feat = aggregate_bureau(
        tiny_home_credit_data.bureau,
        tiny_home_credit_data.bureau_balance,
    )
    assert feat[SK_ID_CURR].is_unique
    assert "HAS_BUREAU_HISTORY" in feat.columns
    assert "BUREAU_CREDIT_COUNT" in feat.columns
    # Client 1 has two bureau credits
    row = feat.loc[feat[SK_ID_CURR] == 1].iloc[0]
    assert row["BUREAU_CREDIT_COUNT"] == 2


def test_installments_collapses_partial_payments(
    tiny_home_credit_data: HomeCreditData,
) -> None:
    feat = aggregate_installments(
        tiny_home_credit_data.installments_payments
    )
    assert feat[SK_ID_CURR].is_unique
    # Client 1: two scheduled installments (partials collapsed)
    row = feat.loc[feat[SK_ID_CURR] == 1].iloc[0]
    assert row["INST_COUNT"] == 2
    assert row["HAS_INSTALLMENT_HISTORY"] == 1


def test_days_sentinel_constant() -> None:
    assert DAYS_SENTINEL == 365243
