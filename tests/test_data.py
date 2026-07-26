"""Tests for Home Credit data loading and validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_home_credit_data
from src.data.validation import DataValidationError, validate_application_pair
from src.features.types import SK_ID_CURR, TARGET_COL


def test_load_drops_xna_and_validates_keys(fixture_csv_dir: Path) -> None:
    data = load_home_credit_data(fixture_csv_dir)
    assert data.xna_gender_rows_dropped == 1
    assert (data.application_train["CODE_GENDER"] == "XNA").sum() == 0
    assert data.application_train[SK_ID_CURR].is_unique
    assert data.application_test[SK_ID_CURR].is_unique
    assert TARGET_COL in data.application_train.columns
    assert TARGET_COL not in data.application_test.columns


def test_application_pair_rejects_target_in_test() -> None:
    train = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "TARGET": [0],
            "CODE_GENDER": ["F"],
            "NAME_EDUCATION_TYPE": ["Higher education"],
        }
    )
    test = pd.DataFrame(
        {
            "SK_ID_CURR": [2],
            "TARGET": [1],
            "CODE_GENDER": ["M"],
            "NAME_EDUCATION_TYPE": ["Higher education"],
        }
    )
    with pytest.raises(DataValidationError, match="must not contain TARGET"):
        validate_application_pair(train, test)


def test_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_home_credit_data(tmp_path)
