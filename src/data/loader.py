"""Load and validate Home Credit CSVs for feature engineering."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from src.data.validation import (
    DataValidationError,
    validate_application_pair,
    validate_table,
)
from src.constants import (
    REQUIRED_KEYS,
    SK_ID_BUREAU,
    SK_ID_CURR,
    TABLE_FILES,
)

PathLike = Union[str, Path]


@dataclass
class HomeCreditData:
    """Validated Home Credit relational tables ready for FE."""

    application_train: pd.DataFrame
    application_test: pd.DataFrame
    bureau: pd.DataFrame
    bureau_balance: pd.DataFrame
    previous_application: pd.DataFrame
    installments_payments: pd.DataFrame
    pos_cash: pd.DataFrame
    credit_card: pd.DataFrame
    xna_gender_rows_dropped: int = 0


def _resolve_data_dir(data_dir: PathLike) -> Path:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {path}")
    return path


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def _drop_xna_gender(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Drop rows with CODE_GENDER == 'XNA' from train and test.

    This is an exclusion forced by the data, applied before training and
    evaluation: XNA is undefined, too rare (4 train rows, 0 test rows, all
    TARGET = 0) for any group metric, and is not redistributed to F/M.
    """
    train_mask = train["CODE_GENDER"] == "XNA"
    test_mask = test["CODE_GENDER"] == "XNA"
    n_dropped = int(train_mask.sum() + test_mask.sum())
    train_clean = train.loc[~train_mask].copy()
    test_clean = test.loc[~test_mask].copy()
    return train_clean, test_clean, n_dropped


def load_home_credit_data(
    data_dir: PathLike = "data",
    nrows: Optional[Dict[str, int]] = None,
) -> HomeCreditData:
    """Load all Home Credit tables, validate keys, drop XNA gender.

    Args:
        data_dir: Directory containing Home Credit CSVs.
        nrows: Optional per-table row limits for smoke tests
            (keys match TABLE_FILES logical names).

    Returns:
        HomeCreditData with validated frames.

    Raises:
        FileNotFoundError: If a required CSV is missing.
        DataValidationError: If key or schema checks fail.
    """
    root = _resolve_data_dir(data_dir)
    limits = nrows or {}
    frames: Dict[str, pd.DataFrame] = {}

    for name, filename in TABLE_FILES.items():
        path = root / filename
        limit = limits.get(name)
        frames[name] = _read_csv(path, nrows=limit)

    validate_application_pair(
        frames["application_train"], frames["application_test"]
    )

    train, test, n_xna = _drop_xna_gender(
        frames["application_train"], frames["application_test"]
    )
    frames["application_train"] = train
    frames["application_test"] = test

    # Re-validate uniqueness after XNA drop
    validate_table(
        train,
        "application_train",
        required_columns=REQUIRED_KEYS["application_train"],
        unique_key=SK_ID_CURR,
    )
    validate_table(
        test,
        "application_test",
        required_columns=REQUIRED_KEYS["application_test"],
        unique_key=SK_ID_CURR,
    )

    validate_table(
        frames["bureau"],
        "bureau",
        required_columns=REQUIRED_KEYS["bureau"],
        unique_key=SK_ID_BUREAU,
    )
    validate_table(
        frames["bureau_balance"],
        "bureau_balance",
        required_columns=REQUIRED_KEYS["bureau_balance"],
    )
    validate_table(
        frames["previous_application"],
        "previous_application",
        required_columns=REQUIRED_KEYS["previous_application"],
        unique_key="SK_ID_PREV",
    )
    for child in (
        "installments_payments",
        "pos_cash",
        "credit_card",
    ):
        validate_table(
            frames[child],
            child,
            required_columns=REQUIRED_KEYS[child],
        )

    if n_xna < 0:
        raise DataValidationError("XNA drop count cannot be negative")

    return HomeCreditData(
        application_train=frames["application_train"],
        application_test=frames["application_test"],
        bureau=frames["bureau"],
        bureau_balance=frames["bureau_balance"],
        previous_application=frames["previous_application"],
        installments_payments=frames["installments_payments"],
        pos_cash=frames["pos_cash"],
        credit_card=frames["credit_card"],
        xna_gender_rows_dropped=n_xna,
    )
