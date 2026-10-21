"""Validation helpers for Home Credit relational tables."""

from typing import Iterable, List, Optional

import pandas as pd

from src.constants import REQUIRED_KEYS, SK_ID_CURR, TARGET_COL


class DataValidationError(ValueError):
    """Raised when a Home Credit table fails schema or key checks."""


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing required columns: {missing}")


def validate_table(
    df: pd.DataFrame,
    name: str,
    required_columns: Optional[List[str]] = None,
    unique_key: Optional[str] = None,
) -> None:
    """Validate required columns and optional uniqueness of a key.

    Args:
        df: Loaded table.
        name: Logical table name for error messages.
        required_columns: Columns that must be present.
        unique_key: Column that must be unique and non-null when set.

    Raises:
        DataValidationError: If schema or key constraints fail.
    """
    if df.empty:
        raise DataValidationError(f"{name}: table is empty")

    cols = required_columns or REQUIRED_KEYS.get(name, [])
    _require_columns(df, cols, name)

    if unique_key is not None:
        if unique_key not in df.columns:
            raise DataValidationError(f"{name}: unique key {unique_key!r} not found")
        if df[unique_key].isna().any():
            raise DataValidationError(
                f"{name}: unique key {unique_key!r} contains nulls"
            )
        if df[unique_key].duplicated().any():
            n_dup = int(df[unique_key].duplicated().sum())
            raise DataValidationError(
                f"{name}: unique key {unique_key!r} has {n_dup} duplicates"
            )


def validate_application_pair(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Validate train/test application tables against FE contracts.

    Args:
        train: Application train frame.
        test: Application test frame.

    Raises:
        DataValidationError: On schema or key failures.
    """
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
    if TARGET_COL in test.columns:
        raise DataValidationError("application_test must not contain TARGET")
    train_feats = set(train.columns) - {TARGET_COL}
    test_feats = set(test.columns)
    if train_feats != test_feats:
        only_train = sorted(train_feats - test_feats)
        only_test = sorted(test_feats - train_feats)
        raise DataValidationError(
            "application train/test feature schemas differ; "
            f"only_train={only_train[:10]}, only_test={only_test[:10]}"
        )
