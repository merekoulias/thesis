"""Application-table cleaning and applicant base preparation."""

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.features.types import (
    AGE_BAND_COL,
    AGE_BAND_EDGES,
    AGE_YEARS_COL,
    DAYS_PER_YEAR,
    DAYS_SENTINEL,
    INTERSECTION_COL,
    PROTECTED_ATTRIBUTES,
    SK_ID_CURR,
    TARGET_COL,
    CleaningReport,
)


def _building_columns_to_drop(columns: List[str]) -> List[str]:
    """Drop _MEDI/_MODE building variants when a matching _AVG exists."""
    col_set = set(columns)
    to_drop: List[str] = []
    for col in columns:
        if col.endswith("_AVG"):
            stem = col[: -len("_AVG")]
            for suffix in ("_MEDI", "_MODE"):
                sibling = f"{stem}{suffix}"
                if sibling in col_set:
                    to_drop.append(sibling)
    return sorted(set(to_drop))


def _replace_days_employed_sentinel(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Replace DAYS_EMPLOYED sentinel with NaN and add anomaly flag."""
    out = df.copy()
    if "DAYS_EMPLOYED" not in out.columns:
        return out, 0
    anom = out["DAYS_EMPLOYED"] == DAYS_SENTINEL
    n_anom = int(anom.sum())
    out["DAYS_EMPLOYED_ANOM"] = anom.astype(np.int8)
    out.loc[anom, "DAYS_EMPLOYED"] = np.nan
    return out, n_anom


def assign_age_band(days_birth: pd.Series) -> pd.Series:
    """Map DAYS_BIRTH to left-closed 10-year age bands.

    Age in years is ``-DAYS_BIRTH / 365.25``. Bands follow ``AGE_BAND_EDGES``
    and are labelled ``"20-29"``, ``"30-39"``, and so on.

    Args:
        days_birth: Age in days at application, as negative integers.

    Returns:
        String band labels aligned to ``days_birth``.

    Raises:
        ValueError: If any age is missing or outside the band edges.
    """
    age_years = -days_birth / DAYS_PER_YEAR
    labels = [
        f"{lo}-{hi - 1}" for lo, hi in zip(AGE_BAND_EDGES[:-1], AGE_BAND_EDGES[1:])
    ]
    bands = pd.cut(age_years, bins=AGE_BAND_EDGES, right=False, labels=labels)
    if bands.isna().any():
        bad = age_years[bands.isna()]
        raise ValueError(
            f"{len(bad)} ages outside [{AGE_BAND_EDGES[0]}, {AGE_BAND_EDGES[-1]}) "
            f"years or missing (e.g. {bad.head(3).tolist()})"
        )
    return bands.astype(str)


def extract_fairness_groups(app: pd.DataFrame) -> pd.DataFrame:
    """Return raw protected attributes and fairness grouping labels.

    The intersectional group is gender x 10-year age band
    (``GENDER_AGE``, e.g. ``"F | 30-39"``).

    Args:
        app: Application frame containing protected attribute columns.

    Returns:
        DataFrame with SK_ID_CURR, CODE_GENDER, DAYS_BIRTH, AGE_YEARS,
        AGE_BAND, and GENDER_AGE.
    """
    missing = [c for c in PROTECTED_ATTRIBUTES if c not in app.columns]
    if missing:
        raise KeyError(f"Missing protected attributes: {missing}")
    groups = app[[SK_ID_CURR] + PROTECTED_ATTRIBUTES].copy()
    groups[AGE_YEARS_COL] = -groups["DAYS_BIRTH"] / DAYS_PER_YEAR
    groups[AGE_BAND_COL] = assign_age_band(groups["DAYS_BIRTH"])
    groups[INTERSECTION_COL] = (
        groups["CODE_GENDER"].astype(str) + " | " + groups[AGE_BAND_COL]
    )
    return groups


def clean_application(
    train: pd.DataFrame,
    test: pd.DataFrame,
    xna_dropped: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, CleaningReport]:
    """Clean application tables and separate the training target.

    Args:
        train: Application train (after XNA gender drop).
        test: Application test (after XNA gender drop).
        xna_dropped: Rows already dropped for CODE_GENDER == XNA.

    Returns:
        Tuple of (train_features, test_features, y_train, cleaning_report).
        Protected attributes remain in the feature frames.
    """
    report = CleaningReport(xna_gender_rows_dropped=xna_dropped)

    train_c, n_train_anom = _replace_days_employed_sentinel(train)
    test_c, n_test_anom = _replace_days_employed_sentinel(test)
    report.days_employed_sentinel_train = n_train_anom
    report.days_employed_sentinel_test = n_test_anom

    if TARGET_COL not in train_c.columns:
        raise KeyError(f"{TARGET_COL} missing from application_train")
    y_train = train_c[TARGET_COL].astype(np.int8).copy()
    train_x = train_c.drop(columns=[TARGET_COL])

    drop_cols = _building_columns_to_drop(list(train_x.columns))
    # Only drop columns present in both frames
    drop_cols = [c for c in drop_cols if c in train_x.columns]
    train_x = train_x.drop(columns=drop_cols, errors="ignore")
    test_x = test_c.drop(columns=drop_cols, errors="ignore")
    report.building_columns_dropped = drop_cols
    report.notes.append(
        "Kept _AVG building variants; dropped matching _MEDI/_MODE. "
        "Standalone _MODE categoricals (e.g. HOUSETYPE_MODE) retained."
    )
    report.notes.append(
        "High-missing columns retained; missingness handled later via "
        "median imputation and optional missing indicators."
    )

    # Align columns (train defines schema)
    test_x = test_x.reindex(columns=train_x.columns)

    return train_x, test_x, y_train, report
