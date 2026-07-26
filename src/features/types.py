"""Shared types for feature engineering outputs."""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.constants import (  # noqa: F401 — re-export for features consumers
    DAYS_SENTINEL,
    INTERSECTION_COL,
    MISSING_TOKEN,
    PROTECTED_ATTRIBUTES,
    REQUIRED_KEYS,
    SK_ID_BUREAU,
    SK_ID_CURR,
    SK_ID_PREV,
    STRUCTURAL_CAT_PLACEHOLDERS,
    TABLE_FILES,
    TARGET_COL,
)


@dataclass
class CleaningReport:
    """Counts of cleaning actions applied during FE."""

    days_employed_sentinel_train: int = 0
    days_employed_sentinel_test: int = 0
    xna_gender_rows_dropped: int = 0
    building_columns_dropped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class FeatureMatrixBundle:
    """Model-ready matrices plus fairness grouping columns."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    ids_train: pd.Series
    ids_test: pd.Series
    fairness_groups_train: pd.DataFrame
    feature_names: List[str]
    numeric_source_columns: List[str]
    categorical_source_columns: List[str]
    cleaning_report: CleaningReport
    fairness_groups_test: Optional[pd.DataFrame] = None
