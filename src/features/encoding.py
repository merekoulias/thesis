"""Train-fitted imputation and one-hot encoding for the feature matrix."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.types import MISSING_TOKEN, SK_ID_CURR


def _is_string_like(series: pd.Series) -> bool:
    """Return True for object/string/category columns."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(series.dtype):
        return True
    if hasattr(pd.api.types, "is_string_dtype") and pd.api.types.is_string_dtype(
        series.dtype
    ):
        return True
    return False


def split_feature_columns(
    df: pd.DataFrame, id_col: str = SK_ID_CURR
) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical feature lists.

    Args:
        df: Feature frame (may include id_col).
        id_col: Identifier column to exclude from features.

    Returns:
        (numeric_columns, categorical_columns).
    """
    cols = [c for c in df.columns if c != id_col]
    categorical = [c for c in cols if _is_string_like(df[c])]
    numeric = [c for c in cols if c not in categorical]
    return numeric, categorical


def add_missing_indicators(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    min_missing_rate: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Add binary missing indicators for high-missing numeric columns.

    High-missing columns are retained; indicators preserve missingness
    signal after median imputation. Indicators are fit from train rates.

    Args:
        train: Training feature frame.
        test: Test feature frame.
        columns: Optional explicit columns. If None, select numeric
            columns with train missing rate >= min_missing_rate.
        min_missing_rate: Threshold for automatic selection.

    Returns:
        (train_with_indicators, test_with_indicators, indicator_names).
    """
    train_out = train.copy()
    test_out = test.copy()

    if columns is None:
        numeric, _ = split_feature_columns(train_out)
        selected: List[str] = []
        n = len(train_out)
        for col in numeric:
            rate = float(train_out[col].isna().sum()) / max(n, 1)
            if rate >= min_missing_rate:
                selected.append(col)
    else:
        selected = [c for c in columns if c in train_out.columns]

    # Always include key application signals when present
    always = [
        "OWN_CAR_AGE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
    for col in always:
        if col in train_out.columns and col not in selected:
            selected.append(col)

    indicator_names: List[str] = []
    train_flags: Dict[str, pd.Series] = {}
    test_flags: Dict[str, pd.Series] = {}
    for col in selected:
        name = f"{col}_MISSING"
        train_flags[name] = train_out[col].isna().astype(np.int8)
        test_flags[name] = test_out[col].isna().astype(np.int8)
        indicator_names.append(name)

    if train_flags:
        train_out = pd.concat(
            [train_out, pd.DataFrame(train_flags, index=train_out.index)],
            axis=1,
        )
        test_out = pd.concat(
            [test_out, pd.DataFrame(test_flags, index=test_out.index)],
            axis=1,
        )

    return train_out, test_out, indicator_names


class FeatureEncoder:
    """Median-impute numerics and one-hot encode all categoricals.

    Policy:
        - One-hot for all nominal categoricals, including binary.
        - No label encoding (avoids fake ordinal structure).
        - Fit on train only; transform test with handle_unknown='ignore'.
        - Keep all dummy columns (drop=None) for tree models.
    """

    def __init__(self) -> None:
        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.feature_names_: List[str] = []
        self._transformer: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame) -> "FeatureEncoder":
        """Fit imputers and one-hot encoder on training features."""
        numeric, categorical = split_feature_columns(X)
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical

        X_fit = X.copy()
        for col in categorical:
            X_fit[col] = (
                X_fit[col]
                .astype("string")
                .fillna(MISSING_TOKEN)
                .astype(str)
            )

        transformers = []
        if numeric:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median",
                                    keep_empty_features=True,
                                ),
                            )
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                    dtype=np.float64,
                                ),
                            )
                        ]
                    ),
                    categorical,
                )
            )

        if not transformers:
            raise ValueError("No feature columns to encode")

        self._transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self._transformer.set_output(transform="pandas")
        self._transformer.fit(X_fit)
        self.feature_names_ = list(self._transformer.get_feature_names_out())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to a dense numeric matrix."""
        if self._transformer is None:
            raise RuntimeError("FeatureEncoder must be fit before transform")

        X_t = X.copy()
        for col in self.categorical_columns_:
            if col not in X_t.columns:
                X_t[col] = MISSING_TOKEN
            X_t[col] = (
                X_t[col].astype("string").fillna(MISSING_TOKEN).astype(str)
            )
        # Ensure all expected columns exist
        for col in self.numeric_columns_:
            if col not in X_t.columns:
                X_t[col] = np.nan

        out = self._transformer.transform(X_t)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, columns=self.feature_names_, index=X.index)
        else:
            out = out.copy()
            out.columns = self.feature_names_
            out.index = X.index
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on X and return the transformed matrix."""
        return self.fit(X).transform(X)
