"""Model-specific preprocessing fitted inside each CV training fold."""

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Return sign(x) * log1p(|x|), elementwise.

    Args:
        values: Input array of any shape.

    Returns:
        Transformed array with the same shape as ``values``.
    """
    return np.sign(values) * np.log1p(np.abs(values))


class SignedLogStandardScaler(BaseEstimator, TransformerMixin):
    """Signed log transform + standardization for non-binary columns.

    Columns whose training values are all 0 or 1 are passed through unchanged.
    Constant non-binary columns get scale 1 and are always transformed to 0, also
    when a later ``transform`` call sees a different value. (The standard deviation
    of a constant column is not exactly 0 in floating point, e.g. 2.2e-16 for a
    column of 5.0, so a ``std > 0`` test alone would blow the column up.)

    Attributes:
        binary_mask_: Boolean mask of passthrough (0/1) columns.
        constant_mask_: Boolean mask, over the non-binary columns only, of those
            that are constant after the signed log transform in training.
        mean_: Means of the transformed non-binary columns.
        scale_: Standard deviations of the transformed non-binary columns
            (1 for constant columns).
        n_features_in_: Number of input columns seen in fit.
    """

    def fit(self, X: Any, y: Optional[Any] = None) -> "SignedLogStandardScaler":
        """Fit means and scales on training data.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Ignored; present for sklearn API compatibility.

        Returns:
            Fitted transformer.
        """
        values = np.asarray(X, dtype=np.float64)
        self.binary_mask_ = np.all((values == 0.0) | (values == 1.0), axis=0)
        transformed = signed_log1p(values[:, ~self.binary_mask_])
        self.mean_ = transformed.mean(axis=0)
        self.constant_mask_ = np.ptp(transformed, axis=0) == 0.0
        self.scale_ = np.where(self.constant_mask_, 1.0, transformed.std(axis=0))
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply signed log and standardization using training-fold statistics.

        Args:
            X: Features with the same number of columns as in ``fit``.

        Returns:
            Transformed array; binary columns are unchanged.

        Raises:
            ValueError: If the column count differs from ``fit``.
        """
        check_is_fitted(self, "mean_")
        values = np.asarray(X, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} columns, got {values.shape[1]}"
            )
        out = values.copy()
        cols = ~self.binary_mask_
        scaled = (signed_log1p(values[:, cols]) - self.mean_) / self.scale_
        scaled[:, self.constant_mask_] = 0.0
        out[:, cols] = scaled
        return out
