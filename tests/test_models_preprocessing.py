"""Tests for SignedLogStandardScaler preprocessing."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.preprocessing import SignedLogStandardScaler, signed_log1p

_WORKED_X_TRAIN = np.array([[0.0, 10.0, 5.0], [1.0, 100.0, 5.0], [0.0, 1000.0, 5.0]])


def test_binary_columns_pass_through() -> None:
    scaler = SignedLogStandardScaler()
    out = scaler.fit_transform(_WORKED_X_TRAIN)
    np.testing.assert_array_equal(out[:, 0], _WORKED_X_TRAIN[:, 0])


def test_nonbinary_columns_standardized() -> None:
    scaler = SignedLogStandardScaler()
    out = scaler.fit_transform(_WORKED_X_TRAIN)
    col1 = out[:, 1]
    assert abs(col1.mean()) < 1e-12
    assert abs(col1.std() - 1.0) < 1e-12

    raw = signed_log1p(_WORKED_X_TRAIN[:, 1])
    mean = raw.mean()
    std = raw.std()
    expected = (raw - mean) / std
    np.testing.assert_allclose(col1, expected, atol=1e-12)


def test_constant_column_becomes_zero() -> None:
    scaler = SignedLogStandardScaler()
    out = scaler.fit_transform(_WORKED_X_TRAIN)
    col2 = out[:, 2]
    assert np.all(col2 == 0.0)
    assert np.isfinite(col2).all()


def test_negative_values_signed() -> None:
    # Two rows so the column is not treated as binary (not all values in {0, 1}).
    x = np.array([[-1000.0, 0.0], [-1.0, 2.0], [0.0, 3.0], [1.0, 4.0], [1000.0, 5.0]])
    scaler = SignedLogStandardScaler()
    out = scaler.fit_transform(x)
    transformed_col = out[:, 0]
    assert np.all(np.diff(transformed_col) > 0)


def test_transform_uses_training_statistics() -> None:
    scaler = SignedLogStandardScaler()
    scaler.fit(_WORKED_X_TRAIN)
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()
    x_val = np.array([[0.0, 50.0, 5.0], [1.0, 200.0, 5.0]])
    x_val_before = x_val.copy()
    out = scaler.transform(x_val)

    # Statistics come from the training column only, computed by hand.
    train_log = signed_log1p(_WORKED_X_TRAIN[:, 1])
    expected_col1 = (signed_log1p(x_val[:, 1]) - train_log.mean()) / train_log.std()
    np.testing.assert_allclose(out[:, 1], expected_col1, atol=1e-12)
    np.testing.assert_array_equal(out[:, 0], x_val[:, 0])

    # Transforming validation data neither refits nor modifies the input.
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.scale_, scale_before)
    np.testing.assert_array_equal(x_val, x_val_before)


def test_constant_training_column_stays_zero_for_new_values() -> None:
    scaler = SignedLogStandardScaler()
    scaler.fit(_WORKED_X_TRAIN)
    x_val = np.array([[0.0, 50.0, 50.0], [1.0, 200.0, 0.0]])
    out = scaler.transform(x_val)
    np.testing.assert_array_equal(out[:, 2], np.zeros(2))


def test_wrong_column_count_raises() -> None:
    scaler = SignedLogStandardScaler()
    scaler.fit(_WORKED_X_TRAIN)
    bad = np.zeros((2, 2))
    with pytest.raises(ValueError, match="Expected 3 columns"):
        scaler.transform(bad)


def test_works_inside_pipeline() -> None:
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=(n, 5))
    x[:, 0] = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    pipe = Pipeline(
        [
            ("prep", SignedLogStandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(x, y)
    proba = pipe.predict_proba(x)
    assert proba.shape == (n, 2)


def test_sklearn_clone() -> None:
    cloned = clone(SignedLogStandardScaler())
    assert isinstance(cloned, SignedLogStandardScaler)
