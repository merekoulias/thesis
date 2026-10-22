"""Tests for the legacy train_lgbm_cv and the training-data contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import PROTECTED_ATTRIBUTES, SK_ID_CURR, TARGET_COL
from src.models.io import load_training_data
from src.models.training import train_lgbm_cv
from src.models.types import OOF_FOLD_COL, OOF_PROBA_COL, TrainingData


def _make_synthetic_training_data(
    n_samples: int = 200,
    n_features: int = 10,
    seed: int = 42,
) -> TrainingData:
    """Create a small stratifiable synthetic dataset for fast CV tests."""
    rng = np.random.default_rng(seed)
    # ~20% positive rate for stratification
    y = (rng.random(n_samples) < 0.2).astype(np.int8)
    if y.sum() == 0 or y.sum() == n_samples:
        y[: n_samples // 5] = 1

    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Encoded protected-attribute columns
    X["cat__CODE_GENDER_F"] = (rng.random(n_samples) < 0.66).astype(float)
    X["cat__CODE_GENDER_M"] = 1.0 - X["cat__CODE_GENDER_F"]
    X["num__DAYS_BIRTH"] = -rng.integers(7500, 25000, size=n_samples).astype(float)

    ids = pd.Series(range(1, n_samples + 1), name=SK_ID_CURR)
    feature_names = list(X.columns)
    return TrainingData(
        X=X, y=pd.Series(y, name=TARGET_COL), ids=ids, feature_names=feature_names
    )


def _write_feature_artifacts(data: TrainingData, out_dir: Path) -> None:
    """Write minimal feature artifacts for io tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data.X.to_parquet(out_dir / "X_train.parquet", index=False)
    data.y.to_frame(TARGET_COL).to_parquet(out_dir / "y_train.parquet", index=False)
    data.ids.to_frame(SK_ID_CURR).to_parquet(out_dir / "ids_train.parquet", index=False)


@pytest.fixture
def fast_model_config() -> dict:
    """Minimal LightGBM config for fast tests."""
    return {
        "seed": 42,
        "n_folds": 3,
        "early_stopping_rounds": 5,
        "lgbm": {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 50,
            "num_leaves": 8,
            "learning_rate": 0.1,
            "verbosity": -1,
            "n_jobs": 1,
        },
        "outputs": {
            "save_fold_models": True,
            "save_feature_importance": True,
            "save_roc_plot": True,
        },
    }


def test_train_lgbm_cv_oof_contract(
    fast_model_config: dict,
) -> None:
    data = _make_synthetic_training_data(n_samples=120)
    result = train_lgbm_cv(data, fast_model_config, return_models=True)

    oof = result.oof_predictions
    assert len(oof) == len(data.X)
    assert oof[SK_ID_CURR].equals(data.ids.reset_index(drop=True))
    assert (oof[TARGET_COL].values == data.y.values).all()
    assert oof[OOF_PROBA_COL].between(0, 1).all()
    assert np.isfinite(oof[OOF_PROBA_COL]).all()
    assert oof[OOF_FOLD_COL].nunique() == fast_model_config["n_folds"]
    assert (oof[OOF_FOLD_COL] >= 0).all()

    assert len(result.fold_results) == fast_model_config["n_folds"]
    assert 0.0 <= result.oof_roc_auc <= 1.0
    assert len(result.fold_roc_aucs) == fast_model_config["n_folds"]
    assert all(0.0 <= auc <= 1.0 for auc in result.fold_roc_aucs)
    assert all(fr.best_iteration > 0 for fr in result.fold_results)
    assert result.feature_importance_gain is not None
    assert result.feature_importance_split is not None

    # Protected columns present in feature importance
    gain_features = set(result.feature_importance_gain["feature"])
    for attr in PROTECTED_ATTRIBUTES:
        assert any(attr in f for f in gain_features)


def test_train_lgbm_cv_reproducible(fast_model_config: dict) -> None:
    data = _make_synthetic_training_data(n_samples=120, seed=7)
    r1 = train_lgbm_cv(data, fast_model_config, return_models=False)
    r2 = train_lgbm_cv(data, fast_model_config, return_models=False)
    np.testing.assert_allclose(
        r1.oof_predictions[OOF_PROBA_COL].values,
        r2.oof_predictions[OOF_PROBA_COL].values,
        rtol=1e-6,
    )


def test_load_training_data_validates(tmp_path: Path) -> None:
    data = _make_synthetic_training_data(n_samples=50)
    _write_feature_artifacts(data, tmp_path)
    loaded = load_training_data(tmp_path)
    assert len(loaded.X) == 50
    assert list(loaded.X.columns) == data.feature_names


def test_load_training_data_missing_protected_attrs(tmp_path: Path) -> None:
    data = _make_synthetic_training_data(n_samples=30)
    data.X = data.X.drop(columns=[c for c in data.X.columns if "CODE_GENDER" in c])
    data.feature_names = list(data.X.columns)
    _write_feature_artifacts(data, tmp_path)
    with pytest.raises(ValueError, match="CODE_GENDER"):
        load_training_data(tmp_path)


def test_load_training_data_duplicate_ids(tmp_path: Path) -> None:
    data = _make_synthetic_training_data(n_samples=20)
    data.ids.iloc[1] = data.ids.iloc[0]
    _write_feature_artifacts(data, tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        load_training_data(tmp_path)
