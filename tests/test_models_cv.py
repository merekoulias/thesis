"""Tests for stratified CV splits and cross_val_predict_model."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.models.cv import cross_val_predict_model, make_stratified_splits
from src.models.registry import get_model_spec
from src.models.training import train_lgbm_cv
from src.models.types import OOF_PROBA_COL, ModelConfig

MODEL_NAMES = ("lgbm", "rf", "logreg", "mlp")


def _resolved_params(config: ModelConfig) -> dict:
    return {**config.fixed_params, **config.default_params}


def test_make_stratified_splits_matches_sklearn():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.2).astype(np.int8)
    seed = 42
    n_splits = 5

    splits = make_stratified_splits(y, n_splits, seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    expected = list(skf.split(np.zeros(n), y))

    assert len(splits) == n_splits
    for (train_a, val_a), (train_b, val_b) in zip(splits, expected):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(val_a, val_b)

    val_union = np.concatenate([val for _, val in splits])
    assert np.array_equal(np.sort(val_union), np.arange(n))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_cross_val_predict_covers_all_rows(
    model_name, make_training_data, tiny_model_configs
):
    data = make_training_data(n_samples=400, seed=0)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    n_splits = 3
    splits = make_stratified_splits(y, n_splits, seed=42)

    config = tiny_model_configs[model_name]
    spec = get_model_spec(model_name)
    params = _resolved_params(config)
    es_rounds = config.early_stopping_rounds

    oof, fits = cross_val_predict_model(
        spec,
        params,
        seed=42,
        X=X,
        y=y,
        splits=splits,
        early_stopping_rounds=es_rounds,
    )

    assert len(fits) == n_splits
    assert oof.shape == (len(y),)
    assert not np.isnan(oof).any()
    assert np.all(np.isfinite(oof))


def test_cross_val_predict_partial_coverage(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=1)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    all_splits = make_stratified_splits(y, 3, seed=42)
    splits = all_splits[:2]

    config = tiny_model_configs["rf"]
    spec = get_model_spec("rf")
    params = _resolved_params(config)

    oof, fits = cross_val_predict_model(spec, params, seed=42, X=X, y=y, splits=splits)

    assert len(fits) == 2
    covered = np.concatenate([val for _, val in splits])
    uncovered = np.setdiff1d(np.arange(len(y)), covered)
    assert np.all(np.isfinite(oof[covered]))
    assert np.all(np.isnan(oof[uncovered]))


def test_overlapping_splits_raise(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=200, seed=2)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    splits = make_stratified_splits(y, 3, seed=42)
    train0, val0 = splits[0]
    train1, val1 = splits[1]
    bad_splits = [
        (train0, val0),
        (train1, np.concatenate([val1, val0[:5]])),
    ]

    config = tiny_model_configs["rf"]
    spec = get_model_spec("rf")
    params = _resolved_params(config)

    with pytest.raises(ValueError, match="overlap"):
        cross_val_predict_model(spec, params, seed=42, X=X, y=y, splits=bad_splits)


def test_matches_legacy_train_lgbm_cv(make_training_data):
    data = make_training_data(n_samples=400, seed=3)
    legacy_cfg = {
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
    }

    legacy = train_lgbm_cv(data, legacy_cfg, return_models=False)

    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    splits = make_stratified_splits(y, legacy_cfg["n_folds"], legacy_cfg["seed"])
    spec = get_model_spec("lgbm")
    oof, fits = cross_val_predict_model(
        spec,
        dict(legacy_cfg["lgbm"]),
        legacy_cfg["seed"],
        X,
        y,
        splits,
        early_stopping_rounds=legacy_cfg["early_stopping_rounds"],
    )

    legacy_proba = legacy.oof_predictions[OOF_PROBA_COL].to_numpy()
    assert np.array_equal(oof, legacy_proba)

    legacy_iters = [fr.best_iteration for fr in legacy.fold_results]
    new_iters = [f.best_iteration for f in fits]
    assert legacy_iters == new_iters
