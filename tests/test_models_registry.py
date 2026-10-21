"""Tests for the model registry and fit_predict adapter."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.models.registry import FINAL_STEP, MODEL_REGISTRY, fit_predict, get_model_spec
from src.models.types import ModelConfig

MODEL_NAMES = ("lgbm", "rf", "logreg", "mlp")


def _resolved_params(config: ModelConfig) -> dict:
    return {**config.fixed_params, **config.default_params}


def test_registry_names():
    assert set(MODEL_REGISTRY) == {"lgbm", "rf", "logreg", "mlp"}
    with pytest.raises(KeyError) as exc_info:
        get_model_spec("xgb")
    assert "available" in str(exc_info.value)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fit_predict_all_models(model_name, make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=400, seed=0)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    X_train, X_pred = X[:300], X[300:]
    y_train = y[:300]

    config = tiny_model_configs[model_name]
    spec = get_model_spec(model_name)
    params = _resolved_params(config)

    out = fit_predict(
        spec, params, seed=42, X_train=X_train, y_train=y_train, X_pred=X_pred
    )

    assert out.proba.shape == (100,)
    assert out.proba.dtype == np.float64
    assert np.all((out.proba >= 0.0) & (out.proba <= 1.0))
    assert np.all(np.isfinite(out.proba))
    assert out.fit_seconds >= 0.0


def test_lgbm_early_stopping(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=400, seed=1)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    n = len(X)
    split = n // 2
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    X_pred = X_val[:50]

    config = tiny_model_configs["lgbm"]
    spec = get_model_spec("lgbm")
    params = _resolved_params(config)

    out = fit_predict(
        spec,
        params,
        seed=42,
        X_train=X_train,
        y_train=y_train,
        X_pred=X_pred,
        eval_set=(X_val, y_val),
        early_stopping_rounds=5,
    )

    assert isinstance(out.best_iteration, int)
    assert 1 <= out.best_iteration <= 40


def test_lgbm_n_estimators_override(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=200, seed=2)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    X_train, X_pred = X[:150], X[150:]

    config = tiny_model_configs["lgbm"]
    spec = get_model_spec("lgbm")
    params = _resolved_params(config)

    out = fit_predict(
        spec,
        params,
        seed=42,
        X_train=X_train,
        y_train=y[:150],
        X_pred=X_pred,
        n_estimators_override=7,
        keep_estimator=True,
    )

    assert out.estimator is not None
    assert out.estimator.n_estimators == 7
    assert out.best_iteration is None


def test_eval_set_rejected_for_non_es_model(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=200, seed=3)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    spec = get_model_spec("rf")
    params = _resolved_params(tiny_model_configs["rf"])

    with pytest.raises(ValueError):
        fit_predict(
            spec,
            params,
            seed=42,
            X_train=X[:100],
            y_train=y[:100],
            X_pred=X[100:],
            eval_set=(X[50:100], y[50:100]),
            early_stopping_rounds=5,
        )


def test_eval_set_requires_rounds(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=200, seed=4)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    spec = get_model_spec("lgbm")
    params = _resolved_params(tiny_model_configs["lgbm"])

    with pytest.raises(ValueError):
        fit_predict(
            spec,
            params,
            seed=42,
            X_train=X[:100],
            y_train=y[:100],
            X_pred=X[100:],
            eval_set=(X[50:100], y[50:100]),
        )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_sample_weight_seam(model_name, make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=5)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    X_train, X_pred = X[:200], X[200:]
    y_train = y[:200]
    n = len(y_train)

    config = tiny_model_configs[model_name]
    spec = get_model_spec(model_name)
    params = _resolved_params(config)

    fit_predict(
        spec,
        params,
        seed=42,
        X_train=X_train,
        y_train=y_train,
        X_pred=X_pred,
        sample_weight=np.ones(n, dtype=np.float64),
    )

    if model_name == "logreg":
        out_uniform = fit_predict(
            spec,
            params,
            seed=42,
            X_train=X_train,
            y_train=y_train,
            X_pred=X_pred,
            sample_weight=np.ones(n, dtype=np.float64),
        )
        weights = np.ones(n, dtype=np.float64)
        weights[y_train == 1] = 5.0
        out_weighted = fit_predict(
            spec,
            params,
            seed=42,
            X_train=X_train,
            y_train=y_train,
            X_pred=X_pred,
            sample_weight=weights,
        )
        assert not np.allclose(out_uniform.proba, out_weighted.proba)


def test_mlp_hidden_layers_tuple(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=200, seed=6)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()

    config = tiny_model_configs["mlp"]
    spec = get_model_spec("mlp")
    params = _resolved_params(config)

    out = fit_predict(
        spec,
        params,
        seed=42,
        X_train=X[:150],
        y_train=y[:150],
        X_pred=X[150:],
        keep_estimator=True,
    )

    assert out.estimator is not None
    assert out.estimator.named_steps[FINAL_STEP].hidden_layer_sizes == (8,)


def test_mlp_convergence_warning_suppressed(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=200, seed=7)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()

    config = tiny_model_configs["mlp"]
    spec = get_model_spec("mlp")
    params = dict(_resolved_params(config))
    params["max_iter"] = 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_predict(
            spec,
            params,
            seed=42,
            X_train=X[:150],
            y_train=y[:150],
            X_pred=X[150:],
        )

    convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(convergence) == 0


@pytest.mark.parametrize(
    "model_name,importance_types",
    [
        ("lgbm", ("gain", "split")),
        ("rf", ("impurity",)),
        ("logreg", ("coef",)),
    ],
)
def test_importance_frames(
    model_name, importance_types, make_training_data, tiny_model_configs
):
    data = make_training_data(n_samples=300, seed=8)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    feature_names = data.feature_names

    config = tiny_model_configs[model_name]
    spec = get_model_spec(model_name)
    params = _resolved_params(config)

    out = fit_predict(
        spec,
        params,
        seed=42,
        X_train=X[:200],
        y_train=y[:200],
        X_pred=X[200:],
        keep_estimator=True,
    )
    assert spec.importance is not None
    frame = spec.importance(out.estimator, feature_names)

    assert list(frame.columns) == ["feature", "importance_type", "importance"]
    for imp_type in importance_types:
        subset = frame[frame["importance_type"] == imp_type]
        assert len(subset) == len(feature_names)


def test_mlp_has_no_importance():
    spec = get_model_spec("mlp")
    assert spec.importance is None


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fit_predict_deterministic(model_name, make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=9)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    X_train, X_pred = X[:200], X[200:]

    config = tiny_model_configs[model_name]
    spec = get_model_spec(model_name)
    params = _resolved_params(config)

    out_a = fit_predict(
        spec, params, seed=42, X_train=X_train, y_train=y[:200], X_pred=X_pred
    )
    out_b = fit_predict(
        spec, params, seed=42, X_train=X_train, y_train=y[:200], X_pred=X_pred
    )
    assert np.array_equal(out_a.proba, out_b.proba)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_return_train_proba(model_name, make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=10)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    X_train, X_pred = X[:200], X[200:]

    config = tiny_model_configs[model_name]
    spec = get_model_spec(model_name)
    params = _resolved_params(config)

    out_default = fit_predict(
        spec, params, seed=42, X_train=X_train, y_train=y[:200], X_pred=X_pred
    )
    assert out_default.train_proba is None

    out_train = fit_predict(
        spec,
        params,
        seed=42,
        X_train=X_train,
        y_train=y[:200],
        X_pred=X_pred,
        return_train_proba=True,
    )
    assert out_train.train_proba.shape == (len(X_train),)
    assert out_train.train_proba.dtype == np.float64
    assert np.all((out_train.train_proba >= 0.0) & (out_train.train_proba <= 1.0))
    assert np.array_equal(out_default.proba, out_train.proba)
