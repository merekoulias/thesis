"""Model registry: build estimators and the unified fit_predict adapter."""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.models.preprocessing import SignedLogStandardScaler
from src.models.types import FitOutput, ModelSpec

FINAL_STEP = "clf"


def _build_lgbm(params: Dict[str, Any], seed: int) -> LGBMClassifier:
    return LGBMClassifier(**params, random_state=seed)


def _build_rf(params: Dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(**params, random_state=seed)


def _build_logreg(params: Dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", SignedLogStandardScaler()),
            ("clf", LogisticRegression(**params, random_state=seed)),
        ]
    )


def _build_mlp(params: Dict[str, Any], seed: int) -> Pipeline:
    mlp_params = dict(params)
    if "hidden_layer_sizes" in mlp_params:
        mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])
    return Pipeline(
        [
            ("prep", SignedLogStandardScaler()),
            ("clf", MLPClassifier(**mlp_params, random_state=seed)),
        ]
    )


def _lgbm_importance(estimator: Any, feature_names: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for importance_type in ("gain", "split"):
        values = estimator.booster_.feature_importance(importance_type=importance_type)
        for feature, importance in zip(feature_names, values):
            rows.append(
                {
                    "feature": feature,
                    "importance_type": importance_type,
                    "importance": float(importance),
                }
            )
    return pd.DataFrame(rows)


def _rf_importance(estimator: Any, feature_names: List[str]) -> pd.DataFrame:
    values = estimator.feature_importances_
    rows = [
        {
            "feature": feature,
            "importance_type": "impurity",
            "importance": float(importance),
        }
        for feature, importance in zip(feature_names, values)
    ]
    return pd.DataFrame(rows)


def _logreg_importance(estimator: Any, feature_names: List[str]) -> pd.DataFrame:
    coef = estimator.named_steps[FINAL_STEP].coef_[0]
    rows = [
        {
            "feature": feature,
            "importance_type": "coef",
            "importance": float(importance),
        }
        for feature, importance in zip(feature_names, coef)
    ]
    return pd.DataFrame(rows)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "lgbm": ModelSpec(
        "lgbm",
        _build_lgbm,
        uses_early_stopping=True,
        importance=_lgbm_importance,
    ),
    "rf": ModelSpec("rf", _build_rf, importance=_rf_importance),
    "logreg": ModelSpec("logreg", _build_logreg, importance=_logreg_importance),
    "mlp": ModelSpec("mlp", _build_mlp, suppress_convergence_warnings=True),
}


def get_model_spec(name: str) -> ModelSpec:
    """Return the registry entry for a model family.

    Args:
        name: Registry name (e.g. ``lgbm``).

    Returns:
        ModelSpec for the requested model.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{name}'; available: {available}")
    return MODEL_REGISTRY[name]


def _predict_positive_proba(
    estimator: Any,
    X: np.ndarray,
    best_iteration: Optional[int],
) -> np.ndarray:
    if best_iteration is not None:
        return estimator.predict_proba(X, num_iteration=best_iteration)[:, 1]
    return estimator.predict_proba(X)[:, 1]


def fit_predict(
    spec: ModelSpec,
    params: Dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    *,
    eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    early_stopping_rounds: Optional[int] = None,
    n_estimators_override: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
    keep_estimator: bool = False,
    return_train_proba: bool = False,
) -> FitOutput:
    """Fit one estimator and return P(y=1) for X_pred.

    Args:
        spec: Registry entry.
        params: Fully resolved parameters (fixed_params + candidate).
        seed: Random seed injected as random_state.
        X_train, y_train: Training rows.
        X_pred: Rows to score (validation fold).
        eval_set: (X_es, y_es) for early stopping; allowed only if
            spec.uses_early_stopping, and then early_stopping_rounds is required.
        early_stopping_rounds: Patience for early stopping.
        n_estimators_override: Replaces params["n_estimators"] (LightGBM refit).
        sample_weight: Optional per-row training weights (seam for later
            pre-processing mitigation; unused by this plan's runs).
        keep_estimator: Return the fitted estimator in FitOutput.estimator.
        return_train_proba: Also predict X_train (in-sample) into
            FitOutput.train_proba, for the fit diagnostics.

    Returns:
        FitOutput(proba float64 1-D, best_iteration or None, fit_seconds, estimator or None).

    Raises:
        ValueError: eval_set given for a model without early stopping, or
            eval_set given without early_stopping_rounds.
    """
    if eval_set is not None:
        if not spec.uses_early_stopping:
            raise ValueError(
                f"eval_set is not supported for model '{spec.name}' "
                "(no early stopping)"
            )
        if early_stopping_rounds is None:
            raise ValueError(
                "early_stopping_rounds is required when eval_set is provided"
            )

    resolved = dict(params)
    if n_estimators_override is not None:
        resolved["n_estimators"] = int(n_estimators_override)

    estimator = spec.build(resolved, seed)

    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        if isinstance(estimator, Pipeline):
            fit_kwargs[f"{FINAL_STEP}__sample_weight"] = sample_weight
        else:
            fit_kwargs["sample_weight"] = sample_weight

    def _fit() -> None:
        if eval_set is not None:
            estimator.fit(
                X_train,
                y_train,
                eval_set=[eval_set],
                eval_metric="auc",
                callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
                **fit_kwargs,
            )
        else:
            estimator.fit(X_train, y_train, **fit_kwargs)

    t0 = time.perf_counter()
    if spec.suppress_convergence_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            _fit()
    else:
        _fit()
    fit_seconds = time.perf_counter() - t0

    if eval_set is not None:
        best_iteration = int(estimator.best_iteration_)
    else:
        best_iteration = None

    proba = _predict_positive_proba(estimator, X_pred, best_iteration).astype(
        np.float64
    )

    train_proba: Optional[np.ndarray] = None
    if return_train_proba:
        train_proba = _predict_positive_proba(
            estimator, X_train, best_iteration
        ).astype(np.float64)

    return FitOutput(
        proba=proba,
        best_iteration=best_iteration,
        fit_seconds=fit_seconds,
        estimator=estimator if keep_estimator else None,
        train_proba=train_proba,
    )
