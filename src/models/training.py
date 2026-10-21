"""LightGBM stratified cross-validation with out-of-fold predictions."""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from src.constants import SK_ID_CURR, TARGET_COL
from src.models.types import (
    OOF_FOLD_COL,
    OOF_PROBA_COL,
    CVResult,
    FoldResult,
    TrainingData,
)


def _sanitize_feature_names(
    columns: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map original feature names to LightGBM-safe identifiers.

    LightGBM rejects feature names with JSON-special characters (e.g. spaces).

    Returns:
        (original_to_safe, safe_to_original) bijection dicts.
    """
    original_to_safe: Dict[str, str] = {}
    safe_to_original: Dict[str, str] = {}
    used: set = set()

    for col in columns:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", col)
        safe = re.sub(r"_+", "_", safe).strip("_")
        if not safe:
            safe = "feature"
        base = safe
        suffix = 0
        while safe in used:
            suffix += 1
            safe = f"{base}_{suffix}"
        used.add(safe)
        original_to_safe[col] = safe
        safe_to_original[safe] = col

    return original_to_safe, safe_to_original


def _rename_features(X: pd.DataFrame, original_to_safe: Dict[str, str]) -> pd.DataFrame:
    """Return X with sanitized column names for LightGBM."""
    return X.rename(columns=original_to_safe)


def _aggregate_importance(
    models: List[Any],
    feature_names: List[str],
    safe_to_original: Dict[str, str],
    importance_type: str,
) -> pd.DataFrame:
    """Average per-fold feature importance across folds."""
    rows = []
    for fold_idx, model in enumerate(models):
        imp = model.booster_.feature_importance(importance_type=importance_type)
        for safe_name, value in zip(feature_names, imp):
            original = safe_to_original.get(safe_name, safe_name)
            rows.append(
                {
                    "feature": original,
                    "fold": fold_idx,
                    "importance": float(value),
                }
            )

    frame = pd.DataFrame(rows)
    agg = (
        frame.groupby("feature", as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": f"mean_{importance_type}"})
        .sort_values(f"mean_{importance_type}", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def train_lgbm_cv(
    data: TrainingData,
    config: Dict[str, Any],
    *,
    return_models: bool = True,
) -> CVResult:
    """Train LightGBM with stratified k-fold CV and collect OOF predictions.

    Args:
        data: Aligned training features, target, and ids.
        config: Model configuration (from configs/model.yaml).
        return_models: Whether to retain fitted fold models.

    Returns:
        CVResult with OOF predictions, fold metrics, and optional models.
    """
    seed = int(config.get("seed", 42))
    n_folds = int(config.get("n_folds", 5))
    early_stopping_rounds = int(config.get("early_stopping_rounds", 50))
    lgbm_params = dict(config.get("lgbm", {}))
    lgbm_params["random_state"] = seed

    X = data.X
    y = data.y
    ids = data.ids
    original_to_safe, safe_to_original = _sanitize_feature_names(data.feature_names)
    X = _rename_features(X, original_to_safe)
    safe_feature_names = list(X.columns)
    n_samples = len(X)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_proba = np.full(n_samples, np.nan, dtype=float)
    fold_assignments = np.full(n_samples, -1, dtype=int)
    fold_results: List[FoldResult] = []
    fold_aucs: List[float] = []
    models: List[Any] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
        )

        val_proba = model.predict_proba(X_val, num_iteration=model.best_iteration_)[
            :, 1
        ]
        oof_proba[val_idx] = val_proba
        fold_assignments[val_idx] = fold_idx

        fold_auc = float(roc_auc_score(y_val, val_proba))
        fold_aucs.append(fold_auc)
        fold_results.append(
            FoldResult(
                fold=fold_idx,
                roc_auc=fold_auc,
                best_iteration=int(model.best_iteration_),
                train_indices=train_idx,
                val_indices=val_idx,
            )
        )
        if return_models:
            models.append(model)

    if np.isnan(oof_proba).any():
        raise RuntimeError("OOF predictions contain NaN; not all rows were scored")

    oof_df = pd.DataFrame(
        {
            SK_ID_CURR: ids.values,
            TARGET_COL: y.values,
            OOF_PROBA_COL: oof_proba,
            OOF_FOLD_COL: fold_assignments,
        }
    )

    oof_auc = float(roc_auc_score(y, oof_proba))
    fpr, tpr, thresholds = roc_curve(y, oof_proba)
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }

    importance_gain: Optional[pd.DataFrame] = None
    importance_split: Optional[pd.DataFrame] = None
    if return_models and models:
        importance_gain = _aggregate_importance(
            models, safe_feature_names, safe_to_original, "gain"
        )
        importance_split = _aggregate_importance(
            models, safe_feature_names, safe_to_original, "split"
        )

    return CVResult(
        oof_predictions=oof_df,
        fold_results=fold_results,
        fold_roc_aucs=fold_aucs,
        mean_fold_roc_auc=float(np.mean(fold_aucs)),
        std_fold_roc_auc=float(np.std(fold_aucs)),
        oof_roc_auc=oof_auc,
        roc_curve=roc_data,
        models=models if return_models else [],
        feature_importance_gain=importance_gain,
        feature_importance_split=importance_split,
        config=config,
    )
