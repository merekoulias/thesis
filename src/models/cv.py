"""Stratified cross-validation splits and generic out-of-fold prediction."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.registry import fit_predict
from src.models.types import FitOutput, ModelSpec, Split


def make_stratified_splits(y: np.ndarray, n_splits: int, seed: int) -> List[Split]:
    """StratifiedKFold(n_splits, shuffle=True, random_state=seed) on y.

    Uses np.zeros(len(y)) as the X placeholder; splits depend only on y and the seed,
    so they equal the legacy folds when given the full y and seed 42.

    Args:
        y: Binary target, one entry per row.
        n_splits: Number of folds.
        seed: Shuffle seed.

    Returns:
        List of (train_idx, val_idx) int64 arrays, one pair per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    placeholder = np.zeros(len(y), dtype=np.float64)
    splits: List[Split] = []
    for train_idx, val_idx in skf.split(placeholder, y):
        splits.append(
            (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(val_idx, dtype=np.int64),
            )
        )
    return splits


def _validation_parts_overlap(splits: Sequence[Split]) -> bool:
    """Return True if any row index appears more than once across validation parts."""
    if not splits:
        return False
    all_val = np.concatenate([val_idx for _, val_idx in splits])
    return bool(np.unique(all_val).size != all_val.size)


def cross_val_predict_model(
    spec: ModelSpec,
    params: Dict[str, Any],
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Split],
    *,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[np.ndarray, List[FitOutput]]:
    """Fit on each split's train part, predict its validation part.

    If spec.uses_early_stopping, the split's validation part is passed as eval_set
    (early stopping on the scored fold). This is used (a) inside the inner loop of
    nested CV, where that fold is an inner fold, and (b) by legacy_check to mirror
    the legacy stratified CV trainer. Never call it with outer folds for real results.

    Returns:
        (oof, fits): oof has len(y) and NaN only for rows in no validation part;
        fits[i] belongs to splits[i] (estimator not kept).

    Raises:
        ValueError: if validation parts overlap.
    """
    if _validation_parts_overlap(splits):
        raise ValueError("Validation indices overlap across splits")

    n_samples = len(y)
    oof = np.full(n_samples, np.nan, dtype=np.float64)
    fits: List[FitOutput] = []

    for train_idx, val_idx in splits:
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]

        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
        es_rounds: Optional[int] = None
        if spec.uses_early_stopping:
            eval_set = (X_val, y[val_idx])
            es_rounds = early_stopping_rounds

        fit_out = fit_predict(
            spec,
            params,
            seed,
            X_train,
            y_train,
            X_val,
            eval_set=eval_set,
            early_stopping_rounds=es_rounds,
        )
        oof[val_idx] = fit_out.proba
        fits.append(fit_out)

    return oof, fits
