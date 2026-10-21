"""CLI: compare generic CV LightGBM OOF predictions to train_lgbm_cv."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.cv import cross_val_predict_model, make_stratified_splits
from src.models.io import load_training_data, load_yaml_config
from src.models.registry import get_model_spec
from src.models.training import train_lgbm_cv
from src.models.types import OOF_FOLD_COL, OOF_PROBA_COL, Split


def _fold_assignments(n_samples: int, splits: Sequence[Split]) -> np.ndarray:
    folds = np.full(n_samples, -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(splits):
        folds[val_idx] = fold_idx
    return folds


def main(argv: Sequence[str] = None) -> None:
    """Run legacy vs new OOF equivalence check on full training data."""
    parser = argparse.ArgumentParser(
        description="Compare train_lgbm_cv OOF predictions to the generic CV core."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("artifacts/features"),
        help="Directory with X_train, y_train, ids_train parquet files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Legacy LightGBM YAML config (same as notebook 08).",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml_config(args.config)
    data = load_training_data(args.features_dir)

    legacy = train_lgbm_cv(data, cfg, return_models=False)

    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    splits = make_stratified_splits(y, int(cfg["n_folds"]), int(cfg["seed"]))
    spec = get_model_spec("lgbm")
    oof, fits = cross_val_predict_model(
        spec,
        dict(cfg["lgbm"]),
        int(cfg["seed"]),
        X,
        y,
        splits,
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
    )

    legacy_proba = legacy.oof_predictions[OOF_PROBA_COL].to_numpy()
    max_abs_diff = float(np.max(np.abs(oof - legacy_proba)))

    legacy_folds = legacy.oof_predictions[OOF_FOLD_COL].to_numpy()
    new_folds = _fold_assignments(len(y), splits)
    folds_identical = bool(np.array_equal(legacy_folds, new_folds))

    legacy_iters = [fr.best_iteration for fr in legacy.fold_results]
    new_iters = [f.best_iteration for f in fits]
    iters_identical = legacy_iters == new_iters

    legacy_auc = float(roc_auc_score(y, legacy_proba))
    new_auc = float(roc_auc_score(y, oof))

    print(f"legacy OOF AUC = {legacy_auc:.6f} | new OOF AUC = {new_auc:.6f}")
    print(f"best iterations legacy = {legacy_iters} | new = {new_iters}")
    print(f"folds identical = {folds_identical} | max |Δp| = {max_abs_diff:.3e}")

    equivalent = max_abs_diff <= 1e-9 and folds_identical and iters_identical
    if equivalent:
        print("EQUIVALENT")
        sys.exit(0)

    print("NOT EQUIVALENT")
    sys.exit(1)


if __name__ == "__main__":
    main()
