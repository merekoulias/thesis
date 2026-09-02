"""Load feature artifacts and model configuration."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml

from src.constants import PROTECTED_ATTRIBUTES, SK_ID_CURR, TARGET_COL
from src.models.types import TrainingData

PathLike = Union[str, Path]


def load_yaml_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature artifact: {path}. "
            "Run: python -m src.features.pipeline"
        )
    return path


def _validate_protected_columns(feature_names: list) -> None:
    """Ensure encoded protected-attribute columns are present in X."""
    missing = []
    for attr in PROTECTED_ATTRIBUTES:
        if not any(attr in col for col in feature_names):
            missing.append(attr)
    if missing:
        raise ValueError(
            f"Encoded protected attributes missing from X: {missing}. "
            f"Expected columns containing {PROTECTED_ATTRIBUTES}."
        )


def load_training_data(features_dir: PathLike) -> TrainingData:
    """Load and validate model-ready training data from feature artifacts.

    Args:
        features_dir: Directory with feature parquet files.

    Returns:
        TrainingData with aligned X, y, and ids.

    Raises:
        FileNotFoundError: If required artifacts are missing.
        ValueError: On schema, alignment, or data-quality failures.
    """
    feat_dir = Path(features_dir)
    X = pd.read_parquet(_require_file(feat_dir, "X_train.parquet"))
    y_frame = pd.read_parquet(_require_file(feat_dir, "y_train.parquet"))
    ids_frame = pd.read_parquet(_require_file(feat_dir, "ids_train.parquet"))

    if TARGET_COL not in y_frame.columns:
        raise ValueError(f"y_train must contain column {TARGET_COL!r}")
    if SK_ID_CURR not in ids_frame.columns:
        raise ValueError(f"ids_train must contain column {SK_ID_CURR!r}")

    y = y_frame[TARGET_COL]
    ids = ids_frame[SK_ID_CURR]

    n = len(X)
    if len(y) != n:
        raise ValueError(
            f"Row count mismatch: X_train={n}, y_train={len(y)}"
        )
    if len(ids) != n:
        raise ValueError(
            f"Row count mismatch: X_train={n}, ids_train={len(ids)}"
        )

    if ids.duplicated().any():
        dupes = int(ids.duplicated().sum())
        raise ValueError(f"ids_train contains {dupes} duplicate SK_ID_CURR values")

    if TARGET_COL in X.columns:
        raise ValueError(f"{TARGET_COL} must not appear in X_train")

    # Binary target check
    unique_targets = set(y.dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"TARGET must be binary 0/1; found values: {sorted(unique_targets)}"
        )
    if y.isna().any():
        raise ValueError("TARGET contains missing values")

    # Numeric, finite features
    non_numeric = [
        c
        for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c])
    ]
    if non_numeric:
        raise ValueError(
            f"X_train has non-numeric columns: {non_numeric[:5]}"
        )

    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_cols = [
            c for c in X.columns if not np.isfinite(X[c].to_numpy(dtype=float)).all()
        ]
        raise ValueError(
            f"X_train has non-finite values in columns: {bad_cols[:5]}"
        )

    feature_names = list(X.columns)
    _validate_protected_columns(feature_names)

    return TrainingData(
        X=X,
        y=y.astype(np.int8),
        ids=ids,
        feature_names=feature_names,
    )


def load_feature_metadata(features_dir: PathLike) -> Dict[str, Any]:
    """Load feature metadata.json if present."""
    meta_path = Path(features_dir) / "metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)
