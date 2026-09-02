"""Shared types for credit-scoring model training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.constants import SK_ID_CURR, TARGET_COL


@dataclass
class TrainingData:
    """Aligned feature matrix, target, and applicant identifiers."""

    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    feature_names: List[str]


@dataclass
class FoldResult:
    """Per-fold training outcome."""

    fold: int
    roc_auc: float
    best_iteration: int
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass
class CVResult:
    """Out-of-fold cross-validation results."""

    oof_predictions: pd.DataFrame
    fold_results: List[FoldResult]
    fold_roc_aucs: List[float]
    mean_fold_roc_auc: float
    std_fold_roc_auc: float
    oof_roc_auc: float
    roc_curve: Dict[str, List[float]]
    models: List[Any] = field(default_factory=list)
    feature_importance_gain: Optional[pd.DataFrame] = None
    feature_importance_split: Optional[pd.DataFrame] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelArtifacts:
    """In-memory model training outputs."""

    cv_result: CVResult
    training_data: TrainingData


# OOF prediction column names
OOF_PROBA_COL = "y_pred_proba"
OOF_FOLD_COL = "fold"
