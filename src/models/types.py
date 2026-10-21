"""Shared types for credit-scoring model training."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# A (train_indices, validation_indices) pair of integer index arrays.
Split = Tuple[np.ndarray, np.ndarray]

# (params, seed) -> unfitted estimator with fit / predict_proba.
EstimatorBuilder = Callable[[Dict[str, Any], int], Any]

# (fitted estimator, feature names) -> long frame: feature, importance_type, importance.
ImportanceFn = Callable[[Any, List[str]], pd.DataFrame]


@dataclass(frozen=True)
class CVSettings:
    """Nested cross-validation settings shared by every model."""

    seed: int
    n_outer_folds: int
    n_inner_folds: int
    n_configs: int
    top_k: int


@dataclass(frozen=True)
class ModelConfig:
    """One model's fixed parameters, default candidate and search space."""

    name: str
    fixed_params: Dict[str, Any]
    default_params: Dict[str, Any]
    search_space: Dict[str, Dict[str, Any]]
    early_stopping_rounds: Optional[int] = None


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry: how to build, fit and explain one model family."""

    name: str
    build: EstimatorBuilder
    uses_early_stopping: bool = False
    suppress_convergence_warnings: bool = False
    importance: Optional[ImportanceFn] = None


@dataclass
class FitOutput:
    """Result of fitting one estimator on one split."""

    proba: np.ndarray
    best_iteration: Optional[int]
    fit_seconds: float
    estimator: Optional[Any] = None
    train_proba: Optional[np.ndarray] = None


@dataclass
class OuterFoldSelection:
    """Candidate ranking inside one outer fold."""

    outer_fold: int
    ranking: List[int]
    mean_inner_auc: Dict[int, float]
    mean_best_iteration: Dict[int, Optional[float]]


@dataclass
class RankedRun:
    """OOF predictions when every outer fold uses its rank-r candidate."""

    rank: int
    oof_predictions: pd.DataFrame
    fold_roc_aucs: List[float]
    oof_roc_auc: float
    oof_brier: float
    oof_log_loss: float
    fold_metrics: List[Dict[str, Any]]
    selected: List[Dict[str, Any]]


@dataclass
class NestedCVResult:
    """Everything produced by one nested-CV run of one model."""

    model_name: str
    candidates: List[Dict[str, Any]]
    inner_scores: pd.DataFrame
    selections: List[OuterFoldSelection]
    runs: Dict[int, RankedRun]
    outer_splits: List[Split]
    feature_importance: Optional[pd.DataFrame]
    runtime_seconds: float


INNER_SCORE_COLUMNS: List[str] = [
    "outer_fold",
    "candidate_id",
    "inner_fold",
    "roc_auc",
    "best_iteration",
    "fit_seconds",
]
