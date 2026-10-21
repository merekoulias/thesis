"""Model-level classification metrics and fit diagnostics (no groups, no threshold)."""

import math
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _validate_binary_labels(y_true: np.ndarray) -> None:
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes for classification metrics")


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Return {"roc_auc", "brier", "log_loss"} (definitions in plan 5.10).

    Args:
        y_true: Binary labels, shape (n,).
        proba: Predicted P(y=1), shape (n,).

    Returns:
        Dictionary of metric name to value.

    Raises:
        ValueError: If ``y_true`` has a single class.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    _validate_binary_labels(y_true)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
    }


def reference_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """Return the no-skill references for predicting the base rate for everyone.

    Args:
        y_true: Binary labels, shape (n,).

    Returns:
        {"base_rate", "brier_reference", "log_loss_reference"} for the empirical
        base rate.

    Raises:
        ValueError: If ``y_true`` has a single class (log loss reference undefined).
    """
    y_true = np.asarray(y_true)
    _validate_binary_labels(y_true)
    base_rate = float(np.mean(y_true))
    brier_reference = base_rate * (1.0 - base_rate)
    log_loss_reference = -(
        base_rate * math.log(base_rate) + (1.0 - base_rate) * math.log(1.0 - base_rate)
    )
    return {
        "base_rate": base_rate,
        "brier_reference": float(brier_reference),
        "log_loss_reference": float(log_loss_reference),
    }


def fold_fit_diagnostics(
    y_train: np.ndarray,
    train_proba: np.ndarray,
    y_val: np.ndarray,
    val_proba: np.ndarray,
    mean_inner_roc_auc: float,
) -> Dict[str, float]:
    """Return train_/val_ metrics, the three gaps and selection_optimism (plan 5.10).

    Args:
        y_train: Outer training labels.
        train_proba: In-sample probabilities on the outer training fold.
        y_val: Outer validation labels.
        val_proba: Out-of-sample probabilities on the outer validation fold.
        mean_inner_roc_auc: Mean inner ROC AUC of the selected candidate.

    Returns:
        Per-fold fit diagnostic metrics.
    """
    train_m = classification_metrics(y_train, train_proba)
    val_m = classification_metrics(y_val, val_proba)
    return {
        "train_roc_auc": train_m["roc_auc"],
        "val_roc_auc": val_m["roc_auc"],
        "roc_auc_gap": train_m["roc_auc"] - val_m["roc_auc"],
        "train_brier": train_m["brier"],
        "val_brier": val_m["brier"],
        "brier_gap": val_m["brier"] - train_m["brier"],
        "train_log_loss": train_m["log_loss"],
        "val_log_loss": val_m["log_loss"],
        "log_loss_gap": val_m["log_loss"] - train_m["log_loss"],
        "selection_optimism": float(mean_inner_roc_auc) - val_m["roc_auc"],
    }


def summarize_folds(folds: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return the means of the three gaps and the selection optimism over folds.

    Args:
        folds: One dict per outer fold from ``fold_fit_diagnostics`` (plus metadata).

    Returns:
        {"mean_roc_auc_gap", "mean_brier_gap", "mean_log_loss_gap",
        "mean_selection_optimism"}.

    Raises:
        ValueError: If ``folds`` is empty.
    """
    if not folds:
        raise ValueError("folds must not be empty")
    return {
        "mean_roc_auc_gap": float(np.mean([f["roc_auc_gap"] for f in folds])),
        "mean_brier_gap": float(np.mean([f["brier_gap"] for f in folds])),
        "mean_log_loss_gap": float(np.mean([f["log_loss_gap"] for f in folds])),
        "mean_selection_optimism": float(
            np.mean([f["selection_optimism"] for f in folds])
        ),
    }
