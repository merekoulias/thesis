"""Tests for src.models.metrics."""

import math

import numpy as np
import pytest

from src.models.metrics import (
    classification_metrics,
    fold_fit_diagnostics,
    reference_metrics,
    summarize_folds,
)


def test_classification_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = classification_metrics(y, p)
    assert m["roc_auc"] == 0.75
    expected_brier = float(np.mean((p - y) ** 2))
    assert m["brier"] == expected_brier
    assert m["brier"] == pytest.approx(0.158125, abs=1e-12)
    hand_ll = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert m["log_loss"] == pytest.approx(hand_ll, abs=1e-12)


def test_reference_metrics():
    y = np.array([1] * 2 + [0] * 23)
    ref = reference_metrics(y)
    assert ref["base_rate"] == 0.08
    assert ref["brier_reference"] == pytest.approx(0.0736, abs=1e-12)
    b = 0.08
    expected_ll = -(b * math.log(b) + (1 - b) * math.log(1 - b))
    assert ref["log_loss_reference"] == pytest.approx(expected_ll, abs=1e-12)
    p_const = np.full(25, 0.08)
    m = classification_metrics(y, p_const)
    assert m["brier"] == pytest.approx(ref["brier_reference"], abs=1e-12)
    assert m["log_loss"] == pytest.approx(ref["log_loss_reference"], abs=1e-12)


def test_gap_signs():
    y_train = np.array([0, 0, 1, 1])
    train_proba = y_train.astype(float)
    y_val = np.array([0, 1, 0, 1])
    val_proba = np.array([0.6, 0.7, 0.4, 0.2])
    mean_inner = 0.9
    d = fold_fit_diagnostics(y_train, train_proba, y_val, val_proba, mean_inner)
    assert d["roc_auc_gap"] > 0
    assert d["brier_gap"] > 0
    assert d["log_loss_gap"] > 0
    assert d["selection_optimism"] == pytest.approx(
        mean_inner - d["val_roc_auc"], abs=1e-12
    )


def test_summarize_folds():
    folds = [
        {
            "roc_auc_gap": 0.1,
            "brier_gap": 0.02,
            "log_loss_gap": 0.03,
            "selection_optimism": 0.01,
        },
        {
            "roc_auc_gap": 0.2,
            "brier_gap": 0.04,
            "log_loss_gap": 0.05,
            "selection_optimism": 0.02,
        },
    ]
    s = summarize_folds(folds)
    assert s["mean_roc_auc_gap"] == pytest.approx(0.15, abs=1e-12)
    assert s["mean_brier_gap"] == pytest.approx(0.03, abs=1e-12)
    assert s["mean_log_loss_gap"] == pytest.approx(0.04, abs=1e-12)
    assert s["mean_selection_optimism"] == pytest.approx(0.015, abs=1e-12)


def test_single_class_raises():
    with pytest.raises(ValueError):
        classification_metrics(np.zeros(5), np.full(5, 0.5))


def test_reference_metrics_single_class_raises():
    with pytest.raises(ValueError):
        reference_metrics(np.zeros(5))


def test_summarize_folds_empty_raises():
    with pytest.raises(ValueError):
        summarize_folds([])
