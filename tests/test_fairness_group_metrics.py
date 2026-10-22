"""Tests for thresholding and group-level metrics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.group_metrics import (
    ALL_GROUP,
    ALL_GROUPING,
    METRIC_SPECS,
    group_counts,
    group_metrics,
    rate,
    utility,
)
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import YHAT_COL


def _add_yhat(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    thr = base_rate_threshold(out["score"].to_numpy(), out["y"].to_numpy())
    out[YHAT_COL] = apply_threshold(out["score"].to_numpy(), thr.threshold)
    return out


def test_threshold_flags_base_rate_share(make_audit_frame) -> None:
    frame = make_audit_frame()
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    assert thr.n_flagged == thr.n_positives
    yhat = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    expected = (frame["score"].to_numpy() > thr.threshold).astype(np.int8)
    np.testing.assert_array_equal(yhat, expected)


def test_threshold_with_ties_warns(caplog) -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    score = np.array([0.5] * 10, dtype=float)
    with caplog.at_level("WARNING"):
        base_rate_threshold(score, y)
    assert any("n_flagged" in rec.message for rec in caplog.records)


def test_toy_rates(toy_audit_frame) -> None:
    y = toy_audit_frame["y"].to_numpy()
    yhat = toy_audit_frame["yhat"].to_numpy()
    f_mask = toy_audit_frame["CODE_GENDER"] == "F"
    m_mask = toy_audit_frame["CODE_GENDER"] == "M"
    expected = {
        "F": {
            "approval_rate": (7, 10),
            "fpr": (2, 8),
            "fnr": (1, 2),
            "tpr": (1, 2),
            "ppv": (1, 3),
            "default_among_approved": (1, 7),
        },
        "M": {
            "approval_rate": (6, 10),
            "fpr": (1, 6),
            "fnr": (1, 4),
            "tpr": (3, 4),
            "ppv": (3, 4),
            "default_among_approved": (1, 6),
        },
    }
    rates_f = {
        "approval_rate": 0.7,
        "fpr": 2 / 8,
        "fnr": 0.5,
        "tpr": 0.5,
        "ppv": 1 / 3,
        "default_among_approved": 1 / 7,
    }
    rates_m = {
        "approval_rate": 0.6,
        "fpr": 1 / 6,
        "fnr": 0.25,
        "tpr": 0.75,
        "ppv": 0.75,
        "default_among_approved": 1 / 6,
    }
    for gender, metrics, rates in [
        ("F", expected["F"], rates_f),
        ("M", expected["M"], rates_m),
    ]:
        mask = f_mask if gender == "F" else m_mask
        for name, (events, denom) in metrics.items():
            assert rate(y[mask], yhat[mask], name) == (events, denom)
            assert abs(events / denom - rates[name]) < 1e-12


def test_toy_counts(toy_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    counts = group_counts(toy_audit_frame, config)
    f_row = counts[
        (counts["grouping"] == "CODE_GENDER") & (counts["group"] == "F")
    ].iloc[0]
    m_row = counts[
        (counts["grouping"] == "CODE_GENDER") & (counts["group"] == "M")
    ].iloc[0]
    assert f_row["n"] == 10 and f_row["positives"] == 2
    assert f_row["negatives"] == 8 and f_row["flagged"] == 3 and f_row["approved"] == 7
    assert m_row["n"] == 10 and m_row["positives"] == 4
    assert m_row["negatives"] == 6 and m_row["flagged"] == 4 and m_row["approved"] == 6
    all_row = counts[
        (counts["grouping"] == ALL_GROUPING) & (counts["group"] == ALL_GROUP)
    ].iloc[0]
    assert all_row["n"] == 20
    assert all_row["positives"] == f_row["positives"] + m_row["positives"]
    assert all_row["negatives"] == f_row["negatives"] + m_row["negatives"]
    assert all_row["flagged"] == f_row["flagged"] + m_row["flagged"]
    assert all_row["approved"] == f_row["approved"] + m_row["approved"]


def test_rates_use_metric_denominator(toy_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    metrics = group_metrics(toy_audit_frame, config, "lgbm", 1)
    for metric_name, denom_name in [
        ("fpr", "negatives"),
        ("fnr", "positives"),
        ("default_among_approved", "approved"),
    ]:
        spec = METRIC_SPECS[metric_name]
        assert spec.denominator == denom_name
        sub = metrics[metrics["metric"] == metric_name]
        counts = group_counts(toy_audit_frame, config)
        merged = sub.merge(
            counts,
            on=["grouping", "group"],
            suffixes=("_m", "_c"),
        )
        col = denom_name if denom_name != "n" else "n"
        np.testing.assert_array_equal(
            merged["denominator"].to_numpy(),
            merged[col].to_numpy(),
        )


def test_group_metrics_shape(make_audit_frame) -> None:
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=50)
    frame = _add_yhat(make_audit_frame())
    metrics = group_metrics(frame, config, "lgbm", 1)
    assert len(metrics) == 18 * 9
    for _, row in metrics.iterrows():
        if np.isnan(row["value"]):
            assert np.isnan(row["ci_low"]) and np.isnan(row["ci_high"])
        else:
            assert row["ci_low"] <= row["value"] <= row["ci_high"]


def test_brier_logloss_all_row(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _add_yhat(make_audit_frame(n=500, seed=3))
    metrics = group_metrics(frame, config, "lgbm", 1)
    y = frame["y"].to_numpy()
    score = frame["score"].to_numpy()
    all_metrics = metrics[
        (metrics["grouping"] == ALL_GROUPING) & (metrics["group"] == ALL_GROUP)
    ]
    brier_row = all_metrics[all_metrics["metric"] == "brier"].iloc[0]
    ll_row = all_metrics[all_metrics["metric"] == "log_loss"].iloc[0]
    assert abs(brier_row["value"] - brier_score_loss(y, score)) < 1e-12
    assert abs(ll_row["value"] - log_loss(y, score, labels=[0, 1])) < 1e-12


def test_auc_bootstrap_deterministic(make_audit_frame) -> None:
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=30)
    frame = _add_yhat(make_audit_frame(n=800, seed=1))
    m1 = group_metrics(frame, config, "lgbm", 1)
    m2 = group_metrics(frame, config, "lgbm", 1)
    pd.testing.assert_frame_equal(m1, m2)


def test_utility_identity(make_audit_frame) -> None:
    frame = make_audit_frame(n=2000, seed=5)
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    assert thr.n_flagged == thr.n_positives
    frame = frame.copy()
    frame[YHAT_COL] = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    run_metrics = {
        "oof_roc_auc": 0.75,
        "mean_fold_roc_auc": 0.74,
        "std_fold_roc_auc": 0.01,
        "fold_roc_aucs": [0.73, 0.74, 0.75, 0.76, 0.77],
        "oof_brier": 0.1,
        "brier_reference": 0.08,
        "oof_log_loss": 0.3,
        "log_loss_reference": 0.28,
    }
    util = utility(frame, thr, run_metrics)
    assert abs(util["tpr"] - util["ppv"]) < 1e-12
    assert abs(util["tpr"] - util["f1"]) < 1e-12
