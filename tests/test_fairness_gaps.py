"""Tests for pairwise gaps and cell summaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.gaps import cell_summary, pair_gaps
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import YHAT_COL, PairSpec


def _one_pair_config() -> object:
    base = load_audit_config(default_fairness_config_path())
    fm = next(p for p in base.pairs if p.pair_id == "F-M")
    return replace(
        base,
        pairs=[fm],
        gap_metrics=["approval_rate", "fpr", "fnr"],
        n_boot=50,
    )


def _scipy_g(table: np.ndarray) -> tuple:
    arr = np.asarray(table, dtype=float)
    g, _, _, _ = chi2_contingency(arr, correction=False, lambda_="log-likelihood")
    return float(g)


def test_toy_gaps(toy_audit_frame) -> None:
    config = _one_pair_config()
    gaps = pair_gaps(toy_audit_frame, config)
    row = gaps[gaps["metric"] == "approval_rate"].iloc[0]
    assert row["diff"] == pytest.approx(0.1, rel=0, abs=1e-12)
    assert row["ratio"] == pytest.approx(7 / 6, rel=0, abs=1e-12)
    assert row["ratio_lower_over_higher"] == pytest.approx(6 / 7, rel=0, abs=1e-12)
    fpr = gaps[gaps["metric"] == "fpr"].iloc[0]
    assert fpr["diff"] == pytest.approx(0.0833333, rel=0, abs=1e-6)
    fnr = gaps[gaps["metric"] == "fnr"].iloc[0]
    assert fnr["diff"] == pytest.approx(0.25, rel=0, abs=1e-12)

    for metric in ("approval_rate", "fpr", "fnr"):
        g_row = gaps[gaps["metric"] == metric].iloc[0]
        ev_a = int(g_row["rate_a"] * g_row["den_a"])
        ev_b = int(g_row["rate_b"] * g_row["den_b"])
        int_table = np.array(
            [[ev_a, g_row["den_a"] - ev_a], [ev_b, g_row["den_b"] - ev_b]],
            dtype=float,
        )
        assert g_row["g_stat"] == pytest.approx(_scipy_g(int_table), rel=0, abs=1e-9)


def test_primary_flag(make_audit_frame) -> None:
    frame = make_audit_frame()
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    frame = frame.copy()
    frame[YHAT_COL] = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=20)
    gaps = pair_gaps(frame, config)
    primary = gaps[gaps["primary"]]
    assert len(primary) == 2
    assert set(primary["metric"]) == {"fpr", "fnr"}
    assert (primary["pair_id"] == "F-M").all()


def test_gap_rows(make_audit_frame) -> None:
    frame = make_audit_frame()
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    frame = frame.copy()
    frame[YHAT_COL] = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=20)
    gaps = pair_gaps(frame, config)
    assert len(gaps) == 9 * 4
    non_approval = gaps[gaps["metric"] != "approval_rate"]
    for col in ("ratio", "ratio_ci_low", "ratio_ci_high", "ratio_lower_over_higher"):
        assert non_approval[col].isna().all()
    approval = gaps[gaps["metric"] == "approval_rate"]
    assert approval["ratio"].notna().all()


def test_sign_convention(toy_audit_frame) -> None:
    config = _one_pair_config()
    forward = pair_gaps(toy_audit_frame, config)
    pair = config.pairs[0]
    swapped = replace(
        config,
        pairs=[PairSpec(pair_id=pair.pair_id, a=pair.b, b=pair.a)],
    )
    backward = pair_gaps(toy_audit_frame, swapped)
    for metric in config.gap_metrics:
        f = forward[forward["metric"] == metric].iloc[0]
        b = backward[backward["metric"] == metric].iloc[0]
        assert b["diff"] == pytest.approx(-f["diff"], rel=0, abs=1e-12)
        if metric == "approval_rate":
            assert b["ratio"] == pytest.approx(1.0 / f["ratio"], rel=0, abs=1e-12)


def test_cell_summary(toy_audit_frame) -> None:
    config = replace(
        load_audit_config(default_fairness_config_path()),
        cell_summary_metrics=["approval_rate", "fpr", "fnr"],
        n_boot=100,
    )
    summary1 = cell_summary(toy_audit_frame, config, "lgbm", 1)
    summary2 = cell_summary(toy_audit_frame, config, "lgbm", 1)
    pd.testing.assert_frame_equal(summary1, summary2)
    assert len(summary1) == 3

    y = toy_audit_frame["y"].to_numpy()
    yhat = toy_audit_frame["yhat"].to_numpy()
    labels = config.groupings["GENDER_AGE"]
    for _, row in summary1.iterrows():
        metric = row["metric"]
        rates = []
        for label in labels:
            mask = (toy_audit_frame["GENDER_AGE"] == label).to_numpy()
            from src.fairness.group_metrics import rate as rate_fn

            ev, den = rate_fn(y[mask], yhat[mask], metric)
            rates.append(ev / den if den else float("nan"))
        expected_gap = float(np.nanmax(rates) - np.nanmin(rates))
        assert row["max_min_gap"] == pytest.approx(expected_gap, rel=0, abs=1e-12)
        max_label = labels[int(np.nanargmax(rates))]
        min_label = labels[int(np.nanargmin(rates))]
        assert row["max_cell"] == max_label
        assert row["min_cell"] == min_label


def test_cell_g_test_df(make_audit_frame) -> None:
    frame = make_audit_frame(n=4000, seed=1)
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    frame = frame.copy()
    frame[YHAT_COL] = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    config = replace(
        load_audit_config(default_fairness_config_path()),
        n_boot=30,
        cell_summary_metrics=["fpr"],
    )
    summary = cell_summary(frame, config, "lgbm", 1)
    assert summary.iloc[0]["df"] == 9
