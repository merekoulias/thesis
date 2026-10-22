"""Tests for pairwise gaps and cell summaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.gaps import LUM_SUMMARY_KEYS, cell_summary, pair_gaps
from src.fairness.group_metrics import rate
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import YHAT_COL, AuditConfig, PairSpec

# Plan 5.7, F - M: (events, denominator) per side and metric.
TOY_TABLES = {
    "approval_rate": ((7, 10), (6, 10)),
    "fpr": ((2, 8), (1, 6)),
    "fnr": ((1, 2), (1, 4)),
    "default_among_approved": ((1, 7), (1, 6)),
}
TOY_DIFFS = {
    "approval_rate": 0.1,
    "fpr": 1 / 12,
    "fnr": 0.25,
    "default_among_approved": 1 / 7 - 1 / 6,
}


def _one_pair_config() -> AuditConfig:
    base = load_audit_config(default_fairness_config_path())
    fm = next(p for p in base.pairs if p.pair_id == "F-M")
    return replace(base, pairs=[fm], n_boot=50)


def _scipy_g(table: np.ndarray) -> float:
    arr = np.asarray(table, dtype=float)
    g, _, _, _ = chi2_contingency(arr, correction=False, lambda_="log-likelihood")
    return float(g)


def _scipy_p(table: np.ndarray) -> float:
    arr = np.asarray(table, dtype=float)
    _, p, _, _ = chi2_contingency(arr, correction=False, lambda_="log-likelihood")
    return float(p)


def test_toy_gaps(toy_audit_frame) -> None:
    config = _one_pair_config()
    gaps = pair_gaps(toy_audit_frame, config)
    assert list(gaps["metric"]) == list(TOY_TABLES)
    z = config.z

    for metric, ((ev_a, den_a), (ev_b, den_b)) in TOY_TABLES.items():
        row = gaps[gaps["metric"] == metric].iloc[0]
        r_a, r_b = ev_a / den_a, ev_b / den_b
        assert row["den_a"] == den_a and row["den_b"] == den_b
        assert row["rate_a"] == pytest.approx(r_a, rel=0, abs=1e-12)
        assert row["rate_b"] == pytest.approx(r_b, rel=0, abs=1e-12)
        assert row["diff"] == pytest.approx(TOY_DIFFS[metric], rel=0, abs=1e-12)
        se = np.sqrt(r_a * (1 - r_a) / den_a + r_b * (1 - r_b) / den_b)
        diff = TOY_DIFFS[metric]
        assert row["diff_ci_low"] == pytest.approx(diff - z * se, rel=0, abs=1e-12)
        assert row["diff_ci_high"] == pytest.approx(diff + z * se, rel=0, abs=1e-12)

        table = np.array([[ev_a, den_a - ev_a], [ev_b, den_b - ev_b]])
        assert row["g_stat"] == pytest.approx(_scipy_g(table), rel=0, abs=1e-12)
        assert row["df"] == 1
        assert row["p_value"] == pytest.approx(_scipy_p(table), rel=0, abs=1e-12)

    approval = gaps[gaps["metric"] == "approval_rate"].iloc[0]
    se_approval = approval["diff"] - approval["diff_ci_low"]
    assert se_approval == pytest.approx(z * np.sqrt(0.045), rel=0, abs=1e-12)
    assert approval["ratio"] == pytest.approx(7 / 6, rel=0, abs=1e-12)
    assert approval["ratio_lower_over_higher"] == pytest.approx(6 / 7, rel=0, abs=1e-12)


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
            ev, den = rate(y[mask], yhat[mask], metric)
            rates.append(ev / den if den else float("nan"))
        assert row["k_groups"] == int(np.sum(~np.isnan(rates)))
        expected_gap = float(np.nanmax(rates) - np.nanmin(rates))
        assert row["max_min_gap"] == pytest.approx(expected_gap, rel=0, abs=1e-12)
        max_label = labels[int(np.nanargmax(rates))]
        min_label = labels[int(np.nanargmin(rates))]
        assert row["max_cell"] == max_label
        assert row["min_cell"] == min_label


@pytest.mark.filterwarnings("error")
def test_cell_summary_fewer_than_two_cells_is_nan() -> None:
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=20)
    # No positives anywhere: fnr has 0 non-empty cells; one cell only: 1 non-empty.
    frames = {
        0: pd.DataFrame(
            {
                "y": np.zeros(8, dtype=np.int8),
                YHAT_COL: np.array([0, 1] * 4, dtype=np.int8),
                "GENDER_AGE": ["F | 20-29"] * 4 + ["M | 20-29"] * 4,
            }
        ),
        1: pd.DataFrame(
            {
                "y": np.array([0, 1, 0, 1], dtype=np.int8),
                YHAT_COL: np.array([0, 1, 1, 0], dtype=np.int8),
                "GENDER_AGE": ["F | 30-39"] * 4,
            }
        ),
    }
    for n_active, frame in frames.items():
        row = cell_summary(frame, config, "lgbm", 1).set_index("metric").loc["fnr"]
        assert row["k_groups"] == n_active
        for key in LUM_SUMMARY_KEYS:
            assert np.isnan(row[key])
        assert (
            row["max_cell"] == row["min_cell"] == ("" if n_active == 0 else "F | 30-39")
        )
        assert row["df"] == 0


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
    y = frame["y"].to_numpy()
    yhat = frame[YHAT_COL].to_numpy()
    for label in config.groupings["GENDER_AGE"]:
        mask = (frame["GENDER_AGE"] == label).to_numpy()
        events, den = rate(y[mask], yhat[mask], "fpr")
        assert 0 < events < den, label
    summary = cell_summary(frame, config, "lgbm", 1)
    assert summary.iloc[0]["df"] == 9
