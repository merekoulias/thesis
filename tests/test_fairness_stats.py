"""Tests for ``src.fairness.stats``."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, chi2_contingency
from sklearn.metrics import roc_auc_score

from analysis.helpers.data_audit import (
    proportion_gap,
    proportion_table,
    worst_group_summary,
)
from src.fairness.stats import (
    bootstrap_auc_ci,
    diff_ci,
    g_test,
    mean_ci,
    proportion_ci,
    ratio_ci,
    seed_for,
    stratified_g_test,
)
from src.fairness.stats import worst_group_summary as fairness_worst_group_summary
from src.fairness.stats import z_value


def test_z_value() -> None:
    assert z_value(0.95) == pytest.approx(1.959964, abs=1e-6)


def test_proportion_ci_known() -> None:
    z = z_value(0.95)
    se_term = z * np.sqrt(0.021)
    rate, lo, hi = proportion_ci(7, 10, z)
    assert rate == pytest.approx(0.7)
    assert lo == pytest.approx(0.7 - se_term)
    assert hi == pytest.approx(0.7 + se_term)
    assert proportion_ci(0, 5, z) == (0.0, 0.0, 0.0)
    assert proportion_ci(5, 5, z) == (1.0, 1.0, 1.0)
    # Clipping without collapse: the raw endpoint leaves [0, 1] on one side only.
    _, lo, hi = proportion_ci(1, 10, z)
    assert lo == 0.0 and 0.0 < hi < 1.0
    _, lo, hi = proportion_ci(9, 10, z)
    assert hi == 1.0 and 0.0 < lo < 1.0
    nan3 = proportion_ci(0, 0, z)
    assert all(np.isnan(v) for v in nan3)


def test_matches_data_audit_helpers() -> None:
    z = z_value(0.95)
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(5, 200))
        events = int(rng.integers(0, n + 1))
        rate, ci_lo, ci_hi = proportion_ci(events, n, z)
        frame = pd.DataFrame({"g": ["a"] * n, "y": [1] * events + [0] * (n - events)})
        tab = proportion_table(frame, ["g"], "y", z=z)
        assert rate == pytest.approx(tab.loc["a", "rate"], rel=1e-12)
        assert ci_lo == pytest.approx(tab.loc["a", "ci_low"], rel=1e-12, abs=1e-12)
        assert ci_hi == pytest.approx(tab.loc["a", "ci_high"], rel=1e-12)

    for _ in range(20):
        n_a = int(rng.integers(10, 300))
        n_b = int(rng.integers(10, 300))
        r_a = float(rng.random())
        r_b = float(rng.random())
        d, d_lo, d_hi = diff_ci(r_a, n_a, r_b, n_b, z)
        gap = proportion_gap(r_a, n_a, r_b, n_b, z=z)
        assert d == pytest.approx(gap["diff"], rel=1e-12, abs=1e-12)
        assert d_lo == pytest.approx(gap["diff_ci_low"], rel=1e-12, abs=1e-12)
        assert d_hi == pytest.approx(gap["diff_ci_high"], rel=1e-12, abs=1e-12)
        q, q_lo, q_hi = ratio_ci(r_a, n_a, r_b, n_b, z)
        if np.isnan(gap["ratio"]):
            assert np.isnan(q) and np.isnan(q_lo) and np.isnan(q_hi)
        else:
            assert q == pytest.approx(gap["ratio"], rel=1e-12, abs=1e-12)
            assert q_lo == pytest.approx(gap["ratio_ci_low"], rel=1e-12, abs=1e-12)
            assert q_hi == pytest.approx(gap["ratio_ci_high"], rel=1e-12, abs=1e-12)

    n_groups = [40, 55, 30, 80, 25]
    rates = [0.12, 0.18, 0.09, 0.22, 0.15]
    helper = worst_group_summary(n_groups, rates, n_boot=200, seed=7)
    ours = fairness_worst_group_summary(n_groups, rates, n_boot=200, seed=7, alpha=0.05)
    for key in helper:
        assert ours[key] == pytest.approx(helper[key], rel=1e-12, abs=1e-12)


def test_diff_ci_toy() -> None:
    z = z_value(0.95)
    diff, _, _ = diff_ci(0.7, 10, 0.6, 10, z)
    assert diff == pytest.approx(0.1)
    half_width = z * 0.2121320
    lo, hi = diff_ci(0.7, 10, 0.6, 10, z)[1:]
    assert diff - lo == pytest.approx(half_width, abs=1e-7)
    assert hi - diff == pytest.approx(half_width, abs=1e-7)


def test_ratio_ci_toy() -> None:
    z = z_value(0.95)
    ratio = 7.0 / 6.0
    q, q_lo, q_hi = ratio_ci(0.7, 10, 0.6, 10, z)
    assert q == pytest.approx(ratio)
    half_log = z * np.sqrt(0.3 / 7.0 + 0.4 / 6.0)
    assert q * np.exp(-half_log) == pytest.approx(q_lo)
    assert q * np.exp(half_log) == pytest.approx(q_hi)


@pytest.mark.parametrize("shape", [(2, 2), (10, 2), (3, 4)])
def test_g_test_matches_scipy(shape: tuple) -> None:
    rng = np.random.default_rng(42)
    for _ in range(5):
        table = rng.integers(1, 50, size=shape)
        g, df, p = g_test(table)
        ref = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        assert g == pytest.approx(ref.statistic, rel=1e-10)
        assert df == ref.dof
        assert p == pytest.approx(ref.pvalue, rel=1e-10)


def test_g_test_zero_rows_and_cols() -> None:
    full = np.array([[5, 0, 3], [0, 0, 0], [2, 0, 4]])
    reduced = np.array([[5, 3], [2, 4]])
    assert g_test(full) == g_test(reduced)
    single_row = np.array([[1, 2, 3]])
    g0, df0, p0 = g_test(single_row)
    assert g0 == 0.0 and df0 == 0 and np.isnan(p0)


def test_stratified_g_test() -> None:
    t1 = np.array([[10, 5], [8, 12]])
    t2 = np.array([[6, 9], [11, 4]])
    g1, df1, _ = g_test(t1)
    g2, df2, _ = g_test(t2)
    g_sum, df_sum, p_sum = stratified_g_test([t1, t2])
    assert g_sum == pytest.approx(g1 + g2)
    assert df_sum == df1 + df2
    assert p_sum == pytest.approx(float(chi2.sf(g_sum, df_sum)))


def test_mean_ci() -> None:
    z = z_value(0.95)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    m, lo, hi = mean_ci(values, z)
    assert m == pytest.approx(2.5)
    se = 0.6454972243679028  # sqrt(5 / 3) / 2, by hand
    assert lo == pytest.approx(2.5 - z * se)
    assert hi == pytest.approx(2.5 + z * se)


def test_mean_ci_degenerate() -> None:
    z = z_value(0.95)
    assert all(np.isnan(v) for v in mean_ci(np.array([]), z))
    m, lo, hi = mean_ci(np.array([0.3]), z)
    assert m == pytest.approx(0.3) and np.isnan(lo) and np.isnan(hi)


def test_bootstrap_auc_ci_deterministic() -> None:
    rng = np.random.default_rng(99)
    n = 500
    y = (rng.random(n) < 0.2).astype(int)
    score = np.clip(rng.normal(y * 0.3 + 0.1, 0.2, n), 1e-6, 1 - 1e-6)
    a = bootstrap_auc_ci(y, score, n_boot=100, seed=11, alpha=0.05)
    b = bootstrap_auc_ci(y, score, n_boot=100, seed=11, alpha=0.05)
    assert a == b
    point, ci_lo, ci_hi = a
    assert point == pytest.approx(roc_auc_score(y, score))
    assert ci_lo <= point <= ci_hi


def test_seed_for_stable() -> None:
    key = "auc|lgbm|1|CODE_GENDER|F"
    s1 = seed_for(42, key)
    s2 = seed_for(42, key)
    assert s1 == s2
    assert 0 <= s1 < 2**32
    assert seed_for(42, "other") != s1
    # Pinned value: guards against any per-process randomness such as hash().
    assert s1 == 367178000


def test_zero_size_groups_give_nan() -> None:
    z = z_value(0.95)
    for out in (
        diff_ci(float("nan"), 0, 0.5, 10, z),
        diff_ci(0.5, 10, float("nan"), 0, z),
        ratio_ci(float("nan"), 0, 0.5, 10, z),
        ratio_ci(0.5, 10, 0.0, 10, z),
    ):
        assert all(np.isnan(v) for v in out)
    assert all(isinstance(v, float) for v in ratio_ci(0.7, 10, 0.6, 10, z))


def test_bootstrap_auc_ci_undefined_groups() -> None:
    score = np.linspace(0.0, 1.0, 5)
    for y, s in [
        (np.zeros(5, dtype=int), score),
        (np.ones(5, dtype=int), score),
        (np.zeros(0, dtype=int), np.zeros(0)),
    ]:
        out = bootstrap_auc_ci(y, s, n_boot=20, seed=1, alpha=0.05)
        assert all(np.isnan(v) for v in out)
