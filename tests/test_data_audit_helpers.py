"""Contract tests for the pre-modelling bias audit helpers."""

import numpy as np
import pandas as pd
import pytest

from analysis.helpers.data_audit import (
    cmh_test,
    discretize,
    feldman_ber_threshold,
    outcome_by_missingness,
    proportion_gap,
    proportion_table,
    worst_group_summary,
)


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["F"] * 10 + ["M"] * 10,
            "stratum": (["a"] * 5 + ["b"] * 5) * 2,
            "y": [1, 0, 0, 0, 0, 1, 1, 0, 0, 0] + [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        }
    )


def test_proportion_table_counts_and_interval_bounds():
    table = proportion_table(_toy_frame(), ["group"], "y")
    assert table.loc["F", "n"] == 10
    assert table.loc["F", "events"] == 3
    assert table.loc["M", "rate"] == pytest.approx(0.5)
    assert (table["ci_low"] >= 0).all() and (table["ci_high"] <= 1).all()
    assert table["share"].sum() == pytest.approx(1.0)


def test_proportion_gap_sign_and_ratio():
    gap = proportion_gap(0.10, 1000, 0.05, 2000)
    assert gap["diff"] == pytest.approx(0.05)
    assert gap["diff_ci_low"] < gap["diff"] < gap["diff_ci_high"]
    assert gap["ratio"] == pytest.approx(2.0)
    assert gap["ratio_ci_low"] < 2.0 < gap["ratio_ci_high"]
    assert 0 <= gap["p_value"] < 0.05


def test_worst_group_summary_corrects_downwards():
    summary = worst_group_summary(
        n=[200, 200, 200], rate=[0.10, 0.12, 0.08], n_boot=200, seed=0
    )
    assert summary["max_min_gap"] == pytest.approx(0.04)
    assert summary["var_single_corrected"] <= summary["var_uncorrected"]
    assert summary["var_double_corrected_ci_low"] >= 0


def test_cmh_odds_ratio_direction():
    result = cmh_test(_toy_frame(), "group", "y", "stratum", group_a="M")
    assert result["k_strata"] == 2
    assert result["mh_odds_ratio"] > 1
    assert 0 <= result["p_value"] <= 1


def test_discretize_keeps_missing_as_category():
    series = pd.Series([float(i) for i in range(30)] + [np.nan])
    binned = discretize(series, n_bins=5)
    assert binned.nunique() == 6
    assert binned.iloc[-1] == "__MISSING__"


def test_outcome_by_missingness_skips_groups_without_both_states():
    indicators = pd.DataFrame({"col": [True, False, True, False, False, False]})
    outcome = pd.Series([1, 0, 1, 0, 0, 1])
    groups = pd.Series(["F", "F", "F", "M", "M", "M"])
    table = outcome_by_missingness(indicators, outcome, groups)
    assert list(table["group"]) == ["F"]
    assert table.iloc[0]["rate_missing"] == pytest.approx(1.0)


def test_feldman_threshold_four_fifths():
    assert feldman_ber_threshold(0.8) == pytest.approx(0.5 - 0.8 / 8)
