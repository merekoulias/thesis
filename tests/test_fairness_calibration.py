"""Tests for calibration deciles and sufficiency tests."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.fairness.calibration import (
    calibration_table,
    decile_bins,
    sufficiency_tests,
)
from src.fairness.config import default_fairness_config_path, load_audit_config


def _gender_frame(score: np.ndarray, gender: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Frame with one age band, so only the gender split varies."""
    return pd.DataFrame(
        {
            "y": y,
            "score": score,
            "CODE_GENDER": gender,
            "AGE_BAND": ["20-29"] * len(score),
            "GENDER_AGE": [f"{g} | 20-29" for g in gender],
        }
    )


def test_decile_bins_equal_frequency() -> None:
    score = np.linspace(0.0, 1.0, 1000)
    _, bins = decile_bins(score, 10)
    assert bins.min() == 0
    assert bins.max() == 9
    for b in range(10):
        assert int((bins == b).sum()) == 100


def test_calibration_counts_sum(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = make_audit_frame()
    table = calibration_table(frame, config)
    n_rows = len(frame)
    for grouping in table["grouping"].unique():
        sub = table[table["grouping"] == grouping]
        assert int(sub["n"].sum()) == n_rows


def test_bins_from_pooled_score() -> None:
    """Disjoint score ranges: pooled deciles put F in bins 0-4 and M in 5-9."""
    config = load_audit_config(default_fairness_config_path())
    rng = np.random.default_rng(7)
    score = np.r_[rng.uniform(0.0, 0.4, 1000), rng.uniform(0.6, 1.0, 1000)]
    gender = np.array(["F"] * 1000 + ["M"] * 1000)
    y = (rng.random(2000) < score).astype(np.int8)
    frame = _gender_frame(score, gender, y)

    table = calibration_table(frame, config)
    gender_rows = table[table["grouping"] == "CODE_GENDER"]
    nonempty = gender_rows[gender_rows["n"] > 0]
    assert sorted(nonempty.loc[nonempty["group"] == "F", "bin"]) == [0, 1, 2, 3, 4]
    assert sorted(nonempty.loc[nonempty["group"] == "M", "bin"]) == [5, 6, 7, 8, 9]
    empty = gender_rows[gender_rows["n"] == 0]
    assert empty[["observed_rate", "mean_score", "calibration_gap"]].isna().all().all()

    tests = sufficiency_tests(frame, config)
    gender_row = tests[tests["grouping"] == "CODE_GENDER"].iloc[0]
    assert gender_row["df"] == 0
    assert gender_row["g_stat"] == 0.0


def test_sufficiency_df_counts_groups_present() -> None:
    """df = sum over bins of (groups present - 1); G = sum of per-bin scipy G."""
    config = load_audit_config(default_fairness_config_path())
    rng = np.random.default_rng(3)
    # F covers [0, 1], M only [0.5, 1]: bins 0 and 1 of the pooled score hold F only.
    score = np.r_[rng.uniform(0.0, 1.0, 1000), rng.uniform(0.5, 1.0, 1000)]
    gender = np.array(["F"] * 1000 + ["M"] * 1000)
    y = (rng.random(2000) < score).astype(np.int8)
    frame = _gender_frame(score, gender, y)

    _, bins = decile_bins(score, config.calibration_bins)
    expected_g = 0.0
    expected_df = 0
    for b in range(config.calibration_bins):
        in_bin = bins == b
        assert len(np.unique(y[in_bin])) == 2
        table = pd.crosstab(gender[in_bin], y[in_bin]).to_numpy()
        if table.shape[0] > 1:
            stat, _, dof, _ = chi2_contingency(
                table, correction=False, lambda_="log-likelihood"
            )
            expected_g += stat
            expected_df += dof
    assert set(np.unique(gender[bins <= 1])) == {"F"}
    assert expected_df == 8

    tests = sufficiency_tests(frame, config)
    gender_row = tests[tests["grouping"] == "CODE_GENDER"].iloc[0]
    assert gender_row["df"] == expected_df
    np.testing.assert_allclose(gender_row["g_stat"], expected_g, rtol=1e-12)


def test_perfectly_calibrated_synthetic() -> None:
    config = load_audit_config(default_fairness_config_path())
    rng = np.random.default_rng(12345)
    n = 20_000
    score = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < score).astype(np.int8)
    gender = rng.choice(["F", "M"], size=n)
    band = rng.choice(config.groupings["AGE_BAND"], size=n)
    frame = pd.DataFrame(
        {
            "y": y,
            "score": score,
            "CODE_GENDER": gender,
            "AGE_BAND": band,
            "GENDER_AGE": [f"{g} | {b}" for g, b in zip(gender, band)],
        }
    )
    table = calibration_table(frame, config)
    all_rows = table[(table["grouping"] == "ALL") & (table["group"] == "all")]
    covered = (
        (all_rows["ci_low"] <= all_rows["mean_score"])
        & (all_rows["mean_score"] <= all_rows["ci_high"])
    ).sum()
    assert covered >= 9
    tests = sufficiency_tests(frame, config)
    assert (tests["p_value"] > 0.01).all()


def test_sufficiency_detects_group_shift() -> None:
    config = load_audit_config(default_fairness_config_path())
    rng = np.random.default_rng(99)
    n = 20_000
    score = rng.uniform(0.0, 1.0, n)
    gender = np.array(["F"] * (n // 2) + ["M"] * (n - n // 2))
    p = np.where(gender == "F", score, np.minimum(1.0, 1.5 * score))
    y = (rng.random(n) < p).astype(np.int8)
    frame = _gender_frame(score, gender, y)
    tests = sufficiency_tests(frame, config)
    gender_row = tests[tests["grouping"] == "CODE_GENDER"].iloc[0]
    assert gender_row["p_value"] < 1e-6
