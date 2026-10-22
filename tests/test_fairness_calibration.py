"""Tests for calibration deciles and sufficiency tests."""

import numpy as np
import pandas as pd

from src.fairness.calibration import (
    calibration_table,
    decile_bins,
    sufficiency_tests,
)
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import YHAT_COL


def _frame_with_yhat(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    thr = base_rate_threshold(out["score"].to_numpy(), out["y"].to_numpy())
    out[YHAT_COL] = apply_threshold(out["score"].to_numpy(), thr.threshold)
    return out


def test_decile_bins_equal_frequency() -> None:
    score = np.linspace(0.0, 1.0, 1000)
    _, bins = decile_bins(score, 10)
    assert bins.min() == 0
    assert bins.max() == 9
    for b in range(10):
        assert int((bins == b).sum()) == 100


def test_calibration_counts_sum(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _frame_with_yhat(make_audit_frame())
    table = calibration_table(frame, config)
    n_rows = len(frame)
    for grouping in table["grouping"].unique():
        sub = table[table["grouping"] == grouping]
        assert int(sub["n"].sum()) == n_rows


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
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    frame[YHAT_COL] = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    table = calibration_table(frame, config)
    all_rows = table[(table["grouping"] == "ALL") & (table["group"] == "all")]
    covered = 0
    for _, row in all_rows.iterrows():
        if row["ci_low"] <= row["mean_score"] <= row["ci_high"]:
            covered += 1
    assert covered >= 9
    tests = sufficiency_tests(frame, config)
    assert (tests["p_value"] > 0.01).all()


def test_sufficiency_detects_group_shift() -> None:
    config = load_audit_config(default_fairness_config_path())
    rng = np.random.default_rng(99)
    n = 20_000
    score = rng.uniform(0.0, 1.0, n)
    gender = np.array(["F"] * (n // 2) + ["M"] * (n - n // 2))
    y = np.zeros(n, dtype=np.int8)
    for i in range(n):
        p = score[i] if gender[i] == "F" else min(1.0, 1.5 * score[i])
        y[i] = int(rng.random() < p)
    frame = pd.DataFrame(
        {
            "y": y,
            "score": score,
            "CODE_GENDER": gender,
            "AGE_BAND": ["20-29"] * n,
            "GENDER_AGE": [f"{g} | 20-29" for g in gender],
        }
    )
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    frame[YHAT_COL] = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    tests = sufficiency_tests(frame, config)
    gender_row = tests[tests["grouping"] == "CODE_GENDER"].iloc[0]
    assert gender_row["p_value"] < 1e-6
