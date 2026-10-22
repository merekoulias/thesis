"""Tests for conditional demographic parity on PTI quintiles."""

import pandas as pd

from src.fairness.conditional import (
    conditional_parity,
    conditional_parity_cells,
)
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.gaps import pair_gaps, pair_mask
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import PTI_Q_COL, YHAT_COL


def _frame_with_yhat(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    thr = base_rate_threshold(out["score"].to_numpy(), out["y"].to_numpy())
    out[YHAT_COL] = apply_threshold(out["score"].to_numpy(), thr.threshold)
    return out


def test_quintile_rows(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _frame_with_yhat(make_audit_frame())
    gaps = pair_gaps(frame, config)
    parity, _ = conditional_parity(frame, config, gaps)
    assert len(parity) == len(config.cdp_pairs) * config.cdp_n_quantiles
    work = frame[frame[PTI_Q_COL].notna()]
    pti_q = work[PTI_Q_COL].to_numpy()

    for pair_id in config.cdp_pairs:
        pair = next(p for p in config.pairs if p.pair_id == pair_id)
        for q in range(1, config.cdp_n_quantiles + 1):
            q_label = f"Q{q}"
            row = parity[
                (parity["pair_id"] == pair_id) & (parity["quintile"] == q_label)
            ].iloc[0]
            q_mask = pti_q == q_label
            n_a = int((pair_mask(work, pair.a) & q_mask).sum())
            n_b = int((pair_mask(work, pair.b) & q_mask).sum())
            assert row["n_a"] == n_a
            assert row["n_b"] == n_b


def test_missing_pti_excluded(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _frame_with_yhat(make_audit_frame())
    gaps = pair_gaps(frame, config)
    _, tests = conditional_parity(frame, config, gaps)
    assert int(tests["n_excluded_missing_pti"].iloc[0]) == 3
    assert (tests["n_excluded_missing_pti"] == 3).all()


def test_stratified_test_df(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _frame_with_yhat(make_audit_frame())
    gaps = pair_gaps(frame, config)
    _, tests = conditional_parity(frame, config, gaps)
    assert (tests["df"] == 5).all()


def test_unconditional_copied(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _frame_with_yhat(make_audit_frame())
    gaps = pair_gaps(frame, config)
    _, tests = conditional_parity(frame, config, gaps)
    for pair_id in config.cdp_pairs:
        gap_row = gaps[
            (gaps["pair_id"] == pair_id) & (gaps["metric"] == "approval_rate")
        ].iloc[0]
        test_row = tests[tests["pair_id"] == pair_id].iloc[0]
        assert test_row["unconditional_diff"] == gap_row["diff"]
        assert test_row["unconditional_diff_ci_low"] == gap_row["diff_ci_low"]
        assert test_row["unconditional_diff_ci_high"] == gap_row["diff_ci_high"]


def test_cells_appendix(make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _frame_with_yhat(make_audit_frame())
    cells = conditional_parity_cells(frame, config)
    n_cells = len(config.groupings["GENDER_AGE"])
    assert len(cells) == n_cells * config.cdp_n_quantiles
    assert cells["approval_rate"].between(0.0, 1.0).all()
