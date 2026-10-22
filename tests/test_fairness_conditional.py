"""Tests for conditional demographic parity on PTI quintiles."""

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from src.fairness.conditional import (
    conditional_parity,
    conditional_parity_cells,
)
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.gaps import pair_gaps, pair_mask
from src.fairness.io import assign_pti_quintiles
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import PTI_COL, PTI_Q_COL, YHAT_COL, AuditConfig


@pytest.fixture
def config() -> AuditConfig:
    return load_audit_config(default_fairness_config_path())


@pytest.fixture
def frame(make_audit_frame) -> pd.DataFrame:
    """Synthetic audit frame with ``yhat`` from the base-rate threshold."""
    out = make_audit_frame()
    thr = base_rate_threshold(out["score"].to_numpy(), out["y"].to_numpy())
    out[YHAT_COL] = apply_threshold(out["score"].to_numpy(), thr.threshold)
    return out


def test_quintile_rows(frame: pd.DataFrame, config: AuditConfig) -> None:
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


def test_quintile_values(frame: pd.DataFrame, config: AuditConfig) -> None:
    """Approval rates, Wald CI and stratified G against direct computation."""
    parity, tests = conditional_parity(frame, config, pair_gaps(frame, config))
    work = frame[frame[PTI_Q_COL].notna()]
    z = config.z

    for pair_id in config.cdp_pairs:
        pair = next(p for p in config.pairs if p.pair_id == pair_id)
        expected_g = 0.0
        expected_df = 0
        for q in range(1, config.cdp_n_quantiles + 1):
            q_label = f"Q{q}"
            in_q = work[PTI_Q_COL].to_numpy() == q_label
            approved_a = work.loc[pair_mask(work, pair.a) & in_q, YHAT_COL] == 0
            approved_b = work.loc[pair_mask(work, pair.b) & in_q, YHAT_COL] == 0
            r_a, r_b = approved_a.mean(), approved_b.mean()
            se = np.sqrt(
                r_a * (1 - r_a) / len(approved_a) + r_b * (1 - r_b) / len(approved_b)
            )
            row = parity[
                (parity["pair_id"] == pair_id) & (parity["quintile"] == q_label)
            ].iloc[0]
            np.testing.assert_allclose(row["approval_rate_a"], r_a, rtol=1e-12)
            np.testing.assert_allclose(row["approval_rate_b"], r_b, rtol=1e-12)
            np.testing.assert_allclose(row["diff"], r_a - r_b, atol=1e-12)
            np.testing.assert_allclose(
                row["diff_ci_low"], r_a - r_b - z * se, atol=1e-12
            )
            np.testing.assert_allclose(
                row["diff_ci_high"], r_a - r_b + z * se, atol=1e-12
            )
            table = np.array(
                [
                    [approved_a.sum(), (~approved_a).sum()],
                    [approved_b.sum(), (~approved_b).sum()],
                ]
            )
            stat, _, dof, _ = chi2_contingency(
                table, correction=False, lambda_="log-likelihood"
            )
            expected_g += stat
            expected_df += dof
        test_row = tests[tests["pair_id"] == pair_id].iloc[0]
        np.testing.assert_allclose(test_row["g_stat"], expected_g, rtol=1e-12)
        assert test_row["df"] == expected_df


def test_missing_pti_excluded(frame: pd.DataFrame, config: AuditConfig) -> None:
    _, tests = conditional_parity(frame, config, pair_gaps(frame, config))
    assert (tests["n_excluded_missing_pti"] == 3).all()


def test_stratified_test_df(frame: pd.DataFrame, config: AuditConfig) -> None:
    _, tests = conditional_parity(frame, config, pair_gaps(frame, config))
    assert (tests["df"] == 5).all()


def test_unconditional_copied(frame: pd.DataFrame, config: AuditConfig) -> None:
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


def test_pti_checks_warn(frame: pd.DataFrame, config: AuditConfig, caplog) -> None:
    """Mismatched cut points / missing count warn; matching values do not."""
    gaps = pair_gaps(frame, config)
    with caplog.at_level(logging.WARNING, logger="src.fairness.conditional"):
        conditional_parity(frame, config, gaps)
    messages = [r.getMessage() for r in caplog.records]
    assert any("PTI missing count 3 != expected 12" in m for m in messages)
    assert any("PTI cut points" in m for m in messages)

    _, cut_points = assign_pti_quintiles(frame[PTI_COL], config.cdp_n_quantiles)
    matching = replace(
        config,
        cdp_expected_cut_points=[round(float(c), 3) for c in cut_points],
        cdp_expected_missing=3,
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.fairness.conditional"):
        conditional_parity(frame, matching, gaps)
    assert caplog.records == []


def test_cells_appendix(frame: pd.DataFrame, config: AuditConfig) -> None:
    cells = conditional_parity_cells(frame, config)
    n_cells = len(config.groupings["GENDER_AGE"])
    assert len(cells) == n_cells * config.cdp_n_quantiles
    assert cells["approval_rate"].between(0.0, 1.0).all()
