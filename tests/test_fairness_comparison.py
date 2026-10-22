"""Tests for data comparison and robustness ranks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fairness.comparison import (
    expected_signs,
    kendall_cells,
    oracle_benchmark,
    proxy_importance,
)
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.gaps import pair_gaps
from src.fairness.robustness import robustness_table
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import GENDER_AGE_COL, YHAT_COL, AuditConfig


def _one_pair_config() -> AuditConfig:
    base = load_audit_config(default_fairness_config_path())
    fm = next(p for p in base.pairs if p.pair_id == "F-M")
    return replace(base, pairs=[fm])


def test_oracle_toy(toy_audit_frame: pd.DataFrame) -> None:
    config = _one_pair_config()
    gaps = pair_gaps(toy_audit_frame, config)
    oracle = oracle_benchmark(toy_audit_frame, config, gaps)
    row = oracle.iloc[0]
    assert row["oracle_approval_a"] == pytest.approx(0.8, rel=0, abs=1e-12)
    assert row["oracle_approval_b"] == pytest.approx(0.6, rel=0, abs=1e-12)
    assert row["oracle_dir"] == pytest.approx(4 / 3, rel=0, abs=1e-12)
    assert row["model_dir"] == pytest.approx(7 / 6, rel=0, abs=1e-12)
    assert row["direction"] == "narrower"


def _frame_fm_base_rates(br_f: float, br_m: float) -> pd.DataFrame:
    n_f, n_m = 100, 100
    y_f = np.array([1] * int(round(br_f * n_f)) + [0] * (n_f - int(round(br_f * n_f))))
    y_m = np.array([1] * int(round(br_m * n_m)) + [0] * (n_m - int(round(br_m * n_m))))
    return pd.DataFrame(
        {
            "y": np.concatenate([y_f, y_m]),
            "CODE_GENDER": ["F"] * n_f + ["M"] * n_m,
        }
    )


def _approval_gap_row(model_dir: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "F-M",
                "metric": "approval_rate",
                "diff": 0.0,
                "ratio": model_dir,
                "ratio_ci_low": 0.1,
                "ratio_ci_high": 10.0,
            }
        ]
    )


def test_direction_labels() -> None:
    config = _one_pair_config()
    # oracle_dir = 0.8/0.6 = 4/3 with br F=0.2, M=0.4 → do = 1/3
    frame = _frame_fm_base_rates(0.2, 0.4)
    cases = [
        (1.5, "wider"),
        (0.5, "reversed"),
        (4 / 3, "equal"),
    ]
    for model_dir, label in cases:
        oracle = oracle_benchmark(frame, config, _approval_gap_row(model_dir))
        assert oracle.iloc[0]["direction"] == label


def test_expected_signs_toy(toy_audit_frame: pd.DataFrame) -> None:
    config = _one_pair_config()
    gaps = pair_gaps(toy_audit_frame, config)
    signs = expected_signs(toy_audit_frame, config, gaps)
    row = signs.iloc[0]
    assert row["base_rate_a"] == pytest.approx(0.2, rel=0, abs=1e-12)
    assert row["base_rate_b"] == pytest.approx(0.4, rel=0, abs=1e-12)
    assert row["expected_sign_eod"] == -1
    assert row["observed_eod"] == pytest.approx(1 / 12, rel=0, abs=1e-12)
    assert not row["eod_matches"]
    assert row["expected_sign_dfnr"] == 1
    assert row["observed_dfnr"] == pytest.approx(0.25, rel=0, abs=1e-12)
    assert row["dfnr_matches"]


def test_kendall_perfect() -> None:
    config = load_audit_config(default_fairness_config_path())
    rows = []
    n_neg = 80
    for i, label in enumerate(config.groupings[GENDER_AGE_COL]):
        n_pos = 10 + 3 * i
        n_flag = 2 + 2 * i
        n_per = n_pos + n_neg
        y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int8)
        yhat = np.zeros(n_per, dtype=np.int8)
        yhat[n_pos : n_pos + n_flag] = 1
        gender, band = label.split(" | ")
        rows.append(
            {
                "SK_ID_CURR": 10_000 + i * n_per + np.arange(n_per),
                "y": y,
                "score": np.linspace(0.1, 0.9, n_per),
                "fold": np.zeros(n_per, dtype=int),
                "CODE_GENDER": np.full(n_per, gender),
                "AGE_BAND": np.full(n_per, band),
                "GENDER_AGE": np.full(n_per, label),
                "PTI": np.linspace(0.1, 0.3, n_per),
                "PTI_Q": np.full(n_per, "Q3"),
                "yhat": yhat,
            }
        )
    frame = pd.DataFrame({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    result = kendall_cells(frame, config)
    assert result["tau"] == pytest.approx(1.0, rel=0, abs=1e-12)


def test_proxy_importance() -> None:
    config = load_audit_config(default_fairness_config_path())
    importance = pd.DataFrame(
        {
            "feature": [
                "cat__CODE_GENDER_F",
                "num__EXT_SOURCE_1",
                "num__EXT_SOURCE_1_MISSING",
                "num__X",
            ],
            "fold": [0, 0, 0, 0],
            "importance_type": ["coef", "coef", "coef", "coef"],
            "importance": [2.0, -3.0, 1.0, 0.5],
        }
    )
    out = proxy_importance(importance, config, "logreg")
    assert out is not None
    assert set(out["feature_group"]) <= set(config.proxy_watch_list) | {"other"}
    ext = out[out["feature_group"] == "ext_source_1"].iloc[0]
    assert ext["n_features"] == 2
    assert ext["best_feature"] == "num__EXT_SOURCE_1"
    assert out["importance_share"].sum() == pytest.approx(1.0, rel=0, abs=1e-12)
    assert proxy_importance(importance, config, "mlp") is None


def test_robustness_table(tmp_path: Path, make_audit_frame) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = make_audit_frame(n=500, seed=1)
    ids = frame["SK_ID_CURR"].to_numpy()
    y = frame["y"].to_numpy()
    fold = frame["fold"].to_numpy()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    groups_path = tmp_path / "groups.parquet"

    groups = frame[["SK_ID_CURR", "CODE_GENDER", "AGE_BAND", "GENDER_AGE"]].copy()
    groups.to_parquet(groups_path, index=False)
    app = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "AMT_ANNUITY": np.linspace(5000, 50000, len(ids)),
            "AMT_INCOME_TOTAL": np.full(len(ids), 100000.0),
        }
    )
    app.to_csv(data_dir / "application_train.csv", index=False)

    for rank in range(1, 6):
        score = np.clip(frame["score"].to_numpy() + 0.02 * (rank - 1), 1e-6, 1 - 1e-6)
        oof = pd.DataFrame(
            {
                "SK_ID_CURR": ids,
                "TARGET": y,
                "y_pred_proba": score,
                "fold": fold,
            }
        )
        if rank == 1:
            path = run_dir / "oof_predictions.parquet"
        else:
            rank_dir = run_dir / "robustness" / f"rank_{rank}"
            rank_dir.mkdir(parents=True)
            path = rank_dir / "oof_predictions.parquet"
        oof.to_parquet(path, index=False)

    table = robustness_table(run_dir, config, groups_path, data_dir)
    assert len(table) == 5 * len(config.robustness_metrics)
    assert set(table["rank"]) == {1, 2, 3, 4, 5}

    rank1_frame = frame.copy()
    thr = base_rate_threshold(
        rank1_frame["score"].to_numpy(), rank1_frame["y"].to_numpy()
    )
    rank1_frame[YHAT_COL] = apply_threshold(
        rank1_frame["score"].to_numpy(), thr.threshold
    )
    gaps = pair_gaps(rank1_frame, config)
    primary = gaps[gaps["pair_id"] == config.primary_pair]
    for metric in config.robustness_metrics:
        gap_row = primary[primary["metric"] == metric].iloc[0]
        rob_row = table[(table["rank"] == 1) & (table["metric"] == metric)].iloc[0]
        assert rob_row["diff"] == pytest.approx(gap_row["diff"], rel=0, abs=1e-12)
        assert rob_row["diff_ci_low"] == pytest.approx(
            gap_row["diff_ci_low"], rel=0, abs=1e-12
        )
        assert rob_row["diff_ci_high"] == pytest.approx(
            gap_row["diff_ci_high"], rel=0, abs=1e-12
        )
