"""Tests for data comparison and robustness ranks."""

from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.fairness.comparison import (
    error_rate_identity,
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
    # oracle_dir = 0.8/0.6 ≈ 4/3 with br F=0.2, M=0.4 → do = 1/3
    frame = _frame_fm_base_rates(0.2, 0.4)
    eps = 1e-15
    cases = [
        (1.5, "wider"),
        (1.2, "narrower"),
        (1.0, "narrower"),
        (0.5, "reversed"),
        (4 / 3, "equal"),
        (4 / 3 + eps, "equal"),
        (4 / 3 - eps, "equal"),
    ]
    for model_dir, label in cases:
        oracle = oracle_benchmark(frame, config, _approval_gap_row(model_dir))
        assert oracle.iloc[0]["direction"] == label, model_dir


def _one_sided_frame() -> pd.DataFrame:
    """Frame with F rows only: the M side of every F-M pair is empty."""
    return pd.DataFrame(
        {
            "y": np.array([1, 0, 0, 0, 1, 0], dtype=np.int8),
            "yhat": np.array([1, 0, 1, 0, 0, 0], dtype=np.int8),
            "CODE_GENDER": ["F"] * 6,
            "AGE_BAND": ["20-29"] * 6,
            "GENDER_AGE": ["F | 20-29"] * 6,
        }
    )


def test_empty_pair_side_gives_nan_not_error() -> None:
    config = _one_pair_config()
    frame = _one_sided_frame()
    gaps = pair_gaps(frame, config)

    oracle = oracle_benchmark(frame, config, gaps).iloc[0]
    assert np.isnan(oracle["oracle_dir"])
    assert oracle["direction"] is None
    assert pd.isna(oracle["oracle_in_model_ci"])

    signs = expected_signs(frame, config, gaps).iloc[0]
    assert np.isnan(signs["observed_eod"])
    for col in (
        "expected_sign_eod",
        "expected_sign_dfnr",
        "eod_matches",
        "dfnr_matches",
        "eod_ci_excludes_0",
    ):
        assert pd.isna(signs[col])


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


def test_error_rate_identity_toy(toy_audit_frame: pd.DataFrame) -> None:
    config = _one_pair_config()
    row = error_rate_identity(toy_audit_frame, config).iloc[0]
    assert row["ppv_a"] == pytest.approx(1 / 3, rel=0, abs=1e-12)
    assert row["ppv_b"] == pytest.approx(3 / 4, rel=0, abs=1e-12)
    assert row["prevalence_factor"] == pytest.approx(0.375, rel=0, abs=1e-12)
    assert row["ppv_factor"] == pytest.approx(6.0, rel=0, abs=1e-12)
    assert row["tpr_factor"] == pytest.approx(2 / 3, rel=0, abs=1e-12)
    assert row["fpr_ratio_observed"] == pytest.approx(1.5, rel=0, abs=1e-12)
    assert row["fpr_ratio_identity"] == pytest.approx(1.5, rel=0, abs=1e-12)
    assert row["fpr_gap_observed"] == pytest.approx(1 / 12, rel=0, abs=1e-12)
    assert row["fpr_gap_base_rate_only"] == pytest.approx(
        (1 / 6) * (0.375 - 1.0), rel=0, abs=1e-12
    )


def test_error_rate_identity_holds(
    make_audit_frame: Callable[..., pd.DataFrame],
) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = make_audit_frame(n=4000, seed=3)
    thr = base_rate_threshold(frame["score"].to_numpy(), frame["y"].to_numpy())
    frame[YHAT_COL] = apply_threshold(frame["score"].to_numpy(), thr.threshold)
    table = error_rate_identity(frame, config)
    assert table["fpr_ratio_observed"].notna().all()
    np.testing.assert_allclose(
        table["fpr_ratio_identity"], table["fpr_ratio_observed"], rtol=1e-12
    )


def test_error_rate_identity_empty_side_is_nan() -> None:
    row = error_rate_identity(_one_sided_frame(), _one_pair_config()).iloc[0]
    assert np.isnan(row["ppv_b"])
    assert np.isnan(row["prevalence_factor"])
    assert np.isnan(row["fpr_ratio_identity"])


def _kendall_frame(config: AuditConfig) -> pd.DataFrame:
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
    return pd.DataFrame({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})


def test_kendall_perfect() -> None:
    config = load_audit_config(default_fairness_config_path())
    result = kendall_cells(_kendall_frame(config), config)
    assert result["tau"] == pytest.approx(1.0, rel=0, abs=1e-12)
    assert result["n_cells"] == 10
    assert len(result["cells"]) == 10


def test_kendall_counts_cells_used() -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = _kendall_frame(config)
    frame = frame[frame[GENDER_AGE_COL] != config.groupings[GENDER_AGE_COL][0]]
    result = kendall_cells(frame, config)
    assert result["n_cells"] == 9
    assert len(result["cells"]) == 10
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
    assert ext["best_feature_rank"] == 1
    assert ext["importance_share"] == pytest.approx(4.0 / 6.5, rel=0, abs=1e-12)
    protected = out[out["feature_group"] == "protected"].iloc[0]
    assert protected["best_feature_rank"] == 2
    other = out[out["feature_group"] == "other"].iloc[0]
    assert other["best_feature"] == "num__X"
    assert other["best_feature_rank"] == 4
    assert out["importance_share"].sum() == pytest.approx(1.0, rel=0, abs=1e-12)
    assert proxy_importance(importance, config, "mlp") is None
    assert proxy_importance(importance, config, "not_a_model") is None
    assert proxy_importance(None, config, "logreg") is None


def test_proxy_importance_ties_are_deterministic() -> None:
    config = load_audit_config(default_fairness_config_path())
    importance = pd.DataFrame(
        {
            "feature": ["num__B", "num__A", "num__C"],
            "fold": [0, 0, 0],
            "importance_type": ["gain"] * 3,
            "importance": [0.0, 0.0, 0.0],
        }
    )
    out = proxy_importance(importance, config, "lgbm")
    assert out is not None
    assert out.iloc[0]["best_feature"] == "num__A"
    assert out.iloc[0]["best_feature_rank"] == 1


def _write_robustness_inputs(
    tmp_path: Path, frame: pd.DataFrame, rank_scores: Dict[int, np.ndarray]
) -> Tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    groups_path = tmp_path / "groups.parquet"

    ids = frame["SK_ID_CURR"].to_numpy()
    frame[["SK_ID_CURR", "CODE_GENDER", "AGE_BAND", "GENDER_AGE"]].to_parquet(
        groups_path, index=False
    )
    pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "AMT_ANNUITY": np.linspace(5000, 50000, len(ids)),
            "AMT_INCOME_TOTAL": np.full(len(ids), 100000.0),
        }
    ).to_csv(data_dir / "application_train.csv", index=False)

    for rank, score in rank_scores.items():
        oof = pd.DataFrame(
            {
                "SK_ID_CURR": ids,
                "TARGET": frame["y"].to_numpy(),
                "y_pred_proba": score,
                "fold": frame["fold"].to_numpy(),
            }
        )
        if rank == 1:
            path = run_dir / "oof_predictions.parquet"
        else:
            rank_dir = run_dir / "robustness" / f"rank_{rank}"
            rank_dir.mkdir(parents=True)
            path = rank_dir / "oof_predictions.parquet"
        oof.to_parquet(path, index=False)
    return run_dir, groups_path, data_dir


def test_robustness_table(
    tmp_path: Path, make_audit_frame: Callable[..., pd.DataFrame]
) -> None:
    config = load_audit_config(default_fairness_config_path())
    frame = make_audit_frame(n=500, seed=1)
    base_score = frame["score"].to_numpy()

    # Each rank gets its own score noise, so thresholds and decisions differ by rank.
    rank_scores: Dict[int, np.ndarray] = {}
    for rank in range(1, 6):
        noise = np.random.default_rng(rank).normal(0, 0.05 * (rank - 1), len(frame))
        rank_scores[rank] = np.clip(base_score + noise, 1e-6, 1 - 1e-6)

    run_dir, groups_path, data_dir = _write_robustness_inputs(
        tmp_path, frame, rank_scores
    )
    table = robustness_table(run_dir, config, groups_path, data_dir)
    assert len(table) == 5 * len(config.robustness_metrics)
    assert set(table["rank"]) == {1, 2, 3, 4, 5}
    assert table["threshold"].nunique() == 5

    y = frame["y"].to_numpy()
    for rank, score in rank_scores.items():
        thr = base_rate_threshold(score, y)
        rank_frame = frame.copy()
        rank_frame["score"] = score
        rank_frame[YHAT_COL] = apply_threshold(score, thr.threshold)
        gaps = pair_gaps(rank_frame, config)
        primary = gaps[gaps["pair_id"] == config.primary_pair]
        rank_rows = table[table["rank"] == rank]
        assert (rank_rows["threshold"] == thr.threshold).all()
        assert (rank_rows["n_flagged"] == thr.n_flagged).all()
        assert rank_rows["roc_auc"].iloc[0] == pytest.approx(
            roc_auc_score(y, score), rel=0, abs=1e-12
        )
        for metric in config.robustness_metrics:
            gap_row = primary[primary["metric"] == metric].iloc[0]
            rob_row = rank_rows[rank_rows["metric"] == metric].iloc[0]
            for col in ("diff", "diff_ci_low", "diff_ci_high"):
                assert rob_row[col] == pytest.approx(gap_row[col], rel=0, abs=1e-12)
