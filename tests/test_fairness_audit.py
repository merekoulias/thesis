"""Tests for the fairness audit runner, outputs, and summary CLI."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.fairness.audit import main as audit_main
from src.fairness.audit import run_audit
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.summary import build_summary
from src.models.runs import IMPORTANCE_FILE, METRICS_FILE, OOF_FILE

EXPECTED_ARTIFACTS = [
    "audit_config.json",
    "threshold.json",
    "utility.json",
    "group_counts.csv",
    "group_metrics.csv",
    "gaps.csv",
    "cell_summary.csv",
    "calibration.csv",
    "calibration_tests.csv",
    "bin_edges.json",
    "conditional_parity.csv",
    "conditional_parity_tests.csv",
    "conditional_parity_cells.csv",
    "pti_quintiles.json",
    "oracle_benchmark.csv",
    "expected_signs.csv",
    "kendall.json",
    "robustness.csv",
    "figures/calibration_CODE_GENDER.png",
    "figures/calibration_AGE_BAND.png",
    "figures/cells_error_rates.png",
    "report.md",
    "audit.log",
]


def _write_oof(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(
        {
            "SK_ID_CURR": frame["SK_ID_CURR"],
            "TARGET": frame["y"],
            "y_pred_proba": frame["score"],
            "fold": frame["fold"],
        }
    )
    oof.to_parquet(path, index=False)


def _write_metrics(path: Path, frame: pd.DataFrame, model: str = "logreg") -> None:
    y = frame["y"].to_numpy()
    score = frame["score"].to_numpy()
    auc = float(roc_auc_score(y, score))
    fold_aucs = [auc, auc, auc]
    payload = {
        "model": model,
        "rank": 1,
        "n_samples": len(frame),
        "n_features": 10,
        "n_folds": 3,
        "oof_roc_auc": auc,
        "oof_brier": float(np.mean((score - y) ** 2)),
        "oof_log_loss": 0.5,
        "base_rate": float(y.mean()),
        "brier_reference": 0.2,
        "log_loss_reference": 0.6,
        "fold_roc_aucs": fold_aucs,
        "mean_fold_roc_auc": float(np.mean(fold_aucs)),
        "std_fold_roc_auc": float(np.std(fold_aucs)),
        "folds": 3,
        "mean_roc_auc_gap": 0.0,
        "mean_brier_gap": 0.0,
        "mean_log_loss_gap": 0.0,
        "mean_selection_optimism": 0.0,
        "runtime_seconds": 1.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_importance(path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature": ["cat__CODE_GENDER_F", "num__EXT_SOURCE_1", "num__DAYS_BIRTH"],
            "importance_type": ["coef", "coef", "coef"],
            "importance": [0.5, 0.3, 0.2],
        }
    )
    df.to_parquet(path, index=False)


def _bootstrap_run_tree(
    tmp_path: Path,
    frame: pd.DataFrame,
    model: str = "logreg",
    with_importance: bool = True,
) -> tuple[Path, Path, Path, Path]:
    runs = tmp_path / "runs"
    features = tmp_path / "features"
    data = tmp_path / "data"
    out = tmp_path / "fairness"
    run_dir = runs / model / "baseline"
    run_dir.mkdir(parents=True)
    _write_oof(frame, run_dir / OOF_FILE)
    for rank in (2, 3, 4, 5):
        perturbed = frame.copy()
        perturbed["score"] = np.clip(perturbed["score"] + 0.01 * rank, 1e-6, 1 - 1e-6)
        _write_oof(
            perturbed,
            run_dir / "robustness" / f"rank_{rank}" / OOF_FILE,
        )
    _write_metrics(run_dir / METRICS_FILE, frame, model=model)
    if with_importance:
        _write_importance(run_dir / IMPORTANCE_FILE)

    groups = frame[["SK_ID_CURR", "CODE_GENDER", "AGE_BAND", "GENDER_AGE"]].copy()
    groups["DAYS_BIRTH"] = -12000
    groups["AGE_YEARS"] = 33.0
    features.mkdir(parents=True, exist_ok=True)
    groups.to_parquet(features / "fairness_groups_train.parquet", index=False)

    data.mkdir(parents=True, exist_ok=True)
    income = 100_000.0
    csv = pd.DataFrame(
        {
            "SK_ID_CURR": frame["SK_ID_CURR"],
            "AMT_ANNUITY": frame["PTI"].fillna(0.1) * income,
            "AMT_INCOME_TOTAL": income,
            "IGNORE": 0,
        }
    )
    csv.to_csv(data / "application_train.csv", index=False)
    return runs, features, data, out


def test_run_audit_end_to_end(
    tmp_path: Path, make_audit_frame: Callable[..., pd.DataFrame]
) -> None:
    frame = make_audit_frame(n=800, seed=1)
    runs, features, data, out = _bootstrap_run_tree(tmp_path, frame)
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=20)
    run_audit("logreg", config, runs, features, data, out, overwrite=True)
    model_out = out / "logreg" / "baseline"
    for rel in EXPECTED_ARTIFACTS:
        assert (model_out / rel).exists(), rel
    assert (model_out / "proxy_importance.csv").exists()
    report = (model_out / "report.md").read_text(encoding="utf-8")
    for section in range(1, 13):
        assert f"## {section}." in report


def test_cli_refuses_existing_output(
    tmp_path: Path, make_audit_frame: Callable[..., pd.DataFrame]
) -> None:
    frame = make_audit_frame(n=400, seed=2)
    runs, features, data, out = _bootstrap_run_tree(tmp_path, frame)
    config_path = default_fairness_config_path()
    argv = [
        "--model",
        "logreg",
        "--config",
        str(config_path),
        "--runs-dir",
        str(runs),
        "--features-dir",
        str(features),
        "--data-dir",
        str(data),
        "--out-dir",
        str(out),
        "--n-boot",
        "20",
        "--overwrite",
    ]
    audit_main(argv)
    argv_no_overwrite = [a for a in argv if a != "--overwrite"]
    with pytest.raises(SystemExit) as exc:
        audit_main(argv_no_overwrite)
    assert exc.value.code != 0


def test_mlp_without_importance(
    tmp_path: Path, make_audit_frame: Callable[..., pd.DataFrame]
) -> None:
    frame = make_audit_frame(n=400, seed=3)
    runs, features, data, out = _bootstrap_run_tree(
        tmp_path, frame, model="mlp", with_importance=False
    )
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=20)
    run_audit("mlp", config, runs, features, data, out, overwrite=True)
    model_out = out / "mlp" / "baseline"
    assert not (model_out / "proxy_importance.csv").exists()
    report = (model_out / "report.md").read_text(encoding="utf-8")
    assert "not available for mlp" in report


def test_summary(tmp_path: Path, make_audit_frame: Callable[..., pd.DataFrame]) -> None:
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=20)
    for model in ("logreg", "lgbm"):
        frame = make_audit_frame(n=500, seed=4 if model == "logreg" else 5)
        runs, features, data, out = _bootstrap_run_tree(tmp_path, frame, model=model)
        run_audit(model, config, runs, features, data, out, overwrite=True)
    summary_dir = build_summary(out, "baseline")
    gaps = pd.read_csv(summary_dir / "primary_gaps.csv")
    assert set(gaps["model"]) == {"logreg", "lgbm"}
    assert set(gaps["rank"]) == {1, 2, 3, 4, 5}
    assert (summary_dir / "figures" / "primary_gaps.png").exists()


def test_audit_deterministic(
    tmp_path: Path, make_audit_frame: Callable[..., pd.DataFrame]
) -> None:
    frame = make_audit_frame(n=600, seed=6)
    runs, features, data, out = _bootstrap_run_tree(tmp_path, frame)
    config = replace(load_audit_config(default_fairness_config_path()), n_boot=20)
    run_audit("logreg", config, runs, features, data, out, overwrite=True)
    gm1 = (out / "logreg" / "baseline" / "group_metrics.csv").read_bytes()
    cs1 = (out / "logreg" / "baseline" / "cell_summary.csv").read_bytes()
    run_audit("logreg", config, runs, features, data, out, overwrite=True)
    gm2 = (out / "logreg" / "baseline" / "group_metrics.csv").read_bytes()
    cs2 = (out / "logreg" / "baseline" / "cell_summary.csv").read_bytes()
    assert gm1 == gm2
    assert cs1 == cs2
