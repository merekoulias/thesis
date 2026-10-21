"""Tests for run persistence and CLIs."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.constants import SK_ID_CURR, TARGET_COL
from src.models.nested import run_nested_cv
from src.models.run import main as run_main
from src.models.runs import (
    IMPORTANCE_FILE,
    METRICS_FILE,
    METRICS_JSON_KEYS,
    OOF_FILE,
    ROBUSTNESS_DIR,
    RUN_CONFIG_FILE,
    SELECTED_PARAMS_FILE,
    TUNING_DIR,
    load_metrics,
    load_oof,
    load_selected_params,
    run_dir_for,
    save_nested_run,
)
from src.models.summary import collect_summary_rows, write_summary_csv
from src.models.types import (
    OOF_FOLD_COL,
    OOF_PROBA_COL,
    CVSettings,
)

CV_TINY = CVSettings(seed=42, n_outer_folds=3, n_inner_folds=2, n_configs=3, top_k=2)


def _tiny_run_config(n_samples: int, n_features: int) -> dict:
    return {
        "model": "lgbm",
        "variant": "baseline",
        "cv_settings": {
            "seed": 42,
            "n_outer_folds": 3,
            "n_inner_folds": 2,
            "n_configs": 3,
            "top_k": 2,
        },
        "model_config": {},
        "features_dir": "artifacts/features",
        "n_samples": n_samples,
        "n_features": n_features,
        "max_rows": None,
        "git_commit": "test",
        "versions": {},
        "started_at": "2020-01-01T00:00:00+00:00",
        "finished_at": "2020-01-01T00:01:00+00:00",
    }


def _write_feature_artifacts(data, out_dir: Path) -> None:
    """Write minimal feature artifacts (same contract as test_models_pipeline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data.X.to_parquet(out_dir / "X_train.parquet", index=False)
    data.y.to_frame(TARGET_COL).to_parquet(out_dir / "y_train.parquet", index=False)
    data.ids.to_frame(SK_ID_CURR).to_parquet(out_dir / "ids_train.parquet", index=False)


def test_run_dir_for() -> None:
    assert run_dir_for("x", "rf") == Path("x/rf/baseline")


def test_save_and_load_round_trip(
    make_training_data, tiny_model_configs, tmp_path
) -> None:
    data = make_training_data(n_samples=300, seed=0)
    result = run_nested_cv(data, tiny_model_configs["lgbm"], CV_TINY)
    run_dir = tmp_path / "lgbm" / "baseline"
    save_nested_run(
        result,
        run_dir,
        _tiny_run_config(len(data.y), data.X.shape[1]),
    )

    required = [
        RUN_CONFIG_FILE,
        OOF_FILE,
        METRICS_FILE,
        SELECTED_PARAMS_FILE,
        IMPORTANCE_FILE,
        f"{TUNING_DIR}/candidates.json",
        f"{TUNING_DIR}/inner_scores.csv",
        f"{TUNING_DIR}/rankings.json",
        f"{ROBUSTNESS_DIR}/rank_2/{OOF_FILE}",
        f"{ROBUSTNESS_DIR}/rank_2/{METRICS_FILE}",
        f"{ROBUSTNESS_DIR}/rank_2/{SELECTED_PARAMS_FILE}",
    ]
    for rel in required:
        assert (run_dir / rel).is_file(), rel

    loaded1 = load_oof(run_dir)
    pd.testing.assert_frame_equal(loaded1, result.runs[1].oof_predictions)
    loaded2 = load_oof(run_dir, rank=2)
    pd.testing.assert_frame_equal(loaded2, result.runs[2].oof_predictions)

    assert loaded1[SK_ID_CURR].dtype == result.runs[1].oof_predictions[SK_ID_CURR].dtype
    assert loaded1[TARGET_COL].dtype.name == "int8"
    assert loaded1[OOF_PROBA_COL].dtype == float
    assert loaded1[OOF_FOLD_COL].dtype in (int, "int64")


def test_selected_params_round_trip(
    make_training_data, tiny_model_configs, tmp_path
) -> None:
    data = make_training_data(n_samples=300, seed=1)
    result = run_nested_cv(data, tiny_model_configs["rf"], CV_TINY)
    run_dir = tmp_path / "rf"
    save_nested_run(
        result,
        run_dir,
        _tiny_run_config(len(data.y), data.X.shape[1]),
    )
    loaded = load_selected_params(run_dir)
    assert set(loaded.keys()) == {0, 1, 2}
    for k, entry in enumerate(result.runs[1].selected):
        assert loaded[k] == entry["params"]


def test_json_files_valid(make_training_data, tiny_model_configs, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=2)
    result = run_nested_cv(data, tiny_model_configs["rf"], CV_TINY)
    run_dir = tmp_path / "rf"
    save_nested_run(
        result,
        run_dir,
        _tiny_run_config(len(data.y), data.X.shape[1]),
    )
    for path in run_dir.rglob("*.json"):
        with path.open(encoding="utf-8") as f:
            json.load(f)
    rankings_path = run_dir / TUNING_DIR / "rankings.json"
    with rankings_path.open(encoding="utf-8") as f:
        rankings = json.load(f)
    assert len(rankings) == CV_TINY.n_outer_folds


def test_metrics_json_keys(make_training_data, tiny_model_configs, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=3)
    result = run_nested_cv(data, tiny_model_configs["rf"], CV_TINY)
    run_dir = tmp_path / "rf"
    save_nested_run(
        result,
        run_dir,
        _tiny_run_config(len(data.y), data.X.shape[1]),
    )
    for rank in (1, 2):
        metrics = load_metrics(run_dir, rank=rank)
        assert list(metrics.keys()) == METRICS_JSON_KEYS
        assert len(metrics["folds"]) == CV_TINY.n_outer_folds


def _write_tiny_configs(config_dir: Path) -> tuple:
    config_dir.mkdir(parents=True, exist_ok=True)
    cv_yaml = config_dir / "cv.yaml"
    cv_yaml.write_text(
        "seed: 42\nn_outer_folds: 3\nn_inner_folds: 2\nn_configs: 3\ntop_k: 2\n",
        encoding="utf-8",
    )
    rf_yaml = config_dir / "rf.yaml"
    rf_yaml.write_text(
        """name: rf
early_stopping_rounds: null
fixed_params:
  n_estimators: 15
  n_jobs: 1
default_params:
  min_samples_leaf: 5
search_space:
  min_samples_leaf: {dist: int_loguniform, low: 2, high: 30}
""",
        encoding="utf-8",
    )
    return cv_yaml, rf_yaml


def test_cli_end_to_end(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=4)
    feat_dir = tmp_path / "features"
    _write_feature_artifacts(data, feat_dir)
    config_dir = tmp_path / "configs"
    cv_yaml, rf_yaml = _write_tiny_configs(config_dir)
    runs_dir = tmp_path / "runs"
    run_main(
        [
            "--model",
            "rf",
            "--features-dir",
            str(feat_dir),
            "--runs-dir",
            str(runs_dir),
            "--cv-config",
            str(cv_yaml),
            "--model-config",
            str(rf_yaml),
        ]
    )
    run_dir = runs_dir / "rf" / "baseline"
    assert (run_dir / METRICS_FILE).is_file()
    assert (run_dir / OOF_FILE).is_file()
    assert (run_dir / "run.log").is_file()


def test_cli_refuses_existing_run(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=5)
    feat_dir = tmp_path / "features"
    _write_feature_artifacts(data, feat_dir)
    config_dir = tmp_path / "configs"
    cv_yaml, rf_yaml = _write_tiny_configs(config_dir)
    runs_dir = tmp_path / "runs"
    base_args = [
        "--model",
        "rf",
        "--features-dir",
        str(feat_dir),
        "--runs-dir",
        str(runs_dir),
        "--cv-config",
        str(cv_yaml),
        "--model-config",
        str(rf_yaml),
    ]
    run_main(base_args)
    with pytest.raises(SystemExit) as exc:
        run_main(base_args)
    assert exc.value.code != 0
    run_main(base_args + ["--overwrite"])


def test_cli_max_rows_guard() -> None:
    with pytest.raises(SystemExit) as exc:
        run_main(["--model", "rf", "--max-rows", "100"])
    assert exc.value.code != 0


def test_summary_table(tmp_path) -> None:
    def _fake_metrics(model: str, rank: int, auc: float) -> dict:
        return {
            "model": model,
            "rank": rank,
            "oof_roc_auc": auc,
            "std_fold_roc_auc": 0.01,
            "oof_brier": 0.1,
            "brier_reference": 0.2,
            "oof_log_loss": 0.3,
            "log_loss_reference": 0.4,
            "mean_roc_auc_gap": 0.05,
            "mean_selection_optimism": 0.001,
            "runtime_seconds": 3600.0,
        }

    for model, auc in (("lgbm", 0.78), ("rf", 0.75)):
        run_dir = tmp_path / model / "baseline"
        run_dir.mkdir(parents=True)
        with (run_dir / METRICS_FILE).open("w", encoding="utf-8") as f:
            json.dump(_fake_metrics(model, 1, auc), f)

    rows = collect_summary_rows(tmp_path, "baseline")
    assert len(rows) == 2
    out_csv = tmp_path / "summary_baseline.csv"
    write_summary_csv(rows, out_csv)
    df = pd.read_csv(out_csv)
    assert list(df.columns) == [
        "model",
        "rank",
        "oof_roc_auc",
        "std_fold_roc_auc",
        "oof_brier",
        "brier_reference",
        "oof_log_loss",
        "log_loss_reference",
        "mean_roc_auc_gap",
        "mean_selection_optimism",
        "runtime_hours",
    ]
    assert set(df["model"]) == {"lgbm", "rf"}
