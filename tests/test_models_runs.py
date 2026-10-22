"""Tests for run persistence and CLIs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.models.run as run_module
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
    _write_json,
    load_metrics,
    load_oof,
    load_selected_params,
    run_dir_for,
    save_nested_run,
)
from src.models.summary import collect_summary_rows
from src.models.summary import main as summary_main
from src.models.summary import write_summary_csv
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
    """Write minimal feature artifacts (same contract as test_models_training)."""
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


def _cli_args(tmp_path: Path, data, seed: int = 0, extra=()) -> list:
    """Write artifacts and tiny configs under tmp_path; return CLI args for rf."""
    feat_dir = tmp_path / "features"
    _write_feature_artifacts(data, feat_dir)
    cv_yaml, rf_yaml = _write_tiny_configs(tmp_path / "configs")
    return [
        "--model",
        "rf",
        "--features-dir",
        str(feat_dir),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--cv-config",
        str(cv_yaml),
        "--model-config",
        str(rf_yaml),
        *extra,
    ]


def test_cli_max_rows_guard(monkeypatch, tmp_path) -> None:
    def _no_load(*args, **kwargs):
        raise AssertionError("data must not be loaded before the guard")

    monkeypatch.setattr(run_module, "load_training_data", _no_load)
    monkeypatch.chdir(tmp_path)  # independent of the working directory
    default_runs = run_module._repo_root() / "artifacts" / "runs"
    with pytest.raises(SystemExit) as exc:
        run_main(
            ["--model", "rf", "--max-rows", "100", "--runs-dir", str(default_runs)]
        )
    assert exc.value.code != 0


def test_cli_max_rows_must_be_below_n_rows(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=6)
    for bad in ("300", "1"):
        with pytest.raises(SystemExit) as exc:
            run_main(_cli_args(tmp_path, data, extra=["--max-rows", bad]))
        assert exc.value.code != 0


def test_cli_max_rows_subsamples(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=7)
    args = _cli_args(tmp_path, data, extra=["--max-rows", "150"])
    run_main(args)
    run_dir = tmp_path / "runs" / "rf" / "baseline"
    assert len(load_oof(run_dir)) == 150
    with (run_dir / RUN_CONFIG_FILE).open(encoding="utf-8") as f:
        assert json.load(f)["max_rows"] == 150


def test_cli_resume_and_overwrite_are_exclusive(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=8)
    with pytest.raises(SystemExit) as exc:
        run_main(_cli_args(tmp_path, data, extra=["--resume", "--overwrite"]))
    assert exc.value.code != 0


def test_cli_model_config_name_mismatch(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=9)
    args = _cli_args(tmp_path, data)
    args[args.index("--model") + 1] = "logreg"  # rf.yaml has name: rf
    with pytest.raises(SystemExit) as exc:
        run_main(args)
    assert exc.value.code != 0


def test_cli_existing_run_checked_before_loading_data(
    make_training_data, monkeypatch, tmp_path
) -> None:
    data = make_training_data(n_samples=300, seed=10)
    args = _cli_args(tmp_path, data)
    run_main(args)

    def _no_load(*a, **k):
        raise AssertionError("data loaded although the run directory exists")

    monkeypatch.setattr(run_module, "load_training_data", _no_load)
    with pytest.raises(SystemExit) as exc:
        run_main(args)
    assert exc.value.code != 0


def test_cli_run_config_contents(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=11)
    run_main(_cli_args(tmp_path, data))
    run_dir = tmp_path / "runs" / "rf" / "baseline"
    with (run_dir / RUN_CONFIG_FILE).open(encoding="utf-8") as f:
        cfg = json.load(f)
    assert list(cfg) == [
        "model",
        "variant",
        "cv_settings",
        "model_config",
        "features_dir",
        "n_samples",
        "n_features",
        "max_rows",
        "git_commit",
        "versions",
        "resumed",
        "started_at",
        "finished_at",
    ]
    assert cfg["model"] == "rf" and cfg["variant"] == "baseline"
    assert cfg["cv_settings"]["n_outer_folds"] == 3
    assert cfg["model_config"]["name"] == "rf"
    assert cfg["n_samples"] == 300 and cfg["n_features"] == data.X.shape[1]
    assert cfg["max_rows"] is None and cfg["resumed"] is False
    assert set(cfg["versions"]) == {"python", "numpy", "pandas", "sklearn", "lightgbm"}
    assert cfg["git_commit"]
    assert cfg["started_at"] <= cfg["finished_at"]


def test_cli_resume_on_finished_run(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=12)
    args = _cli_args(tmp_path, data)
    run_main(args)
    run_dir = tmp_path / "runs" / "rf" / "baseline"
    first = load_oof(run_dir)
    first_runtime = load_metrics(run_dir)["runtime_seconds"]
    run_main(args + ["--resume"])
    pd.testing.assert_frame_equal(load_oof(run_dir), first)
    with (run_dir / RUN_CONFIG_FILE).open(encoding="utf-8") as f:
        assert json.load(f)["resumed"] is True
    # tuning compute of the first session is carried over, not lost
    assert load_metrics(run_dir)["runtime_seconds"] >= 0.5 * first_runtime


def test_cli_resume_without_run_dir_starts_fresh(make_training_data, tmp_path) -> None:
    data = make_training_data(n_samples=300, seed=13)
    run_main(_cli_args(tmp_path, data, extra=["--resume"]))
    run_dir = tmp_path / "runs" / "rf" / "baseline"
    assert (run_dir / METRICS_FILE).is_file()
    with (run_dir / RUN_CONFIG_FILE).open(encoding="utf-8") as f:
        assert json.load(f)["resumed"] is False


def test_cli_does_not_leak_log_handlers(make_training_data, tmp_path) -> None:
    import logging

    data = make_training_data(n_samples=300, seed=14)
    before = list(logging.getLogger().handlers)
    level = logging.getLogger().level
    run_main(_cli_args(tmp_path, data))
    assert logging.getLogger().handlers == before
    assert logging.getLogger().level == level


def test_git_commit_marks_dirty_tree(monkeypatch) -> None:
    def _fake(args):
        return "abc123" if args[0] == "rev-parse" else " M src/models/run.py"

    monkeypatch.setattr(run_module, "_git_output", _fake)
    assert run_module._git_commit() == "abc123-dirty"
    monkeypatch.setattr(
        run_module, "_git_output", lambda a: "abc123" if a[0] == "rev-parse" else ""
    )
    assert run_module._git_commit() == "abc123"


def test_write_json_rejects_nan_and_converts_numpy(tmp_path) -> None:
    path = tmp_path / "x.json"
    _write_json(path, {1: np.int64(2), "a": np.float64(0.5), "b": np.bool_(True)})
    assert json.loads(path.read_text()) == {"1": 2, "a": 0.5, "b": True}
    with pytest.raises(ValueError):
        _write_json(path, {"bad": float("nan")})


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


def _write_fake_run(root: Path, model: str, ranks, runtime_seconds: float = 7200.0):
    base = root / model / "baseline"
    for rank in ranks:
        sub = base if rank == 1 else base / ROBUSTNESS_DIR / f"rank_{rank}"
        sub.mkdir(parents=True, exist_ok=True)
        metrics = {
            "model": model,
            "rank": rank,
            "oof_roc_auc": 0.7 + 0.01 * rank,
            "std_fold_roc_auc": 0.0123456,
            "oof_brier": 0.0661234,
            "brier_reference": 0.0742,
            "oof_log_loss": 0.2381234,
            "log_loss_reference": 0.2805,
            "mean_roc_auc_gap": 0.05,
            "mean_selection_optimism": -0.001,
            "runtime_seconds": runtime_seconds,
        }
        (sub / METRICS_FILE).write_text(json.dumps(metrics), encoding="utf-8")


def test_summary_includes_robustness_ranks_sorted(tmp_path) -> None:
    _write_fake_run(tmp_path, "rf", [1, 3, 2])
    _write_fake_run(tmp_path, "lgbm", [1, 2])
    rows = collect_summary_rows(tmp_path, "baseline")
    assert [(r["model"], r["rank"]) for r in rows] == [
        ("lgbm", 1),
        ("lgbm", 2),
        ("rf", 1),
        ("rf", 2),
        ("rf", 3),
    ]
    assert rows[0]["runtime_hours"] == pytest.approx(2.0)


def test_summary_main_writes_csv_and_prints_table(tmp_path, capsys) -> None:
    _write_fake_run(tmp_path, "rf", [1, 2])
    summary_main(["--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2 + 2  # header, separator, two rows
    assert out[0].startswith("| model | rank | oof_roc_auc |")
    assert out[2].startswith("| rf | 1 | 0.7100 | 0.0123 | 0.0661 | 0.0742 |")
    assert out[2].endswith("| 2.0000 |")
    df = pd.read_csv(tmp_path / "summary_baseline.csv")
    assert list(df["rank"]) == [1, 2]
    # the CSV keeps full precision; only the printed table is rounded
    assert df.loc[0, "std_fold_roc_auc"] == pytest.approx(0.0123456, abs=0)
    assert df.loc[0, "runtime_hours"] == pytest.approx(2.0)


def test_summary_ignores_other_variants_and_missing_dir(tmp_path) -> None:
    _write_fake_run(tmp_path, "rf", [1])
    assert collect_summary_rows(tmp_path, "other") == []
    assert collect_summary_rows(tmp_path / "nope", "baseline") == []
