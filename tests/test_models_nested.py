"""Tests for nested cross-validation."""

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import src.models.nested as nested_mod
from src.constants import SK_ID_CURR, TARGET_COL
from src.models.cv import cross_val_predict_model, make_stratified_splits
from src.models.metrics import classification_metrics
from src.models.nested import inner_splits_absolute, run_nested_cv
from src.models.registry import fit_predict
from src.models.types import (
    INNER_SCORE_COLUMNS,
    OOF_FOLD_COL,
    OOF_PROBA_COL,
    CVSettings,
    OuterFoldSelection,
)

CV_TINY = CVSettings(seed=42, n_outer_folds=3, n_inner_folds=2, n_configs=3, top_k=2)

FOLD_METRIC_KEYS = {
    "outer_fold",
    "candidate_id",
    "mean_inner_roc_auc",
    "train_roc_auc",
    "val_roc_auc",
    "roc_auc_gap",
    "train_brier",
    "val_brier",
    "brier_gap",
    "train_log_loss",
    "val_log_loss",
    "log_loss_gap",
    "selection_optimism",
}


def _run_nested(data, model_config, checkpoint_dir=None, resume=False):
    return run_nested_cv(
        data,
        model_config,
        CV_TINY,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )


@pytest.mark.parametrize("model_name", ["lgbm", "rf", "logreg", "mlp"])
def test_nested_oof_contract(model_name, make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=0)
    result = _run_nested(data, tiny_model_configs[model_name])
    oof = result.runs[1].oof_predictions
    expected_cols = [SK_ID_CURR, TARGET_COL, OOF_PROBA_COL, OOF_FOLD_COL]
    assert list(oof.columns) == expected_cols
    assert len(oof) == 300
    assert not oof.isna().any().any()
    proba = oof[OOF_PROBA_COL].to_numpy()
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    y = data.y.to_numpy()
    expected_splits = make_stratified_splits(y, 3, 42)
    expected_fold = np.full(len(y), -1, dtype=int)
    for k, (_, va) in enumerate(expected_splits):
        expected_fold[va] = k
    assert set(oof[OOF_FOLD_COL].unique()) == {0, 1, 2}
    np.testing.assert_array_equal(oof[OOF_FOLD_COL].to_numpy(), expected_fold)
    np.testing.assert_array_equal(oof[SK_ID_CURR].to_numpy(), data.ids.to_numpy())


def test_nested_inner_scores_shape(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=1)
    for model_name in ("rf", "lgbm"):
        result = _run_nested(data, tiny_model_configs[model_name])
        scores = result.inner_scores
        assert len(scores) == 3 * 3 * 2
        assert list(scores.columns) == INNER_SCORE_COLUMNS
        if model_name == "rf":
            assert scores["best_iteration"].isna().all()
        else:
            assert (scores["best_iteration"] >= 1).all()


def test_ranking_ties_broken_by_smaller_candidate_id(
    make_training_data, tiny_model_configs
):
    data = make_training_data(n_samples=300, seed=2)
    with mock.patch("src.models.nested.roc_auc_score", return_value=0.5) as mocked:
        result = _run_nested(data, tiny_model_configs["rf"])
        assert mocked.called
    for sel in result.selections:
        assert sel.ranking == [0, 1, 2]
        assert all(auc == 0.5 for auc in sel.mean_inner_auc.values())


def test_ranking_matches_inner_scores(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=2)
    result = _run_nested(data, tiny_model_configs["rf"])
    scores = result.inner_scores
    for k, sel in enumerate(result.selections):
        fold = scores[scores["outer_fold"] == k]
        expected = fold.groupby("candidate_id")["roc_auc"].mean()
        assert set(sel.mean_inner_auc) == set(expected.index)
        for cid, auc in sel.mean_inner_auc.items():
            assert auc == pytest.approx(expected[cid], abs=1e-12)
        assert sel.ranking == sorted(expected.index, key=lambda c: (-expected[c], c))
        assert len(set(sel.mean_inner_auc.values())) > 1


def test_top_k_runs(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=3)
    result = _run_nested(data, tiny_model_configs["rf"])
    assert set(result.runs) == {1, 2}
    for k, sel in enumerate(result.selections):
        cid_rank2 = sel.ranking[1]
        assert result.runs[2].selected[k]["candidate_id"] == cid_rank2


def test_lgbm_refit_uses_mean_best_iteration(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=4)
    result = _run_nested(data, tiny_model_configs["lgbm"])
    for k, sel in enumerate(result.selections):
        cid = sel.ranking[0]
        mean_it = sel.mean_best_iteration[cid]
        expected_n = max(1, int(round(mean_it)))
        assert result.runs[1].selected[k]["params"]["n_estimators"] == expected_n


def test_inner_splits_never_touch_outer_validation(make_training_data):
    data = make_training_data(n_samples=300, seed=5)
    y = data.y.to_numpy()
    outer_splits = make_stratified_splits(y, 3, 42)
    for outer_tr, outer_va in outer_splits:
        inner = inner_splits_absolute(outer_tr, y, 2, 42)
        all_inner = np.concatenate([np.concatenate([tr, va]) for tr, va in inner])
        assert set(all_inner).issubset(set(outer_tr))
        assert np.intersect1d(all_inner, outer_va).size == 0
        val_parts = [va for _, va in inner]
        assert np.sort(np.concatenate(val_parts)).tolist() == np.sort(outer_tr).tolist()


def test_outer_validation_used_only_for_prediction(
    make_training_data, tiny_model_configs
):
    data = make_training_data(n_samples=300, seed=6)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy()
    outer_splits = make_stratified_splits(y, 3, 42)
    cv_calls = []
    fit_calls = []

    def spy_cv(spec, params, seed, X_in, y_in, splits, **kwargs):
        cv_calls.append((X_in.copy(), splits))
        return cross_val_predict_model(spec, params, seed, X_in, y_in, splits, **kwargs)

    def spy_fit(spec, params, seed, X_train, y_train, X_pred, **kwargs):
        fit_calls.append((X_train.copy(), X_pred.copy(), kwargs))
        return fit_predict(spec, params, seed, X_train, y_train, X_pred, **kwargs)

    with mock.patch("src.models.nested.cross_val_predict_model", spy_cv):
        with mock.patch("src.models.nested.fit_predict", spy_fit):
            _run_nested(data, tiny_model_configs["lgbm"])

    # Inner CV (n_configs candidates per outer fold) sees only the outer training rows,
    # and its inner validation parts partition exactly those rows.
    assert len(cv_calls) == 3 * CV_TINY.n_configs
    for i, (X_in, splits) in enumerate(cv_calls):
        outer_tr, _ = outer_splits[i // CV_TINY.n_configs]
        np.testing.assert_array_equal(X_in, X[outer_tr])
        validation_rows = np.sort(np.concatenate([va for _, va in splits]))
        np.testing.assert_array_equal(validation_rows, np.arange(len(outer_tr)))

    # Refits (top_k ranks x outer folds) train on the outer training rows, predict the
    # outer validation rows, and never get an eval_set (no early stopping on them).
    assert len(fit_calls) == CV_TINY.top_k * 3
    for i, (X_train, X_pred, kwargs) in enumerate(fit_calls):
        outer_tr, outer_va = outer_splits[i % 3]
        np.testing.assert_array_equal(X_train, X[outer_tr])
        np.testing.assert_array_equal(X_pred, X[outer_va])
        assert kwargs.get("eval_set") is None
        assert kwargs.get("early_stopping_rounds") is None


def test_nested_deterministic(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=7)
    config = tiny_model_configs["logreg"]
    r1 = _run_nested(data, config)
    r2 = _run_nested(data, config)
    pd.testing.assert_frame_equal(
        r1.runs[1].oof_predictions, r2.runs[1].oof_predictions
    )
    s1 = r1.inner_scores.drop(columns=["fit_seconds"])
    s2 = r2.inner_scores.drop(columns=["fit_seconds"])
    pd.testing.assert_frame_equal(s1, s2)


def test_resume_gives_identical_result(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=8)
    config = tiny_model_configs["rf"]
    ckpt = tmp_path / "tuning"
    call_count = {"n": 0}
    real_cv = None

    def counting_cv(*args, **kwargs):
        call_count["n"] += 1
        return real_cv(*args, **kwargs)

    real_cv = cross_val_predict_model
    with mock.patch.object(nested_mod, "cross_val_predict_model", counting_cv):
        full = _run_nested(data, config, checkpoint_dir=ckpt, resume=False)
        first_count = call_count["n"]

    scores_path = ckpt / "inner_scores.csv"
    lines = scores_path.read_text(encoding="utf-8").splitlines()
    truncated = "\n".join(lines[:8]) + "\n"
    scores_path.write_text(truncated, encoding="utf-8")

    call_count["n"] = 0
    with mock.patch.object(nested_mod, "cross_val_predict_model", counting_cv):
        resumed = _run_nested(data, config, checkpoint_dir=ckpt, resume=True)
        resume_count = call_count["n"]

    assert first_count == 9
    assert resume_count == 6
    pd.testing.assert_frame_equal(
        full.runs[1].oof_predictions, resumed.runs[1].oof_predictions
    )
    np.testing.assert_array_equal(
        full.inner_scores["roc_auc"].to_numpy(),
        resumed.inner_scores["roc_auc"].to_numpy(),
    )


def test_resumed_runtime_includes_earlier_tuning_seconds(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=8)
    config = tiny_model_configs["rf"]
    ckpt = tmp_path / "tuning"
    first = _run_nested(data, config, checkpoint_dir=ckpt, resume=False)
    carried = pd.read_csv(ckpt / "inner_scores.csv")["fit_seconds"].sum()

    def no_tuning(*args, **kwargs):
        raise AssertionError("every pair is finished; nothing may be re-tuned")

    with mock.patch.object(nested_mod, "cross_val_predict_model", no_tuning):
        resumed = _run_nested(data, config, checkpoint_dir=ckpt, resume=True)

    assert resumed.runtime_seconds >= carried
    assert resumed.runtime_seconds >= 0.5 * first.runtime_seconds


def test_resume_with_changed_candidates_raises(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=9)
    config = tiny_model_configs["rf"]
    ckpt = tmp_path / "tuning"
    _run_nested(data, config, checkpoint_dir=ckpt, resume=False)
    changed = tiny_model_configs["rf"]
    changed = type(changed)(
        name=changed.name,
        fixed_params=changed.fixed_params,
        default_params={**changed.default_params, "min_samples_leaf": 99},
        search_space=changed.search_space,
        early_stopping_rounds=changed.early_stopping_rounds,
    )
    with pytest.raises(ValueError, match="Search candidates changed"):
        _run_nested(data, changed, checkpoint_dir=ckpt, resume=True)


def test_existing_checkpoint_without_resume_raises(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=10)
    config = tiny_model_configs["rf"]
    ckpt = tmp_path / "tuning"
    _run_nested(data, config, checkpoint_dir=ckpt, resume=False)
    with pytest.raises(FileExistsError):
        _run_nested(data, config, checkpoint_dir=ckpt, resume=False)


def test_resume_without_checkpoint_starts_fresh(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=13)
    config = tiny_model_configs["rf"]
    fresh = _run_nested(data, config)
    resumed = _run_nested(data, config, checkpoint_dir=tmp_path / "tuning", resume=True)
    pd.testing.assert_frame_equal(
        fresh.runs[1].oof_predictions, resumed.runs[1].oof_predictions
    )
    assert (tmp_path / "tuning" / "inner_scores.csv").exists()


def test_resume_with_orphan_scores_raises(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=13)
    ckpt = tmp_path / "tuning"
    ckpt.mkdir()
    (ckpt / "inner_scores.csv").write_text(",".join(INNER_SCORE_COLUMNS) + "\n")
    with pytest.raises(ValueError, match="without"):
        _run_nested(data, tiny_model_configs["rf"], checkpoint_dir=ckpt, resume=True)
    assert not (ckpt / "candidates.json").exists()


def test_orphan_scores_without_resume_leave_no_files(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=13)
    ckpt = tmp_path / "tuning"
    ckpt.mkdir()
    (ckpt / "inner_scores.csv").write_text(",".join(INNER_SCORE_COLUMNS) + "\n")
    with pytest.raises(FileExistsError):
        _run_nested(data, tiny_model_configs["rf"], checkpoint_dir=ckpt)
    assert not (ckpt / "candidates.json").exists()
    assert not (ckpt / "fingerprint.json").exists()


@pytest.mark.parametrize("cut", [4, 12, 40])
def test_resume_drops_torn_last_row(
    cut, make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=14)
    config = tiny_model_configs["rf"]
    ckpt = tmp_path / "tuning"
    full = _run_nested(data, config, checkpoint_dir=ckpt)
    path = ckpt / "inner_scores.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    # header + 3 complete pairs (6 rows) + first row of pair 4 + torn second row
    path.write_text("\n".join(lines[:8]) + "\n" + lines[8][:cut], encoding="utf-8")
    resumed = _run_nested(data, config, checkpoint_dir=ckpt, resume=True)
    assert [s.ranking for s in full.selections] == [
        s.ranking for s in resumed.selections
    ]
    np.testing.assert_array_equal(
        full.inner_scores["roc_auc"].to_numpy(),
        resumed.inner_scores["roc_auc"].to_numpy(),
    )
    assert not resumed.inner_scores.isna().drop(columns="best_iteration").any().any()
    pd.testing.assert_frame_equal(
        full.runs[1].oof_predictions, resumed.runs[1].oof_predictions
    )


def test_resume_with_changed_run_raises(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=15)
    config = tiny_model_configs["rf"]
    ckpt = tmp_path / "tuning"
    _run_nested(data, config, checkpoint_dir=ckpt)

    other_cv = CVSettings(
        seed=42, n_outer_folds=4, n_inner_folds=2, n_configs=3, top_k=2
    )
    with pytest.raises(ValueError, match="n_outer_folds"):
        run_nested_cv(data, config, other_cv, checkpoint_dir=ckpt, resume=True)

    other_data = make_training_data(n_samples=300, seed=99)
    with pytest.raises(ValueError, match="data_sha256"):
        _run_nested(other_data, config, checkpoint_dir=ckpt, resume=True)

    other_fixed = type(config)(
        name=config.name,
        fixed_params={**config.fixed_params, "n_estimators": 7},
        default_params=config.default_params,
        search_space=config.search_space,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    with pytest.raises(ValueError, match="fixed_params"):
        _run_nested(data, other_fixed, checkpoint_dir=ckpt, resume=True)

    # top_k only affects the refits, so changing it is allowed.
    more_ranks = CVSettings(
        seed=42, n_outer_folds=3, n_inner_folds=2, n_configs=3, top_k=3
    )
    result = run_nested_cv(data, config, more_ranks, checkpoint_dir=ckpt, resume=True)
    assert set(result.runs) == {1, 2, 3}


def test_inner_scores_dtypes_match_with_and_without_checkpoint(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=16)
    for model_name in ("rf", "lgbm"):
        config = tiny_model_configs[model_name]
        in_memory = _run_nested(data, config).inner_scores
        on_disk = _run_nested(
            data, config, checkpoint_dir=tmp_path / model_name
        ).inner_scores
        assert list(in_memory.columns) == INNER_SCORE_COLUMNS
        assert in_memory.dtypes.astype(str).to_dict() == (
            on_disk.dtypes.astype(str).to_dict()
        )
        assert in_memory["best_iteration"].dtype == np.float64
        pd.testing.assert_frame_equal(
            in_memory.drop(columns="fit_seconds"), on_disk.drop(columns="fit_seconds")
        )


def test_early_stopping_model_without_best_iteration_raises(
    make_training_data, tiny_model_configs
):
    data = make_training_data(n_samples=300, seed=17)
    real_build = nested_mod._build_selections

    def build_without_iterations(inner_scores):
        selections = real_build(inner_scores)
        return [
            OuterFoldSelection(
                s.outer_fold,
                s.ranking,
                s.mean_inner_auc,
                {cid: None for cid in s.mean_best_iteration},
            )
            for s in selections
        ]

    with mock.patch.object(nested_mod, "_build_selections", build_without_iterations):
        with pytest.raises(ValueError, match="best iteration"):
            _run_nested(data, tiny_model_configs["lgbm"])


def test_candidates_json_is_list_of_id_and_params(
    make_training_data, tiny_model_configs, tmp_path
):
    data = make_training_data(n_samples=300, seed=18)
    ckpt = tmp_path / "tuning"
    result = _run_nested(data, tiny_model_configs["rf"], checkpoint_dir=ckpt)
    stored = json.loads((ckpt / "candidates.json").read_text(encoding="utf-8"))
    assert [c["candidate_id"] for c in stored] == [0, 1, 2]
    assert [c["params"] for c in stored] == result.candidates


@pytest.mark.parametrize("model_name", ["lgbm", "rf", "logreg", "mlp"])
def test_fold_metrics(model_name, make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=11)
    result = _run_nested(data, tiny_model_configs[model_name])
    run = result.runs[1]
    assert len(run.fold_metrics) == 3
    for k, fm in enumerate(run.fold_metrics):
        assert set(fm.keys()) == FOLD_METRIC_KEYS
        assert fm["val_roc_auc"] == run.fold_roc_aucs[k]
        assert fm["roc_auc_gap"] == pytest.approx(
            fm["train_roc_auc"] - fm["val_roc_auc"], abs=1e-12
        )
    pooled = classification_metrics(
        run.oof_predictions[TARGET_COL].to_numpy(),
        run.oof_predictions[OOF_PROBA_COL].to_numpy(),
    )
    assert run.oof_brier == pytest.approx(pooled["brier"], abs=1e-12)
    assert run.oof_log_loss == pytest.approx(pooled["log_loss"], abs=1e-12)


def test_feature_importance_rank1_only(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=12)
    lgbm = _run_nested(data, tiny_model_configs["lgbm"])
    assert lgbm.feature_importance is not None
    assert set(lgbm.feature_importance["fold"].unique()) == {0, 1, 2}
    assert set(lgbm.feature_importance["importance_type"].unique()) == {
        "gain",
        "split",
    }
    # rank 1 only: 3 folds x n_features x 2 types, not top_k times that
    assert len(lgbm.feature_importance) == 3 * len(data.feature_names) * 2
    mlp = _run_nested(data, tiny_model_configs["mlp"])
    assert mlp.feature_importance is None
