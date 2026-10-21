"""Tests for nested cross-validation."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.constants import SK_ID_CURR, TARGET_COL
from src.models.cv import make_stratified_splits
from src.models.metrics import classification_metrics
from src.models.nested import inner_splits_absolute, run_nested_cv
from src.models.types import (
    INNER_SCORE_COLUMNS,
    OOF_FOLD_COL,
    OOF_PROBA_COL,
    CVSettings,
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


def test_ranking_matches_mean_inner_auc(make_training_data, tiny_model_configs):
    data = make_training_data(n_samples=300, seed=2)
    config = tiny_model_configs["rf"]
    with mock.patch("src.models.nested.roc_auc_score", return_value=0.5) as mocked:
        result = _run_nested(data, config)
        assert mocked.called
    for sel in result.selections:
        ranking = sel.ranking
        mean_auc = sel.mean_inner_auc
        assert ranking[0] == min(mean_auc.keys(), key=lambda c: (-mean_auc[c], c))


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
    config = tiny_model_configs["rf"]
    outer_sizes: list[int] = []
    inner_sizes: list[int] = []

    def spy_cv(spec, params, seed, X, y, splits, **kwargs):
        inner_sizes.append(len(X))
        from src.models.cv import cross_val_predict_model as real_cv

        return real_cv(spec, params, seed, X, y, splits, **kwargs)

    def spy_fit(spec, params, seed, X_train, y_train, X_pred, **kwargs):
        outer_sizes.append(len(X_train))
        from src.models.registry import fit_predict as real_fit

        return real_fit(spec, params, seed, X_train, y_train, X_pred, **kwargs)

    y = data.y.to_numpy()
    outer_splits = make_stratified_splits(y, 3, 42)
    max_outer_train = max(len(tr) for tr, _ in outer_splits)

    with mock.patch("src.models.nested.cross_val_predict_model", spy_cv):
        with mock.patch("src.models.nested.fit_predict", spy_fit):
            _run_nested(data, config)

    assert all(s <= max_outer_train for s in inner_sizes)
    assert all(s <= max_outer_train for s in outer_sizes)


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

    import src.models.nested as nested_mod
    from src.models.cv import cross_val_predict_model

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

    assert resume_count < 9
    assert resume_count < first_count
    pd.testing.assert_frame_equal(
        full.runs[1].oof_predictions, resumed.runs[1].oof_predictions
    )
    np.testing.assert_array_equal(
        full.inner_scores["roc_auc"].to_numpy(),
        resumed.inner_scores["roc_auc"].to_numpy(),
    )


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
    mlp = _run_nested(data, tiny_model_configs["mlp"])
    assert mlp.feature_importance is None
