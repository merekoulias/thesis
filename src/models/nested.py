"""Nested cross-validation with checkpointing, selection and top-k refits."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.constants import SK_ID_CURR, TARGET_COL
from src.models.cv import cross_val_predict_model, make_stratified_splits
from src.models.metrics import (
    classification_metrics,
    fold_fit_diagnostics,
    reference_metrics,
)
from src.models.registry import fit_predict, get_model_spec
from src.models.search import sample_candidates
from src.models.types import (
    INNER_SCORE_COLUMNS,
    OOF_FOLD_COL,
    OOF_PROBA_COL,
    CVSettings,
    ModelConfig,
    NestedCVResult,
    OuterFoldSelection,
    RankedRun,
    Split,
    TrainingData,
)

logger = logging.getLogger(__name__)

CANDIDATES_FILE = "candidates.json"
INNER_SCORES_FILE = "inner_scores.csv"


def inner_splits_absolute(
    outer_train_idx: np.ndarray,
    y: np.ndarray,
    n_inner: int,
    seed: int,
) -> List[Split]:
    """Inner splits of one outer training fold, as absolute row indices of the full data.

    Args:
        outer_train_idx: Row indices of the outer training fold.
        y: Full-data binary target.
        n_inner: Number of inner folds.
        seed: Shuffle seed for inner StratifiedKFold.

    Returns:
        List of (train_idx, val_idx) pairs in full-data index space.
    """
    y_outer = y[outer_train_idx]
    inner_rel = make_stratified_splits(y_outer, n_inner, seed)
    return [(outer_train_idx[tr], outer_train_idx[va]) for tr, va in inner_rel]


def _candidates_payload(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"candidate_id": i, "params": c} for i, c in enumerate(candidates)]


def _candidates_equal(
    stored: List[Dict[str, Any]], candidates: List[Dict[str, Any]]
) -> bool:
    expected = _candidates_payload(candidates)
    return json.loads(json.dumps(stored)) == json.loads(json.dumps(expected))


def _write_candidates(path: Path, candidates: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_candidates_payload(candidates), fh, indent=2)


def _load_candidates(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _complete_pairs(scores: pd.DataFrame, n_inner_folds: int) -> Set[Tuple[int, int]]:
    if scores.empty:
        return set()
    complete: Set[Tuple[int, int]] = set()
    grouped = scores.groupby(["outer_fold", "candidate_id"], sort=False)
    for (outer_fold, candidate_id), group in grouped:
        if len(group) == n_inner_folds:
            complete.add((int(outer_fold), int(candidate_id)))
    return complete


def _prune_and_rewrite_scores(path: Path, n_inner_folds: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=INNER_SCORE_COLUMNS)
    scores = pd.read_csv(path)
    if scores.empty:
        return scores
    keep_rows: List[pd.DataFrame] = []
    grouped = scores.groupby(["outer_fold", "candidate_id"], sort=False)
    for _, group in grouped:
        if len(group) == n_inner_folds:
            keep_rows.append(group)
    if keep_rows:
        pruned = pd.concat(keep_rows, ignore_index=True)
    else:
        pruned = pd.DataFrame(columns=INNER_SCORE_COLUMNS)
    pruned.to_csv(path, index=False)
    return pruned


def _append_score_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", header=write_header, index=False)


def _build_selections(inner_scores: pd.DataFrame) -> List[OuterFoldSelection]:
    selections: List[OuterFoldSelection] = []
    for k in sorted(inner_scores["outer_fold"].unique()):
        fold_df = inner_scores[inner_scores["outer_fold"] == k]
        g = fold_df.groupby("candidate_id", sort=True)
        mean_auc: Dict[int, float] = {}
        mean_it: Dict[int, Optional[float]] = {}
        for cid, grp in g:
            cid_int = int(cid)
            mean_auc[cid_int] = float(grp["roc_auc"].mean())
            it_mean = grp["best_iteration"].mean()
            if pd.isna(it_mean):
                mean_it[cid_int] = None
            else:
                mean_it[cid_int] = float(it_mean)
        ranking = sorted(mean_auc.keys(), key=lambda c: (-mean_auc[c], c))
        selections.append(
            OuterFoldSelection(
                outer_fold=int(k),
                ranking=ranking,
                mean_inner_auc=mean_auc,
                mean_best_iteration=mean_it,
            )
        )
    return selections


def _read_inner_scores(
    scores_path: Optional[Path], inner_scores: pd.DataFrame
) -> pd.DataFrame:
    if scores_path is not None:
        return pd.read_csv(scores_path)
    return inner_scores


def run_nested_cv(
    data: TrainingData,
    model_config: ModelConfig,
    cv_settings: CVSettings,
    *,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = False,
) -> NestedCVResult:
    """Nested CV: tune on inner folds, refit the top_k candidates per outer fold.

    Args:
        data: Validated training data (from io.load_training_data or a fixture).
        model_config: Model config (fixed params, defaults, search space).
        cv_settings: Shared nested-CV settings.
        checkpoint_dir: If given, tuning progress is appended to
            checkpoint_dir / "inner_scores.csv" and candidates are written to
            checkpoint_dir / "candidates.json".
        resume: If True and the checkpoint exists, skip finished
            (outer_fold, candidate_id) pairs.

    Returns:
        NestedCVResult.
    """
    t0 = time.perf_counter()
    spec = get_model_spec(model_config.name)
    X = data.X.to_numpy(dtype=np.float64)
    y = data.y.to_numpy().astype(np.int8)
    ids = data.ids.to_numpy()

    candidates = sample_candidates(
        model_config.default_params,
        model_config.search_space,
        cv_settings.n_configs,
        cv_settings.seed,
    )
    outer_splits = make_stratified_splits(
        y, cv_settings.n_outer_folds, cv_settings.seed
    )

    scores_path: Optional[Path] = None
    candidates_path: Optional[Path] = None
    done_pairs: Set[Tuple[int, int]] = set()
    inner_scores = pd.DataFrame(columns=INNER_SCORE_COLUMNS)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        candidates_path = checkpoint_dir / CANDIDATES_FILE
        scores_path = checkpoint_dir / INNER_SCORES_FILE

        if candidates_path.exists():
            if resume:
                stored = _load_candidates(candidates_path)
                if not _candidates_equal(stored, candidates):
                    raise ValueError(
                        "Search candidates changed since checkpoint was written"
                    )
            else:
                raise FileExistsError(f"Checkpoint already exists: {candidates_path}")
        else:
            if resume:
                raise FileNotFoundError(
                    f"Cannot resume: missing checkpoint file {candidates_path}"
                )
            _write_candidates(candidates_path, candidates)

        if resume and scores_path.exists():
            inner_scores = _prune_and_rewrite_scores(
                scores_path, cv_settings.n_inner_folds
            )
            done_pairs = _complete_pairs(inner_scores, cv_settings.n_inner_folds)
        elif scores_path is not None and scores_path.exists() and not resume:
            raise FileExistsError(f"Checkpoint already exists: {scores_path}")

    model_tag = model_config.name
    logger.info(
        "[%s] candidates: %d | outer folds: %d | inner folds: %d | top_k: %d | rows: %d",
        model_tag,
        cv_settings.n_configs,
        cv_settings.n_outer_folds,
        cv_settings.n_inner_folds,
        cv_settings.top_k,
        len(y),
    )

    for k, (outer_tr, outer_va) in enumerate(outer_splits):
        assert np.intersect1d(outer_tr, outer_va).size == 0

        skipped = 0
        X_tr, y_tr = X[outer_tr], y[outer_tr]
        inner_rel = make_stratified_splits(
            y_tr, cv_settings.n_inner_folds, cv_settings.seed
        )

        for cid, cand in enumerate(candidates):
            if (k, cid) in done_pairs:
                skipped += 1
                continue

            params = {**model_config.fixed_params, **cand}
            oof_inner, fits = cross_val_predict_model(
                spec,
                params,
                cv_settings.seed,
                X_tr,
                y_tr,
                inner_rel,
                early_stopping_rounds=model_config.early_stopping_rounds,
            )
            rows: List[Dict[str, Any]] = []
            inner_aucs: List[float] = []
            inner_its: List[float] = []
            total_fit = 0.0
            for j, (_, va) in enumerate(inner_rel):
                auc = float(roc_auc_score(y_tr[va], oof_inner[va]))
                inner_aucs.append(auc)
                best_it = fits[j].best_iteration
                if best_it is not None:
                    inner_its.append(float(best_it))
                total_fit += fits[j].fit_seconds
                rows.append(
                    {
                        "outer_fold": k,
                        "candidate_id": cid,
                        "inner_fold": j,
                        "roc_auc": auc,
                        "best_iteration": best_it,
                        "fit_seconds": fits[j].fit_seconds,
                    }
                )

            if scores_path is not None:
                _append_score_rows(scores_path, rows)
            else:
                inner_scores = pd.concat(
                    [inner_scores, pd.DataFrame(rows)], ignore_index=True
                )

            mean_inner = float(np.mean(inner_aucs))
            std_inner = float(np.std(inner_aucs, ddof=0))
            if inner_its:
                best_it_log = int(round(float(np.mean(inner_its))))
            else:
                best_it_log = 0
            logger.info(
                "[%s] outer %d/%d | cand %02d/%d | inner AUC %.4f ± %.4f | "
                "best_it %d | %.1f s",
                model_tag,
                k + 1,
                cv_settings.n_outer_folds,
                cid + 1,
                cv_settings.n_configs,
                mean_inner,
                std_inner,
                best_it_log,
                total_fit,
            )

        if skipped:
            logger.info(
                "[%s] outer %d/%d | skipped %d finished candidates (resume)",
                model_tag,
                k + 1,
                cv_settings.n_outer_folds,
                skipped,
            )

        current_scores = _read_inner_scores(scores_path, inner_scores)
        sel_fold = _build_selections(current_scores)[k]
        best_cid = sel_fold.ranking[0]
        best_auc = sel_fold.mean_inner_auc[best_cid]
        logger.info(
            "[%s] outer %d/%d | best cand %02d (inner AUC %.4f)",
            model_tag,
            k + 1,
            cv_settings.n_outer_folds,
            best_cid + 1,
            best_auc,
        )

        del X_tr, y_tr

    inner_scores = _read_inner_scores(scores_path, inner_scores)
    selections = _build_selections(inner_scores)

    runs: Dict[int, RankedRun] = {}
    importance_parts: List[pd.DataFrame] = []

    for rank in range(1, cv_settings.top_k + 1):
        oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_col = np.full(len(y), -1, dtype=np.int64)
        selected: List[Dict[str, Any]] = []
        fold_aucs: List[float] = []
        fold_metrics: List[Dict[str, Any]] = []

        for k, (outer_tr, outer_va) in enumerate(outer_splits):
            sel = selections[k]
            cid = sel.ranking[rank - 1]
            params = {**model_config.fixed_params, **candidates[cid]}
            override: Optional[int] = None
            if spec.uses_early_stopping:
                mean_it = sel.mean_best_iteration[cid]
                if mean_it is not None:
                    override = max(1, int(round(mean_it)))

            out = fit_predict(
                spec,
                params,
                cv_settings.seed,
                X[outer_tr],
                y[outer_tr],
                X[outer_va],
                n_estimators_override=override,
                keep_estimator=(rank == 1 and spec.importance is not None),
                return_train_proba=True,
            )
            oof[outer_va] = out.proba
            fold_col[outer_va] = k
            val_auc = float(roc_auc_score(y[outer_va], out.proba))
            fold_aucs.append(val_auc)

            mean_inner_auc = sel.mean_inner_auc[cid]
            diag = fold_fit_diagnostics(
                y[outer_tr],
                out.train_proba,
                y[outer_va],
                out.proba,
                mean_inner_auc,
            )
            resolved = dict(params)
            if override is not None:
                resolved["n_estimators"] = override
            fold_metrics.append(
                {
                    "outer_fold": k,
                    "candidate_id": cid,
                    "mean_inner_roc_auc": mean_inner_auc,
                    **diag,
                }
            )
            selected.append(
                {
                    "outer_fold": k,
                    "candidate_id": cid,
                    "mean_inner_roc_auc": mean_inner_auc,
                    "params": resolved,
                }
            )

            train_auc = diag["train_roc_auc"]
            val_brier = diag["val_brier"]
            n_est_log = resolved.get("n_estimators", "—")
            logger.info(
                "[%s] refit rank %d | outer %d/%d | AUC %.4f (train %.4f) | "
                "Brier %.4f | n_estimators %s | %.1f s",
                model_tag,
                rank,
                k + 1,
                cv_settings.n_outer_folds,
                val_auc,
                train_auc,
                val_brier,
                n_est_log,
                out.fit_seconds,
            )

            if rank == 1 and out.estimator is not None and spec.importance is not None:
                imp = spec.importance(out.estimator, data.feature_names)
                imp.insert(1, "fold", k)
                importance_parts.append(imp)

        assert not np.isnan(oof).any()
        oof_df = pd.DataFrame(
            {
                SK_ID_CURR: ids,
                TARGET_COL: y,
                OOF_PROBA_COL: oof,
                OOF_FOLD_COL: fold_col,
            }
        )
        pooled = classification_metrics(y, oof)
        runs[rank] = RankedRun(
            rank=rank,
            oof_predictions=oof_df,
            fold_roc_aucs=fold_aucs,
            oof_roc_auc=pooled["roc_auc"],
            oof_brier=pooled["brier"],
            oof_log_loss=pooled["log_loss"],
            fold_metrics=fold_metrics,
            selected=selected,
        )

        ref_m = reference_metrics(y)
        logger.info(
            "[%s] rank %d OOF AUC %.4f | Brier %.4f (ref %.4f) | log loss %.4f "
            "(ref %.4f) | fold AUCs %s",
            model_tag,
            rank,
            pooled["roc_auc"],
            pooled["brier"],
            ref_m["brier_reference"],
            pooled["log_loss"],
            ref_m["log_loss_reference"],
            fold_aucs,
        )

    feature_importance: Optional[pd.DataFrame]
    if importance_parts:
        feature_importance = pd.concat(importance_parts, ignore_index=True)
    else:
        feature_importance = None

    runtime_seconds = time.perf_counter() - t0
    return NestedCVResult(
        model_name=model_config.name,
        candidates=candidates,
        inner_scores=inner_scores,
        selections=selections,
        runs=runs,
        outer_splits=outer_splits,
        feature_importance=feature_importance,
        runtime_seconds=runtime_seconds,
    )
