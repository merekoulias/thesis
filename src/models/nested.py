"""Nested cross-validation with checkpointing, selection and top-k refits."""

import hashlib
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
    CANDIDATES_FILE,
    FINGERPRINT_FILE,
    INNER_SCORE_COLUMNS,
    INNER_SCORES_FILE,
    OOF_FOLD_COL,
    OOF_PROBA_COL,
    CVSettings,
    ModelConfig,
    NestedCVResult,
    OuterFoldSelection,
    RankedRun,
    Split,
    TrainingData,
    candidates_payload,
)

logger = logging.getLogger(__name__)

FINGERPRINT_CHUNK_ROWS = 20_000

_SCORE_DTYPES: Dict[str, Any] = {
    "outer_fold": np.int64,
    "candidate_id": np.int64,
    "inner_fold": np.int64,
    "roc_auc": np.float64,
    "best_iteration": np.float64,
    "fit_seconds": np.float64,
}
_SCORE_KEY = ["outer_fold", "candidate_id"]
_SCORE_REQUIRED = ["outer_fold", "candidate_id", "inner_fold", "roc_auc", "fit_seconds"]


def inner_splits_absolute(
    outer_train_idx: np.ndarray,
    y: np.ndarray,
    n_inner: int,
    seed: int,
) -> List[Split]:
    """Inner splits of one outer training fold, as absolute indices of the full data.

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


def _json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _candidates_equal(
    stored: List[Dict[str, Any]], candidates: List[Dict[str, Any]]
) -> bool:
    expected = candidates_payload(candidates)
    return _json_round_trip(stored) == _json_round_trip(expected)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _data_sha256(
    X: np.ndarray, y: np.ndarray, ids: np.ndarray, feature_names: List[str]
) -> str:
    """Hash the features, target, ids and column names (X in bounded-memory chunks)."""
    digest = hashlib.sha256()
    digest.update("\x1f".join(feature_names).encode("utf-8"))
    digest.update(np.ascontiguousarray(ids).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    for start in range(0, X.shape[0], FINGERPRINT_CHUNK_ROWS):
        chunk = X[start : start + FINGERPRINT_CHUNK_ROWS]
        digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def _run_fingerprint(
    model_config: ModelConfig,
    cv_settings: CVSettings,
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Everything besides the candidates that the stored inner scores depend on.

    ``top_k`` is deliberately absent: it only affects the refits, not the tuning.
    """
    return {
        "model": model_config.name,
        "seed": cv_settings.seed,
        "n_outer_folds": cv_settings.n_outer_folds,
        "n_inner_folds": cv_settings.n_inner_folds,
        "fixed_params": model_config.fixed_params,
        "early_stopping_rounds": model_config.early_stopping_rounds,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "data_sha256": _data_sha256(X, y, ids, feature_names),
    }


def _check_fingerprint(path: Path, fingerprint: Dict[str, Any]) -> None:
    if not path.exists():
        raise ValueError(f"Checkpoint has no {FINGERPRINT_FILE}; cannot verify it")
    stored = _read_json(path)
    expected = _json_round_trip(fingerprint)
    changed = sorted(
        key
        for key in set(stored) | set(expected)
        if stored.get(key) != expected.get(key)
    )
    if changed:
        raise ValueError(
            "Checkpoint does not match this run (changed: " + ", ".join(changed) + ")"
        )


def _normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the inner-score table with the canonical columns and dtypes."""
    return scores[INNER_SCORE_COLUMNS].astype(_SCORE_DTYPES).reset_index(drop=True)


def _scores_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return _normalize_scores(pd.DataFrame(rows, columns=INNER_SCORE_COLUMNS))


def _prune_scores(scores: pd.DataFrame, n_inner_folds: int) -> pd.DataFrame:
    """Keep only (outer_fold, candidate_id) pairs with every inner fold, fully written.

    Drops rows with missing required fields (e.g. a torn last line) and any pair
    whose ``inner_fold`` values are not exactly 0..n_inner_folds-1.
    """
    if list(scores.columns) != INNER_SCORE_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {INNER_SCORES_FILE}: {list(scores.columns)}"
        )
    valid = scores.dropna(subset=_SCORE_REQUIRED)
    expected = list(range(n_inner_folds))
    complete: List[pd.DataFrame] = []
    for _, group in valid.groupby(_SCORE_KEY, sort=False):
        if sorted(group["inner_fold"].astype(int)) == expected:
            complete.append(group)
    if not complete:
        return _scores_frame([])
    return _normalize_scores(pd.concat(complete, ignore_index=True))


def _prune_and_rewrite_scores(path: Path, n_inner_folds: int) -> pd.DataFrame:
    pruned = _prune_scores(pd.read_csv(path), n_inner_folds)
    pruned.to_csv(path, index=False)
    return pruned


def _append_score_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=INNER_SCORE_COLUMNS)
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


def _current_scores(
    scores_path: Optional[Path], rows: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Inner scores so far: re-read from the CSV when checkpointing, else from memory.

    Re-reading makes resumed and uninterrupted runs select from identical values.
    """
    if scores_path is not None:
        return _normalize_scores(pd.read_csv(scores_path))
    return _scores_frame(rows)


def _prepare_checkpoint(
    checkpoint_dir: Path,
    candidates: List[Dict[str, Any]],
    fingerprint: Dict[str, Any],
    resume: bool,
    n_inner_folds: int,
) -> Tuple[Path, Set[Tuple[int, int]], float]:
    """Validate or create the checkpoint files.

    Returns:
        (scores_path, finished (outer_fold, candidate_id) pairs, summed
        ``fit_seconds`` of those finished pairs = tuning compute done by earlier
        sessions).
    """
    candidates_path = checkpoint_dir / CANDIDATES_FILE
    scores_path = checkpoint_dir / INNER_SCORES_FILE
    fingerprint_path = checkpoint_dir / FINGERPRINT_FILE

    if not resume:
        for existing in (candidates_path, scores_path, fingerprint_path):
            if existing.exists():
                raise FileExistsError(f"Checkpoint already exists: {existing}")
        _write_json(candidates_path, candidates_payload(candidates))
        _write_json(fingerprint_path, fingerprint)
        return scores_path, set(), 0.0

    if not candidates_path.exists():
        if scores_path.exists():
            raise ValueError(
                f"{INNER_SCORES_FILE} exists without {CANDIDATES_FILE}; "
                "cannot verify the checkpoint"
            )
        logger.info("No checkpoint found in %s; starting fresh", checkpoint_dir)
        _write_json(candidates_path, candidates_payload(candidates))
        _write_json(fingerprint_path, fingerprint)
        return scores_path, set(), 0.0

    if not _candidates_equal(_read_json(candidates_path), candidates):
        raise ValueError("Search candidates changed since checkpoint was written")
    _check_fingerprint(fingerprint_path, fingerprint)

    if not scores_path.exists():
        return scores_path, set(), 0.0
    pruned = _prune_and_rewrite_scores(scores_path, n_inner_folds)
    done = {
        (int(k), int(c)) for k, c in zip(pruned["outer_fold"], pruned["candidate_id"])
    }
    return scores_path, done, float(pruned["fit_seconds"].sum())


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
            checkpoint_dir / "inner_scores.csv"; the candidates and a fingerprint
            of the run (CV settings, fixed params, data hash) are written to
            checkpoint_dir / "candidates.json" and "fingerprint.json".
        resume: If True, continue from the checkpoint in ``checkpoint_dir`` (start
            fresh if there is none), skipping finished (outer_fold, candidate_id)
            pairs. Half-written pairs are dropped and recomputed. If False, an
            existing checkpoint raises FileExistsError.

    Returns:
        NestedCVResult. ``runtime_seconds`` is the wall time of this call plus the
        ``fit_seconds`` of the tuning pairs finished by earlier sessions (a lower
        bound of their wall time), so a resumed run does not under-report compute.

    Raises:
        FileExistsError: ``checkpoint_dir`` already holds checkpoint files and
            ``resume`` is False.
        ValueError: ``resume`` is True but the stored candidates or run
            fingerprint differ from this run, the checkpoint cannot be verified,
            or an early-stopping model has no recorded best iteration.
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
    done_pairs: Set[Tuple[int, int]] = set()
    carried_fit_seconds = 0.0
    score_rows: List[Dict[str, Any]] = []

    if checkpoint_dir is not None:
        fingerprint = _run_fingerprint(
            model_config, cv_settings, X, y, ids, data.feature_names
        )
        scores_path, done_pairs, carried_fit_seconds = _prepare_checkpoint(
            Path(checkpoint_dir),
            candidates,
            fingerprint,
            resume,
            cv_settings.n_inner_folds,
        )

    model_tag = model_config.name
    logger.info(
        "[%s] candidates: %d | outer folds: %d | inner folds: %d | top_k: %d | "
        "rows: %d",
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
                score_rows.extend(rows)

            mean_inner = float(np.mean(inner_aucs))
            std_inner = float(np.std(inner_aucs, ddof=0))
            best_it_log = (
                str(int(round(float(np.mean(inner_its))))) if inner_its else "-"
            )
            logger.info(
                "[%s] outer %d/%d | cand %02d/%d | inner AUC %.4f ± %.4f | "
                "best_it %s | %.1f s",
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

        sel_fold = _build_selections(_current_scores(scores_path, score_rows))[k]
        best_cid = sel_fold.ranking[0]
        logger.info(
            "[%s] outer %d/%d | best cand id %02d (inner AUC %.4f)",
            model_tag,
            k + 1,
            cv_settings.n_outer_folds,
            best_cid,
            sel_fold.mean_inner_auc[best_cid],
        )

        del X_tr, y_tr

    inner_scores = _current_scores(scores_path, score_rows)
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
                if mean_it is None:
                    raise ValueError(
                        f"No inner best iteration recorded for candidate {cid} "
                        f"in outer fold {k}"
                    )
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
            "(ref %.4f) | fold AUCs [%s]",
            model_tag,
            rank,
            pooled["roc_auc"],
            pooled["brier"],
            ref_m["brier_reference"],
            pooled["log_loss"],
            ref_m["log_loss_reference"],
            ", ".join(f"{auc:.4f}" for auc in fold_aucs),
        )

    feature_importance: Optional[pd.DataFrame]
    if importance_parts:
        feature_importance = pd.concat(importance_parts, ignore_index=True)
    else:
        feature_importance = None

    runtime_seconds = time.perf_counter() - t0 + carried_fit_seconds
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
