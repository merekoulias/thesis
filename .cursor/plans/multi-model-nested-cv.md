# Plan: multi-model credit scorers with nested tuning (Phase 1, Step 6 — models only)

> **Executor:** Cursor Composer. **Reviewer:** Claude Code (Opus).
> Work through the phases **in order**, each started with the execution prompt in section 6. Do not start phase N+1 until every exit criterion of phase N is met **and** the phase review (section 7) has returned `PASS` or `PASS WITH NITS`.
> If something in this plan is ambiguous or contradicts the code, **stop and ask**. Do not guess.

---

## Progress log (executor fills this in)

**BASE commit** (HEAD of `main` before Phase 0; written in Phase 0): `03a9797`

Phase commits are found with `git log --oneline --grep "phase <N>"` (no hash column, so the plan never needs a commit just to record a hash).

| Phase | Title | Status (`todo` / `in progress` / `done`) | Review verdict | Notes |
| ----- | ----- | ---------------------------------------- | -------------- | ----- |
| 0 | Setup, formatter configuration and baseline snapshot | done | PASS WITH NITS (Claude Code, 2026-10-21: extra log-only commit `20a4e25`; step 1 deviation benign; CLAUDE.md "No linter config is checked in" stale → Phase 9) | BASE `03a9797`; commits `5c5d351` (plan+formatters), `2926c22` (black/isort, 15 files); pytest 28 passed before/after (7.47s / 7.29s); features 493×307507 verified; legacy `oof_roc_auc` 0.7841216640666608; `configs/models/` created empty; step 1: clean tree (plan already tracked, not only untracked plans) |
| 1 | Types, config files, settings loader, candidate sampler | done | PASS WITH NITS (Claude Code, 2026-10-21: `validate_distribution` errors from `load_model_config` lack the file path and non-numeric uniform/loguniform bounds (e.g. YAML `1e-4` → str) raise a bare TypeError; deep-copy test does not mutate a nested value; `default_model_config_path` docstring lacks Args; fold into Phase 3's `settings.py` update) | pytest 47 passed (19 new in test_models_settings.py, full suite 5.67s); black/isort check OK; MLP sample_candidates(20,42)→20; no ground-rule-1 files touched |
| 2 | `SignedLogStandardScaler` preprocessing | done | PASS WITH NITS (Claude Code, 2026-10-21: `constant_mask_` deviation from the reference is justified, because the reference gives std ≈ 2.2e-16 for the worked example's constant column and outputs −1 instead of 0, but it is undocumented (no "executor notes" exist; `constant_mask_` is missing from Attributes; inner `std > 0` `np.where` is dead code); `test_transform_uses_training_statistics` passes even if `transform` refits on its input (manual check reads the fitted attributes after `transform`) and does not exercise forced-zero for a train-constant column with a different validation value; `abs(col1.std()) - 1.0` typo makes that assertion one-sided; commit message differs from the plan's; real data: 252 non-binary, 0 constant columns per outer fold, 2 columns change binary status between folds, max validation |z| ≈ 1305 (`POS_LATEST_CANCELED_MEAN`); put the fixes in `models: phase 2 review fixes`) | pytest preprocessing 8 passed, full suite 57 passed (5.54s); black/isort check OK; constant-column handling extended for fp: the plan's reference gives std 2.2e-16 and output -1 for the worked example's constant column, so `constant_mask_` (via `np.ptp`) forces such columns to 0; review fixes applied (committed in `090ae74`): hand-computed training statistics and no-refit check in `test_transform_uses_training_statistics`, new `test_constant_training_column_stays_zero_for_new_values`, two-sided std assertion, `constant_mask_` documented, dead `np.where` removed; pytest 58 passed |
| 3 | Model registry and `fit_predict` adapter | done | PASS WITH NITS (Claude Code, 2026-10-21: `settings.py` dropped `Tuple` from the typing import but `_CV_KEYS: Tuple[str, ...]` still uses it — import works only because Python 3.14 evaluates annotations lazily; `settings.__annotations__` raises NameError and Python < 3.14 fails on import → re-add `Tuple`; importance functions `zip` without `strict=True` silently truncate on a feature-name length mismatch, and `test_importance_frames` checks row counts only, not feature order/values; `test_mlp_convergence_warning_suppressed` has no control showing the warning fires unsuppressed (verified by hand: 1 warning with the flag off); no test that `sample_weight=None` passes no fit kwargs (verified by hand: `{}`); `fit_predict` Returns docstring omits `train_proba` and is 93 cols; `early_stopping_rounds` without `eval_set` silently ignored; 10 `X does not have valid feature names` UserWarnings are a lightgbm 4.6 + sklearn 1.9 quirk (numpy fit sets `feature_names_in_` = `Column_i`), harmless, not to be suppressed here; commit message differs from the plan's; Phase 2 Notes still say "uncommitted" although the fixes are in `090ae74` (not named `phase 2 review fixes`); put the fixes in `models: phase 3 review fixes`) | pytest registry+settings 48 passed, full suite 85 passed (5.87s); black/isort check OK; `grep KNOWN_MODELS/EARLY_STOPPING` settings.py empty; `.fit(` only registry.py + training.py; 27 tests in test_models_registry.py; LGBM tests emit sklearn feature-name warnings on numpy X_pred (10 warnings); review fixes applied (committed in `dcb748a`): `Tuple` import restored, `zip(strict=True)` in importance functions, importance order/value test, unsuppressed control in the convergence-warning test, no-extra-fit-kwargs test, `early_stopping_rounds` without `eval_set` now raises (+ test), docstring Returns/Raises; pytest 87 passed (29 in test_models_registry.py), black/isort OK |
| 4 | Generic CV core and legacy equivalence check | done | PASS WITH NITS (Claude Code, 2026-10-21: legacy_check re-run by reviewer → EQUIVALENT, AUC 0.784122 both, iterations [240, 411, 297, 349, 527] both, max \|Δp\| 0, exit 0, 4:30 wall; `legacy_check.main(argv: Sequence[str] = None)` needs `Optional[Sequence[str]]`; `_validation_parts_overlap` is a per-row Python loop (0.18 s on 307k rows) → vectorize with `np.bincount`, type `seen`; `cross_val_predict_model` rebuilds an identical `FitOutput` (fit_predict already returns `estimator=None`) → append `fit_out` directly; `make_stratified_splits` / `legacy_check.main` docstrings lack Google Args/Returns sections; test uses literal `"y_pred_proba"` instead of `OOF_PROBA_COL`; cv.py docstring rephrase vs P4.1 is benign (the plan's own text fails its grep); commit message differs from the plan's (double space, title); Phase 3 fixes were committed as `dcb748a` "Enhance model registry…", not `models: phase 3 review fixes`, and Phase 3 Notes still say "uncommitted"; 5 sklearn `X does not have valid feature names` warnings per legacy_check run are the known Phase 3 lightgbm/sklearn quirk; put the fixes in `models: phase 4 review fixes`) | pytest 95 passed (8 new in test_models_cv.py, full suite 6.24s); black/isort check OK; legacy_check: legacy OOF AUC = 0.784122 \| new OOF AUC = 0.784122; best iterations [240, 411, 297, 349, 527] both; folds identical = True; max \|Δp\| = 0.000e+00; EQUIVALENT (~4.6 min wall); grep cv.py lgbm/LGBM/lightgbm empty (docstring rephrased vs P4.1 to satisfy); review fixes applied (committed in `2f2c432`): `Optional[Sequence[str]]` for `legacy_check.main`, vectorised overlap check (`np.unique`), `fits.append(fit_out)`, Google Args/Returns docstrings, `OOF_PROBA_COL` in the test, Phase 3 Notes corrected; re-run after fixes: black/isort OK, pytest 95 passed |
| 5 | Nested CV with checkpoint/resume and top-k refits | done | PASS WITH NITS (Claude Code, 2026-10-21: leakage audit clean — `X[outer_va]` only as refit `X_pred`, inner CV gets `X[outer_tr]` only, refit has no `eval_set`, in-sample proba on outer-train rows only; checkpoint and in-memory paths give identical rankings/OOF; reviewer re-ran: pytest metrics+nested 24 passed in 11.7 s, full suite 119 passed, black/isort OK. Minors: without a checkpoint `inner_scores` has all-`object` dtypes (`best_iteration` holds `None`; built by `pd.concat` onto an empty frame) vs int/float from the CSV → collect rows in a list, build one frame with `INNER_SCORE_COLUMNS`, cast `best_iteration` to float64; `resume=True` with no `candidates.json` raises `FileNotFoundError`, contradicting the docstring ("if the checkpoint exists") and crashing the Phase 6 `--resume` path for a run dir without `tuning/` → start fresh instead; `candidates.json` is written before the orphan-CSV `FileExistsError` (side effect) → check both files before writing; torn last CSV row (e.g. `1,0,`) counts as a complete pair and changed the rank-1 selection in a reviewer test → also require `inner_fold` = {0..n−1} and no NaN `roc_auc`/`fit_seconds` when pruning; resume only fingerprints the candidates: changed `n_outer_folds` or different data/`--max-rows` silently reuse old scores (verified) → store and compare a fingerprint (cv_settings, n_samples, n_features, hash of ids/y) or guard it in Phase 6 `run.py`; ES model with `mean_best_iteration is None` silently refits with `n_estimators` 2000 → raise; `test_outer_validation_used_only_for_prediction` uses rf, asserts `≤ max outer-train` and never checks `eval_set` → use lgbm, exact per-fold sizes, assert no `eval_set` in refit calls; "best cand %02d" logs `cid + 1` while files use 0-based ids → log the id. Nits: ranking test recomputes from the same dict and never checks `mean_inner_auc` against `inner_scores` nor `ranking == [0, 1, 2]` under the tie; feature-importance test does not prove rank-1 only (row count); resume test could assert exactly 6 calls; non-ES log shows `best_it 0`; fold-AUC log not rounded; lines > 88 (metrics.py:39, :102; nested.py:47, :232); `reference_metrics` returns inf and `summarize_folds([])` zeros instead of raising; tests use `list[int]`; commit message differs from the plan's; Phase 4 fixes committed as `2f2c432` "Enhance documentation…", not `models: phase 4 review fixes`, and Phase 4 Notes still say "uncommitted"; put the fixes in `models: phase 5 review fixes`) | pytest metrics+nested 24 passed in 13.0s (<60s); full suite 119 passed in 17.4s; black/isort check OK; grep outer_va: X[outer_va] only in fit_predict (line 367) plus allowed y[outer_va]/oof/fold/disjoint uses; no print in nested.py; review fixes applied (committed in `95147b7`, "Enhance metrics and nested CV functionality"): all minors and nits fixed — `inner_scores` built from a row list with canonical dtypes in both paths; `resume=True` without a checkpoint starts fresh; checkpoint files are checked before anything is written; pruning requires `inner_fold` = 0..n−1 and no missing `roc_auc`/`fit_seconds`; new `fingerprint.json` (model, seed, folds, `fixed_params`, `early_stopping_rounds`, n_samples/n_features, data sha256) must match on resume (plan 5.8 tree and P5.1/P5.2/P5.4 updated); early-stopping model without a recorded best iteration raises; log shows the 0-based candidate id, `-` for non-ES best_it and rounded fold AUCs; `reference_metrics` / `summarize_folds` raise on single-class / empty input; lines ≤ 88; stronger tests (exact-array leakage spy on lgbm, tie/ranking tests, torn-row, fingerprint, dtype, exact resume call counts, importance row count); pytest 132 passed, black/isort OK |
| 6 | Run persistence, CLI, summary, smoke runs | done | PASS WITH NITS (Claude Code, 2026-10-21: reviewer re-ran standard checks (black/isort OK, pytest 141 passed in 28.5 s, test_models_runs.py 5.9 s), a logreg smoke run (0.7576714763600358, identical) and a pandas check of all 8 smoke OOFs (schema/dtypes, ids/TARGET = subsample, fold = `make_stratified_splits(y_sub, 3, 42)`, tree = 5.8, run_config keys in order, lgbm `n_estimators` = round(mean best_it)). Minors: the executor's resume check never interrupted anything (smoke `run.log` shows the "interrupted" run finishing all 9 pairs and the rank-1 OOF before the resume skipped 3+3+3), because `cmd &` in a non-interactive shell starts Python with SIGINT = SIG_IGN (verified), so `kill -INT` is a no-op, and the P6.5 grep `cand 02/03` never matches the log format `cand 02/3`; the reviewer re-ran it with `kill -TERM` (8 rows checkpointed, resume skipped 3 + 1, recomputed 5, AUC identical, max \|Δp\| 0, inner `roc_auc` identical) → fix the P6.5 recipe (`kill -TERM`, grep `outer 2/3 \| cand 01/3`) and use `pkill -TERM` in the Phase 7 stop recipe; `runtime_seconds`/`started_at` cover only the last process (resumed logreg 11–18 s vs 23.6 s uninterrupted; smoke summary shows the resumed value) → accumulate over sessions or record `resumed` + total inner `fit_seconds`; `git_commit` has no dirty flag (smoke runs say `95147b7` while run.py was uncommitted) → `git describe --always --dirty`; `test_cli_max_rows_guard` depends on the CWD (fails from another dir with FileNotFoundError in io.py) and would read the real `artifacts/features` if the guard regressed → pass the default runs dir explicitly and monkeypatch `load_training_data`; run.py `choices=[...]` duplicates the registry → `sorted(MODEL_REGISTRY)`; `--max-rows` = n passes validation but `train_test_split` raises; summary `main`, robustness rows in the summary, run_config contents, `--resume`/`--overwrite` exclusivity and `--model` mismatch untested. Nits: runs.py imports `nested` for file-name constants (4.3 allows `types`) and duplicates `_candidates_payload`/`_write_json`; literals `"TARGET"`, `"baseline"`, `"run.log"` instead of constants; `json.dump` without `allow_nan=False`; FileHandler never closed; existing-run check runs after loading the full data; `save_nested_run` docstring says "resolved" path; summary CSV stores 4-decimal strings; summary table in Notes abbreviated; commit message differs from the plan's; Phase 5 fixes committed as `95147b7` "Enhance metrics…", not `models: phase 5 review fixes`, and Phase 5 Notes still say "uncommitted"; put the fixes in `models: phase 6 review fixes`) | pytest 141 passed (9 new in test_models_runs.py, full suite 31.4s, slowest test 2.1s); black/isort OK; smoke 20k×4 (logreg ~31s, lgbm ~39s, mlp ~33s, rf ~23s wall): rank-1 OOF AUC logreg=0.7577, lgbm=0.7542, mlp=0.7482, rf=0.7347; OOF 20000 rows, no NaN, fold∈{0,1,2}; resume logreg: `skipped 3 finished candidates (resume)`, AUC 0.7576714763600358 unchanged (ref from first smoke loop); summary (after the review fixes, 8 rows, columns: model, rank, oof_roc_auc, std_fold_roc_auc, oof_brier, brier_reference, oof_log_loss, log_loss_reference, mean_roc_auc_gap, mean_selection_optimism, runtime_hours): lgbm/1: 0.7542 0.0023 0.0686 0.0742 0.2486 0.2806 0.2005 -0.0061 0.0051; lgbm/2: 0.7416 0.0026 0.0693 0.0742 0.2528 0.2806 0.2302 -0.0041 0.0051; logreg/1: 0.7577 0.0020 0.0682 0.0742 0.2471 0.2806 0.0424 -0.0198 0.0063; logreg/2: 0.7474 0.0028 0.0695 0.0742 0.2533 0.2806 0.0680 -0.0336 0.0063; mlp/1: 0.7482 0.0053 0.0688 0.0742 0.2500 0.2806 0.0858 -0.0288 0.0060; mlp/2: 0.7273 0.0097 0.0705 0.0742 0.2632 0.2806 0.1503 -0.0160 0.0060; rf/1: 0.7347 0.0054 0.0706 0.0742 0.2586 0.2806 0.1382 -0.0083 0.0056; rf/2: 0.7377 0.0066 0.0707 0.0742 0.2590 0.2806 0.1002 -0.0147 0.0056; git status clean under artifacts/. Review fixes applied (uncommitted): resume recipe fixed in P6.5 (`rm -rf` the run dir, `kill -TERM`, wait for `outer 2/3 \| cand 01/3`) and `pkill -TERM` in Phase 7; recipe re-run on real data: killed after 8 of 18 tuning rows (exit 143, no `metrics.json`), resume skipped 3 + 1 pairs, AUC 0.7576714763600358, max \|Δp\| 0; `runtime_seconds` = last session wall time + `fit_seconds` of pairs finished earlier (resumed logreg 22.8 s vs 23.6 s uninterrupted); `run_config.json` gets `resumed` and `git_commit` a `-dirty` suffix; `--max-rows` must be in [2, n−1]; existing-run check before loading data, `--overwrite` deletes after the data loaded; model choices from `MODEL_REGISTRY`; log handlers added for the run only and removed/closed afterwards (no `basicConfig(force=True)`); `runs.py` no longer imports `nested` (file-name constants and `candidates_payload` moved to `types.py`), `_json_default` removed, `allow_nan=False` in both JSON writers, `TARGET_COL` / `DEFAULT_VARIANT` / `LOG_FILE` constants; summary CSV keeps full precision; 15 new tests (pytest 156 passed, black/isort OK), `test_cli_max_rows_guard` is CWD-independent and never loads data; deliberately left: `_write_json` exists in both `nested.py` (plain checkpoint JSON) and `runs.py` (numpy-safe) because `nested` may not import `runs`; Phase 5 Notes corrected (fixes were committed in `95147b7`) |
| 7 | Full runs (4 models) | done | PASS WITH NITS (Claude Code, 2026-10-22: reviewer re-ran black/isort (OK), pytest (156 passed, 36 s), `python -m src.models.summary` (CSV byte-identical to the committed one) and an own verification script: all 4 trees = 5.8 with ranks 2–5 (mlp without importance), 300 unique inner rows each, `candidates.json` = a fresh `sample_candidates` from the YAML, all 20 OOF files 307,507 rows, columns/dtypes `int64/int8/float64/int64`, no NaN, p ∈ [0, 1], `SK_ID_CURR`/`TARGET` = `ids_train`/`y_train` in order, `fold` = `make_stratified_splits(y, 5, 42)`, pooled/fold AUC, Brier and log loss recomputed = `metrics.json` (< 1e-12), all sanity flags met; lgbm `n_estimators` = round(mean inner best_it) and candidate = `ranking[r−1]` for ranks 1–5; refits from `selected_params.json` reproduce the stored OOF bit-exactly (lgbm fold 4, logreg fold 0, max \|Δp\| 0); all runs at clean `0480a66`, not resumed; `run.log`s and stdout have no tracebacks or ConvergenceWarnings, only the known 350 lightgbm/sklearn feature-name UserWarnings. Results: rank-1 candidate identical in all 5 folds for logreg (17), mlp (3), rf (8), lgbm 8 in 4 folds and 19 in fold 4; candidate 0 never wins (best rank 4); tuning gain over candidate 0 (mean inner AUC) logreg ≤ 0.0005, lgbm/rf ≈ 0.002–0.003, mlp 0.003–0.005; rank 1–5 OOF AUC spread lgbm 0.0005, logreg 0.0003, rf 0.0030, mlp 0.0054 (> its fold std 0.0048; rank 2 beats rank 1); every fold's validation Brier/log loss below the references, `mean_selection_optimism` −0.0011 to −0.0026 (\|·\| < fold std), outer fold 1 easiest for every model (optimism ≈ −0.010); no 5.10 warning sign. Report-only: rf AUC rises monotonically with `max_features` (log2 < sqrt < 0.05 < 0.1) and the top 3 candidates all use 0.1, the largest value offered (range truncated by the deliberate runtime cap, 8.1 → thesis caveat, no change); lgbm fold-4 winner (cand 19) sits near the upper edges of `min_child_samples`, `colsample_bytree`, `reg_lambda`; nested lgbm 0.7862 > legacy 0.7841 despite the legacy optimism, consistent with the ≈ 0.002 tuning gain. Minors: commit labels break the `--grep` convention — the Phase 7 commit is `cdb7a06` "models: phase 7b -  Full runs (4 models)" (plan: `models: phase 7 - full runs`), `1e0cfbc` "models: phase 7 - Enhance nested CV…" is only the Phase 6 verdict (1 plan line; its body describes code it does not contain), and `0480a66` "models: phase 8 - Refine nested CV…" is the Phase 6 review fixes (the code the full runs used), so `--grep "phase 8"` will hit it → do not rewrite history; use exactly `models: phase 8 - cleanup` next, and Phase 6 Notes still say "Review fixes applied (uncommitted)" (they are `0480a66`). Nits: Notes abbreviate the summary table (ranks 2–5 of logreg/mlp/rf as ranges, no header) → Phase 9 should take the table from `artifacts/runs/summary_baseline.csv`; the verification script asked for in 6.2 is neither shown nor kept; Python warnings reach only `full_runs_stdout.log`, never `run.log` (optional later: `logging.captureWarnings(True)` in `run.py::_attach_log_file`); progress lines count candidates 1-based (`cand 09/20`) while `best cand id 08` is 0-based (as in P5.3; say so in the Phase 9 docs); the Phase 7 time table is off (logreg 2.39 h vs 1–1.5 h, lgbm 1.25 h vs 2–3 h, mlp 2.59 h, rf 3.42 h vs 4–7 h) → use the measured times in Phase 9) | Phase 7a user chain completed (rf metrics 2026-10-22 08:20, `full_runs_stdout.log` 1252 lines, no Traceback); precondition metadata 493×307507 OK; black/isort check OK; pytest 156 passed in 41.9s; verification script PASS (trees 5.8, inner_scores 300×4, OOF 307507 rows ranks 1–5, ids/folds vs `make_stratified_splits(y,5,42)`, sanity AUCs lgbm=0.7862 logreg=0.7767 mlp=0.7738 rf=0.7637, all brier/log_loss below reference); `python -m src.models.summary` → `summary_baseline.csv`; summary table: lgbm/1 0.7862 0.0036 0.0659 0.0742 0.2373 0.2805 0.0711 -0.0026 1.25h; lgbm/2 0.7859 0.0035 0.0659 0.0742 0.2374 0.2805 0.0901 -0.0026 1.25h; lgbm/3 0.7859 0.0034 0.0660 0.0742 0.2374 0.2805 0.0783 -0.0028 1.25h; lgbm/4 0.7860 0.0037 0.0659 0.0742 0.2374 0.2805 0.0870 -0.0031 1.25h; lgbm/5 0.7857 0.0031 0.0660 0.0742 0.2375 0.2805 0.0665 -0.0030 1.25h; logreg/1–5 0.7764–0.7767 ~2.39h; mlp/1 0.7738 0.0048 0.0669 0.0742 0.2416 0.2805 0.0075 -0.0011 2.59h; mlp/2–5 0.7686–0.7740; rf/1 0.7637 0.0041 0.0683 0.0742 0.2477 0.2805 0.1002 -0.0021 3.42h; rf/2–5 0.7607–0.7626; no code changes |
| 8 | Cleanup of unneeded code, files and dependencies | done | PASS WITH NITS (Claude Code, 2026-10-22: reviewer re-ran black/isort (OK), pytest (154 passed = 156 − the 2 removed pipeline tests), `python -m src.models.legacy_check` (EQUIVALENT, AUC 0.784122 both, iterations [240, 411, 297, 349, 527] both, max \|Δp\| 0, exit 0, 4:25 wall), `import analysis.helpers.data_audit` and the `from src.models import …` line (OK; `ModelArtifacts` gone, `import *` works, lightgbm still lazily imported), `pip install -r requirements.txt --dry-run` (OK); removed-symbol grep over src/tests/analysis incl. notebooks → no hits; third-party imports of src/tests/analysis (.py + notebooks) = numpy, pandas, sklearn, lightgbm, yaml, scipy, matplotlib, seaborn, pytest, IPython, all covered by requirements.txt (pyarrow via pandas parquet, IPython via jupyter); renamed `test_models_training.py` keeps all 5 `train_lgbm_cv` / `load_training_data` tests; Phase 8 commit touches no Ground-rule-1 file outside the Phase 8 list (`training.py`, `io.py`, `configs/model.yaml`, `analysis/`, `FoldResult`/`CVResult`/`TrainingData`/`OOF_PROBA_COL`/`load_yaml_config` unchanged). Minors: `git grep "fe\.py\|feature_importance\.py"` is not empty — the plan matches itself (criterion cannot pass as written; should exclude `.cursor/plans`) and `analysis/notebooks/00_dataset_overview.ipynb:944` (markdown cell: "before modeling in `fe.py`") is a real stale reference that Ground rule 1 blocks; executor correctly reported instead of editing → user to decide whether to allow a markdown-only edit (no re-execution) to "in `src/features/`"; `CLAUDE.md:22` `pytest tests/test_models_pipeline.py` now points to a deleted file and is in neither the Phase 8 list nor the Phase 9 instructions/grep → change to `tests/test_models_training.py` in Phase 9 and add `test_models_pipeline` to the Phase 9 grep; item 9 Notes say "user did not confirm deletion", which is not a recorded answer (artifacts/models/ still present, 33 MB) → record the user's actual yes/no; commit labels again break the convention despite the Phase 7 review — `14c146e` is "models: phase 8 -  Cleanup of unneeded code, files and dependencies" (plan: `models: phase 8 - cleanup`), and `ed4de7d` "models: phase 9 - Refine nested CV functionality…" only commits the Phase 7 verdict (1 plan line; body describes code it does not contain), so `--grep "phase 9"` will hit it → do not rewrite history; name Phase 9 exactly `models: phase 9 - documentation`. Nits: `src/README.md:41` "Run model" command was already rewritten (a Phase 9 item; correct, flags exist), while `:13`, `:22`, `:38` still say LightGBM only → Phase 9; `__getattr__` in `src/models/__init__.py:17` has no return annotation (pre-existing pattern) → `-> Any`; `fast_model_config["outputs"]` in `test_models_training.py:67` and `outputs:` in `configs/model.yaml` were read only by the removed `save_model_artifacts` (now dead; model.yaml frozen → mention in Phase 9 docs); version floors in requirements.txt (`scikit-learn>=1.0.0`, `lightgbm>=3.0.0`) predate the plan and are far below the tested 1.9 / 4.6 → optional follow-up) | black/isort check OK; pytest 154 passed in 37.1s (removed 2 pipeline tests); `git grep` pipeline symbols clean in src/tests/analysis; `fe.py`/`feature_importance.py` grep still hits plan + `analysis/notebooks/00_dataset_overview.ipynb` (analysis/ out of scope per ground rule 1); requirements dry-run OK; legacy_check EQUIVALENT legacy/new AUC 0.784122, max \|Δp\| 0 (~4.0 min wall); `artifacts/models/` retained — user did not confirm deletion (item 9); after review (2026-10-22): user answered **yes** to item 9 → `artifacts/models/` deleted (gitignored, 33 MB); user allowed a markdown-only edit of `analysis/notebooks/00_dataset_overview.ipynb` (cell "5. Implications for later EDA": "before modeling in `fe.py`" → "before modeling (done in `src/features/`)", not re-executed) → `git grep "fe\.py\|feature_importance\.py" -- ':!.cursor/plans'` now finds nothing; `CLAUDE.md:22` example command changed to `pytest tests/test_models_training.py` (runs: 5 passed); nits fixed: `src/README.md` layout comment, `models/` row and Status line describe the four models / nested CV; `__getattr__ -> Any` in `src/models/__init__.py`; dead `outputs` block removed from `fast_model_config`; `requirements.txt` floors raised to the tested `scikit-learn>=1.9`, `lightgbm>=4.6`; `configs/model.yaml` `outputs:` (frozen) added to the Phase 9 legacy-baseline docs bullet |
| 9 | Documentation | todo | | |

---

## 1. Goal and scope

Build the credit-scoring part of Phase 1, Step 6. Four models are trained on the **same frozen feature matrix**, each tuned with **the same nested-CV budget on ROC AUC only**:

| Registry name | Model | Library |
| ------------- | ----- | ------- |
| `lgbm` | LightGBM (existing baseline family) | `lightgbm.LGBMClassifier` |
| `rf` | Random forest | `sklearn.ensemble.RandomForestClassifier` |
| `logreg` | Logistic regression (L2) | `sklearn.linear_model.LogisticRegression` |
| `mlp` | Multi-layer perceptron | `sklearn.neural_network.MLPClassifier` |

Each run writes out-of-fold (OOF) predictions with the **same schema as the legacy baseline**, so a later fairness plan can consume any model unchanged.

### In scope

- New modules in `src/models/`, new configs in `configs/models/`, new tests in `tests/`.
- Nested CV: 5 outer folds × 3 inner folds, 20 candidates per model, selection on mean inner ROC AUC.
- Refit of the top 5 candidates per outer fold (rank 1 = main run; ranks 2–5 = robustness runs).
- Checkpoint/resume for long runs.
- A one-off check that the new code reproduces the legacy LightGBM OOF predictions exactly.
- black + isort as the project formatters (Phase 0).
- Cleanup of code, files, dependencies and doc references made unnecessary by the new runner (Phase 8).
- Docs update (Phase 9).

### Out of scope (do NOT implement)

- **Fairness metrics, thresholds, calibration, group reports** → separate plan (`src/fairness/`).
- **Mitigation methods** (reweighing, ExponentiatedGradient, ThresholdOptimizer, adversarial debiasing, …) → separate plan. Only one seam is prepared now: `fit_predict(..., sample_weight=...)`.
- **Feature engineering changes.** `src/features/` and `artifacts/features/` are frozen.
- **Any use of `X_test`.**
- **Reading `fairness_groups_*.parquet`** in `src/models/`.
- Changing `train_lgbm_cv` or anything notebook `08` depends on (see Ground rules and Phase 8).
- Editing `docs/decisions.md` or `docs/open-questions.md` (the reviewer maintains them).

---

## 2. Decisions this plan implements (summary — authoritative text is in `docs/decisions.md`)

1. **Tuning:** random search, **20 candidates per model**; candidate 0 = the model's `default_params`, candidates 1–19 sampled with seed 42. The same candidate list is used in every outer fold. Selection metric: **mean inner ROC AUC**. Never a fairness metric.
2. **Nested CV:** outer = `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (identical to the legacy folds); inner = `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` applied to each outer training fold. The outer validation fold is **never** seen during tuning or early stopping.
3. **LightGBM early stopping:** during tuning, on the inner validation fold. At refit (full outer training fold), no early stopping: `n_estimators = round(mean inner best_iteration of that candidate)`.
4. **MLP epochs:** `max_iter` (epochs) is a searched hyperparameter; `early_stopping=False`. (sklearn's MLP early stopping uses accuracy, which is meaningless at an 8% positive rate.)
5. **Frozen features:** `artifacts/features/X_train.parquet` (493 columns, 307,507 rows) is used as-is by all models. Logistic regression and the MLP get a model-specific `SignedLogStandardScaler` **inside** their sklearn `Pipeline`, so it is fitted per training fold.
6. **No class weights, no resampling** (natural 8.07% positive rate), same as the legacy baseline.
7. **Robustness check:** ranks 2–5 per outer fold are refitted and their OOF predictions are saved.
8. **Frozen hyperparameters for later mitigation:** the per-outer-fold selected parameters are saved in `selected_params.json`; the mitigation plan will reuse them.
9. **Legacy LightGBM code:** `src/models/training.py` (`train_lgbm_cv`), `src/models/io.py`, `configs/model.yaml` and `analysis/` stay (notebook `08` uses `train_lgbm_cv`; `legacy_check` compares against it). The legacy CLI `src/models/pipeline.py`, `ModelArtifacts` and the `artifacts/models/` outputs are superseded by the new runner and are removed in Phase 8.
10. **Metrics kept per model:** see section 5.9.

---

## 3. Ground rules for the executor

Numbering: sections 1–8 are plan-wide; subsections inside a phase are prefixed with `P` (e.g. `P5.2` = Phase 5, subsection 2), so `5.8` always means the shared reference and `P5.2` the Phase 5 algorithm.

1. **Do not modify** these files: `src/models/training.py`, `src/models/pipeline.py`, `src/models/io.py`, `configs/model.yaml`, anything under `src/data/`, `src/features/`, `analysis/`, `artifacts/features/`, `tests/test_models_pipeline.py`, `docs/decisions.md`, `docs/open-questions.md`. Exceptions, each only in the phase named: the formatting-only commit of Phase 0 (black + isort over `src/`, `tests/`, `analysis/helpers/`); **adding** new types to `src/models/types.py` (Phase 1); appending fixtures to `tests/conftest.py` (Phase 3); the removals and edits listed in Phase 8.
2. **Style** (from `.cursor/rules/python-src.mdc`): formatting is done by the tools, never by hand. **black** (line length 88) and **isort** (profile `black`) are installed in `.venv` and configured in `pyproject.toml` (Phase 0). Before every commit run `black src tests analysis/helpers && isort src tests analysis/helpers`. What the tools do not cover is your job: Google-style docstrings on every public function/class; type hints on all public functions using `typing` (`Optional[X]`, `Dict`, `List`, `Tuple` — **not** `X | None`, not `dict[...]`); custom types in `src/models/types.py`; no mutable default arguments.
3. **No hard-coded hyperparameters, seeds, fold counts or paths in `src/`** — they come from `configs/models/*.yaml` or CLI arguments. Library-level constants (file names, column names) go in module-level `UPPER_CASE` constants.
4. **No new runtime dependencies.** Use only numpy, pandas, scikit-learn 1.9, lightgbm 4.6, pyyaml, pyarrow, matplotlib in `src/`. Do **not** import `scipy` in `src/`. (black and isort are dev tools added in Phase 0; Phase 8 only aligns `requirements.txt` with imports that already exist.)
5. **Tests use tiny synthetic data only** (≤ 600 rows). Never read `data/*.csv` or `artifacts/` in tests. The full new test suite must run in **< 90 s**. Exception: Phase 1 tests may load the small YAML files in `configs/models/`.
6. Run everything from the repo root with `source .venv/bin/activate`.
7. **Git:** work directly on **`main`** (no feature branch). Phase 0 records the starting commit as **BASE**; reviewers diff against it. Make **one commit per phase** after its exit criteria pass, message `models: phase N - <short title>`, including the Progress-log update (and any verdict the reviewer wrote into the log). Fixes after a review go in a separate commit `models: phase N review fixes`. Do not push. Do not amend, rebase or squash. Never run `git reset --hard`, `git checkout -- .`, `git restore .`, `git stash` or `git clean` to undo work; if something must be undone, stop and ask.
8. **Large data:** `X` is 307,507 × 493 float64 (~1.2 GB). Convert it once with `data.X.to_numpy(dtype=np.float64)`; index with integer arrays; `del` large intermediates (`X_outer_train`, etc.) at the end of each outer fold.
9. If an exit criterion fails and the fix is not obvious, **stop and report**. Do not loosen tolerances or thresholds written in this plan.

### 3.1 Standard checks (part of the exit criteria of every phase)

```bash
source .venv/bin/activate
black --check src tests analysis/helpers
isort --check-only src tests analysis/helpers
pytest -q
```

All three must exit with code 0. (Before Phase 0's formatting commit, black and isort report existing files; that is expected.)

---

## 4. Target architecture

### 4.1 File tree (new = ★, modified = ✎, removed = ✗, kept as is = ·)

```
pyproject.toml               ★ black + isort configuration (Phase 0)
requirements.txt             ✎ dev tools (Phase 0); aligned with real imports (Phase 8)
configs/
  model.yaml                 · legacy LightGBM config (notebook 08, legacy_check)
  paths.yaml                 ✎ runs_dir replaces models_dir (Phase 8)
  models/                    ★
    cv.yaml                  ★ shared nested-CV settings (equal budget)
    cv_smoke.yaml            ★ tiny settings for smoke runs
    lgbm.yaml                ★
    rf.yaml                  ★
    logreg.yaml              ★
    mlp.yaml                 ★
src/models/
  __init__.py                ✎ exports (Phase 8)
  types.py                   ✎ add new dataclasses/aliases (Phase 1); ModelArtifacts removed (Phase 8)
  io.py                      · load_training_data(), load_yaml_config()  (reused, not changed)
  training.py                · legacy train_lgbm_cv() (notebook 08, legacy_check)
  pipeline.py                ✗ legacy CLI, removed in Phase 8 (superseded by run.py)
  settings.py                ★ load + validate cv.yaml and <model>.yaml
  search.py                  ★ sample_candidates()
  preprocessing.py           ★ SignedLogStandardScaler
  registry.py                ★ MODEL_REGISTRY, get_model_spec(), fit_predict()
  cv.py                      ★ make_stratified_splits(), cross_val_predict_model()
  legacy_check.py            ★ CLI: new CV core vs train_lgbm_cv on full data
  metrics.py                 ★ ROC AUC, Brier, log loss, no-skill reference, fit diagnostics
  nested.py                  ★ run_nested_cv()
  runs.py                    ★ save/load run directories
  run.py                     ★ CLI: python -m src.models.run --model <name>
  summary.py                 ★ CLI: python -m src.models.summary
tests/
  conftest.py                ✎ add synthetic fixtures (append only, Phase 3)
  test_models_pipeline.py    ✎ renamed to test_models_training.py; pipeline tests removed (Phase 8)
  test_models_settings.py    ★
  test_models_preprocessing.py ★
  test_models_registry.py    ★
  test_models_cv.py          ★
  test_models_metrics.py     ★
  test_models_nested.py      ★
  test_models_runs.py        ★
```

### 4.2 Data flow

```text
artifacts/features/{X_train,y_train,ids_train}.parquet
        │  src.models.io.load_training_data()   (existing contract checks)
        ▼
TrainingData ──► X: np.ndarray float64, y: np.ndarray int8, ids
        │
        │  configs/models/cv.yaml  ──► CVSettings
        │  configs/models/<m>.yaml ──► ModelConfig ──► sample_candidates() ──► 20 candidates
        ▼
src.models.nested.run_nested_cv()
   for outer fold k in 0..4:                         (outer = legacy folds)
       for candidate c in 0..19:
           inner 3-fold CV on outer-train(k)  ──► 3 inner AUCs   ──► tuning/inner_scores.csv (checkpoint)
       rank candidates by mean inner AUC
       for rank r in 1..5: refit candidate on full outer-train(k), predict outer-val(k)
        ▼
src.models.runs.save_nested_run()
        ▼
artifacts/runs/<model>/baseline/
   oof_predictions.parquet   (rank 1)        ◄── later: src/fairness/ reads this
   metrics.json, selected_params.json, run_config.json, feature_importance.parquet
   tuning/{candidates.json, fingerprint.json, inner_scores.csv, rankings.json}
   robustness/rank_{2..5}/{oof_predictions.parquet, metrics.json, selected_params.json}
   run.log
```

### 4.3 Module responsibilities and dependencies

| Module | Responsibility | May import from |
| ------ | -------------- | --------------- |
| `types.py` | dataclasses, type aliases, column/file constants | stdlib, numpy, pandas, `src.constants` |
| `settings.py` | parse + validate YAML into `CVSettings` / `ModelConfig` | `types`, `registry`, `io.load_yaml_config` |
| `search.py` | deterministic candidate sampling | `types`, numpy |
| `preprocessing.py` | `SignedLogStandardScaler` | numpy, sklearn |
| `registry.py` | build estimators, `fit_predict`, importances | `types`, `preprocessing`, sklearn, lightgbm |
| `cv.py` | splits + generic CV loop (no LightGBM-specific code) | `types`, `registry`, sklearn |
| `metrics.py` | model-level metrics and fit diagnostics (no groups, no threshold) | numpy, sklearn |
| `nested.py` | nested CV, checkpoint, selection, refits | `types`, `cv`, `registry`, `search`, `metrics` |
| `runs.py` | write/read run directories | `types` |
| `run.py` / `summary.py` / `legacy_check.py` | thin CLIs | everything above + `io` |

**Rule:** `cv.py`, `nested.py`, `runs.py` must contain **no** `if model_name == "lgbm"` branches. All model-specific behaviour lives in `ModelSpec` (registry).

---

## 5. Shared reference: configs, types and contracts

### 5.1 `configs/models/cv.yaml`

```yaml
# Shared nested-CV settings. Identical for every credit-scoring model (equal tuning budget).
# See docs/decisions.md, row "Hyperparameter tuning".
seed: 42
n_outer_folds: 5   # same outer folds as the legacy baseline (StratifiedKFold, shuffle, seed 42)
n_inner_folds: 3   # inner stratified CV inside each outer training fold
n_configs: 20      # candidate 0 = default_params, candidates 1..19 = random draws
top_k: 5           # rank 1 = main run; ranks 2..5 = robustness runs
```

### 5.2 `configs/models/cv_smoke.yaml`

```yaml
# Tiny settings for smoke runs on a subsample (never used for thesis results).
seed: 42
n_outer_folds: 3
n_inner_folds: 2
n_configs: 3
top_k: 2
```

### 5.3 Model YAML schema

Every `configs/models/<name>.yaml` has exactly these top-level keys:

| Key | Type | Meaning |
| --- | ---- | ------- |
| `name` | str | registry name; must equal the file stem |
| `early_stopping_rounds` | int or `null` | required int ≥ 1 for models whose spec has `uses_early_stopping=True` (only `lgbm`); must be `null`/absent otherwise |
| `fixed_params` | mapping | passed to every fit, never searched |
| `default_params` | mapping | candidate 0; **keys must equal the keys of `search_space`** |
| `search_space` | mapping name → distribution | see 5.4 |

Forbidden keys anywhere in `fixed_params` / `default_params` / `search_space`: `random_state`, `seed` (the seed is injected by the registry). `fixed_params` and `search_space` must not share keys.

### 5.4 Distribution spec (implemented in `search.py`, no scipy)

| `dist` | Required keys | Sample (with `rng = np.random.default_rng(seed)`) | Returned type |
| ------ | ------------- | ------------------------------------------------- | ------------- |
| `uniform` | `low < high` | `rng.uniform(low, high)` | float rounded to 6 significant digits |
| `loguniform` | `0 < low < high` | `exp(rng.uniform(log(low), log(high)))` | float rounded to 6 s.d. |
| `int_uniform` | ints `low < high` | `rng.integers(low, high, endpoint=True)` | Python `int` |
| `int_loguniform` | ints `1 ≤ low < high` | `int(round(exp(rng.uniform(log(low), log(high)))))` | Python `int` |
| `choice` | non-empty list `values` | `values[rng.integers(len(values))]` (deep-copied) | element as-is (may be `None`, str, number, list) |

Rounding to 6 significant digits: `float(f"{value:.6g}")`. Parameters are sampled in **sorted key order** for each candidate, so YAML key order never changes the result.

### 5.5 The four model YAML files (copy exactly)

`configs/models/lgbm.yaml`

```yaml
# LightGBM credit scorer. Candidate 0 = legacy configs/model.yaml values.
name: lgbm
early_stopping_rounds: 50     # inner validation fold during tuning only
fixed_params:
  objective: binary
  metric: auc
  boosting_type: gbdt
  learning_rate: 0.05
  n_estimators: 2000          # upper bound while tuning; refit uses the mean inner best iteration
  max_depth: -1
  subsample_freq: 1           # needed for `subsample` to have any effect (legacy left it at 0)
  verbosity: -1
  n_jobs: -1
default_params:
  num_leaves: 31
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.0
  reg_lambda: 0.0
search_space:
  num_leaves: {dist: int_uniform, low: 15, high: 127}
  min_child_samples: {dist: int_loguniform, low: 20, high: 500}
  subsample: {dist: uniform, low: 0.6, high: 1.0}
  colsample_bytree: {dist: uniform, low: 0.2, high: 0.9}
  reg_alpha: {dist: loguniform, low: 0.001, high: 10.0}
  reg_lambda: {dist: loguniform, low: 0.001, high: 10.0}
```

`configs/models/rf.yaml`

```yaml
# Random forest credit scorer. n_estimators is fixed (not a tuning parameter for RF).
name: rf
early_stopping_rounds: null
fixed_params:
  n_estimators: 200
  criterion: gini
  bootstrap: true
  n_jobs: -1
default_params:
  max_features: sqrt
  min_samples_leaf: 50
  max_depth: null
  max_samples: null
search_space:
  max_features: {dist: choice, values: [sqrt, log2, 0.05, 0.1]}
  min_samples_leaf: {dist: int_loguniform, low: 5, high: 300}
  max_depth: {dist: choice, values: [null, 12, 20, 30]}
  max_samples: {dist: choice, values: [null, 0.5, 0.75]}
```

`configs/models/logreg.yaml`

```yaml
# L2 logistic regression. SignedLogStandardScaler is applied inside the pipeline.
# scikit-learn >= 1.8 deprecates `penalty`; the default l1_ratio=0.0 is pure L2, so neither is set.
name: logreg
early_stopping_rounds: null
fixed_params:
  solver: lbfgs
  max_iter: 2000
  tol: 0.0001
default_params:
  C: 1.0
search_space:
  C: {dist: loguniform, low: 0.0001, high: 10.0}
```

`configs/models/mlp.yaml`

```yaml
# MLP credit scorer. SignedLogStandardScaler is applied inside the pipeline.
# max_iter = number of epochs and is searched; early_stopping is off because sklearn's
# MLP early stopping monitors accuracy, which is uninformative at an 8% positive rate.
# Pilot (164k rows): alpha=1.0, lr=1e-3, 5 epochs gave AUC 0.77; weak alpha overfits after ~10 epochs.
name: mlp
early_stopping_rounds: null
fixed_params:
  activation: relu
  solver: adam
  early_stopping: false
  shuffle: true
  tol: 0.0001
  n_iter_no_change: 10
default_params:
  hidden_layer_sizes: [128, 64]
  alpha: 1.0
  learning_rate_init: 0.001
  batch_size: 512
  max_iter: 5
search_space:
  hidden_layer_sizes: {dist: choice, values: [[64], [128], [64, 32], [128, 64], [256, 128]]}
  alpha: {dist: loguniform, low: 0.001, high: 10.0}
  learning_rate_init: {dist: loguniform, low: 0.0001, high: 0.003}
  batch_size: {dist: choice, values: [256, 512, 1024]}
  max_iter: {dist: int_uniform, low: 3, high: 20}
```

### 5.6 New types (add to `src/models/types.py`, below the existing content)

```python
from typing import Callable, Tuple  # merge into the existing typing import line

# A (train_indices, validation_indices) pair of integer index arrays.
Split = Tuple[np.ndarray, np.ndarray]

# (params, seed) -> unfitted estimator with fit / predict_proba.
EstimatorBuilder = Callable[[Dict[str, Any], int], Any]

# (fitted estimator, feature names) -> long frame: feature, importance_type, importance.
ImportanceFn = Callable[[Any, List[str]], pd.DataFrame]


@dataclass(frozen=True)
class CVSettings:
    """Nested cross-validation settings shared by every model."""

    seed: int
    n_outer_folds: int
    n_inner_folds: int
    n_configs: int
    top_k: int


@dataclass(frozen=True)
class ModelConfig:
    """One model's fixed parameters, default candidate and search space."""

    name: str
    fixed_params: Dict[str, Any]
    default_params: Dict[str, Any]
    search_space: Dict[str, Dict[str, Any]]
    early_stopping_rounds: Optional[int] = None


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry: how to build, fit and explain one model family."""

    name: str
    build: EstimatorBuilder
    uses_early_stopping: bool = False
    suppress_convergence_warnings: bool = False
    importance: Optional[ImportanceFn] = None


@dataclass
class FitOutput:
    """Result of fitting one estimator on one split."""

    proba: np.ndarray
    best_iteration: Optional[int]
    fit_seconds: float
    estimator: Optional[Any] = None
    train_proba: Optional[np.ndarray] = None


@dataclass
class OuterFoldSelection:
    """Candidate ranking inside one outer fold."""

    outer_fold: int
    ranking: List[int]
    mean_inner_auc: Dict[int, float]
    mean_best_iteration: Dict[int, Optional[float]]


@dataclass
class RankedRun:
    """OOF predictions when every outer fold uses its rank-r candidate."""

    rank: int
    oof_predictions: pd.DataFrame
    fold_roc_aucs: List[float]
    oof_roc_auc: float
    oof_brier: float
    oof_log_loss: float
    fold_metrics: List[Dict[str, Any]]
    selected: List[Dict[str, Any]]


@dataclass
class NestedCVResult:
    """Everything produced by one nested-CV run of one model."""

    model_name: str
    candidates: List[Dict[str, Any]]
    inner_scores: pd.DataFrame
    selections: List[OuterFoldSelection]
    runs: Dict[int, RankedRun]
    outer_splits: List[Split]
    feature_importance: Optional[pd.DataFrame]
    runtime_seconds: float


INNER_SCORE_COLUMNS: List[str] = [
    "outer_fold",
    "candidate_id",
    "inner_fold",
    "roc_auc",
    "best_iteration",
    "fit_seconds",
]
```

### 5.7 OOF contract (unchanged from legacy)

`oof_predictions.parquet` columns, in this order: `SK_ID_CURR` (int), `TARGET` (int8, 0/1), `y_pred_proba` (float64 in [0, 1]), `fold` (int, 0..n_outer−1). One row per training applicant, no NaN, same row order as `ids_train.parquet`. Use the existing constants `SK_ID_CURR`, `TARGET_COL` (`src.constants`) and `OOF_PROBA_COL`, `OOF_FOLD_COL` (`src.models.types`).

### 5.8 Run directory contract

```
artifacts/runs/<model>/baseline/
  run_config.json
  oof_predictions.parquet        # rank 1, schema 5.7
  metrics.json
  selected_params.json
  feature_importance.parquet     # only if the spec has an importance function
  run.log
  tuning/
    candidates.json
    fingerprint.json         # written by run_nested_cv when checkpointing (resume guard)
    inner_scores.csv
    rankings.json
  robustness/
    rank_2/ oof_predictions.parquet, metrics.json, selected_params.json
    ...
    rank_<top_k>/ ...
```

Example `metrics.json`:

```json
{
  "model": "lgbm",
  "rank": 1,
  "n_samples": 307507,
  "n_features": 493,
  "n_folds": 5,
  "oof_roc_auc": 0.7853,
  "oof_brier": 0.0669,
  "oof_log_loss": 0.2401,
  "base_rate": 0.0807,
  "brier_reference": 0.0742,
  "log_loss_reference": 0.2805,
  "fold_roc_aucs": [0.7821, 0.7902, 0.7836, 0.7881, 0.7826],
  "mean_fold_roc_auc": 0.7853,
  "std_fold_roc_auc": 0.0032,
  "folds": [
    {
      "outer_fold": 0,
      "candidate_id": 7,
      "mean_inner_roc_auc": 0.7841,
      "train_roc_auc": 0.8612,
      "val_roc_auc": 0.7821,
      "roc_auc_gap": 0.0791,
      "train_brier": 0.0612,
      "val_brier": 0.0671,
      "brier_gap": 0.0059,
      "train_log_loss": 0.2190,
      "val_log_loss": 0.2405,
      "log_loss_gap": 0.0215,
      "selection_optimism": 0.0020
    }
  ],
  "mean_roc_auc_gap": 0.0789,
  "mean_brier_gap": 0.0058,
  "mean_log_loss_gap": 0.0213,
  "mean_selection_optimism": 0.0015,
  "runtime_seconds": 8123.4
}
```

`folds` has one entry per outer fold (only one shown). Definitions are in 5.10.

(The numbers above are illustrative only.)

Example `selected_params.json` (one entry per outer fold; `params` = fully resolved parameters actually used at refit, i.e. `fixed_params` + candidate + any override):

```json
[
  {
    "outer_fold": 0,
    "candidate_id": 7,
    "mean_inner_roc_auc": 0.78412,
    "params": {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
               "n_estimators": 412, "num_leaves": 48, "min_child_samples": 133,
               "subsample": 0.84211, "colsample_bytree": 0.31876,
               "reg_alpha": 0.0213, "reg_lambda": 2.7712, "...": "..."}
  }
]
```

Example `tuning/candidates.json`:

```json
[
  {"candidate_id": 0, "params": {"num_leaves": 31, "min_child_samples": 20, "subsample": 0.8,
                                  "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0}},
  {"candidate_id": 1, "params": {"colsample_bytree": 0.54187, "min_child_samples": 61,
                                  "num_leaves": 102, "reg_alpha": 0.0071, "reg_lambda": 1.3021,
                                  "subsample": 0.92244}}
]
```

Example `tuning/inner_scores.csv` (`best_iteration` empty for models without early stopping):

```csv
outer_fold,candidate_id,inner_fold,roc_auc,best_iteration,fit_seconds
0,0,0,0.7795312877,233,21.4
0,0,1,0.7811270554,301,24.9
0,0,2,0.7779104332,268,23.0
0,1,0,0.7820018723,415,30.2
```

Example `tuning/rankings.json`:

```json
[{"outer_fold": 0, "ranking": [7, 1, 0, 12, 3, "..."],
  "mean_inner_auc": {"0": 0.77952, "1": 0.78101, "7": 0.78412, "...": 0.0},
  "mean_best_iteration": {"0": 267.3, "1": 401.0, "...": 0.0}}]
```

`run_config.json` keys: `model`, `variant` (`"baseline"`), `cv_settings` (dict), `model_config` (dict of the `ModelConfig`), `features_dir`, `n_samples`, `n_features`, `max_rows` (int or null), `git_commit` (output of `git rev-parse HEAD`, suffixed `-dirty` if tracked files have uncommitted changes, or `"unknown"`), `versions` (`python`, `numpy`, `pandas`, `sklearn`, `lightgbm`), `resumed` (bool: the run continued an existing checkpoint), `started_at`, `finished_at` (ISO-8601 UTC; of the last session only).

JSON rules: write with `indent=2`, `encoding="utf-8"`; convert numpy scalars to Python types; dict keys that are ints become strings.

### 5.9 What every run records, and why

| Recorded | File | Why |
| -------- | ---- | --- |
| OOF predicted probabilities (rank 1) | `oof_predictions.parquet` | The single input of every later metric (TPR at `t_m`, per-group AUC, FPR/FNR gaps, approval rates, calibration). Out-of-fold means each applicant is scored by a model that never saw them in training or tuning. |
| OOF probabilities of ranks 2–5 | `robustness/rank_<r>/` | Robustness check: how much the fairness gaps move between models of almost equal AUC (Hurlin et al. 2024 found fairness verdicts change with hyperparameters). |
| Pooled OOF ROC AUC | `metrics.json` | The utility metric of the decision "Utility metrics": threshold-free, unaffected by the 8% base rate, the tuning criterion, and comparable with the legacy 0.7841 and Kozodoi et al. |
| Pooled OOF **Brier score** and **log loss**, with the no-skill references | `metrics.json` | Quality of the probabilities themselves (calibration + sharpness), which ROC AUC ignores. Needed to read the group calibration curves of the fairness stage: a model that is miscalibrated for everyone (random forest probabilities are typically squeezed toward the middle) must not be mistaken for a group-specific calibration gap. The references (everyone gets the base rate) show whether a model beats no information at all. |
| **Fit diagnostics** per outer fold: training vs validation ROC AUC, Brier and log loss, their gaps, and the selection optimism (mean inner AUC of the chosen candidate − its outer validation AUC) | `metrics.json` → `folds` | Shows how much each model overfits its training fold and how optimistic the tuning step was. Descriptive only (see 5.10). |
| ROC AUC per outer fold, mean, std | `metrics.json` | Stability. A difference between two models smaller than the fold spread is not meaningful. |
| Inner ROC AUC per candidate × inner fold; LightGBM best iteration | `tuning/inner_scores.csv` | Evidence of the equal tuning budget; shows the gain over candidate 0 (defaults); shows whether a search range was too narrow. |
| Candidate ranking per outer fold | `tuning/rankings.json` | Audit of the selection rule; source of the robustness ranks. |
| Resolved parameters per outer fold | `selected_params.json` | The frozen hyperparameters for the mitigation runs; reproducibility. |
| Feature importance of the rank-1 models (LightGBM gain/split, RF impurity, logistic-regression coefficients; none for the MLP) | `feature_importance.parquet` | Descriptive: how strongly each model relies on the protected columns and on the proxies found in the Step 5 audit. |
| Runtime, package versions, git commit, configs | `metrics.json`, `run_config.json` | Reproducibility and compute cost. |

Deliberately **not** computed in `src/models/`:

- **Anything that needs the threshold `t_m` or the protected groups** (TPR at `t_m`, per-group AUC, approval rates, error-rate gaps, calibration). The threshold is a fairness-stage decision, and keeping `src/models/` blind to the groups guarantees tuning never sees fairness information. The fairness plan computes all of these from the OOF files.
- **Accuracy and F1.** At `t_m` (flags 8.07% = the base rate) FP = FN, so precision = recall = F1 = TPR, and accuracy = 1 − 2·FN/n is a monotone function of TPR: they add nothing beyond TPR at `t_m` (decision "Utility metrics"). Accuracy also rewards a model that flags nobody (91.9%).
- **Fitted fold models.** Large (random forest) and refittable exactly from `selected_params.json` and the seed.

### 5.10 Metric definitions and how to read the fit diagnostics

All computed by `src/models/metrics.py` from `y` (0/1) and predicted probabilities `p`; no groups, no threshold.

| Name | Definition | Better |
| ---- | ---------- | ------ |
| `roc_auc` | `sklearn.metrics.roc_auc_score(y, p)` | higher |
| `brier` | `sklearn.metrics.brier_score_loss(y, p)` = mean of `(p − y)²` | lower |
| `log_loss` | `sklearn.metrics.log_loss(y, p, labels=[0, 1])` | lower |
| `base_rate` | `mean(y)` of the evaluated rows | — |
| `brier_reference` | `base_rate · (1 − base_rate)` (Brier of predicting the base rate for everyone; 0.0742 on the full data) | a model must be below it |
| `log_loss_reference` | `−(b·ln b + (1 − b)·ln(1 − b))`, `b` = base rate (0.2805 on the full data) | a model must be below it |
| `train_*` | metric on the **outer training fold**, predicted by the model refitted on that same fold (in-sample) | — |
| `val_*` | metric on the **outer validation fold** (out-of-sample) | — |
| `roc_auc_gap` | `train_roc_auc − val_roc_auc` | closer to 0 |
| `brier_gap`, `log_loss_gap` | `val_* − train_*` (sign chosen so that positive = worse on unseen data, like `roc_auc_gap`) | closer to 0 |
| `selection_optimism` | `mean_inner_roc_auc − val_roc_auc` of the chosen candidate | closer to 0 |
| `mean_*` | mean of the per-fold values over the outer folds | — |

How to read them (for the docs and the thesis text, not for any automatic decision):

- **Overfitting that matters shows up out of sample.** Every model is selected on held-out (inner) AUC and evaluated on outer folds it never saw, so the OOF metrics already include any harm from overfitting. The train–validation gaps describe *how much* a model memorises; they are not a selection criterion and nothing is retuned because of them.
- **Gaps are comparable within a model family, not across families.** A random forest of deep trees fits its own bootstrap samples almost perfectly, so its training AUC is close to 1 by construction; boosted trees also have large gaps while generalising well. Logistic regression typically has a tiny gap. Compare gaps across candidates, folds and ranks of the same model.
- **Warning signs** worth reporting: validation Brier or log loss above the no-skill reference; `selection_optimism` clearly larger than the fold-to-fold spread of AUC (tuning picked a lucky candidate); a validation log loss that is much worse than the training one while AUC is similar (overconfident probabilities, relevant to the calibration diagnostics).
- `selection_optimism` mixes two effects: the winner's curse of picking the best of 20 (positive) and the smaller inner training sets (2/3 of the outer training fold, which lowers inner AUC, negative). Small values of either sign are normal.

---

## Phase 0 — Setup, formatter configuration and baseline snapshot

**Goal:** a clean starting point on `main`, working formatters, and a recorded reference.

Steps:

1. `git checkout main` and `git status --short`. The only expected entry is the untracked `.cursor/plans/` folder. Anything else → stop and ask.
2. Record BASE: `git rev-parse --short HEAD` → write it into the Progress log ("BASE commit").
3. `source .venv/bin/activate && pytest -q` → record the number of passing tests in the Notes of Phase 0.
4. Confirm `artifacts/features/` contains `X_train.parquet`, `y_train.parquet`, `ids_train.parquet`, `metadata.json`, and that `metadata.json` has `"n_features": 493` and `"n_train": 307507`.
5. Confirm `artifacts/models/metrics.json` has `oof_roc_auc` = `0.7841216640666608` (legacy reference). Do not delete or regenerate `artifacts/models/` (Phase 8 decides about it).
6. Formatters. black 26.10.1 and isort 9.0.2 are already installed in `.venv`; check with `black --version` and `isort --version` (if missing: `pip install "black>=26.10" "isort>=9.0"`). Create `pyproject.toml` at the repo root with exactly:
   ```toml
   [tool.black]
   line-length = 88

   [tool.isort]
   profile = "black"
   line_length = 88
   known_first_party = ["src", "analysis"]
   ```
   In `requirements.txt`, add these lines directly after `pytest>=7.0.0`:
   ```
   # Development tools (formatting; config in pyproject.toml)
   black>=26.10
   isort>=9.0
   ```
7. Commit A: `git add .cursor/plans/multi-model-nested-cv.md pyproject.toml requirements.txt && git commit -m "models: phase 0 - plan and formatter config"`.
8. Formatting baseline: `black src tests analysis/helpers && isort src tests analysis/helpers`. Expected: black reformats about 15 files; isort fixes `src/features/previous.py`, `src/models/training.py`, `src/data/loader.py`. This commit must be **formatting only**: `git diff --stat` lists only `.py` files under those three folders. Run `pytest -q` → same pass count as step 3. Commit B: `git commit -am "style: apply black and isort"`. Notebooks are not formatted.
9. `mkdir -p configs/models` (stays empty and untracked until Phase 1).

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] On `main`; BASE recorded; commits A and B exist (`git log --oneline -3`); `git status --short` is empty.
- [ ] `pytest -q` pass count identical before and after commit B.
- [ ] Feature artifacts verified (493 features, 307,507 rows).
- [ ] Legacy reference AUC recorded.

Phase 0 gets a short review (checklist in 7.2).

---

## Phase 1 — Types, config files, settings loader, candidate sampler

**Files:** ✎ `src/models/types.py`; ★ `configs/models/{cv,cv_smoke,lgbm,rf,logreg,mlp}.yaml`; ★ `src/models/search.py`; ★ `src/models/settings.py`; ★ `tests/test_models_settings.py`.

Note: `settings.py` validates the model name against the registry, but the registry is built in Phase 3. In Phase 1, implement the name check against a module-level constant in `settings.py`:

```python
KNOWN_MODELS: Tuple[str, ...] = ("lgbm", "rf", "logreg", "mlp")
EARLY_STOPPING_MODELS: Tuple[str, ...] = ("lgbm",)
```

In Phase 3 these two constants are **replaced** by lookups in `MODEL_REGISTRY` (`name in MODEL_REGISTRY`, `MODEL_REGISTRY[name].uses_early_stopping`).

### P1.1 `types.py`

Add exactly the content of section 5.6. Do not change existing classes.

### P1.2 Config files

Create the six YAML files with the exact content of sections 5.1, 5.2 and 5.5.

### P1.3 `src/models/search.py`

Public API:

```python
SUPPORTED_DISTRIBUTIONS: Tuple[str, ...] = (
    "uniform", "loguniform", "int_uniform", "int_loguniform", "choice",
)


def validate_distribution(name: str, spec: Dict[str, Any]) -> None:
    """Raise ValueError if a search-space entry is malformed (see plan 5.4)."""


def sample_candidates(
    default_params: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
    n_configs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return n_configs candidates: [deepcopy(default_params), draw_1, ..., draw_{n-1}].

    One rng = np.random.default_rng(seed) is created once and used for all draws.
    Within each draw, parameters are sampled in sorted(search_space) order.
    """
```

Implementation notes:

- `validate_distribution` messages must name the parameter, e.g. `ValueError("search_space['C']: loguniform needs 0 < low < high, got low=0.0, high=10.0")`.
- Unknown `dist` → `ValueError("search_space['x']: unknown dist 'gaussian'; expected one of (...)")`.
- `int_uniform` / `int_loguniform` require `low` and `high` to be `int` (not `bool`).
- `sample_candidates` with `n_configs < 1` → `ValueError`.
- Returned values must be plain Python types (`int`, `float`, `str`, `None`, `list`), so `json.dumps` works without a custom encoder.

Example (shape only; values are what your implementation produces):

```python
>>> cands = sample_candidates({"C": 1.0}, {"C": {"dist": "loguniform", "low": 1e-4, "high": 10.0}}, 3, 42)
>>> cands[0]
{'C': 1.0}
>>> len(cands), type(cands[1]["C"])
(3, <class 'float'>)
```

### P1.4 `src/models/settings.py`

Public API:

```python
def load_cv_settings(path: PathLike) -> CVSettings:
    """Load configs/models/cv.yaml. Validates: all five keys present and int;
    n_outer_folds >= 2; n_inner_folds >= 2; n_configs >= 1; 1 <= top_k <= n_configs."""


def load_model_config(path: PathLike) -> ModelConfig:
    """Load configs/models/<name>.yaml and validate it (plan 5.3/5.4)."""


def default_model_config_path(model_name: str) -> Path:
    """Return <repo>/configs/models/<model_name>.yaml (resolved from this file's location)."""


def default_cv_config_path() -> Path:
    """Return <repo>/configs/models/cv.yaml."""
```

Use `src.models.io.load_yaml_config` to read YAML. Repo root = `Path(__file__).resolve().parents[2]` (same pattern as `pipeline._default_config_path`). `PathLike = Union[str, Path]` (define locally, as `pipeline.py` does).

Validation in `load_model_config` (each failure → `ValueError` with a clear message that includes the file path):

1. Top-level keys ⊆ `{name, early_stopping_rounds, fixed_params, default_params, search_space}`; `name`, `fixed_params`, `default_params`, `search_space` required.
2. `name` in `KNOWN_MODELS` and equal to the file stem (`Path(path).stem`).
3. `set(default_params) == set(search_space)`.
4. `set(fixed_params) & set(search_space) == set()`.
5. No `random_state` or `seed` key in `fixed_params`, `default_params` or `search_space`.
6. Every `search_space` entry passes `validate_distribution`.
7. If `name` in `EARLY_STOPPING_MODELS`: `early_stopping_rounds` is an int ≥ 1. Otherwise it must be absent or `None`.

### P1.5 Tests — `tests/test_models_settings.py`

| Test | Asserts |
| ---- | ------- |
| `test_real_cv_configs_load` | `load_cv_settings` on `configs/models/cv.yaml` → `CVSettings(42, 5, 3, 20, 5)`; `cv_smoke.yaml` → `(42, 3, 2, 3, 2)` |
| `test_real_model_configs_load` | each of the 4 YAMLs loads; `lgbm.early_stopping_rounds == 50`; others `None` |
| `test_sample_candidates_first_is_default` | candidate 0 equals `default_params` and is a copy (mutating it does not change the input) |
| `test_sample_candidates_deterministic` | two calls with seed 42 are equal; seed 43 differs |
| `test_sample_candidates_key_order_irrelevant` | same space with reversed key insertion order → identical candidates |
| `test_sample_candidates_types_and_bounds` | for 200 draws on a space with all five dists: floats within `[low, high]`, ints are `int` within bounds, choice values ∈ `values`, loguniform > 0 |
| `test_sample_candidates_json_serializable` | `json.dumps(sample_candidates(...))` works for each real model config |
| `test_invalid_model_configs` (parametrized) | write bad YAML into `tmp_path` and expect `ValueError` for: unknown dist; `default_params` keys ≠ `search_space` keys; `random_state` in `fixed_params`; key in both fixed and search; `lgbm` without `early_stopping_rounds`; `rf` with `early_stopping_rounds: 10`; `name` ≠ file stem; loguniform with `low: 0` |
| `test_invalid_cv_settings` (parametrized) | `top_k > n_configs`; `n_inner_folds: 1`; missing key → `ValueError` |

For the bad-YAML tests, write a helper in the test file that dumps a dict with `yaml.safe_dump` to `tmp_path / f"{name}.yaml"`.

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `pytest -q tests/test_models_settings.py` passes.
- [ ] `pytest -q` (whole suite) passes; no previously passing test fails.
- [ ] `python -c "from src.models.settings import load_model_config, default_model_config_path as p; from src.models.search import sample_candidates as s; c=load_model_config(p('mlp')); print(len(s(c.default_params, c.search_space, 20, 42)))"` prints `20`.
- [ ] Before committing, `git status --short` lists only files named in this phase; no file from Ground rule 1 is modified.
- [ ] Commit `models: phase 1 - types, configs, settings, sampler`.

---

## Phase 2 — `SignedLogStandardScaler`

**Files:** ★ `src/models/preprocessing.py`; ★ `tests/test_models_preprocessing.py`.

Purpose: model-specific scaling for logistic regression and the MLP, fitted inside each training fold via a sklearn `Pipeline`. Binary 0/1 columns (one-hot dummies, missing indicators, flags) pass through unchanged; every other column gets `sign(x) * log1p(|x|)` and is then standardized with the training-fold mean and standard deviation. (On the real matrix, 252 of 493 columns are non-binary.)

Reference implementation (use it; adapt docstrings to Google style):

```python
"""Model-specific preprocessing fitted inside each CV training fold."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Return sign(x) * log1p(|x|), elementwise."""
    return np.sign(values) * np.log1p(np.abs(values))


class SignedLogStandardScaler(BaseEstimator, TransformerMixin):
    """Signed log transform + standardization for non-binary columns.

    Columns whose training values are all 0 or 1 are passed through unchanged.
    Constant non-binary columns get scale 1 (they become all zeros).

    Attributes:
        binary_mask_: Boolean mask of passthrough (0/1) columns.
        mean_: Means of the transformed non-binary columns.
        scale_: Standard deviations of the transformed non-binary columns (0 -> 1).
        n_features_in_: Number of input columns seen in fit.
    """

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64)
        self.binary_mask_ = np.all((values == 0.0) | (values == 1.0), axis=0)
        transformed = signed_log1p(values[:, ~self.binary_mask_])
        self.mean_ = transformed.mean(axis=0)
        std = transformed.std(axis=0)
        self.scale_ = np.where(std > 0.0, std, 1.0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "mean_")
        values = np.asarray(X, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} columns, got {values.shape[1]}"
            )
        out = values.copy()
        cols = ~self.binary_mask_
        out[:, cols] = (signed_log1p(values[:, cols]) - self.mean_) / self.scale_
        return out
```

Worked example (use in a test):

```python
X_train = np.array([[0.0, 10.0, 5.0], [1.0, 100.0, 5.0], [0.0, 1000.0, 5.0]])
# col 0: binary -> passthrough [0, 1, 0]
# col 1: log1p -> [2.3979, 4.6151, 6.9088] -> standardized, mean 0, std 1
# col 2: constant 5 -> log1p(5)=1.7918 for all rows, std 0 -> scale 1 -> all 0.0
```

### Tests — `tests/test_models_preprocessing.py`

| Test | Asserts |
| ---- | ------- |
| `test_binary_columns_pass_through` | col 0 of the worked example unchanged after `fit_transform` |
| `test_nonbinary_columns_standardized` | col 1 output has mean ≈ 0 and std ≈ 1 (`atol=1e-12`) and equals `(signed_log1p(x) - m) / s` computed by hand in the test |
| `test_constant_column_becomes_zero` | col 2 output all `0.0`, no NaN/inf |
| `test_negative_values_signed` | input `[-1000, -1, 0, 1, 1000]` (with a second row set making it non-binary) is monotonic after transform |
| `test_transform_uses_training_statistics` | fit on `X_train`, transform a different `X_val`; result equals manual computation with train mean/std (no refit leakage) |
| `test_wrong_column_count_raises` | `ValueError` on transform with a different number of columns |
| `test_works_inside_pipeline` | `Pipeline([("prep", SignedLogStandardScaler()), ("clf", LogisticRegression())]).fit(X, y)` on 200 synthetic rows; `predict_proba` shape `(n, 2)` |
| `test_sklearn_clone` | `sklearn.base.clone(SignedLogStandardScaler())` works (no `__init__` params) |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `pytest -q tests/test_models_preprocessing.py` passes; full `pytest -q` passes.
- [ ] Transformer never modifies its input array in place (add an assertion in one test: input copy equals input after `transform`).
- [ ] Commit `models: phase 2 - signed-log standard scaler`.

---

## Phase 3 — Model registry and `fit_predict` adapter

**Files:** ★ `src/models/registry.py`; ✎ `src/models/settings.py` (replace `KNOWN_MODELS` / `EARLY_STOPPING_MODELS` with registry lookups); ✎ `tests/conftest.py` (append fixtures); ★ `tests/test_models_registry.py`.

### P3.1 Builders

```python
def _build_lgbm(params: Dict[str, Any], seed: int) -> LGBMClassifier:
    return LGBMClassifier(**params, random_state=seed)


def _build_rf(params: Dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(**params, random_state=seed)


def _build_logreg(params: Dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", SignedLogStandardScaler()),
            ("clf", LogisticRegression(**params, random_state=seed)),
        ]
    )


def _build_mlp(params: Dict[str, Any], seed: int) -> Pipeline:
    mlp_params = dict(params)
    if "hidden_layer_sizes" in mlp_params:
        mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])
    return Pipeline(
        [
            ("prep", SignedLogStandardScaler()),
            ("clf", MLPClassifier(**mlp_params, random_state=seed)),
        ]
    )
```

The final step of every pipeline is named `"clf"`. Define `FINAL_STEP = "clf"` as a module constant and use it.

### P3.2 Importance functions

All return a long `pd.DataFrame` with columns `feature`, `importance_type`, `importance` (float), one row per feature per type, features in input order.

| Model | `importance_type` values | Source |
| ----- | ------------------------ | ------ |
| `lgbm` | `gain`, `split` | `estimator.booster_.feature_importance(importance_type=...)` |
| `rf` | `impurity` | `estimator.feature_importances_` |
| `logreg` | `coef` | `estimator.named_steps["clf"].coef_[0]` (signed, on the transformed scale) |
| `mlp` | — | `importance=None` |

### P3.3 Registry

```python
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "lgbm": ModelSpec("lgbm", _build_lgbm, uses_early_stopping=True,
                      importance=_lgbm_importance),
    "rf": ModelSpec("rf", _build_rf, importance=_rf_importance),
    "logreg": ModelSpec("logreg", _build_logreg, importance=_logreg_importance),
    "mlp": ModelSpec("mlp", _build_mlp, suppress_convergence_warnings=True),
}


def get_model_spec(name: str) -> ModelSpec:
    """Raise KeyError("Unknown model 'xgb'; available: ['lgbm', 'logreg', 'mlp', 'rf']")."""
```

### P3.4 `fit_predict` — the single place where any model is fitted

```python
def fit_predict(
    spec: ModelSpec,
    params: Dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    *,
    eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    early_stopping_rounds: Optional[int] = None,
    n_estimators_override: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
    keep_estimator: bool = False,
    return_train_proba: bool = False,
) -> FitOutput:
    """Fit one estimator and return P(y=1) for X_pred.

    Args:
        spec: Registry entry.
        params: Fully resolved parameters (fixed_params + candidate).
        seed: Random seed injected as random_state.
        X_train, y_train: Training rows.
        X_pred: Rows to score (validation fold).
        eval_set: (X_es, y_es) for early stopping; allowed only if
            spec.uses_early_stopping, and then early_stopping_rounds is required.
        early_stopping_rounds: Patience for early stopping.
        n_estimators_override: Replaces params["n_estimators"] (LightGBM refit).
        sample_weight: Optional per-row training weights (seam for later
            pre-processing mitigation; unused by this plan's runs).
        keep_estimator: Return the fitted estimator in FitOutput.estimator.
        return_train_proba: Also predict X_train (in-sample) into
            FitOutput.train_proba, for the fit diagnostics.

    Returns:
        FitOutput(proba float64 1-D, best_iteration or None, fit_seconds, estimator or None).

    Raises:
        ValueError: eval_set given for a model without early stopping, or
            eval_set given without early_stopping_rounds.
    """
```

Required behaviour:

1. `resolved = dict(params)`; if `n_estimators_override is not None`: `resolved["n_estimators"] = int(n_estimators_override)`.
2. `estimator = spec.build(resolved, seed)`.
3. Fit kwargs: if `sample_weight is not None`: for a `Pipeline` → `{f"{FINAL_STEP}__sample_weight": sample_weight}`, else `{"sample_weight": sample_weight}`. If `None`, pass **nothing** (this keeps LightGBM bit-identical to the legacy call).
4. If `eval_set is not None` (early-stopping path, LightGBM only):
   ```python
   estimator.fit(
       X_train, y_train,
       eval_set=[eval_set],
       eval_metric="auc",
       callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
       **fit_kwargs,
   )
   best_iteration = int(estimator.best_iteration_)
   proba = estimator.predict_proba(X_pred, num_iteration=best_iteration)[:, 1]
   ```
5. Else: `estimator.fit(X_train, y_train, **fit_kwargs)`; `best_iteration = None`; `proba = estimator.predict_proba(X_pred)[:, 1]`.
6. If `spec.suppress_convergence_warnings`: wrap the fit in `warnings.catch_warnings()` + `warnings.simplefilter("ignore", category=ConvergenceWarning)` (from `sklearn.exceptions`). Do not suppress anything else, and nothing for other models.
7. If `return_train_proba`: `train_proba = predict_proba(X_train)[:, 1]` as float64, using `num_iteration=best_iteration` on the early-stopping path; otherwise `None`. This prediction is not part of `fit_seconds`.
8. Time the fit with `time.perf_counter()`.
9. Return `proba.astype(np.float64)`.

### P3.5 Update `settings.py`

Replace `KNOWN_MODELS` / `EARLY_STOPPING_MODELS` with `name in MODEL_REGISTRY` and `MODEL_REGISTRY[name].uses_early_stopping`. Phase 1 tests must still pass unchanged.

### P3.6 Test fixtures — append to `tests/conftest.py`

```python
@pytest.fixture
def make_training_data():
    """Factory for small synthetic TrainingData with a learnable signal."""
    from src.constants import SK_ID_CURR, TARGET_COL
    from src.models.types import TrainingData

    def _make(n_samples: int = 400, n_features: int = 8, seed: int = 0) -> TrainingData:
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            rng.normal(size=(n_samples, n_features)),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        X["cat__CODE_GENDER_F"] = (rng.random(n_samples) < 0.66).astype(float)
        X["cat__CODE_GENDER_M"] = 1.0 - X["cat__CODE_GENDER_F"]
        X["num__DAYS_BIRTH"] = -rng.integers(7500, 25000, size=n_samples).astype(float)
        logits = -1.5 + 1.5 * X["feat_0"] - 1.0 * X["feat_1"]
        y = (rng.random(n_samples) < 1.0 / (1.0 + np.exp(-logits))).astype(np.int8)
        ids = pd.Series(np.arange(1, n_samples + 1), name=SK_ID_CURR)
        return TrainingData(
            X=X, y=pd.Series(y, name=TARGET_COL), ids=ids, feature_names=list(X.columns)
        )

    return _make


@pytest.fixture
def tiny_model_configs():
    """Fast ModelConfig per registry model (for unit tests only)."""
    from src.models.types import ModelConfig

    return {
        "lgbm": ModelConfig(
            name="lgbm",
            fixed_params={"objective": "binary", "n_estimators": 40, "learning_rate": 0.1,
                          "verbosity": -1, "n_jobs": 1, "subsample_freq": 1},
            default_params={"num_leaves": 7, "subsample": 0.8},
            search_space={"num_leaves": {"dist": "int_uniform", "low": 4, "high": 15},
                          "subsample": {"dist": "uniform", "low": 0.6, "high": 1.0}},
            early_stopping_rounds=5,
        ),
        "rf": ModelConfig(
            name="rf",
            fixed_params={"n_estimators": 15, "n_jobs": 1},
            default_params={"min_samples_leaf": 5},
            search_space={"min_samples_leaf": {"dist": "int_loguniform", "low": 2, "high": 30}},
        ),
        "logreg": ModelConfig(
            name="logreg",
            fixed_params={"max_iter": 500},
            default_params={"C": 1.0},
            search_space={"C": {"dist": "loguniform", "low": 0.01, "high": 10.0}},
        ),
        "mlp": ModelConfig(
            name="mlp",
            fixed_params={"early_stopping": False},
            default_params={"hidden_layer_sizes": [8], "alpha": 0.1, "max_iter": 5},
            search_space={
                "hidden_layer_sizes": {"dist": "choice", "values": [[4], [8]]},
                "alpha": {"dist": "loguniform", "low": 0.001, "high": 1.0},
                "max_iter": {"dist": "int_uniform", "low": 3, "high": 8},
            },
        ),
    }
```

(`numpy as np`, `pandas as pd`, `pytest` are already imported in `conftest.py`.)

### P3.7 Tests — `tests/test_models_registry.py`

| Test | Asserts |
| ---- | ------- |
| `test_registry_names` | `set(MODEL_REGISTRY) == {"lgbm", "rf", "logreg", "mlp"}`; `get_model_spec("xgb")` raises `KeyError` mentioning available names |
| `test_fit_predict_all_models` (parametrized over 4 names) | with `tiny_model_configs[name]` params (`{**fixed, **default}`) on 300 train / 100 pred rows: `proba.shape == (100,)`, dtype float64, all in `[0, 1]`, finite; `fit_seconds >= 0` |
| `test_lgbm_early_stopping` | with `eval_set` + `early_stopping_rounds=5`: `best_iteration` is an int, `1 <= best_iteration <= 40` |
| `test_lgbm_n_estimators_override` | `n_estimators_override=7, keep_estimator=True` → `estimator.n_estimators == 7`, `best_iteration is None` |
| `test_eval_set_rejected_for_non_es_model` | `rf` with `eval_set` → `ValueError` |
| `test_eval_set_requires_rounds` | `lgbm` with `eval_set` but no rounds → `ValueError` |
| `test_sample_weight_seam` (parametrized over 4 names) | fitting with `sample_weight=np.ones(n)` works; for `logreg`, weights `w` with `w[y==1] = 5.0` change the predictions (`not np.allclose`) |
| `test_mlp_hidden_layers_tuple` | `keep_estimator=True` → `estimator.named_steps["clf"].hidden_layer_sizes == (8,)` |
| `test_mlp_convergence_warning_suppressed` | `with warnings.catch_warnings(record=True) as w: warnings.simplefilter("always"); fit_predict(mlp, max_iter=1 ...)` → no `ConvergenceWarning` in `w` |
| `test_importance_frames` (parametrized lgbm/rf/logreg) | columns `["feature", "importance_type", "importance"]`; each type has exactly `len(feature_names)` rows; `mlp` spec has `importance is None` |
| `test_fit_predict_deterministic` (parametrized over 4) | two calls with the same seed → `np.array_equal` probas |
| `test_return_train_proba` (parametrized over 4) | `return_train_proba=True` → `train_proba.shape == (len(X_train),)`, float64, in [0, 1]; default → `train_proba is None`; `proba` identical with and without the flag |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `pytest -q tests/test_models_registry.py tests/test_models_settings.py` pass; full suite passes.
- [ ] `grep -n "KNOWN_MODELS\|EARLY_STOPPING_MODELS" src/models/settings.py` returns nothing.
- [ ] `fit_predict` is the **only** function in `src/models/` (outside legacy `training.py`) that calls `.fit(` on a model: `grep -n "\.fit(" src/models/*.py` shows hits only in `registry.py`, `preprocessing.py` (none expected — it defines `fit`), and `training.py`.
- [ ] Commit `models: phase 3 - model registry and fit_predict`.

---

## Phase 4 — Generic CV core and legacy equivalence check

**Files:** ★ `src/models/cv.py`; ★ `src/models/legacy_check.py`; ★ `tests/test_models_cv.py`.

### P4.1 `cv.py`

```python
def make_stratified_splits(y: np.ndarray, n_splits: int, seed: int) -> List[Split]:
    """StratifiedKFold(n_splits, shuffle=True, random_state=seed) on y.

    Uses np.zeros(len(y)) as the X placeholder; splits depend only on y and the seed,
    so they equal the legacy folds when given the full y and seed 42.
    Returns a list of (train_idx, val_idx) int64 arrays.
    """


def cross_val_predict_model(
    spec: ModelSpec,
    params: Dict[str, Any],
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Split],
    *,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[np.ndarray, List[FitOutput]]:
    """Fit on each split's train part, predict its validation part.

    If spec.uses_early_stopping, the split's validation part is passed as eval_set
    (early stopping on the scored fold). This is used (a) inside the inner loop of
    nested CV, where that fold is an inner fold, and (b) by legacy_check to mirror
    train_lgbm_cv. Never call it with outer folds for real results.

    Returns:
        (oof, fits): oof has len(y) and NaN only for rows in no validation part;
        fits[i] belongs to splits[i] (estimator not kept).
    Raises:
        ValueError: if validation parts overlap.
    """
```

No LightGBM-specific code here: decide the `eval_set` only via `spec.uses_early_stopping`.

### P4.2 `legacy_check.py` (CLI)

```
python -m src.models.legacy_check [--features-dir artifacts/features] [--config configs/model.yaml]
```

Steps:

1. `cfg = load_yaml_config(config)`; `data = load_training_data(features_dir)`.
2. Legacy: `legacy = train_lgbm_cv(data, cfg, return_models=False)`.
3. New: `X = data.X.to_numpy(dtype=np.float64)`, `y = data.y.to_numpy()`, `splits = make_stratified_splits(y, cfg["n_folds"], cfg["seed"])`, `oof, fits = cross_val_predict_model(get_model_spec("lgbm"), dict(cfg["lgbm"]), cfg["seed"], X, y, splits, early_stopping_rounds=cfg["early_stopping_rounds"])`.
4. Compare: `max_abs_diff = np.max(np.abs(oof - legacy.oof_predictions[OOF_PROBA_COL].to_numpy()))`; also fold assignment equality and best iterations (`[f.best_iteration for f in fits]` vs `[fr.best_iteration for fr in legacy.fold_results]`).
5. Print, e.g.:
   ```
   legacy OOF AUC = 0.784122 | new OOF AUC = 0.784122
   best iterations legacy = [240, 411, 297, 349, 527] | new = [240, 411, 297, 349, 527]
   folds identical = True | max |Δp| = 0.000e+00
   EQUIVALENT
   ```
6. Exit code 0 if `max_abs_diff <= 1e-9` and folds identical and best iterations identical; else print `NOT EQUIVALENT` and `sys.exit(1)`.

(The reviewer verified on a 30,000-row subsample that LightGBM on a float64 NumPy array gives bit-identical OOF predictions to the legacy DataFrame call. Expect `max |Δp| = 0`.)

### P4.3 Tests — `tests/test_models_cv.py`

| Test | Asserts |
| ---- | ------- |
| `test_make_stratified_splits_matches_sklearn` | equals `StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y)` exactly; validation parts partition `range(n)` |
| `test_cross_val_predict_covers_all_rows` (parametrized over 4 models, tiny configs) | no NaN in `oof`; `len(fits) == n_splits` |
| `test_cross_val_predict_partial_coverage` | with only the first 2 of 3 splits, rows of split 3 are NaN, others finite |
| `test_overlapping_splits_raise` | `ValueError` |
| `test_matches_legacy_train_lgbm_cv` | on `make_training_data(n_samples=400)` with the existing legacy-style config (`seed 42, n_folds 3, early_stopping_rounds 5, lgbm: {objective: binary, metric: auc, n_estimators: 50, num_leaves: 8, learning_rate: 0.1, verbosity: -1, n_jobs: 1}`): `np.array_equal` of OOF probabilities with `train_lgbm_cv(...)`, and identical best iterations |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `pytest -q tests/test_models_cv.py` passes; full suite passes.
- [ ] **Full-data equivalence:** `python -m src.models.legacy_check` exits 0 and prints `EQUIVALENT` with legacy OOF AUC ≈ 0.784122 (both runs ~2 min each). Paste the printed output in the Progress log Notes.
- [ ] `grep -n "lgbm\|LGBM\|lightgbm" src/models/cv.py` returns nothing.
- [ ] Commit `models: phase 4 - generic CV core and legacy check`.

---

## Phase 5 — Nested CV with checkpoint/resume and top-k refits

**Files:** ★ `src/models/metrics.py`; ★ `tests/test_models_metrics.py`; ★ `src/models/nested.py`; ★ `tests/test_models_nested.py`.

### P5.0 `src/models/metrics.py` (write and test this first)

```python
def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Return {"roc_auc", "brier", "log_loss"} (definitions in plan 5.10)."""


def reference_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """Return {"base_rate", "brier_reference", "log_loss_reference"} for predicting the base rate."""


def fold_fit_diagnostics(
    y_train: np.ndarray,
    train_proba: np.ndarray,
    y_val: np.ndarray,
    val_proba: np.ndarray,
    mean_inner_roc_auc: float,
) -> Dict[str, float]:
    """Return train_/val_ metrics, the three gaps and selection_optimism (plan 5.10)."""


def summarize_folds(folds: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return mean_roc_auc_gap, mean_brier_gap, mean_log_loss_gap, mean_selection_optimism."""
```

Rules: inputs are 1-D arrays of equal length; raise `ValueError` if `y_true` has a single class (AUC undefined). Return plain Python floats.

Tests — `tests/test_models_metrics.py`:

| Test | Asserts |
| ---- | ------- |
| `test_classification_metrics_known_values` | `y = [0, 0, 1, 1]`, `p = [0.1, 0.4, 0.35, 0.8]` → `roc_auc == 0.75`, `brier == mean((p − y)²) == 0.158125`, `log_loss` equals the hand formula `−mean(y·ln p + (1−y)·ln(1−p))` to 1e-12 |
| `test_reference_metrics` | `y` with 2 positives in 25 → `base_rate 0.08`, `brier_reference 0.0736`, `log_loss_reference` equals the formula to 1e-12; and `classification_metrics(y, np.full(25, 0.08))` gives Brier and log loss equal to the references |
| `test_gap_signs` | train perfect (`p = y`), val noisy → `roc_auc_gap > 0`, `brier_gap > 0`, `log_loss_gap > 0`; `selection_optimism == mean_inner − val_roc_auc` |
| `test_summarize_folds` | means of two hand-made fold dicts |
| `test_single_class_raises` | `ValueError` |


### P5.1 Public API

```python
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
            checkpoint_dir / "inner_scores.csv"; the candidates and a fingerprint of the
            run (CV settings, fixed params, data hash) are written to
            checkpoint_dir / "candidates.json" and "fingerprint.json".
        resume: If True and the checkpoint exists, skip finished
            (outer_fold, candidate_id) pairs.

    Returns:
        NestedCVResult.
    """


def inner_splits_absolute(outer_train_idx: np.ndarray, y: np.ndarray, n_inner: int, seed: int) -> List[Split]:
    """Inner splits of one outer training fold, as absolute row indices of the full data."""
```

### P5.2 Algorithm (implement exactly)

```python
spec = get_model_spec(model_config.name)
X = data.X.to_numpy(dtype=np.float64)
y = data.y.to_numpy().astype(np.int8)
ids = data.ids.to_numpy()
candidates = sample_candidates(model_config.default_params, model_config.search_space,
                               cv_settings.n_configs, cv_settings.seed)
outer_splits = make_stratified_splits(y, cv_settings.n_outer_folds, cv_settings.seed)

# --- checkpoint setup ---
# candidates.json: if it exists and resume=True, it must equal [{"candidate_id": i, "params": c}]
#   (compare after json round-trip); otherwise raise ValueError("Search candidates changed ...").
# If it exists and resume=False: raise FileExistsError (the CLI handles --overwrite).
#   Check candidates.json, inner_scores.csv and fingerprint.json BEFORE writing anything.
# fingerprint.json: {model, seed, n_outer_folds, n_inner_folds, fixed_params,
#   early_stopping_rounds, n_samples, n_features, data_sha256}; on resume it must equal this
#   run's fingerprint, else ValueError("Checkpoint does not match this run (changed: ...)").
#   top_k is not part of it (it only affects the refits).
# inner_scores.csv: if resume, read it; keep only (outer_fold, candidate_id) groups with exactly
#   n_inner_folds rows (inner_fold values exactly 0..n_inner-1, no missing roc_auc/fit_seconds,
#   so a torn last line is dropped too); rewrite the file with those rows.
# resume=True with no checkpoint files yet is NOT an error: start fresh (the Phase 7 chain
#   always passes --resume, also for models that have not started).

for k, (outer_tr, outer_va) in enumerate(outer_splits):
    X_tr, y_tr = X[outer_tr], y[outer_tr]
    inner_rel = make_stratified_splits(y_tr, cv_settings.n_inner_folds, cv_settings.seed)
    for cid, cand in enumerate(candidates):
        if (k, cid) already done: continue
        params = {**model_config.fixed_params, **cand}
        oof_inner, fits = cross_val_predict_model(
            spec, params, cv_settings.seed, X_tr, y_tr, inner_rel,
            early_stopping_rounds=model_config.early_stopping_rounds)
        rows = [
            {"outer_fold": k, "candidate_id": cid, "inner_fold": j,
             "roc_auc": roc_auc_score(y_tr[va], oof_inner[va]),
             "best_iteration": fits[j].best_iteration,   # None -> empty cell / NaN
             "fit_seconds": fits[j].fit_seconds}
            for j, (_, va) in enumerate(inner_rel)
        ]
        append rows (checkpoint: append to CSV and flush; header only when creating)
        log one line (see P5.3)
    del X_tr, y_tr

inner_scores = full table (re-read from CSV when checkpointing, so resumed and
               uninterrupted runs select from identical values)

selections = []
for k: g = inner_scores[outer_fold == k].groupby("candidate_id")
       mean_auc = g["roc_auc"].mean(); mean_it = g["best_iteration"].mean()  (NaN -> None)
       ranking = sorted(candidate ids, key=lambda c: (-mean_auc[c], c))
       selections.append(OuterFoldSelection(k, ranking, mean_auc dict, mean_it dict))

runs = {}
for r in 1..top_k:
    oof = np.full(len(y), np.nan); fold_col = np.full(len(y), -1)
    selected = []; fold_aucs = []; fold_metrics = []
    for k, (outer_tr, outer_va):
        cid = selections[k].ranking[r - 1]
        params = {**fixed_params, **candidates[cid]}
        override = None
        if spec.uses_early_stopping:
            override = max(1, int(round(selections[k].mean_best_iteration[cid])))
        out = fit_predict(spec, params, seed, X[outer_tr], y[outer_tr], X[outer_va],
                          n_estimators_override=override,
                          keep_estimator=(r == 1 and spec.importance is not None),
                          return_train_proba=True)
        oof[outer_va] = out.proba; fold_col[outer_va] = k
        fold_aucs.append(roc_auc_score(y[outer_va], out.proba))
        diag = fold_fit_diagnostics(y[outer_tr], out.train_proba, y[outer_va], out.proba,
                                    selections[k].mean_inner_auc[cid])
        fold_metrics.append({"outer_fold": k, "candidate_id": cid,
                             "mean_inner_roc_auc": selections[k].mean_inner_auc[cid], **diag})
        resolved = dict(params); if override: resolved["n_estimators"] = override
        selected.append({"outer_fold": k, "candidate_id": cid,
                         "mean_inner_roc_auc": selections[k].mean_inner_auc[cid],
                         "params": resolved})
        if r == 1 and out.estimator is not None:
            imp = spec.importance(out.estimator, data.feature_names); imp.insert(1, "fold", k)
            collect imp
    assert not np.isnan(oof).any()
    oof_df = DataFrame({SK_ID_CURR: ids, TARGET_COL: y, OOF_PROBA_COL: oof, OOF_FOLD_COL: fold_col})
    pooled = classification_metrics(y, oof)
    runs[r] = RankedRun(r, oof_df, fold_aucs, pooled["roc_auc"], pooled["brier"],
                        pooled["log_loss"], fold_metrics, selected)

feature_importance = concat(collected) or None
return NestedCVResult(model_config.name, candidates, inner_scores, selections, runs,
                      outer_splits, feature_importance, runtime_seconds)
```

Additional rules:

- Assert at the start of each outer fold: `np.intersect1d(outer_tr, outer_va).size == 0`.
- The outer validation rows `X[outer_va]` are used **only** in the refit `fit_predict` call as `X_pred`. Nowhere else. (`return_train_proba=True` predicts `X[outer_tr]`, the rows the refit was trained on.)
- Use `logging.getLogger(__name__)`; no `print` in `nested.py`.
- Reading the CSV back: `pd.read_csv(path)`; `best_iteration` becomes float with NaN for non-ES models. Write it from `None` as an empty cell (`csv` / pandas does this when the value is `NaN`).
- Write floats with full precision (pandas default `to_csv` is fine).

### P5.3 Example log lines (logger INFO)

```
[lgbm] candidates: 20 | outer folds: 5 | inner folds: 3 | top_k: 5 | rows: 307507
[lgbm] outer 1/5 | cand 01/20 | inner AUC 0.7795 ± 0.0013 | best_it 267 | 69.3 s
[lgbm] outer 1/5 | cand 02/20 | inner AUC 0.7810 ± 0.0011 | best_it 401 | 92.0 s
[lgbm] outer 1/5 | skipped 7 finished candidates (resume)
[lgbm] outer 1/5 | best cand 07 (inner AUC 0.7841)
[lgbm] refit rank 1 | outer 1/5 | AUC 0.7823 (train 0.8612) | Brier 0.0671 | n_estimators 412 | 41.2 s
[lgbm] rank 1 OOF AUC 0.7853 | Brier 0.0669 (ref 0.0742) | log loss 0.2401 (ref 0.2805) | fold AUCs [0.7823, 0.7902, ...]
```

### P5.4 Tests — `tests/test_models_nested.py`

Use `make_training_data(n_samples=300)`, `tiny_model_configs`, and `CVSettings(seed=42, n_outer_folds=3, n_inner_folds=2, n_configs=3, top_k=2)`.

| Test | Asserts |
| ---- | ------- |
| `test_nested_oof_contract` (parametrized over 4 models) | `runs[1].oof_predictions` columns `[SK_ID_CURR, TARGET, y_pred_proba, fold]`; 300 rows; no NaN; probas in [0, 1]; `fold` values `{0,1,2}` and equal to the fold of `make_stratified_splits(y, 3, 42)`; `SK_ID_CURR` equals `data.ids` in order |
| `test_nested_inner_scores_shape` | `len(inner_scores) == 3 * 3 * 2`, columns == `INNER_SCORE_COLUMNS`; `best_iteration` all NaN for `rf`, all ≥ 1 for `lgbm` |
| `test_ranking_ties_broken_by_smaller_candidate_id`, `test_ranking_matches_inner_scores` | with `roc_auc_score` patched to 0.5 every ranking is `[0, 1, 2]`; without the patch `mean_inner_auc` equals the per-candidate mean of `inner_scores` and `ranking` is sorted by `(-auc, id)` |
| `test_top_k_runs` | `set(runs) == {1, 2}`; rank 2 uses `ranking[1]` in each fold (`selected[k]["candidate_id"]`) |
| `test_lgbm_refit_uses_mean_best_iteration` | `selected[k]["params"]["n_estimators"] == round(mean_best_iteration[cid])` |
| `test_inner_splits_never_touch_outer_validation` | for each outer split: every index from `inner_splits_absolute(...)` ∈ `outer_tr`, ∩ `outer_va` = ∅, inner validation parts partition `outer_tr` |
| `test_outer_validation_used_only_for_prediction` | with `lgbm`, spy on `src.models.nested.fit_predict` and `cross_val_predict_model`: every inner-CV call receives exactly `X[outer_tr]` and its validation parts partition it; every refit call trains on exactly `X[outer_tr]`, predicts exactly `X[outer_va]` and gets no `eval_set` / `early_stopping_rounds` |
| `test_nested_deterministic` | two runs → identical `oof_predictions` and identical `inner_scores` except `fit_seconds` |
| `test_resume_gives_identical_result` | run with `checkpoint_dir=tmp_path`; truncate `inner_scores.csv` to header + first 7 rows (leaves a half-finished pair); rerun with `resume=True` → identical rank-1 OOF and identical `roc_auc` column; and the uninterrupted run made exactly 9 and the resumed run exactly 6 `cross_val_predict_model` calls (spy) |
| `test_resume_with_changed_candidates_raises` | after a checkpointed run, rerun with a different `default_params` and `resume=True` → `ValueError` |
| `test_existing_checkpoint_without_resume_raises` | `FileExistsError` |
| `test_resume_without_checkpoint_starts_fresh` | `resume=True` on an empty `checkpoint_dir` runs normally and equals a run with `resume=False` |
| `test_resume_drops_torn_last_row` (cut at 4 / 12 / 40 characters) | a torn last CSV line is dropped; resumed rankings, `roc_auc` column and OOF equal the uninterrupted run |
| `test_resume_with_orphan_scores_raises`, `test_orphan_scores_without_resume_leave_no_files` | `inner_scores.csv` without `candidates.json`: `ValueError` on resume, `FileExistsError` otherwise; no checkpoint file is written |
| `test_resume_with_changed_run_raises` | changed `n_outer_folds`, data or `fixed_params` → `ValueError`; changed `top_k` is allowed |
| `test_inner_scores_dtypes_match_with_and_without_checkpoint` | identical columns and dtypes (`best_iteration` float64) in both paths |
| `test_early_stopping_model_without_best_iteration_raises` | `ValueError` instead of a silent default `n_estimators` |
| `test_candidates_json_is_list_of_id_and_params` | stored candidates equal `result.candidates` |
| `test_fold_metrics` (parametrized over 4 models) | each `runs[r].fold_metrics` has one dict per outer fold with all keys of 5.10; `val_roc_auc == fold_roc_aucs[k]`; `roc_auc_gap == train_roc_auc − val_roc_auc`; `oof_brier`, `oof_log_loss` equal `classification_metrics` on the OOF frame |
| `test_feature_importance_rank1_only` | `lgbm`: `feature_importance` has `fold` values `{0,1,2}`, types `{gain, split}`; `mlp`: `None` |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `pytest -q tests/test_models_metrics.py tests/test_models_nested.py` passes in < 60 s; full suite passes.
- [ ] `grep -n "outer_va" src/models/nested.py` shows `X[outer_va]` only in the refit `fit_predict(...)` call (plus `y[outer_va]` for the fold AUC, fold assignment, OOF assignment and the disjointness assertion).
- [ ] No `print(` in `nested.py`.
- [ ] Commit `models: phase 5 - nested CV with checkpoint and top-k refits`.

---

## Phase 6 — Run persistence, CLI, summary, smoke runs

**Files:** ★ `src/models/runs.py`; ★ `src/models/run.py`; ★ `src/models/summary.py`; ★ `tests/test_models_runs.py`.

### P6.1 `runs.py`

```python
DEFAULT_VARIANT = "baseline"
OOF_FILE = "oof_predictions.parquet"
METRICS_FILE = "metrics.json"
SELECTED_PARAMS_FILE = "selected_params.json"
RUN_CONFIG_FILE = "run_config.json"
IMPORTANCE_FILE = "feature_importance.parquet"
TUNING_DIR = "tuning"
ROBUSTNESS_DIR = "robustness"


def run_dir_for(runs_dir: PathLike, model_name: str, variant: str = DEFAULT_VARIANT) -> Path:
    """Return runs_dir / model_name / variant."""


def save_nested_run(result: NestedCVResult, run_dir: PathLike, run_config: Dict[str, Any]) -> Path:
    """Write the run directory (plan 5.8). Rank 1 at the top level; ranks >= 2 under
    robustness/rank_<r>/. Writes tuning/rankings.json from result.selections.
    tuning/candidates.json and tuning/inner_scores.csv are (re)written from the result
    so a run without checkpointing is still complete (tuning/fingerprint.json is only
    written by run_nested_cv when it checkpoints)."""


def load_oof(run_dir: PathLike, rank: int = 1) -> pd.DataFrame:
    """Load OOF predictions of rank 1 (top level) or rank r (robustness/rank_r)."""


def load_metrics(run_dir: PathLike, rank: int = 1) -> Dict[str, Any]: ...


def load_selected_params(run_dir: PathLike, rank: int = 1) -> Dict[int, Dict[str, Any]]:
    """Return {outer_fold: resolved params} — the frozen hyperparameters for later mitigation runs."""
```

`metrics.json` content per rank: exactly the keys of the example in 5.8, in that order: `model`, `rank`, `n_samples`, `n_features`, `n_folds`, `oof_roc_auc`, `oof_brier`, `oof_log_loss`, `base_rate`, `brier_reference`, `log_loss_reference` (from `reference_metrics` on the OOF `TARGET`), `fold_roc_aucs`, `mean_fold_roc_auc`, `std_fold_roc_auc` (`np.std`, ddof 0, same as legacy), `folds` (= `RankedRun.fold_metrics`), the four `mean_*` values from `summarize_folds`, `runtime_seconds` (whole run, same value for every rank; wall time of the last session plus the `fit_seconds` of tuning pairs finished by earlier sessions, so a resumed run does not under-report compute).

### P6.2 `run.py` (CLI)

```
python -m src.models.run --model {lgbm,rf,logreg,mlp}
    [--features-dir artifacts/features]
    [--runs-dir artifacts/runs]
    [--cv-config configs/models/cv.yaml]
    [--model-config configs/models/<model>.yaml]
    [--max-rows N]
    [--resume | --overwrite]
```

`def main(argv: Optional[List[str]] = None) -> None` (so tests can call `main([...])`). Behaviour:

1. Resolve config paths (defaults from `settings.default_cv_config_path()` / `default_model_config_path(model)`), load and validate them. `--model` must equal `model_config.name`.
2. `data = load_training_data(features_dir)`.
3. `--max-rows N` (must satisfy `2 <= N < n_rows`): stratified subsample with `train_test_split(np.arange(n), train_size=N, stratify=y, random_state=cv_settings.seed)`; sort the selected indices; rebuild `TrainingData` with `reset_index(drop=True)`. **Guard:** if `--max-rows` is given and `Path(runs_dir).resolve() == (repo_root / "artifacts/runs").resolve()`, exit with error `"--max-rows requires a --runs-dir other than artifacts/runs"`.
4. `run_dir = run_dir_for(runs_dir, model)`; the existing-run check happens **before** the data is loaded. If `run_dir / METRICS_FILE` exists or `run_dir / "tuning"` exists:
   - `--overwrite` → `shutil.rmtree(run_dir)` (after the data loaded successfully);
   - `--resume` → continue (and if the run directory does not exist yet, `--resume` simply starts a fresh run);
   - neither → exit with error `"Run directory exists: <path>. Use --resume or --overwrite."`.
5. Logging: a stderr `StreamHandler` and a `FileHandler(run_dir / "run.log", mode="a", encoding="utf-8")` with format `%(asctime)s %(levelname)s %(message)s`, attached to the root logger (level INFO) for the duration of the run and removed/closed afterwards (also on error).
6. `result = run_nested_cv(data, model_config, cv_settings, checkpoint_dir=run_dir / "tuning", resume=args.resume)`.
7. Build `run_config` (5.8) and `save_nested_run(...)`.
8. Print one final line: `Wrote artifacts/runs/lgbm/baseline | rank-1 OOF AUC=0.7853 | fold AUCs=[...] | 2h 15m`.

### P6.3 `summary.py` (CLI)

```
python -m src.models.summary [--runs-dir artifacts/runs] [--variant baseline]
```

For every `<runs_dir>/<model>/<variant>/metrics.json` and its `robustness/rank_*/metrics.json`, collect rows `model, rank, oof_roc_auc, std_fold_roc_auc, oof_brier, brier_reference, oof_log_loss, log_loss_reference, mean_roc_auc_gap, mean_selection_optimism, runtime_hours`. Sort by `model`, `rank`. Write `<runs_dir>/summary_<variant>.csv` and print a Markdown table:

```
| model | rank | oof_roc_auc | std_fold_roc_auc | oof_brier | brier_reference | oof_log_loss | log_loss_reference | mean_roc_auc_gap | mean_selection_optimism | runtime_hours |
| ----- | ---- | ----------- | ---------------- | --------- | --------------- | ------------ | ------------------ | ---------------- | ----------------------- | ------------- |
| lgbm  | 1    | 0.7853      | 0.0032           | 0.0669    | 0.0742          | 0.2401       | 0.2805             | 0.0789           | 0.0015                  | 2.26          |
| lgbm  | 2    | 0.7851      | ...              | ...       | 0.0742          | ...          | 0.2805             | ...              | ...                     | 2.26          |
```

Floats with 4 decimals in the printed table; the CSV keeps full precision. (No threshold metrics: TPR at the base-rate threshold belongs to the fairness plan.)

### P6.4 Tests — `tests/test_models_runs.py`

| Test | Asserts |
| ---- | ------- |
| `test_save_and_load_round_trip` | after `save_nested_run` of a tiny `lgbm` result (top_k=2): all files of 5.8 exist; `load_oof(run_dir)` equals `runs[1].oof_predictions`; `load_oof(run_dir, rank=2)` equals rank 2; dtypes as in 5.7 |
| `test_selected_params_round_trip` | `load_selected_params` keys `{0,1,2}` (int), params equal `selected[k]["params"]` |
| `test_json_files_valid` | every `.json` loads with `json.load`; `rankings.json` has one entry per outer fold |
| `test_run_dir_for` | `run_dir_for("x", "rf") == Path("x/rf/baseline")` |
| `test_cli_end_to_end` | write `make_training_data(300)` feature parquets (as `_write_feature_artifacts` does in `test_models_pipeline.py`) into `tmp_path/features`; write a tiny `rf` model YAML and a `cv` YAML (3/2/3/2) into `tmp_path`; call `main(["--model", "rf", "--features-dir", ..., "--runs-dir", tmp_path/"runs", "--cv-config", ..., "--model-config", ...])`; outputs exist |
| `test_cli_refuses_existing_run` | second `main` call without flags → `SystemExit` with non-zero code; with `--overwrite` → succeeds |
| `test_cli_max_rows_guard` | `--max-rows 100` with `--runs-dir` = the absolute `<repo>/artifacts/runs` → `SystemExit` non-zero; `load_training_data` is monkeypatched to fail and the test runs from another CWD |
| `test_summary_table` | two fake run dirs with metrics → `summary_baseline.csv` has the expected rows and columns |
| `test_cli_max_rows_must_be_below_n_rows`, `test_cli_max_rows_subsamples`, `test_cli_resume_and_overwrite_are_exclusive`, `test_cli_model_config_name_mismatch`, `test_cli_existing_run_checked_before_loading_data`, `test_cli_run_config_contents`, `test_cli_resume_on_finished_run`, `test_cli_resume_without_run_dir_starts_fresh`, `test_cli_does_not_leak_log_handlers`, `test_git_commit_marks_dirty_tree`, `test_write_json_rejects_nan_and_converts_numpy` | CLI edge cases, `run_config.json` keys in order, `resumed` flag, runtime carried over a resume, no leaked log handlers, `-dirty` suffix, strict JSON |
| `test_summary_includes_robustness_ranks_sorted`, `test_summary_main_writes_csv_and_prints_table`, `test_summary_ignores_other_variants_and_missing_dir` | summary rows for ranks ≥ 2, sort order, printed table rounding vs full-precision CSV |
| `test_metrics_json_keys` | the saved `metrics.json` of rank 1 and rank 2 has exactly the keys of 5.8 in order; `folds` has one entry per outer fold |

Note: the tiny `rf` YAML in `test_cli_end_to_end` must be named `rf.yaml` (the name must equal the file stem).

### P6.5 Smoke runs on real data (20,000-row stratified subsample)

```bash
for m in logreg lgbm mlp rf; do
  python -m src.models.run --model $m --max-rows 20000 \
    --cv-config configs/models/cv_smoke.yaml --runs-dir artifacts/runs_smoke --overwrite
done
python -m src.models.summary --runs-dir artifacts/runs_smoke
```

Then test resume on one model:

```bash
rm -rf artifacts/runs_smoke/logreg   # not --overwrite: that deletes only after the data is loaded, so a stale run.log would satisfy the wait below
python -m src.models.run --model logreg --max-rows 20000 \
  --cv-config configs/models/cv_smoke.yaml --runs-dir artifacts/runs_smoke &
PID=$!
# wait until outer fold 2 has started (so some candidates are finished and some are not), then stop it
until grep -q "outer 2/3 | cand 01/3" artifacts/runs_smoke/logreg/baseline/run.log 2>/dev/null; do sleep 1; done
kill -TERM $PID; wait $PID   # exit code 143
python -m src.models.run --model logreg --max-rows 20000 \
  --cv-config configs/models/cv_smoke.yaml --runs-dir artifacts/runs_smoke --resume
```

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `pytest -q` (full suite) passes; new tests < 90 s in total (`pytest -q --durations=10` to check).
- [ ] All four smoke runs finish (each < 10 min); each `artifacts/runs_smoke/<m>/baseline/` matches the tree in 5.8 with `robustness/rank_2/`.
- [ ] Every smoke `oof_predictions.parquet` has 20,000 rows, no NaN, `fold` ∈ {0,1,2}; rank-1 OOF AUC > 0.65 for every model.
- [ ] The run was really interrupted (`tuning/inner_scores.csv` has fewer than 18 data rows and `metrics.json` does not exist before the resume; a background `&` job ignores SIGINT in a non-interactive shell, so use SIGTERM), and the resumed `logreg` smoke run logs `skipped ... finished candidates (resume)` and its rank-1 OOF AUC equals, to 1e-12, the AUC of the uninterrupted `logreg` run from the loop above (write that AUC down **before** starting the interrupted run, because `--overwrite` deletes it).
- [ ] Paste the smoke summary table in the Progress log Notes.
- [ ] `artifacts/` is gitignored — confirm `git status` shows no files under `artifacts/`.
- [ ] Commit `models: phase 6 - run persistence, CLI, summary`.

---

## Phase 7 — Full runs (4 models)

**Precondition:** the feature matrix is final (decision "PTI not a model feature": PTI is not added to `X`). Re-check that `artifacts/features/metadata.json` still reports 493 features and 307,507 rows; if not, stop and ask.

Run **one model at a time** (each uses all CPU cores). Recommended order and rough durations on the 16-core laptop (based on single-fit timings of ~10–30 s for logreg/MLP/LightGBM and ~30–80 s for RF on 164k rows; 300 tuning fits + 25 refits per model):

| Order | Command | Rough time |
| ----- | ------- | ---------- |
| 1 | `python -m src.models.run --model logreg` | ~1–1.5 h |
| 2 | `python -m src.models.run --model lgbm` | ~2–3 h |
| 3 | `python -m src.models.run --model mlp` | ~2–4 h |
| 4 | `python -m src.models.run --model rf` | ~4–7 h |

**Phase 7a is run by the user, not by the executor.** The user starts all four models as **one sequential, resumable background chain** in the macOS Terminal app (not in Cursor's integrated terminal, so closing Cursor or the agent chat cannot kill it). Before starting: Phases 0–6 done and reviewed, laptop on mains power, lid open, automatic macOS updates deferred for the night.

```bash
cd <repo root>
source .venv/bin/activate
mkdir -p artifacts/runs
nohup caffeinate -ims sh -c 'for m in logreg lgbm mlp rf; do [ -f "artifacts/runs/$m/baseline/metrics.json" ] && continue; python -m src.models.run --model "$m" --resume || exit 1; done' \
  >> artifacts/runs/full_runs_stdout.log 2>&1 &
tail -f artifacts/runs/logreg/baseline/run.log      # Ctrl-C stops only the viewing, not the runs
```

- The chain runs the models **one after another** (each uses all cores) and stops at the first failure.
- Models whose `metrics.json` already exists are skipped, and `--resume` continues an interrupted model from its checkpoint (on a model that has not started yet it simply starts fresh). So **the same command both starts and restarts the chain** after a crash, reboot or sleep.
- `caffeinate -ims` keeps the Mac awake only while the chain runs and stops by itself when it ends; nothing needs to be reverted.
- Check progress: `tail -n 20 artifacts/runs/full_runs_stdout.log`, `ls artifacts/runs/*/baseline/metrics.json`, `pgrep -fl src.models.run`.
- Stop deliberately: `pkill -TERM -f src.models.run` (the chain then exits; restart later with the same command). Do not rely on SIGINT: a job started with `&` in a non-interactive shell ignores it.

When `artifacts/runs/rf/baseline/metrics.json` exists and `full_runs_stdout.log` has no traceback, the user sends the Phase 7b prompt (section 6.2) to the executor, which verifies the outputs and commits the Progress-log update.

After all four (in Phase 7b): `python -m src.models.summary`.

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `artifacts/runs/{logreg,lgbm,mlp,rf}/baseline/` each match the tree in 5.8 with `robustness/rank_2` … `rank_5`.
- [ ] Each `oof_predictions.parquet` (rank 1 and ranks 2–5) has 307,507 rows, no NaN, `SK_ID_CURR` identical in order to `artifacts/features/ids_train.parquet`, and `fold` identical to the fold assignment of `make_stratified_splits(y, 5, 42)` (= the legacy folds).
- [ ] Each `tuning/inner_scores.csv` has exactly `5 × 20 × 3 = 300` rows.
- [ ] Sanity flags (do **not** tune further if one fails — stop and report to the reviewer): rank-1 OOF AUC `lgbm ≥ 0.775`, `logreg ≥ 0.760`, `mlp ≥ 0.750`, `rf ≥ 0.740`; for every model `oof_brier < brier_reference` and `oof_log_loss < log_loss_reference`. For reference: the legacy LightGBM has OOF AUC 0.7841 (slightly optimistic), Brier 0.0661 and log loss 0.2381 against references 0.0742 / 0.2805, and single-split pilots gave logreg ≈ 0.774, MLP ≈ 0.770, RF (100 trees) ≈ 0.758.
- [ ] `artifacts/runs/summary_baseline.csv` exists; paste the printed table into the Progress log Notes.
- [ ] No code changes in this phase. (If a bug is found, fix it in a separate commit `models: phase 7 fix - <what>`, rerun the affected tests, and tell the reviewer.)

---

## Phase 8 — Cleanup of unneeded code, files and dependencies

**Goal:** remove what the new runner makes unnecessary and fix stale references, without touching anything notebook `08` or `legacy_check` needs.

**Keep (do not remove):** `src/models/training.py` (`train_lgbm_cv`, `FoldResult`, `CVResult`), `src/models/io.py`, `configs/model.yaml`, `analysis/helpers/data_audit.py`, and the tests of `train_lgbm_cv` / `load_training_data`.

Before removing a symbol or file, run `git grep -n "<name>"` and confirm the only users are the ones named in the table. If anything else uses it, stop and ask.

| # | Item | Action | Why |
| - | ---- | ------ | --- |
| 1 | `src/models/pipeline.py` (`train_credit_scorer`, `save_model_artifacts`, legacy CLI) | `git rm` | Superseded by `run.py` + `runs.py`; not used by notebook `08` or `legacy_check`. |
| 2 | `ModelArtifacts` in `src/models/types.py` | delete the class | Only used by `pipeline.py`. |
| 3 | `src/models/__init__.py` | Remove the lazy exports `train_credit_scorer`, `save_model_artifacts`, `ModelArtifacts`; keep `CVResult`, `TrainingData`; add `run_nested_cv`, `get_model_spec`, `MODEL_REGISTRY`, `load_oof`, `load_selected_params`, `CVSettings`, `ModelConfig`, `NestedCVResult` (same lazy `__getattr__` pattern). Docstring: `"""Credit scoring models (LightGBM, random forest, logistic regression, MLP)."""` | Public API = new runner. |
| 4 | `tests/test_models_pipeline.py` | `git mv` to `tests/test_models_training.py`; delete `test_save_model_artifacts_round_trip`, `test_train_credit_scorer_from_artifacts` and the `src.models.pipeline` import; module docstring `"""Tests for the legacy train_lgbm_cv and the training-data contract."""`. In `tests/test_models_runs.py`, update the comment/reference to `_write_feature_artifacts` if it names the old file. | Test files mirror `src/` (`training.py`, `io.py`). |
| 5 | `configs/paths.yaml` | Replace `models_dir: artifacts/models` with `runs_dir: artifacts/runs`. | The legacy output folder is no longer produced. |
| 6 | `requirements.txt` | Remove `featuretools`, `plotly`, `psutil` and the `setuptools` pin (no imports anywhere; the Featuretools scripts were deleted in commit `07428fb`). Add `scipy>=1.10` (imported directly by `analysis/helpers/data_audit.py`) and `jupyter>=1.0` (notebooks run with `jupyter nbconvert`). | Requirements match real imports. Do not uninstall anything from `.venv`. |
| 7 | References to the deleted `fe.py` / `feature_importance.py` | Delete the bullet in `CLAUDE.md` (Architecture), the two lines in the tree of `docs/thesis-overview.md`, and the bullet in `src/README.md` (Conventions). | The files no longer exist. |
| 8 | `CLAUDE.md` line 10, sentence "(`analysis/notebooks/README.md` still says "audit only" — that is outdated; `docs/decisions.md` is authoritative.)" | Delete that parenthetical sentence only. | The README was already fixed. |
| 9 | `artifacts/models/` (local, gitignored, ~28 MB) | **Ask the user.** Delete only after an explicit yes. | No code reads it after this phase; `legacy_check` recomputes the legacy OOF in memory. |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `git grep -n "train_credit_scorer\|save_model_artifacts\|ModelArtifacts\|src.models.pipeline" -- src tests analysis` → no hits (docs are fixed in Phase 9).
- [ ] `git grep -n "fe\.py\|feature_importance\.py"` → no hits.
- [ ] `git grep -n -i "featuretools\|plotly\|psutil" -- requirements.txt` → no hits; `pip install -r requirements.txt --dry-run` succeeds.
- [ ] `python -m src.models.legacy_check` still prints `EQUIVALENT` (the `train_lgbm_cv` path is intact).
- [ ] `python -c "import analysis.helpers.data_audit"` works (notebook `08` imports).
- [ ] `python -c "from src.models import run_nested_cv, get_model_spec, load_oof, load_selected_params, TrainingData, CVResult"` works.
- [ ] The user's answer about `artifacts/models/` is recorded in the Notes.
- [ ] Commit `models: phase 8 - cleanup`.

---

## Phase 9 — Documentation

**Files:** ✎ `docs/credit_scoring.md`, ✎ `CLAUDE.md`, ✎ `src/README.md`, ✎ `docs/thesis-overview.md`, ✎ `.cursor/rules/thesis-context.mdc` (one line only), ✎ `.cursor/rules/python-src.mdc` (style lines only), optionally ✎ `docs/glossary.md`. No code changes.

1. **`docs/credit_scoring.md`:** restructure into:
   - Goal (4 models, nested CV, OOF for fairness) — replace the "Hyperparameter tuning: None" row.
   - Models table (registry name, estimator, preprocessing, early stopping).
   - Nested CV procedure (outer/inner folds, 20 candidates with candidate 0 = defaults, selection rule, LightGBM refit with mean inner best iteration, MLP epochs searched, top-5 refits for robustness) with the data-flow diagram of 4.2.
   - Configuration: `configs/models/cv.yaml` and the four model files (links, and a table of search spaces).
   - Outputs: the run-directory contract (5.8), what is recorded and why (5.9), and the metric definitions and reading guide (5.10).
   - How to run: `run`, `summary`, `legacy_check`, smoke runs, `--resume`.
   - Results: the Phase 7 summary table with its date.
   - Legacy baseline: `train_lgbm_cv` / `configs/model.yaml` remain for notebook `08` and `legacy_check`; the legacy CLI was removed in Phase 8; its OOF AUC (0.7841) is slightly optimistic because early stopping used the outer validation fold. The `outputs:` block of `configs/model.yaml` is no longer read by any code (only the removed `save_model_artifacts` used it; the file stays frozen).
   - Downstream: fairness reads `artifacts/runs/<model>/baseline/oof_predictions.parquet` (+ `robustness/rank_<r>/` for the robustness check) and joins `fairness_groups_train.parquet` on `SK_ID_CURR`.
2. **`CLAUDE.md`:** in *Commands*, replace the legacy `src.models.pipeline` command with the `run`, `summary` and `legacy_check` commands, and add `black src tests analysis/helpers && isort src tests analysis/helpers`. In the style paragraph, replace "No linter config is checked in." with "Formatting: black + isort, configured in `pyproject.toml`." In *Architecture*, rewrite the `src/models/` bullet (registry, `fit_predict`, generic CV, nested CV, run directories, `train_lgbm_cv` kept for notebook `08`) and change the `src/fairness/` bullet's input to `artifacts/runs/<model>/baseline/oof_predictions.parquet`. Keep everything else.
3. **`src/README.md`:** update the `models/` row and the *Status* commands. (Already done in the Phase 8 review fixes: layout comment, `models/` row, *Status* `src/models/` line and "Run model" command; only check them.)
4. **`docs/thesis-overview.md`:** "Credit scoring" row → status `Done`, text "LightGBM, random forest, logistic regression, MLP; nested CV (5 × 3, 20 candidates each); OOF predictions per model in `artifacts/runs/`". Update the repository tree (no `fe.py`, new `configs/models/`, `artifacts/runs/`).
5. **`.cursor/rules/thesis-context.mdc`:** change the line `` - `src/` = **Feature engineering** across all tables, then **credit scoring model** (LightGBM on engineered features). `` to mention the four models. No other change.
6. **`.cursor/rules/python-src.mdc`:** under *Style*, replace "Follow Black code formatting" / "Use isort for import sorting" with "Format with `black` and `isort` (config in `pyproject.toml`); run them before every commit."
7. Do **not** edit `docs/decisions.md` / `docs/open-questions.md`; the reviewer updates the decision rows.

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `grep -rn "artifacts/models\|src.models.pipeline" CLAUDE.md docs src/README.md .cursor/rules` → no hits.
- [ ] Every command written in the changed docs runs as written (at least `--help` for each CLI).
- [ ] No absolute local paths in changed files: `git diff <BASE> --name-only | xargs grep -n "/Users/"` → nothing.
- [ ] Commit `models: phase 9 - documentation`.

---

## 6. Execution prompts (for Cursor Composer)

### 6.1 Per-phase execution prompt (template)

Copy, replace `<N>` and `<TITLE>`, and append the phase notes from 6.2.

```text
Execute Phase <N> ("<TITLE>") of .cursor/plans/multi-model-nested-cv.md, and only that phase.

1. Read first, in full: sections 1–5 of the plan (Goal and scope, Decisions, Ground rules
   including 3.1, Target architecture, Shared reference) and Phase <N>. Then read every
   existing file you will import from or edit in this phase.
2. Check the Progress log: the previous phase must be `done` with review verdict PASS or
   PASS WITH NITS (Phase 0 has no predecessor). If not, stop and tell me.
3. Set Phase <N> to `in progress` in the Progress log.
4. Implement exactly what Phase <N> specifies: file names, public signatures, constants,
   config values and test names as written; reuse the code given in the plan. Touch only
   the files this phase names. Do not implement anything from later phases. Do not add
   dependencies.
5. If the plan is ambiguous, contradicts the code, or an exit criterion fails and the fix
   is not obvious: stop and ask me. Never loosen tolerances or thresholds, never skip,
   weaken or delete tests to make them pass.
6. When the code is written, from the repo root:
   a. source .venv/bin/activate
   b. black src tests analysis/helpers && isort src tests analysis/helpers
   c. run this phase's tests, then the full suite: pytest -q
   d. go through every exit criterion of Phase <N> one by one, run its command and keep
      the output.
7. Update the Progress log row of Phase <N>: status `done`, Notes with the key evidence
   (test counts, AUCs, timings, anything unexpected). Then `git status --short` (only this
   phase's files, plus the plan), `git add` exactly those files, and commit
   "models: phase <N> - <short title>".
8. Reply with: files changed; an exit-criteria table (criterion → met / not met →
   evidence); deviations from the plan and why; open questions. End with
   "Phase <N> ready for review." Then stop. Do not start the next phase.
```

### 6.2 Phase notes (append to the template)

| Phase | Notes to append |
| ----- | --------------- |
| 0 | No code. Follow the steps in order; there are two commits (A: plan + formatter config, B: formatting only). Report which files black and isort changed. |
| 1 | `settings.py` uses the `KNOWN_MODELS` / `EARLY_STOPPING_MODELS` constants until Phase 3. Copy the YAML blocks of 5.1, 5.2, 5.5 exactly (comments included). |
| 2 | Use the reference implementation in Phase 2 as written; only add Google-style docstrings. |
| 3 | Append the two fixtures to `tests/conftest.py`; do not change existing fixtures. After replacing the constants in `settings.py`, run `python -c "import src.models.settings"` to rule out a circular import. |
| 4 | `python -m src.models.legacy_check` runs on the full data (two 5-fold LightGBM CVs, ~4–5 min, ~3 GB RAM). Run it in the foreground and paste its full output into the Notes. |
| 5 | The most important phase for correctness. Follow the algorithm in P5.2 line by line and the leakage rules under it. Write the tests in the order of the table and run them as you go. |
| 6 | Smoke runs take about 10–30 min in total. Use the background + `kill -TERM` recipe in P6.5 for the resume check (no manual Ctrl-C needed) and confirm the interruption happened before resuming (`wc -l` of `inner_scores.csv`, no `metrics.json`). Write down the uninterrupted `logreg` smoke AUC before the interrupted run. |
| 7a | **Not sent to the executor: the user starts the runs manually** with the command in Phase 7. The executor does nothing in 7a. |
| 7b | Verify (send when the chain has finished: `artifacts/runs/rf/baseline/metrics.json` exists and `artifacts/runs/full_runs_stdout.log` has no traceback): run `python -m src.models.summary`, check every exit criterion of Phase 7 with a short Python script (show the script and its output), fill in the Notes, set `done`, commit the Progress-log update as "models: phase 7 - full runs". If a run failed, report the traceback and stop. |
| 8 | Run `git grep` for every item before removing it. For item 9 (`artifacts/models/`), ask me and wait for the answer; delete only on an explicit yes. |
| 9 | Documentation only; no code changes. Use the Phase 7 summary table from the Notes for the Results section. |

### 6.3 Fix-after-review prompt

```text
The review of Phase <N> of .cursor/plans/multi-model-nested-cv.md returned <FAIL / PASS WITH NITS>.
Findings:
<paste the findings>

Fix exactly these findings: all blockers and majors, and the minors and nits unless I
marked them "skip". Change nothing else. Then run black + isort, the standard checks (3.1)
and every exit criterion of Phase <N> again, update the Notes of Phase <N>, and commit
"models: phase <N> review fixes". Reply with a table: finding → what you changed
(file:line) → evidence it is fixed. Then stop.
```

---

## 7. Review prompts (executed by Claude Code)

### 7.1 Per-phase review prompt (template)

Copy, replace `<N>` and `<TITLE>`, and append the phase-specific checklist from 7.2.

```text
Review Phase <N> ("<TITLE>") of .cursor/plans/multi-model-nested-cv.md.

Context: MSc thesis repo (bias in ML credit scoring). The plan was executed by a smaller
model. Read, in this order: the plan's sections 1–5 and Phase <N>; docs/decisions.md rows
"Hyperparameter tuning", "Frozen feature set", "Model and mitigation code structure",
"Classifier and evaluation metric"; CLAUDE.md.

Do:
1. Find the phase commit(s) with `git log --oneline --grep "phase <N>"` (including any
   "phase <N> review fixes" commit) and inspect them: `git show --stat <commit>` and
   `git show <commit>`. Also check `git status` for uncommitted leftovers.
2. Verify every exit criterion of Phase <N> yourself (run the commands; do not trust the
   Progress log). Run `source .venv/bin/activate && pytest -q`.
3. Check plan conformance: file names, public signatures, config values, constants, schema,
   and the out-of-scope list (no fairness metrics, no mitigation, no feature changes,
   no X_test, no fairness_groups in src/models/, Ground rule 1 respected).
4. Check correctness: data leakage across outer/inner folds, determinism/seeding,
   off-by-one in fold/rank indexing, NaN handling, dtype of OOF columns, silent
   exception swallowing, warnings suppressed too broadly.
5. Check style: run the standard checks (plan 3.1: black --check, isort --check-only,
   pytest); then what the tools miss: Google docstrings, typing (Optional[...], no
   `X | None`), no hard-coded hyperparameters/paths in src/, no mutable defaults, tests
   use tiny synthetic data only.
6. Run the phase-specific checks below.

Report:
- Verdict: PASS / PASS WITH NITS / FAIL.
- Findings ranked by severity (blocker / major / minor / nit), each with file:line,
  what is wrong, why it matters, and the concrete fix.
- Exit-criteria table: criterion → met / not met → evidence (command output).
Do not change code unless I ask. If the verdict is PASS or PASS WITH NITS, write the
verdict into the plan's Progress log.
```

### 7.2 Phase-specific checklists

**Phase 0**
- BASE recorded and equal to the parent of commit A.
- `pyproject.toml` and the `requirements.txt` lines exactly as in Phase 0.
- Commit B is formatting only: `git show --stat <B>` lists only `.py` files under `src/`, `tests/`, `analysis/helpers/`; skim `git show <B>` for any non-formatting change (renamed names, changed literals, removed code).
- Same pytest pass count before and after.

**Phase 1**
- `sample_candidates` creates one RNG for all draws, samples in sorted key order, and returns candidate 0 as a deep copy of `default_params`.
- Rounding to 6 significant digits does not push values outside `[low, high]` in a way that breaks validation (check edge cases like `high=1.0`).
- `int_*` dists reject floats and bools; `loguniform` rejects `low <= 0`.
- YAML files match plan 5.1/5.2/5.5 exactly (`diff` against the plan's blocks).
- New types added without altering existing ones; `typing` imports merged, not duplicated.

**Phase 2**
- Binary detection uses training data only; transform never refits; input not mutated.
- No division by zero; constant columns → 0; dtype float64.
- Works with `sklearn.base.clone` and inside `Pipeline`.

**Phase 3**
- `fit_predict` passes **no** extra kwargs when `sample_weight is None` (bit-identical to legacy).
- `sample_weight` routed to the final `clf` step for pipelines.
- `ConvergenceWarning` suppressed only for `mlp`, only around `fit`, and only that category.
- `random_state` injected for all four models; MLP `hidden_layer_sizes` converted to tuple.
- `return_train_proba` predicts the training rows with the same iteration count as the validation prediction and is excluded from `fit_seconds`.
- Importance frames: correct mapping of features to values (order), types per model.
- `settings.py` now uses the registry; no circular import at runtime (`python -c "import src.models.settings"`).

**Phase 4**
- `cv.py` free of model-specific branches.
- Run `python -m src.models.legacy_check` yourself and confirm `EQUIVALENT` and max |Δp| ≤ 1e-9.
- Splits generated from `y` only and equal to the legacy folds.

**Phase 5**
- Leakage audit: trace every use of `outer_va`; inner CV gets only `X[outer_tr]`; the LightGBM refit has no `eval_set`.
- Selection uses mean inner AUC with deterministic tie-breaking; the same candidate list in every outer fold.
- Resume: half-finished `(outer_fold, candidate_id)` pairs are dropped and recomputed; candidate-change detection works; selection reads the CSV in both paths.
- `n_estimators` override = `round(mean best_iteration)`, min 1; resolved params saved.
- Memory: large arrays deleted per outer fold; no unnecessary DataFrame copies of `X`.
- Logging uses `logging`, no prints.
- `metrics.py`: formulas and gap signs match 5.10 (check `test_classification_metrics_known_values` by hand); `log_loss` passes `labels=[0, 1]`; the in-sample predictions are of the outer training rows only.

**Phase 6**
- Run directory matches 5.8 exactly; JSON files valid, keys stringified, numpy types converted.
- CLI guards: existing run, `--max-rows` with the default runs dir, `--resume` vs `--overwrite` exclusive.
- `run_config.json` contains git commit, versions, timestamps, `max_rows`.
- Re-run one smoke model yourself and verify the OOF contract with a short pandas script.
- Summary CSV/table correct for top-level and robustness ranks.

**Phase 7**
- Verify every exit criterion with a script (row counts, `SK_ID_CURR` order, `fold` equality with `make_stratified_splits(y, 5, 42)`, 300 inner rows, ranks 2–5 present).
- Inspect `tuning/rankings.json`: are selected candidates stable across outer folds? Is candidate 0 ever selected? Is the best candidate at the edge of a search range (suggests the range is too narrow — report, do not change)?
- Compare rank-1 vs ranks 2–5 OOF AUC spread per model.
- Calibration and fit: Brier and log loss vs the no-skill references; train–validation gaps per model (read within a family, 5.10); `selection_optimism` vs the fold spread of AUC. Flag any warning sign listed in 5.10.
- Check `run.log` for warnings/errors (e.g. LightGBM warnings, convergence messages for logreg).
- Report the summary table and any sanity flag that failed.

**Phase 8**
- For every removed symbol/file, `git grep` confirms no remaining user (including notebooks: `grep -l` over `analysis/notebooks/*.ipynb`).
- Nothing notebook `08` needs was touched: `train_lgbm_cv`, `FoldResult`, `CVResult`, `TrainingData`, `OOF_PROBA_COL`, `load_yaml_config`, `configs/model.yaml`.
- `requirements.txt` matches the imports in `src/`, `tests/`, `analysis/` (`git grep -h "^import\|^from"` plus the notebook imports); nothing needed was dropped.
- The renamed test file still contains every `train_lgbm_cv` / `load_training_data` test.
- `artifacts/models/` deleted only if the user said yes.

**Phase 9**
- Docs match the implemented behaviour (commands, file names, defaults); no stale "LightGBM only" statements outside the legacy section.
- `CLAUDE.md` still accurate elsewhere; no `artifacts/models` or `src.models.pipeline` left.
- Then update `docs/decisions.md` (rows "Hyperparameter tuning", "Frozen feature set", "Model and mitigation code structure", "Classifier and evaluation metric": status → `closed (YYYY-MM-DD)` only if the user agrees) and `docs/open-questions.md`, following `.cursor/skills/sync-thesis-decisions/SKILL.md`.

### 7.3 Final review prompt (whole implementation)

```text
Final review of the whole implementation of .cursor/plans/multi-model-nested-cv.md
(commits on main from BASE, recorded in the Progress log, to HEAD).

Read the full plan, docs/decisions.md (rows "Hyperparameter tuning", "Frozen feature set",
"Model and mitigation code structure", "Classifier and evaluation metric", "Decision
threshold", "Utility metrics"), CLAUDE.md, and docs/credit_scoring.md.

Do:
1. `git log --oneline <BASE>..HEAD` and `git diff <BASE>..HEAD --stat`; confirm one commit
   per phase (+ any documented fix commits) and that no file changed outside Ground rule 1
   and its exceptions.
2. Run the standard checks (plan 3.1) and `pytest -q --durations=15`.
3. Run `python -m src.models.legacy_check` and `python -m src.models.summary`.
4. Methodology audit against the decisions:
   - equal tuning budget and identical procedure for all four models (same cv.yaml, same
     candidate count, same outer and inner folds);
   - tuning only on ROC AUC; no fairness information anywhere in src/models/;
   - outer validation folds never used for tuning or early stopping (trace the code);
   - frozen features (no transformations outside fold-internal pipelines);
   - no class weights / resampling; X_test unused;
   - OOF contract identical to the legacy schema so src/fairness/ can consume any run;
   - selected_params.json is sufficient to refit every model per outer fold for later
     mitigation runs (frozen hyperparameters).
5. Results audit: per model, rank-1 OOF AUC, Brier and log loss vs their no-skill
   references, train–validation gaps and selection optimism (read as in plan 5.10),
   fold spread, spread across ranks 1–5,
   selected candidates per outer fold, any candidate at a search-range edge, runtime.
   Compare LightGBM nested vs legacy (expect slightly lower or equal; the legacy number is
   optimistic) and explain any surprise.
6. Architecture audit: adding a fifth model should need only a registry entry + YAML
   (+ an importance function); adding a pre-processing mitigation should need only a
   sample_weight provider. Point out anything that would force edits to cv.py / nested.py /
   runs.py.
7. Code quality: duplication, dead code, unclear names, missing docstrings/types,
   overly broad exception handling or warning suppression, performance hot spots
   (copies of X, per-row Python loops).
8. Cleanup audit: nothing notebook 08 or legacy_check needs was removed; requirements.txt
   matches the real imports; no stale references to removed files.
9. Docs: commands in CLAUDE.md / credit_scoring.md / src/README.md work as written.

Report:
- Overall verdict: DONE / DONE WITH FOLLOW-UPS / NOT DONE.
- Findings ranked by severity with file:line and concrete fixes.
- A short results summary (table of models × OOF AUC × runtime) written for the thesis
  author, plus any methodological caveat they should mention in the thesis text
  (e.g. inner early stopping on the scored inner fold, MLP epochs as a hyperparameter,
  LR/MLP preprocessing).
- Proposed updates to docs/decisions.md and docs/open-questions.md (do not apply them
  without the user's confirmation).
```

---

## 8. Appendix

### 8.1 Why these design choices (for the executor's understanding — do not change them)

- **Candidate 0 = defaults:** guarantees each model's tuned result is at least as good (on inner CV) as its untuned default, and counts toward the equal budget of 20.
- **Same candidate list in every outer fold:** the outer folds then differ only in data, which makes the ranking comparison across folds meaningful.
- **LightGBM refit without early stopping:** using the outer validation fold for early stopping would leak it into the model (the legacy baseline does this).
- **MLP epochs searched:** sklearn's MLP early stopping uses accuracy; with 8% positives accuracy barely moves, so it stops arbitrarily. Pilot: weak regularization overfits after ~10 epochs (AUC 0.72 → 0.67), strong `alpha` gives 0.77.
- **Brier score and log loss next to ROC AUC:** AUC only ranks applicants; the fairness stage also reads the probabilities (calibration by group). Overall probability quality must be known first so a model-wide miscalibration is not read as a group effect.
- **Fit diagnostics are descriptive:** nested CV already protects the OOF results from overfitting; the gaps explain behaviour, they do not drive any choice.
- **RF `n_estimators` fixed at 200:** more trees only reduce variance; tuning it wastes budget. `max_features` limited to ≤ 0.1 of 493 columns to keep fit time manageable.
- **`SignedLogStandardScaler` only for logreg/MLP:** trees are invariant to monotone transforms; linear and neural models need comparable scales and are hurt by heavy tails (e.g. `AMT_INCOME_TOTAL`).

### 8.2 Troubleshooting

| Symptom | Likely cause | Action |
| ------- | ------------ | ------ |
| `legacy_check` reports small non-zero Δp | passing `sample_weight=None` explicitly to LightGBM, using float32, or a different `n_jobs` | pass no kwargs when weights are `None`; keep float64; use the legacy config unchanged |
| LightGBM warning `No further splits with positive gain` floods the log | `verbosity` not `-1` | check `fixed_params` |
| `ConvergenceWarning` from logreg | `max_iter` too low or scaler missing | confirm the pipeline has `prep`; report if it persists with `max_iter: 2000` |
| RF run much slower than expected | `max_features` float too large or `n_jobs` not `-1` | check the YAML; do not change the search space without asking |
| Memory error | copies of `X` kept across outer folds | `del` per-fold arrays; never convert `X` back to a DataFrame inside loops |
| Resume raises "Search candidates changed" | YAML or `cv.yaml` edited after the run started | restore the config or start over with `--overwrite` |
