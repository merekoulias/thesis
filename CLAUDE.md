# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc thesis on **bias in ML credit scoring and bias mitigation** (pre-, in-, post-processing) using the Kaggle **Home Credit Default Risk** dataset. The goal is one LightGBM credit scorer built from all relational tables → fairness audit → mitigation. Predictive performance is nice-to-have, not a research question. Internal validation only — no Kaggle submission files.

- Protected attributes: `CODE_GENDER` (mandatory) and intersectionality via `CODE_GENDER` × `DAYS_BIRTH`. Rows with `CODE_GENDER == 'XNA'` are dropped at load time.
- Protected attributes are **included** as model features (one-hot encoded in `X`) and also kept raw for fairness grouping. (`analysis/notebooks/README.md` still says "audit only" — that is outdated; `docs/decisions.md` is authoritative.)
- All repo text (code, comments, docs) is in English. The written thesis itself will be in Greek.

## Commands

Run everything from the repo root using the local virtualenv (`.venv`, Python 3.14). `pytest.ini` sets `pythonpath = .` so `src.*` imports resolve.

```bash
source .venv/bin/activate
pip install -r requirements.txt

pytest                                              # all tests (testpaths = tests)
pytest tests/test_models_pipeline.py                # one file
pytest tests/test_features_pipeline.py::test_name   # one test

# Stage 1: feature engineering (reads data/*.csv → artifacts/features/)
python -m src.features.pipeline --data-dir data --artifacts-dir artifacts/features

# Stage 2: LightGBM CV (reads artifacts/features/ → artifacts/models/), ~2 min on full data
python -m src.models.pipeline --features-dir artifacts/features --models-dir artifacts/models [--config configs/model.yaml]

# Execute an EDA notebook in place
jupyter nbconvert --to notebook --execute --inplace analysis/notebooks/00_dataset_overview.ipynb
```

Style for `src/`: Black (88 cols), isort, Google-style docstrings, type hints on all public functions using `typing` (`Optional[X]`, not `X | None`), custom types in each package's `types.py`. No linter config is checked in.

## Architecture

Pipeline flow is `src/data/` → `src/features/` → `src/models/` → `src/fairness/`, with stages decoupled through parquet artifacts on disk (`artifacts/`, gitignored). Raw CSVs live in `data/` (gitignored). Paths default to `configs/paths.yaml`; model hyperparameters, seed, and fold count live in `configs/model.yaml` — don't hard-code them in packages.

- **`src/data/`** — `load_home_credit_data()` returns a `HomeCreditData` bundle of all tables, validates `SK_ID_CURR` / `SK_ID_PREV` / `SK_ID_BUREAU` keys, and drops XNA gender rows.
- **`src/features/`** — `pipeline.build_features()` orchestrates:
  1. `application.clean_application()` cleans the applicant "spine" (`DAYS_EMPLOYED` 365243 sentinel → NaN + `DAYS_EMPLOYED_ANOM` flag; drop `_MEDI`/`_MODE` building duplicates in favour of `_AVG`; split out `TARGET`) and `extract_fairness_groups()` copies raw `CODE_GENDER`/`DAYS_BIRTH` and adds `AGE_YEARS`, 10-year `AGE_BAND` (`assign_age_band()`, edges in `src/constants.py`) and `GENDER_AGE`.
  2. One module per child table (`bureau.py`, `previous.py`, `installments.py`, `pos_cash.py`, `credit_card.py`) aggregates one-to-many history to **one row per `SK_ID_CURR`** (via `SK_ID_PREV` / `SK_ID_BUREAU` as needed), with column prefixes `BUREAU_`, `PREV_`, `INST_`, `POS_`, `CC_` and a `HAS_*_HISTORY` flag.
  3. Left-join aggregates onto the spine (applicant count must not change), align train/test columns.
  4. `encoding.py`: missing indicators for high-missing numerics → median impute → one-hot **all** categoricals (no label encoding). Fit on train only; test uses `handle_unknown="ignore"`.
  Outputs: `X_train/X_test/y_train/ids_*/fairness_groups_*.parquet` + `metadata.json`.
- **`src/models/`** — `io.load_training_data()` enforces the data contract (row alignment, unique IDs, no `TARGET` in `X`, finite numeric `X`, encoded protected columns present — it raises if they're missing). `training.train_lgbm_cv()` runs `StratifiedKFold` with per-fold early stopping, sanitizes feature names for LightGBM and maps importances back. No class weights/resampling, no tuning, no threshold. Outputs `oof_predictions.parquet` (`SK_ID_CURR`, `TARGET`, `y_pred_proba`, `fold`), `metrics.json`, ROC, fold models, importances. `X_test` is not used. Baseline OOF ROC AUC ≈ 0.784.
- **`src/fairness/`** — **not yet implemented**. It should consume `artifacts/models/oof_predictions.parquet` + `artifacts/features/fairness_groups_train.parquet` (join on `SK_ID_CURR`). Planned metrics: demographic parity, equal opportunity, equalized odds, worst-group disparity over gender × 10-year age bands; one mitigation method per stage (exact methods TBD). Use OOF predictions and report cell counts per intersectional cell. Threshold selection belongs here: one group-blind threshold per model that flags the base-rate share (see `docs/decisions.md`).
- **`analysis/notebooks/`** — EDA notebooks `00`–`07` (one per table/domain, `07` = protected attributes/proxies) and `08_pre_modeling_bias_audit` (Phase 1, Step 5: data-level disparities, missingness and proxies; needs `artifacts/features/`). Out of scope there: credit-model training and fairness metrics on predictions. Notebook helpers (statistics, the auxiliary gender-predictability classifier) live in `analysis/helpers/`, not in `src/fairness/`. Notebooks resolve `data/` relative to the repo root. Consolidated findings in `docs/EDA.md`.
- **`fe.py`, `feature_importance.py`** (repo root) — legacy Featuretools/importance scripts, reference only. New work goes in `src/`.
- **`tests/`** — light contract tests on tiny in-memory fixtures (`tests/conftest.py::tiny_home_credit_data`): key integrity, output shapes, which columns go into `X`. Don't add tests that need the full CSVs.

Home Credit tables are large (millions of rows in `bureau_balance`, `installments_payments`, etc.): prefer vectorized pandas, avoid unnecessary copies, free big intermediates.

## Docs and decisions

Detailed write-ups live in `docs/` (`feature_engineering.md`, `child_tables_aggregation.md`, `encoding.md`, `credit_scoring.md`, `EDA.md`, `glossary.md`, `thesis-overview.md`). Keep them in sync when changing pipeline behaviour.

`docs/decisions.md`, `docs/open-questions.md`, and `docs/experiments.md` are **gitignored internal docs** but exist locally — read `decisions.md` and `open-questions.md` before making architectural or methodological choices. Decision statuses: `open` → `accepted` → `closed`; never edit a `closed` row, and don't record decisions the user hasn't stated. When a decision is resolved, update both files together (see `.cursor/skills/sync-thesis-decisions/SKILL.md`).

Cursor rules in `.cursor/rules/` (`thesis-context.mdc`, `python-src.mdc`, `python-data-analyst.mdc`) and skills in `.cursor/skills/` carry the same conventions; the EDA notebook skill defines the required section baseline for notebooks `02`–`06` and forbids absolute local paths in notebook content.

## Thesis notes (Obsidian vault)

Research and writing notes live outside the repo in an Obsidian vault at `/Users/c4t/c4t-vault/1-Projects/Uni/thesis/`. Only these folders are relevant; ignore the rest of the vault:

- `bibliography/` — references intended for the thesis, with notes on each.
- `Notes/audit/` — detailed notes on bibliography sections about fairness auditing (bias detection, metrics).
- `Notes/mitigation/` — detailed notes on bibliography sections about bias mitigation (pre-, in-, post-processing).
- `Notes/phases/` — the phases of the thesis implementation.
- `Notes/writing-notes/` — notes and decisions written while working through the phases and the bibliography; these are later turned into the thesis text.