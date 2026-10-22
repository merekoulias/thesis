# Credit scoring model

LightGBM baseline credit scorer implemented under [`src/models/`](../src/models/). Consumes feature artifacts from [`src/features/`](../src/features/) and produces out-of-fold (OOF) predicted probabilities for downstream fairness analysis.

Parent overview: [feature_engineering.md](feature_engineering.md). Configuration: [`configs/model.yaml`](../configs/model.yaml).

---

## Goal

Train a reproducible baseline classifier on the engineered applicant matrix:

- **Classifier:** `LGBMClassifier` (gradient-boosted trees)
- **Validation:** 5-fold stratified cross-validation on `application_train` only
- **Early stopping:** Per-fold validation AUC with patience from config
- **Primary utility metric:** ROC AUC (monitoring only; not a research question)
- **OOF predictions:** One predicted probability per training applicant for unbiased fairness evaluation
- **No Kaggle submission** and no evaluation on unlabeled `application_test`

Protected attributes remain **in `X`** per [decisions.md](decisions.md): `CODE_GENDER` one-hot (`cat__CODE_GENDER_*`) and `DAYS_BIRTH` numeric (`num__DAYS_BIRTH`). Raw grouping labels (gender, 10-year age band, gender × age) live in `fairness_groups_train.parquet` from the feature stage.

---

## Pipeline overview

```text
artifacts/features/
  X_train.parquet, y_train.parquet, ids_train.parquet
        │
        ▼
load_training_data()  — validate alignment, numeric X, protected cols present
        │
        ▼
train_lgbm_cv()  — StratifiedKFold(5), per-fold early stopping
        │
        ▼
artifacts/models/
  oof_predictions.parquet, metrics.json, roc_curve.*, fold_models/, importance
```

**Modules:**

| Module | Role |
| ------ | ---- |
| [`src/models/io.py`](../src/models/io.py) | Load feature parquets and YAML config; validate data contract |
| [`src/models/training.py`](../src/models/training.py) | Stratified CV, LightGBM fit/predict, OOF assembly, feature-name sanitization |
| [`src/models/pipeline.py`](../src/models/pipeline.py) | `train_credit_scorer()`, `save_model_artifacts()`, CLI |
| [`src/models/types.py`](../src/models/types.py) | `TrainingData`, `CVResult`, `ModelArtifacts` |

---

## Data contract (inputs)

Reads from `artifacts/features/` (output of `python -m src.features.pipeline`):

| Artifact | Required | Role |
| -------- | -------- | ---- |
| `X_train.parquet` | Yes | Numeric feature matrix (493 columns after F1 run) |
| `y_train.parquet` | Yes | Binary `TARGET` |
| `ids_train.parquet` | Yes | `SK_ID_CURR` traceability |
| `metadata.json` | Optional | Feature names and cleaning counts |

**Validation checks:**

- Row alignment across `X`, `y`, and `ids`
- Unique `SK_ID_CURR`; no `TARGET` in `X`
- Binary non-missing `TARGET`; finite numeric `X`
- Columns containing `CODE_GENDER` and `DAYS_BIRTH` must be present

`X_test` is **not** used by the baseline scorer.

---

## Model configuration

[`configs/model.yaml`](../configs/model.yaml):

| Setting | Value | Notes |
| ------- | ----- | ----- |
| `seed` | 42 | Fold shuffling and LightGBM randomness |
| `n_folds` | 5 | Stratified on `TARGET` |
| `early_stopping_rounds` | 50 | Validation AUC patience per fold |
| Class imbalance | None | Natural ~8.1% positive rate preserved |
| Hyperparameter tuning | None | Fixed baseline parameters |

LightGBM feature names are sanitized internally (spaces and special characters replaced) so training succeeds on one-hot column names; importance outputs map back to original names.

---

## Outputs

Written to `artifacts/models/` (gitignored):

| Artifact | Description |
| -------- | ----------- |
| `oof_predictions.parquet` | `SK_ID_CURR`, `TARGET`, `y_pred_proba`, `fold` |
| `metrics.json` | Fold/overall ROC AUC, best iterations, sample/feature counts, config |
| `roc_curve.json` / `roc_curve.png` | Aggregate OOF ROC curve |
| `fold_models/fold_*.txt` | Per-fold LightGBM booster files |
| `feature_importance_gain.parquet` | Mean gain importance across folds |
| `feature_importance_split.parquet` | Mean split importance across folds |
| `fold_details.json` | Per-fold AUC, best iteration, train/val indices |

**Not produced:** hard class predictions, classification threshold, or Kaggle submission CSV.

---

## How to run

```bash
# From repo root, with .venv activated

# 1. Feature engineering (if not already done)
python -m src.features.pipeline \
  --data-dir data \
  --artifacts-dir artifacts/features

# 2. Nested-CV run per model (writes artifacts/runs/<model>/baseline/)
python -m src.models.run --model lgbm \
  --features-dir artifacts/features \
  --runs-dir artifacts/runs

# Cross-model summary of the runs
python -m src.models.summary

# Legacy check: generic CV core reproduces train_lgbm_cv (configs/model.yaml)
python -m src.models.legacy_check
```

The legacy CLI `src/models/pipeline.py` and its `artifacts/models/` outputs were removed in the models plan's cleanup phase; the rest of this page still describes the legacy LightGBM baseline and is rewritten by that plan's documentation phase.

---

## Baseline result (M1)

From [experiments.md](experiments.md):

| Metric | Value |
| ------ | ----- |
| OOF ROC AUC | 0.7841 |
| Fold AUCs | 0.7807, 0.7898, 0.7824, 0.7870, 0.7809 |
| Runtime | ~1m 53s (MacBook, 307,507 train rows, 493 features) |

---

## Downstream: fairness

The baseline fairness audit is implemented in [`src/fairness/`](../src/fairness/). Full documentation: [fairness_audit.md](fairness_audit.md).

It consumes, per model and variant:

- `artifacts/runs/<model>/<variant>/oof_predictions.parquet` — OOF predicted probabilities (and robustness ranks 2–5 under `robustness/rank_*/`)
- `artifacts/features/fairness_groups_train.parquet` — raw `CODE_GENDER`, `DAYS_BIRTH`, `AGE_BAND`, and `GENDER_AGE` labels

Join on `SK_ID_CURR` (OOF row order is preserved). Hard predictions are made in the fairness stage: one group-blind threshold per model/rank at the `1 − base rate` quantile of OOF scores (8.07% flagged on full data). See [decisions.md](decisions.md) (rows "Decision threshold", "Rate orientation").

```bash
python -m src.fairness.audit --all --overwrite
python -m src.fairness.summary
```

---

## Related docs

- [Feature engineering](feature_engineering.md)
- [Encoding](encoding.md)
- [Decisions](decisions.md)
- [Experiments](experiments.md)
- [EDA findings](EDA.md)
