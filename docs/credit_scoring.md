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

Protected attributes (`CODE_GENDER`, `NAME_EDUCATION_TYPE`) remain **encoded in `X`** per [decisions.md](decisions.md). Raw grouping labels live in `fairness_groups_train.parquet` from the feature stage.

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
- Encoded columns containing `CODE_GENDER` and `NAME_EDUCATION_TYPE` must be present

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

# 2. Train baseline scorer
python -m src.models.pipeline \
  --features-dir artifacts/features \
  --models-dir artifacts/models

# Optional: custom config
python -m src.models.pipeline --config configs/model.yaml
```

**Python API:**

```python
from src.models import train_credit_scorer, save_model_artifacts

artifacts = train_credit_scorer(features_dir="artifacts/features")
save_model_artifacts(artifacts, "artifacts/models")
```

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

The fairness layer (`src/fairness/`, not yet implemented) should consume:

- `artifacts/models/oof_predictions.parquet` — predicted probabilities
- `artifacts/features/fairness_groups_train.parquet` — raw gender, education, intersection labels

Join on row order or `SK_ID_CURR`. Threshold selection for hard predictions is deferred to the fairness stage.

---

## Related docs

- [Feature engineering](feature_engineering.md)
- [Encoding](encoding.md)
- [Decisions](decisions.md)
- [Experiments](experiments.md)
- [EDA findings](EDA.md)
