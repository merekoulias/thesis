# Source pipeline (`src/`)

Implementation of the thesis credit-scoring and fairness pipeline. EDA lives under `[analysis/notebooks/](../analysis/notebooks/)`; do not put exploratory notebooks here.

Raw Home Credit CSVs stay in repo-root `[data/](../data/)` . Pipeline outputs go under `artifacts/` (gitignored) when generated.

## Layout

```
src/
├── data/        # Load CSVs, validate keys, drop XNA, prepare splits
├── features/    # Application cleaning + child-table aggregates → client matrix
├── models/      # Four scorers, nested CV, OOF predictions, utility metrics
└── fairness/    # Fairness metrics + mitigation (thesis core; implement after the scorer)
```


| Package     | Responsibility                                                                                                                                                                                                                                                                                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data/`     | Load relational tables; check `SK_ID_CURR` / `SK_ID_PREV` / `SK_ID_BUREAU`; drop `CODE_GENDER == 'XNA'`; return validated frames for FE. No model training.                                                                                                                                                                                                                    |
| `features/` | Clean application fields (e.g. `DAYS_EMPLOYED` sentinel); aggregate bureau / previous / installments / POS / credit card to one row per client; build feature matrix `X` and target `y`. Protected attributes (`CODE_GENDER`, `DAYS_BIRTH`) are **included** in model features per [docs/decisions.md](../docs/decisions.md). Also retain them raw, with 10-year `AGE_BAND` and `GENDER_AGE` labels, for fairness grouping. |
| `models/`   | Train LightGBM, random forest, logistic regression and an MLP with nested CV (tuning on ROC AUC only); save out-of-fold predicted probabilities, selected hyperparameters and metrics (ROC AUC, Brier, log loss) per run under `artifacts/runs/`. Legacy `train_lgbm_cv` kept for notebook `08`. No Kaggle submission.                                                       |
| `fairness/` | Demographic parity, equal opportunity, equalized odds, worst-group (gender × 10-year age bands); pre / in / post mitigation. One group-blind threshold per model that flags the base-rate share. Runs on OOF scores from `models/`.                                                                                                                                                                                                           |


**Flow:** `data/` → `features/` → `models/` → `fairness/`

## Conventions

- All repo text in English unless explicitly told otherwise.
- Prefer small, importable modules over notebook-style scripts.
- Configuration (paths, seeds, fold count, feature lists) belongs in `[configs/](../configs/)`, not hard-coded throughout packages.


## Status

- `src/data/` and `src/features/` implement feature engineering (load → clean → aggregate → encode).
- `src/models/` implements nested CV for four credit scorers (`lgbm`, `rf`, `logreg`, `mlp`) with OOF predictions per run.
- `src/fairness/` remains TODO.
- Run features: `python -m src.features.pipeline --data-dir data --artifacts-dir artifacts/features`
- Run model: `python -m src.models.run --model <name> --features-dir artifacts/features --runs-dir artifacts/runs`
- Run tests: `pytest tests/`

See [docs/thesis-overview.md](../docs/thesis-overview.md) and [docs/experiments.md](../docs/experiments.md).

## Related docs

- [Credit scoring](../docs/credit_scoring.md)
- [Feature engineering](../docs/feature_engineering.md)
- [Child-table aggregation](../docs/child_tables_aggregation.md)
- [Encoding](../docs/encoding.md)
- [Thesis overview](../docs/thesis-overview.md)
- [Decisions](../docs/decisions.md)
- [EDA findings](../docs/EDA.md)
- [Glossary](../docs/glossary.md)

