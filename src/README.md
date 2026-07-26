# Source pipeline (`src/`)

Implementation of the thesis credit-scoring and fairness pipeline. EDA lives under `[analysis/notebooks/](../analysis/notebooks/)`; do not put exploratory notebooks here.

Raw Home Credit CSVs stay in repo-root `[data/](../data/)` . Pipeline outputs go under `artifacts/` (gitignored) when generated.

## Layout

```
src/
├── data/        # Load CSVs, validate keys, drop XNA, prepare splits
├── features/    # Application cleaning + child-table aggregates → client matrix
├── models/      # LightGBM CV, OOF predictions, utility metrics
└── fairness/    # Fairness metrics + mitigation (thesis core; implement after the scorer)
```


| Package     | Responsibility                                                                                                                                                                                                                                                                                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data/`     | Load relational tables; check `SK_ID_CURR` / `SK_ID_PREV` / `SK_ID_BUREAU`; drop `CODE_GENDER == 'XNA'`; return validated frames for FE. No model training.                                                                                                                                                                                                                    |
| `features/` | Clean application fields (e.g. `DAYS_EMPLOYED` sentinel); aggregate bureau / previous / installments / POS / credit card to one row per client; build feature matrix `X` and target `y`. Protected attributes (`CODE_GENDER`, `NAME_EDUCATION_TYPE`) are **included** in model features per [docs/decisions.md](../docs/decisions.md). Also retain them for fairness grouping. |
| `models/`   | Train LightGBM with cross-validation; save out-of-fold predicted probabilities and metrics (ROC AUC for monitoring). No Kaggle submission.                                                                                                                                                                                                                                     |
| `fairness/` | Demographic parity, equal opportunity, equalized odds, worst-group (gender × education); pre / in / post mitigation. Runs on OOF (or holdout) scores from `models/`.                                                                                                                                                                                                           |


**Flow:** `data/` → `features/` → `models/` → `fairness/`

## Conventions

- All repo text in English unless explicitly told otherwise.
- Prefer small, importable modules over notebook-style scripts.
- Configuration (paths, seeds, fold count, feature lists) belongs in `[configs/](../configs/)`, not hard-coded throughout packages.
- Legacy Featuretools / importance scripts at repo root (`fe.py`, `feature_importance.py`) are reference only — new work goes here under `src/`.



## Status

- `src/data/` and `src/features/` implement the minimal feature-engineering
  pipeline (load → clean → aggregate → encode).
- `src/models/` and `src/fairness/` remain TODO.
- Run features: `python -m src.features.pipeline --data-dir data --artifacts-dir artifacts/features`
- Run tests: `pytest tests/`

See [docs/thesis-overview.md](../docs/thesis-overview.md) and [docs/experiments.md](../docs/experiments.md).

## Related docs

- [Thesis overview](../docs/thesis-overview.md)
- [Decisions](../docs/decisions.md)
- [EDA findings](../docs/EDA.md)
- [Glossary](../docs/glossary.md)

