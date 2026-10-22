# Thesis overview

## Subject

Study bias in machine-learning credit scoring and evaluate bias mitigation techniques within a theoretically grounded framework. The thesis covers pre-processing, in-processing, and post-processing approaches, with emphasis on fairness across groups and intersectional subgroups.

Deliverables: **code repository**, **written thesis**, and **presentation**. Internal validation only — a Kaggle leaderboard submission is not a goal.

## Research questions

1. **Bias detection:** To what extent does the credit scoring model exhibit fairness disparities across gender and gender × age (10-year age bands), measured by demographic parity, equal opportunity, and equalized odds?
2. **Mitigation:** How do pre-, in-, and post-processing mitigation techniques affect fairness metrics relative to predictive performance? Which category of techniques (pre-, in-, post-) do better?
3. **Intersectionality:** Repeat the mitigation step fro intersectional groups.



# Out of scope

1. **Performance** is nor a research quetion neither a task. Better performance of the creadit scoring model is a nice to have and not a mandatory target.



## Dataset

**Home Credit Default Risk** — [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk).

- Single empirical dataset for this thesis (no additional benchmark datasets planned in code).
- Borrower-level credit default prediction; relational tables mirror real bank credit-scoring pipelines.
- Detailed table descriptions and schema relationships: `[table_descriptions](../analysis/notebooks/00_dataset_overview.ipynb)` and `[table_relationships](../home_credit.png)`.
- Column annotations: `data/HomeCredit_columns_description.csv`.
- EDA note on `_AVG` / `_MEDI` / `_MODE` features: `[glossary](../docs/glossary.md)`.

**Target:** `TARGET` — 1 if the client had payment difficulties on early installments, 0 otherwise.

**Protected attributes (included in model features and used for fairness grouping):**


| Attribute | Column        | Notes                                                                   |
| --------- | ------------- | ----------------------------------------------------------------------- |
| Gender    | `CODE_GENDER` | Mandatory. Drop `XNA` rows (4 in training set). One-hot encoded in `X`. |
| Age       | `DAYS_BIRTH`  | Used with gender for intersectionality. One-hot encoded in `X`.         |


Also monitor proxy features (region, occupation, income type) that may encode protected-attribute information.

## Workflow (EDA → feature engineering → credit scoring → fairness)


| Stage               | Location              | Status | Role                                                                                                                                       |
| ------------------- | --------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| EDA                 | `analysis/notebooks/` | Done   | Notebooks per table/domain; scripts/modules for shared helpers and heavy jobs. See notebook list in [decisions.md](decisions.md).          |
| Feature engineering | `src/features/`       | Done   | Load → clean → aggregate → encode to model-ready matrices. See [feature_engineering.md](feature_engineering.md).                           |
| Credit scoring      | `src/models/`         | TODO   | LightGBM 5-fold stratified CV with OOF predictions. See [credit_scoring.md](credit_scoring.md). More ML credit scoring models to be added. |
| Fairness            | `src/fairness/`       | Baseline audit done (Step 6); mitigation TODO | OOF audit, tables and reports; see [fairness_audit.md](fairness_audit.md).                                                                 |


**Not in scope:** model performance as a research question. Better predictive performance is nice-to-have, not a thesis target.

## Repository layout

```
thesis/
├── data/                          # Home Credit CSVs (gitignored)
├── analysis/
│   └── notebooks/                 # EDA notebooks (see decisions.md)
├── src/
│   ├── data/                      # load CSVs, validate keys, drop XNA, splits
│   ├── features/                  # aggregates + application cleaning → client matrix
│   ├── models/                    # LightGBM CV, save OOF preds + metrics
│   └── fairness/                  # metrics + mitigation (thesis core; add when ready)
├── configs/                       # paths, model params, protected-attr names
├── tests/                         # light tests on joins / shapes / model contracts
└── docs/                          # Project documentation
```



## Related documentation

- [Glossary](glossary.md) — terms and identifiers

