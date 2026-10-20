# EDA notebooks

Exploratory analysis of the Home Credit Default Risk dataset. Notebooks are separate from feature engineering and credit-scoring code.

Consolidated findings: [docs/EDA.md](../../docs/EDA.md).

## Notebooks


| Notebook                             | Focus                                                           |
| ------------------------------------ | --------------------------------------------------------------- |
| `00_dataset_overview.ipynb`          | Inventory, grains, join keys, coverage, train/test alignment    |
| `01_application_eda.ipynb`           | Current applications (`application_train` / `application_test`) |
| `02_bureau_eda.ipynb`                | External bureau credits + monthly balances                      |
| `03_previous_applications_eda.ipynb` | Previous Home Credit applications                               |
| `04_installments_eda.ipynb`          | Installment payment history                                     |
| `05_pos_cash_eda.ipynb`              | POS / cash monthly balances                                     |
| `06_credit_card_eda.ipynb`           | Credit-card monthly balances                                    |
| `07_protected_attributes.ipynb`      | Gender, age, gender × 10-year age bands, proxy candidates       |
| `08_pre_modeling_bias_audit.ipynb`   | Pre-modelling bias audit (Phase 1, Step 5): outcome gaps with intervals, stratified gaps, missingness and history coverage by group, proxies and joint predictability of gender |

`08_pre_modeling_bias_audit.ipynb` also needs the stage-1 feature artifacts in `artifacts/features/` (`python -m src.features.pipeline`). Its statistics and the auxiliary gender classifier live in [analysis/helpers/data_audit.py](../helpers/data_audit.py).


## Out of scope

- Credit scoring model training
- Null-importance feature selection
- Fairness metrics on model predictions / mitigation (notebook `08` describes the labelled data only; its auxiliary classifier predicts gender, not default)

Protected attributes are `CODE_GENDER` and age (`DAYS_BIRTH`). They are **included** as model features and also kept raw for fairness grouping (gender × 10-year age bands); see [docs/decisions.md](../../docs/decisions.md).

## How to run

Place Home Credit CSVs in `data/` at the repo root. Run notebooks from the repo root or from this directory (each notebook resolves `data/` relative to the repo root).

Example:

```bash
jupyter nbconvert --to notebook --execute --inplace analysis/notebooks/00_dataset_overview.ipynb
```



### Column meanings

`data/HomeCredit_columns_description.csv`

## Related docs

- [Thesis overview](../../docs/thesis-overview.md)
- [Glossary](../../docs/glossary.md)
- [EDA findings](../../docs/EDA.md)

