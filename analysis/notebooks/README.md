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
| `07_protected_attributes.ipynb`      | Protected attributes and proxy candidates (audit only)          |




## Out of scope

- Credit scoring model training
- Null-importance feature selection
- Fairness metrics / mitigation

Protected attributes (`CODE_GENDER`, `NAME_EDUCATION_TYPE`) are for audit and grouping only, not model features.

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

