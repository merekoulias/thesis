# Glossary

Terms, identifiers, and conventions used in this project.

---

## Dataset identifiers

| Term | Column / file | Description |
|------|---------------|-------------|
| Client ID | `SK_ID_CURR` | Primary key in `application_{train,test}.csv`; links to bureau and previous applications |
| Bureau credit ID | `SK_ID_BUREAU` | Primary key in `bureau.csv`; links to `bureau_balance.csv` |
| Previous application ID | `SK_ID_PREV` | Primary key in `previous_application.csv`; links to installments, POS/cash, credit card tables |
| Target | `TARGET` | Binary default label (train only): 1 = payment difficulties, 0 = otherwise |

## Application tables

| File | Granularity |
|------|-------------|
| `application_train.csv` | One row per loan application (with `TARGET`) |
| `application_test.csv` | One row per loan application (no `TARGET`) |
| `bureau.csv` | One row per external credit record per client |
| `bureau_balance.csv` | One row per month per bureau credit |
| `previous_application.csv` | One row per prior Home Credit application |
| `installments_payments.csv` | One row per installment payment (or missed payment) |
| `POS_CASH_balance.csv` | One row per month per POS/cash loan |
| `credit_card_balance.csv` | One row per month per credit card |

See [`dataset-desc.md`](../dataset-desc.md) for full descriptions.

## Protected and sensitive attributes

| Term | Column | Role in this project |
|------|--------|----------------------|
| Gender | `CODE_GENDER` | Mandatory protected attribute (`M`, `F`; the 4 `XNA` rows are excluded) |
| Age | `DAYS_BIRTH` | Protected attribute; age in years is `−DAYS_BIRTH / 365.25` |
| Age band | `AGE_BAND` | 10-year band, left-closed: `20-29`, `30-39`, `40-49`, `50-59`, `60-69` |
| Intersectional group | `GENDER_AGE` | Gender × age band (10 cells); subgroup for disaggregated fairness metrics |

Protected attributes are **included** as model features (gender one-hot, `DAYS_BIRTH` numeric) and also kept raw in `fairness_groups_*` for grouping and mitigation. Education (`NAME_EDUCATION_TYPE`) is an ordinary model feature and a proxy candidate, not a protected attribute.

## Feature suffix conventions

From Kaggle discussions ([`dataset-annotation-meanings.md`](../dataset-annotation-meanings.md)):

| Suffix | Use when |
|--------|----------|
| `_AVG` | Feature distribution is approximately normal |
| `_MEDI` | Feature distribution is skewed |
| `_MODE` | Feature is discrete (e.g. counts) |

`_AVG`, `_MEDI`, and `_MODE` variants of the same building attribute are highly correlated — typically keep only one.

## Script roles

| Location | Role | Thesis model? |
|----------|------|---------------|
| `analysis/notebooks/` | EDA — explore and document the data (one notebook per domain) | No |
| `analysis/scripts/` or modules | Shared helpers and heavy EDA jobs | No |

## Modeling terms

| Term | Meaning in this repo |
|------|----------------------|
| OOF predictions | Out-of-fold predicted probabilities from cross-validation; used for unbiased fairness evaluation |
| Null importances | Feature importance computed on shuffled targets; features beating the null distribution are retained |
| DFS | Deep Feature Synthesis (Featuretools) |

## Fairness metrics (planned)

| Metric | Abbreviation | What it compares across groups |
|--------|--------------|-------------------------------|
| Demographic parity | DP | Approval rate, `P(Ŷ = 0 \| group)` |
| Equal opportunity | EO | Approval rate among applicants without payment difficulties (`Y = 0`), i.e. equal FPR across groups when `Ŷ = 1` means flagged |
| Equalized odds | EOd | Both TPR and FPR across groups |
| Worst-group disparity | — | Maximum fairness gap (max−min) over the gender × age-band cells |

Conventions: `Y = 1` = payment difficulties; `Ŷ = 1` = flagged as high risk, `Ŷ = 0` = credit granted. Formulas use flags; selection rates are reported as approval rates. Each model gets one group-blind threshold that flags the base-rate share of applications (8.07%). See [decisions.md](decisions.md).

## Output paths

| Path | Contents |
|------|----------|
| `data/` | Raw Home Credit CSVs |

## Abbreviations

| Abbrev | Expansion |
|--------|-----------|
| EDA | Exploratory data analysis |
| FE | Feature engineering |
| CV | Cross-validation |
| AUC | Area under the ROC curve |
| LGBM | LightGBM gradient boosting classifier |
