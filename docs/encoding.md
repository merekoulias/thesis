# Encoding

Detailed description of how the joined applicant matrix becomes a **fully numeric, model-ready** `X_train` / `X_test`. Parent overview: [feature_engineering.md](feature_engineering.md). Aggregation that precedes encoding: [child_tables_aggregation.md](child_tables_aggregation.md).

Implementation: `[src/features/encoding.py](../src/features/encoding.py)`, orchestrated by `[src/features/pipeline.py](../src/features/pipeline.py)`.

---

## Goal

After aggregation, each applicant is one row, but columns are still mixed:

- Numbers with missing values (income, debt sums, delays, …)
- Strings / categoricals (`CODE_GENDER`, `NAME_EDUCATION_TYPE`, contract types, …)
- Already-numeric flags (`DAYS_EMPLOYED_ANOM`, `HAS_BUREAU_HISTORY`, aggregated counts, …)

This pipeline builds a **generic numeric matrix**: median-imputed numerics + one-hot categoricals, with train/test column alignment. That matrix is what `src/models/` will consume later.

Protected attributes are encoded into `X_train` and `X_test` (fit on train only). Raw gender / education / intersection labels are kept separately in `fairness_groups_*` for fairness evaluation.

---



## Where encoding sits

```text
clean application
  → aggregate child tables  (some categoricals already one-hot + count)
  → left-join to one row per applicant
  → add missing indicators
  → FeatureEncoder: fit on train / transform test
  → X_train, X_test (all numeric)
```

Encoding has **two layers**:

1. Early encoding inside child-table aggregations
2. Final train-fitted `FeatureEncoder` on the applicant matrix

---



## Layer 1 — Encoding inside aggregations

Child tables have many rows per applicant and categorical fields (e.g. `CREDIT_ACTIVE`, `NAME_CONTRACT_STATUS`).

Approach:

1. One-hot the categorical at row level (`pd.get_dummies`)
2. Sum those dummy columns per `SK_ID_CURR` (counts)

Example:


| SK_ID_CURR | CREDIT_ACTIVE |
| ---------- | ------------- |
| 100        | Active        |
| 100        | Closed        |
| 100        | Active        |


Becomes applicant-level counts:


| SK_ID_CURR | …_Active | …_Closed |
| ---------- | -------- | -------- |
| 100        | 2        | 1        |


For history tables, encoding therefore produces **count features**, already numeric when joined onto the application spine. See [child_tables_aggregation.md](child_tables_aggregation.md).

---



## Layer 2 — Final applicant-matrix encoding

Class: `FeatureEncoder` in `[encoding.py](../src/features/encoding.py)`.

### A. Split columns by type


| Type        | Rule                             | Examples                                               |
| ----------- | -------------------------------- | ------------------------------------------------------ |
| Categorical | string / object / category dtype | `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `HOUSETYPE_MODE` |
| Numeric     | everything else                  | amounts, days, rates, flags, aggregated counts         |


`SK_ID_CURR` is excluded from `X` (stored as `ids_train` / `ids_test`).

### B. Missing indicators (before imputation)

For selected high-missing **numeric** columns, create a **new binary column** next to the original:

- `OWN_CAR_AGE_MISSING = 1` if originally missing, else `0`

Selection:

- Any numeric with train missing rate ≥ 50%, and always when present:
  - `OWN_CAR_AGE`
  - `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`

Purpose in plain words:

- Keep the original column (useful when present; later filled if missing)
- After median imputation, “was missing?” would otherwise disappear
- The `*_MISSING` flag preserves that signal

High-missing columns are **retained**, not dropped by default. Missingness itself can be informative in credit data.

### C. Numeric pipeline

- `SimpleImputer(strategy="median")`
- Medians learned from **train only**, then applied to test
- Columns that are entirely missing in train are kept and filled safely (`keep_empty_features=True`)

No standard scaling: not required for LightGBM tree models.

### D. Categorical pipeline (one-hot for all)


| Choice              | Policy                                                     | Why                                            |
| ------------------- | ---------------------------------------------------------- | ---------------------------------------------- |
| Encoding type       | One-hot for **all** nominal categoricals, including binary | Avoid fake order from label encoding           |
| Label encoding      | **Not used**                                               | Would make arbitrary integers look ordinal     |
| Missing categories  | Fill with token `__MISSING__`, then one-hot                | Missing becomes an explicit category           |
| Unknown test levels | `OneHotEncoder(handle_unknown="ignore")`                   | Unseen category → zeros for that feature group |
| Dummy drop          | Keep all levels (`drop=None`)                              | Fine for trees; simpler to audit               |
| Fit scope           | Fit on train only; transform test                          | No leakage from test categories                |


Example for `CODE_GENDER`:


| CODE_GENDER | …_F | …_M |
| ----------- | --- | --- |
| F           | 1   | 0   |
| M           | 0   | 1   |


Sklearn names features with prefixes such as:

- `num__AMT_INCOME_TOTAL`
- `cat__CODE_GENDER_F`
- `cat__NAME_EDUCATION_TYPE_Higher education`

---



## Train vs test discipline

Anti-leakage rules:

1. Fit the encoder on training features only
2. Transform test with that fitted encoder
3. Enforce the same feature names and column order in `X_train` and `X_test`

Also:

- Categorical missing token applied consistently
- Unknown test categories do not create new columns
- Numeric medians come from train only

---



## What the final encoded dataset looks like

- **One row per applicant**
- **All columns numeric**
- No raw string categoricals in `X`
- No `TARGET` in `X` (`y_train` only)
- Train and test share the same feature columns
- Encoded protected attributes present in `X`
- Raw protected attributes outside `X` in `fairness_groups_*`

Conceptually:

```text
X row =
  [imputed application numerics + missing flags]
+ [one-hot application categoricals]
+ [already-numeric child aggregates / count-dummies]
+ [history flags]
```

---



## Comparison to common alternatives


| Approach                                                   | Used here? | Notes                                                            |
| ---------------------------------------------------------- | ---------- | ---------------------------------------------------------------- |
| One-hot all nominals (incl. binary)                        | **Yes**    | Default thesis baseline                                          |
| Label-encode binary + one-hot multi-level (Koehrsen-style) | No         | Label encoding avoided to prevent fake ordinality                |
| LightGBM native categorical dtype                          | No         | Explicit numeric matrix preferred for clarity / fairness tooling |
| Target / WOE encoding                                      | No         | Out of scope for minimal FE                                      |
| Featuretools automatic encoding                            | No         | Explicit non-goal                                                |


---



## Why this encoding fits the thesis

1. Clear and auditable for the methods chapter
2. Safe with tree models (no false ordinal structure)
3. Compatible with fairness work (protected attrs in `X`; raw groups retained)
4. Leakage-aware (fit on train only)
5. Minimal but complete before modeling

Related: [feature_engineering.md](feature_engineering.md), [child_tables_aggregation.md](child_tables_aggregation.md), [EDA.md](EDA.md).