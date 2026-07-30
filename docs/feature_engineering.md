# Feature engineering

Minimal Home Credit feature-engineering pipeline implemented under `[src/](../src/)`. EDA lives in `[analysis/notebooks/](../analysis/notebooks/)`; findings that drive FE choices are summarized in [EDA.md](EDA.md).

**Flow:** `src/data/` → `src/features/` → `src/models/` → `src/fairness/`

---

## Goal

Produce a **model-ready applicant matrix** for the thesis credit scorer:

- **One row per applicant** (`SK_ID_CURR`), aligned with `TARGET`
- **Numeric columns only** after encoding
- Sensible handling of missing values and anomalies
- Multi-table history compressed into client-level features
- Categorical encoding suitable for LightGBM

Protected attributes (`CODE_GENDER`, `NAME_EDUCATION_TYPE`) are **included** in model features ([decisions.md](decisions.md)). Raw values are also retained for fairness grouping.

---



## Non-goals (for this phase)

- Featuretools / deep feature synthesis
- Null-importance or automated feature selection
- Fairness metrics or mitigation
- LightGBM training

---



## Pipeline overview

```text
Load & validate CSVs (drop CODE_GENDER == XNA)
        │
        ▼
Clean application spine (sentinels, building dedup, split y)
        │
        ├─► fairness_groups (raw gender, education, intersection)
        │
        ▼
Aggregate child tables → one summary row per SK_ID_CURR
        │
        ▼
Left-join summaries onto application (still 1 row per applicant)
        │
        ▼
Missing indicators + median impute + one-hot encode
        │
        ▼
X_train, y_train, X_test, ids_*, fairness_groups_*
```


| Stage             | Module(s)                                                                                   | Role                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Load / validate   | `[src/data/loader.py](../src/data/loader.py)`, `[validation.py](../src/data/validation.py)` | Read Home Credit CSVs; check keys; drop `XNA` gender                      |
| Application clean | `[src/features/application.py](../src/features/application.py)`                             | Sentinel handling, building-column dedup, extract `y` and fairness groups |
| Child aggregates  | `bureau.py`, `previous.py`, `installments.py`, `pos_cash.py`, `credit_card.py`              | Compress one-to-many history to client level                              |
| Orchestration     | `[src/features/pipeline.py](../src/features/pipeline.py)`                                   | Join, encode, save artifacts                                              |
| Encoding          | `[src/features/encoding.py](../src/features/encoding.py)`                                   | Missing indicators, median impute, one-hot                                |


Detailed write-ups:

- **Child-table aggregation:** [child_tables_aggregation.md](child_tables_aggregation.md)
- **Encoding:** [encoding.md](encoding.md)

---



## Data contract (outputs)


| Output                   | Description                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------ |
| `X_train`                | One row per training applicant; numeric; includes encoded protected attributes       |
| `y_train`                | `TARGET`, aligned to `X_train`                                                       |
| `X_test`                 | Same columns as `X_train`, one row per test applicant                                |
| `ids_train` / `ids_test` | `SK_ID_CURR` for traceability                                                        |
| `fairness_groups_train`  | Raw `CODE_GENDER`, `NAME_EDUCATION_TYPE`, and `GENDER_EDUCATION` intersection labels |
| Metadata                 | Feature names, source numeric/categorical columns, cleaning counts                   |


Artifacts are written under `artifacts/features/` (gitignored), not under `data/`.

---



## Application cleaning (spine)

The application tables are already one row per applicant. Cleaning before joins:

1. `DAYS_EMPLOYED` **sentinel** — values equal to `365243` (~1000 years) are not real tenure. Replace with `NaN` and add `DAYS_EMPLOYED_ANOM` so unemployment / N-A coding remains visible after imputation. See [EDA.md](EDA.md) §01.
2. **Building redundancy** — where `_AVG` / `_MEDI` / `_MODE` triples exist with near-duplicate correlation, keep `_AVG` and drop matching `_MEDI` / `_MODE`. Standalone categoricals such as `HOUSETYPE_MODE` are retained.
3. `TARGET` — moved to `y_train` only; never a feature.
4. **Protected attributes** — kept in the feature frame for encoding into `X`; also copied raw into fairness groups.

High-missing application fields (e.g. building averages, `OWN_CAR_AGE`) are **not dropped by default**. Missingness is handled later via indicators + median imputation ([encoding.md](encoding.md)).

---



## Child-table aggregation (summary)

Relational tables are one-to-many. They cannot be stacked as extra rows for the classifier. Each domain is aggregated to **one summary row per** `SK_ID_CURR`, then **left-joined** onto the application spine.

- Applicants without history stay in the matrix; history flags such as `HAS_CREDIT_CARD_HISTORY` become `0`.
- Origin of features is visible via column prefixes (`BUREAU_`, `PREV_`, `INST_`, `POS_`, `CC_`), not a separate source column.

**Full procedure, per-domain feature choices, and join logic:** [child_tables_aggregation.md](child_tables_aggregation.md).

---



## Encoding (summary)

After the wide client matrix is built:

1. Add binary missing indicators for high-missing numerics (and key application fields).
2. Median-impute numerics (fit on train only).
3. One-hot encode **all** nominal categoricals, including binary ones (no label encoding).
4. Fit on train; transform test with `handle_unknown="ignore"`.

Child-table categoricals are often one-hot’d **before** aggregation and stored as per-applicant counts.

**Full encoding policy and examples:** [encoding.md](encoding.md).

---



## Final dataset shape

O**ne new row per applicant**.

Each row is a wide numeric vector: cleaned application fields + history summaries + missing/history flags + one-hot categoricals. Train and test share the same feature columns. `TARGET` is only in `y_train`.

---



## How to run

```bash
# From repo root, with .venv activated
pytest tests/

python -m src.features.pipeline \
  --data-dir data \
  --artifacts-dir artifacts/features
```

Config defaults: `[configs/paths.yaml](../configs/paths.yaml)`.

---



## Related docs

- [Child-table aggregation](child_tables_aggregation.md)
- [Encoding](encoding.md)
- [EDA findings](EDA.md)
- [Glossary](glossary.md)

