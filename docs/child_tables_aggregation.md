# Child-table aggregation

Detailed description of how Home Credit relational (child) tables are compressed into **one summary row per applicant** and joined onto the application spine. Parent overview: [feature_engineering.md](feature_engineering.md).

Implementation: `[src/features/pipeline.py](../src/features/pipeline.py)` plus per-domain modules under `[src/features/](../src/features/)`.

---

## Why aggregate?

Raw Home Credit data is **relational**. A supervised credit model needs a flat table: one row → one prediction.


| Table                   | What one row means                       | Typical multiplicity    |
| ----------------------- | ---------------------------------------- | ----------------------- |
| `application_*`         | Current loan application (the applicant) | 1 per `SK_ID_CURR`      |
| `bureau`                | One external credit                      | Many per applicant      |
| `bureau_balance`        | One month of one bureau credit           | Many per `SK_ID_BUREAU` |
| `previous_application`  | One earlier Home Credit application      | Many per applicant      |
| `installments_payments` | One payment event                        | Many per previous app   |
| `POS_CASH_balance`      | One month of a POS/cash loan             | Many per previous app   |
| `credit_card_balance`   | One month of a card account              | Many per previous app   |


If the 100th applicant has 4 bureau credits and 25 installment rows, those raw rows cannot be fed as “the applicant.” They must be **summarized** into numbers attached to `SK_ID_CURR`.

Join keys: `SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU` (see [EDA.md](EDA.md) §00).

---



## High-level procedure

```text
application (1 row / applicant)
        │
        │  left join on SK_ID_CURR
        │
        ├─ bureau aggregates        (bureau_balance → bureau → client)
        ├─ previous aggregates      (previous → client)
        ├─ installment aggregates   (payment events → scheduled inst. → client)
        ├─ POS/cash aggregates      (months → previous → client)
        └─ credit-card aggregates   (months → previous → client)
                │
                ▼
        wide matrix: still 1 row per applicant
```

In code:

1. Clean application base (`clean_application`).
2. Aggregate each child domain to client level once (`_aggregate_all_children`).
3. Left-join those summaries onto train and test application frames (`_build_client_matrix`).
4. Continue with missing indicators and encoding ([encoding.md](encoding.md)).

**Left join** means every applicant stays. If they have no credit-card history, card columns are missing and `HAS_CREDIT_CARD_HISTORY` is set to `0`. Sparse domains do not delete applicants.

---



## What “aggregate” means

Take many rows that share an ID and collapse them with summary statistics (`count`, `sum`, `mean`, `min`, `max`, rates/shares).

Example — three previous loans for applicant `100`:


| SK_ID_CURR | SK_ID_PREV | AMT_CREDIT | STATUS   |
| ---------- | ---------- | ---------- | -------- |
| 100        | A          | 1000       | Approved |
| 100        | B          | 2000       | Refused  |
| 100        | C          | 1500       | Approved |


After aggregation (illustrative):


| SK_ID_CURR | PREV_APP_COUNT | AMT_CREDIT mean | Approved count | Refused count |
| ---------- | -------------- | --------------- | -------------- | ------------- |
| 100        | 3              | 1500            | 2              | 1             |


Those are **new columns about history**, not new applicant rows.

---



## Design approach for which columns to create

Column choice is **manual and EDA-guided**, not Featuretools DFS and not “every column × every aggregation.”

Rules:

1. Prefer interpretable **credit-risk signals** (delinquency, utilization, obligations, outcomes, recency, coverage).
2. Use standard aggregation verbs and rates/shares.
3. Always keep a **history-presence flag** when coverage is incomplete (`HAS_*_HISTORY`).
4. Encode child categoricals as **one-hot then count** per applicant (see [encoding.md](encoding.md)).
5. Clean dirty values first (sentinels, negatives, coded `-1`), then aggregate.
6. Skip or soften redundant / near-empty fields called out in EDA.

Guided by FE implications in [EDA.md](EDA.md) §§02–06.

---



## Domain-by-domain procedure



### Bureau + bureau balance

**Modules:** `[src/features/bureau.py](../src/features/bureau.py)`

1. Aggregate `bureau_balance` months → one summary per `SK_ID_BUREAU`
  (history length, DPD-status rates, latest month/status).
2. Join onto `bureau`.
3. Aggregate all bureau credits → one summary per `SK_ID_CURR`
  (credit count, active/closed shares, debt/credit/annuity sums and means, overdue stats, balance-history signal).
4. Clip/flag anomalous negative debt/limit values.
5. One-hot + count categoricals such as `CREDIT_ACTIVE`, `CREDIT_TYPE`.
6. Set `HAS_BUREAU_HISTORY = 1` on rows with bureau credits.



### Previous applications

**Module:** `[src/features/previous.py](../src/features/previous.py)`

1. Replace schedule-date sentinel `365243` with missing + anomaly flags.
2. Treat `SELLERPLACE_AREA == -1` as missing/coded.
3. For very high-missing interest fields, rely mainly on missing indicators.
4. Aggregate by `SK_ID_CURR`: application count, amount sums/means, `DAYS_DECISION` recency, credit/application ratio, outcome/product/channel counts.
5. Set `HAS_PREV_APP_HISTORY = 1`.



### Installments

**Module:** `[src/features/installments.py](../src/features/installments.py)`

Payment events are not unique on `(SK_ID_PREV, NUM_INSTALMENT_NUMBER)` because of partial payments. Two steps:

1. Collapse to scheduled installment: sum paid, max delay, underpayment flag.
2. Aggregate to `SK_ID_CURR`: installment count, late rate, underpay rate, mean/max delay, payment-ratio stats, missing-payment rate.
3. Set `HAS_INSTALLMENT_HISTORY = 1`.



### POS / cash balance

**Module:** `[src/features/pos_cash.py](../src/features/pos_cash.py)`

1. Aggregate months → `SK_ID_PREV` (DPD stats, completed/active rates, remaining installments, history length, latest status).
2. Aggregate previous apps → `SK_ID_CURR`.
3. Set `HAS_POS_CASH_HISTORY = 1`.



### Credit card balance

**Module:** `[src/features/credit_card.py](../src/features/credit_card.py)`

Coverage is sparse (~28% of train clients), so the history flag is especially important.

1. Derive utilization (where limit > 0), over-limit, drawing activity, payment-below-minimum.
2. Keep one receivable field family (`AMT_TOTAL_RECEIVABLE`), not all redundant variants.
3. Aggregate months → previous → client: utilization, DPD, drawings, history length.
4. Set `HAS_CREDIT_CARD_HISTORY = 1`.

---



## How tables are combined

Each child aggregator returns:

- key: `SK_ID_CURR`
- many numeric summary columns
- a history flag (`HAS_*_HISTORY = 1`)

Then:

```text
application
  LEFT JOIN bureau_feats      USING (SK_ID_CURR)
  LEFT JOIN previous_feats    USING (SK_ID_CURR)
  LEFT JOIN installment_feats USING (SK_ID_CURR)
  LEFT JOIN pos_feats         USING (SK_ID_CURR)
  LEFT JOIN credit_card_feats USING (SK_ID_CURR)
```

Missing history flags are filled with `0` after the join.

---



## What the joined matrix looks like



### Shape

- **One row per applicant** (same applicants as cleaned application train/test)
- Wide: application columns + all child summary columns side by side
- Not a stacked multi-row history table



### Origin of columns

There is **no** single column like `SOURCE=bureau`. Origin is encoded in **name prefixes**:


| Prefix / pattern        | Origin                             |
| ----------------------- | ---------------------------------- |
| (mostly original names) | Application                        |
| `BUREAU_…`, `BB_…`      | Bureau / bureau balance            |
| `PREV_…`                | Previous applications              |
| `INST_…`                | Installments                       |
| `POS_…`                 | POS / cash                         |
| `CC_…`                  | Credit card                        |
| `HAS_*_HISTORY`         | Coverage / presence of that domain |




### Tiny picture

Before:

```text
Applicant 100
  application: 1 row
  bureau: 4 rows
  previous: 3 rows
  installments: 40 rows
  pos: 20 rows
  credit card: 0 rows
```

After join (before final encoding):

```text
Applicant 100 → 1 wide row
  [app fields…] [bureau summaries…] [prev…] [inst…] [pos…]
  [cc columns empty + HAS_CREDIT_CARD_HISTORY = 0]
```

---



## What is not produced here

- Raw child-table rows in the modeling matrix
- Deep feature synthesis / automatic primitive explosion
- Importance-based pruning of aggregates (optional later, after modeling)

Next step after this join: [encoding.md](encoding.md).