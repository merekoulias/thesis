# EDA: Way of work & Findings

Consolidated findings from Home Credit exploratory analysis notebooks. Source notebooks live in `analysis/notebooks/`.

---

## Way of work

EDA was run **table by table**, separate from credit-scoring implementation:


| Notebook                             | Tables                                            |
| ------------------------------------ | ------------------------------------------------- |
| `00_dataset_overview.ipynb`          | All Home Credit CSVs (inventory, joins, coverage) |
| `01_application_eda.ipynb`           | `application_train.csv`, `application_test.csv`   |
| `02_bureau_eda.ipynb`                | `bureau.csv`, `bureau_balance.csv`                |
| `03_previous_applications_eda.ipynb` | `previous_application.csv`                        |
| `04_installments_eda.ipynb`          | `installments_payments.csv`                       |
| `05_pos_cash_eda.ipynb`              | `POS_CASH_balance.csv`                            |
| `06_credit_card_eda.ipynb`           | `credit_card_balance.csv`                         |
| `07_protected_attributes.ipynb`      | Protected attributes + proxies (`application_*`)  |


**Method (each notebook):**

1. State purpose and out-of-scope items (no model training, no null-importance selection, no fairness metrics).
2. Load data with repo-relative paths; report shape, memory, grain, and keys.
3. Schema, missingness, cardinality, numeric ranges, and key integrity.
4. Domain-specific checks (coverage/joins, statuses, anomalies, payment/DPD behavior).
5. A few labeled plots; interpretation in markdown after evidence.
6. FE / join / leakage notes; bullet summary of findings.

**Conventions carried into FE:**

- Join keys: `SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`.
- Protected attributes (`CODE_GENDER`, age from `DAYS_BIRTH`) are **included** in model features (gender one-hot encoded, `DAYS_BIRTH` numeric) and also retained raw in `fairness_groups_*`, with `AGE_BAND` and `GENDER_AGE` labels, for audit. See [decisions.md](decisions.md).
- Day/month fields are relative to the current application date.
- One-to-many child tables must be aggregated to `SK_ID_CURR` (often via `SK_ID_PREV` / `SK_ID_BUREAU` first).
- Column meanings: `data/HomeCredit_columns_description.csv`.

Protected-attribute deep dive is in `07_protected_attributes.ipynb`.

---



## 00 — Dataset overview

**Scope:** Full relational map of Home Credit CSVs before table-specific EDA.

### Findings

- **Tables loaded:** 9 CSVs spanning 58.9M total rows. Largest: `bureau_balance` (27.3M), `installments_payments` (13.6M), `POS_CASH_balance` (10.0M). Application tables: **307,511 train** and **48,744 test** clients.
- **Relational pattern:** `application_*` → `bureau` / `previous_application` (one-to-many via `SK_ID_CURR`); `bureau` → `bureau_balance` (via `SK_ID_BUREAU`); `previous_application` → installments, POS/cash, and credit card (one-to-many via `SK_ID_PREV`).
- **Coverage (train clients):** Highest — `installments_payments` (94.8%), `previous_application` (94.7%), `POS_CASH_balance` (94.1%). Lowest — `credit_card_balance` (28.3%), `bureau_balance` via bureau (30.0%). Direct `bureau` coverage: 85.7%.
- **Multiplicity:** Covered clients typically have several bureau/previous-app records (median 4) and many installment/POS rows (median 22–25 per client).
- **Train vs test:** 121 shared columns; **only** `TARGET` **is train-only**. Schemas align for feature engineering.



### Short summary

The dataset is a clean relational graph around current applications. Most auxiliary tables cover >85% of train clients except credit-card history and bureau monthly balances. FE must aggregate one-to-many children; train/test feature schemas match aside from `TARGET`.

---



## 01 — Application EDA

**Scope:** Current loan applications (`application_train` / `application_test`).

### Findings

- **Train / test:** 307,511 × 122 and 48,744 × 121; 121 shared columns; only `TARGET` is train-only. `SK_ID_CURR` has no nulls or duplicates in either split.
- **TARGET imbalance:** 8.1% defaults (`TARGET=1`) vs 91.9% non-defaults — minority positive class.
- **Missingness:** 41 train columns (mostly building `_AVG`/`_MEDI`/`_MODE` and `OWN_CAR_AGE`) have ≥50% missing; imputation or exclusion decisions belong in later stage.
- **DAYS_EMPLOYED anomaly:** Sentinel `365243` (values >50 years employment); 55,374 train rows (18.0%) and 9,274 test rows (19.0%); anomaly default rate 5.4% vs 8.7% for non-anomaly rows. Those values must be replaced sentinel with `NaN`.
- **Correlations with TARGET:** Strongest negative: `EXT_SOURCE_3` (−0.18), `EXT_SOURCE_2` (−0.16), `EXT_SOURCE_1` (−0.16). Strongest positive among top features: `DAYS_BIRTH` (+0.08). Exploratory only.
- **Building redundancy:** 14 attribute groups have all three `_AVG`/`_MEDI`/`_MODE` variants; mean pairwise |r| often >0.98 — keep one variant per group in FE.
- **Fairness audit inputs:** `CODE_GENDER` (4 `XNA` rows, excluded before training and evaluation) and age from `DAYS_BIRTH`, plus proxy candidates (`NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_INCOME_TYPE`, `REGION_*`, income/housing) for `07_protected_attributes.ipynb`.



### Short summary

Application tables are ID-clean and train/test aligned. Main FE issues are class imbalance, high-missing building/car-age fields, the `DAYS_EMPLOYED` sentinel, redundant building suffixes, and retaining protected attributes in `X` while also keeping raw groups for fairness audit.

---



## 02 — Bureau EDA

**Scope:** External credit bureau credits and monthly status (`bureau`, `bureau_balance`).

### Findings

- **Grain / keys:** `bureau` is 1,716,428 credits × 17 columns; `SK_ID_BUREAU` is unique with no nulls. `bureau_balance` is 27,299,925 monthly rows × 3 columns; unique on (`SK_ID_BUREAU`, `MONTHS_BALANCE`) with no nulls.
- **Missingness:** `AMT_ANNUITY` (71.5%) and `AMT_CREDIT_MAX_OVERDUE` (65.5%) exceed 50% missing. Closed-credit / limit fields also have substantial missingness. `bureau_balance` has zero missing values.
- **Coverage:** 85.7% of train and 86.8% of test clients have at least one bureau credit. Clients with no bureau history have a higher raw train default rate (10.1% vs 7.7%). Only 45% of bureau credits have monthly balance history; 95% of distinct balance credit IDs join back to `bureau`.
- **Status / type:** Credits are mostly Closed (63%) or Active (37%); Bad debt is tiny. Dominant types: Consumer credit and Credit card. Currency is almost entirely `currency 1`.
- **Overdue rarity:** `CREDIT_DAY_OVERDUE > 0` on 0.25% of bureau rows. In balance history, DPD statuses `1`–`5` are 1.3% of months but appear on 103k credits — useful after aggregation. Status codes: `C` closed, `0` no DPD, `X` unknown, `1`–`5` DPD buckets.
- **Anomalies:** A few `DAYS_*` values outside a 100-year window (notably `DAYS_CREDIT_ENDDATE`). Some negative `AMT_CREDIT_SUM_DEBT` / `AMT_CREDIT_SUM_LIMIT` values (credit-card rows; likely sign/encoding quirks) — clip or flag in FE.
- **Balance horizon:** `MONTHS_BALANCE` ranges from 0 to −96; median history length per credit with balances is 26 months.
- **FE implications:** Aggregate bureau to `SK_ID_CURR` (counts, active share, debt sums, max overdue). Aggregate `bureau_balance` to `SK_ID_BUREAU` first (late-month rate, recent `STATUS`), then to client. Add indicators for missing bureau history and missing balance history.
- **Leakage / fairness:** Day fields are application-relative. Bureau presence may relate to thin-file / socioeconomic factors — audit later; do not use as a protected-attribute model feature.



### Short summary

Bureau history is broad but incomplete for monthly balances. Aggregate credit-level then client-level features; treat high-missing amount fields, extreme day offsets, and negative debt/limit values carefully; keep missing-history indicators.

---



## 03 — Previous applications EDA

**Scope:** Prior Home Credit applications (`previous_application`).

### Findings

- **Grain / keys:** 1,670,214 rows × 37 columns. `SK_ID_PREV` is unique; `SK_ID_CURR` / `SK_ID_PREV` have no nulls.
- **Coverage:** 94.7% of train and 98.1% of test clients have previous Home Credit application history. Clients with previous history have a higher raw train default rate (8.2% vs 6.0%).
- **Applications per client:** Mean 4.9, median 4, max 77. FE should aggregate by `SK_ID_CURR`.
- **Missingness:** `RATE_INTEREST_PRIVILEGED` and `RATE_INTEREST_PRIMARY` (99.6%), `AMT_DOWN_PAYMENT` and `RATE_DOWN_PAYMENT` (53.6%). Several post-approval date/insurance fields are 40.3% missing.
- **Outcomes:** Approved (62.1%), Canceled (18.9%), Refused (17.4%), Unused offer (1.6%). Reject reasons are meaningful mainly for refused rows, while `XAP` is mostly not-applicable.
- **Products and amounts:** Cash loans and Consumer loans dominate. `AMT_CREDIT` differs from `AMT_APPLICATION` by design; final credit is below requested amount in 21.4% of rows. `AMT_APPLICATION == 0` in 23.5% of rows.
- **Operational anomalies:** `SELLERPLACE_AREA == -1` is common (45.7%) with a very large max — treat as coded/dirty. Many categoricals use `XNA` as a structural placeholder.
- **Temporal fields:** `DAYS_DECISION` ranges from −2922 to −1 (always before current application). Post-approval schedule columns often contain sentinel `365243`, especially `DAYS_FIRST_DRAWING` (56.0% of rows) — convert to missing/indicator in FE.
- **FE / leakage / fairness:** Aggregate to `SK_ID_CURR`; keep `SK_ID_PREV` for child tables. Outcome/reject/approval fields are historical but target-like — review before modeling. Product, channel, seller industry, rejection reason, and insurance may act as socioeconomic proxies.



### Short summary

Previous applications give dense Home Credit history. Aggregate outcomes, amounts, and recency to the client; handle interest/down-payment missingness, `XNA`/`XAP`, `SELLERPLACE_AREA` codes, and `365243` date sentinels; review approval/reject fields for leakage risk.

---



## 04 — Installments EDA

**Scope:** Scheduled and actual installment payments (`installments_payments`).

### Findings

- **Grain / keys:** 13,605,401 rows × 8 columns. IDs clean with no exact duplicate rows. Grain is a **payment event**: `SK_ID_PREV + NUM_INSTALMENT_NUMBER` is not unique because partial payments create multiple rows for the same scheduled installment.
- **Missingness:** Only `DAYS_ENTRY_PAYMENT` and `AMT_PAYMENT` incomplete (0.021%, 2,905 rows), always missing together (unpaid/unrecorded payments).
- **Coverage:** 94.8% of train and 98.4% of test clients. Clients with installment history have higher raw train default (8.2% vs 6.0%). 91% of installment `SK_ID_PREV` values join to `previous_application`; only 57% of previous applications have installment rows.
- **Volume:** Median 25 payment rows per client (mean 40, max 372); median 10 payment rows per previous application with installments.
- **Timing:** Early/on time 91.6%; late 8.4%; delays >30 days 0.3% and >90 days 0.1%.
- **Amounts:** Exact payments 89%; underpayment 9.5%; overpayment 1.3% — underpayment often from split partial payments.
- **Versions:** `NUM_INSTALMENT_VERSION` mostly `1` (62%) or `0` (30%; credit-card calendars per dictionary). Some previous applications change version over time.
- **FE implications:** Aggregate within `SK_ID_PREV + NUM_INSTALMENT_NUMBER` first (sum paid amount, max delay), then to `SK_ID_PREV` and `SK_ID_CURR`. Features: late rate, underpay rate, mean/max delay, installment counts, missing-payment indicator.
- **Leakage / fairness:** Day fields are application-relative historical payments. Payment behavior may correlate with socioeconomic factors for later audit.



### Short summary

Installments are the densest repayment signal. Almost no missingness except rare unpaid rows.

---



## 05 — POS / cash balance EDA

**Scope:** Monthly POS and cash loan snapshots (`POS_CASH_balance`).

### Findings

- **Grain / keys:** 10,001,358 rows × 8 columns. Unique on (`SK_ID_PREV`, `MONTHS_BALANCE`); IDs clean; no exact duplicates.
- **Missingness:** Only `CNT_INSTALMENT` and `CNT_INSTALMENT_FUTURE` (0.26%), usually missing together.
- **Coverage:** 94.1% of train and 98.1% of test clients. Clients with POS/cash history have higher raw train default (8.2% vs 6.7%). 97% of POS/cash `SK_ID_PREV` values join to `previous_application`; only 54% of previous applications have POS/cash rows.
- **History:** `MONTHS_BALANCE` from −96 to −1; median 10 monthly snapshots per previous application (mean 10.7, max 96).
- **Status:** Mostly `Active` (91.5%), then `Completed` (7.4%). Rare `Demand` months have much higher DPD rates (50%).
- **Term consistency:** Completed months almost always have `CNT_INSTALMENT_FUTURE == 0`. Remaining > term is rare (0.07%).
- **DPD:** `SK_DPD > 0` on 3.0% of months; `SK_DPD_DEF > 0` on 1.1%. Raw DPD without tolerated DPD is common among DPD months (low-amount tolerance in dictionary).
- **FE implications:** Aggregate months → `SK_ID_PREV` → `SK_ID_CURR`. Features: max/mean DPD, DPD-month rate, remaining-installment stats, completed-month rate, history length, freshest status, missing-history indicator.
- **Leakage / fairness:** Month and DPD fields are application-relative. Rare statuses (`Demand`, `Amortized debt`) are informative but handle carefully.



### Short summary

POS/cash gives clean monthly delinquency and term-progress history for most clients. Aggregate DPD and remaining-term features to previous apps then clients; keep missing-history indicators.

---



## 06 — Credit card balance EDA

**Scope:** Monthly credit-card account snapshots (`credit_card_balance`).

### Findings

- **Grain / keys:** 3,840,312 rows × 23 columns. Unique on (`SK_ID_PREV`, `MONTHS_BALANCE`); IDs clean; no exact duplicates.
- **Missingness:** Drawing-detail fields and `AMT_PAYMENT_CURRENT` missing 20% (as a block). `AMT_INST_MIN_REGULARITY` and `CNT_INSTALMENT_MATURE_CUM` missing together (8%). No column >50% missing.
- **Coverage:** Only 28.3% of train and 34.2% of test clients — much sparser than POS/installments/bureau. Clients with card history have slightly higher raw train default (8.7% vs 7.8%). 72% of card `SK_ID_PREV` values join to `previous_application`; 5.6% of all previous apps have card rows, but 48% of revolving previous apps do.
- **History:** `MONTHS_BALANCE` from −96 to −1; median 21 monthly snapshots per card (mean 37, max 96).
- **Status / amounts:** Mostly `Active` (96.3%). 56% of months have zero balance; 20% have zero credit limit. Utilization (limit > 0) has median near 0 but 11% of those months are over limit (`utilization > 1`).
- **Payments / drawings:** Drawing activity in 16% of months. Payment below minimum due 4% of comparable rows. Receivable fields nearly redundant (`AMT_RECIVABLE`  `AMT_TOTAL_RECEIVABLE`).
- **DPD:** `SK_DPD > 0` on 4.0% of months; `SK_DPD_DEF > 0` on 2.3%. `Demand` concentrates delinquency. Negative balances/receivables exist — clip or flag in FE.
- **FE implications:** Missing-card-history indicator is critical. Then utilization, max/mean DPD, drawing intensity, payment-below-minimum rate, history length. Keep one receivable field, not all three.
- **Leakage / fairness:** Month, balance, payment, and DPD are application-relative. Rare statuses such as `Demand` handle carefully. Utilization/delinquency may correlate with socioeconomic factors for later audit.



### Short summary

Credit-card history is informative but sparse. Prioritize availability indicators; then utilization, DPD, drawings, and under-min payments; drop redundant receivable columns.

---



## 07 — Protected attributes EDA

**Scope:** Gender, age, gender × 10-year age-band intersections, and proxy candidates on application tables (fairness audit; no model fairness metrics on predictions in EDA).

### Findings

- **Protected attributes:** `CODE_GENDER` and age (`DAYS_BIRTH`, in years `−DAYS_BIRTH / 365.25`), present in train and test. Both are model features (gender one-hot, `DAYS_BIRTH` numeric); raw values plus `AGE_BAND` and `GENDER_AGE` labels are kept in `fairness_groups_*` for grouping.
- **XNA:** 4 train rows with `CODE_GENDER == 'XNA'` (0.001%; 0 in test), all `TARGET = 0`. Excluded before training and evaluation as an **exclusion forced by the data**: no group metric can be estimated for 4 rows with no positives, and the rows are not redistributed to F/M because `XNA` is undefined. See [decisions.md](decisions.md).
- **Gender (train, no XNA):** 65.8% F / 34.2% M. Raw default rates: F 7.00%, M 10.14% (gap 3.14 pp). Descriptive outcome gap only — not a model fairness metric.
- **Age:** no missing values, no sentinel; ages 20.5–69.1 in train and 20.1–69.0 in test. Women are older (median 44.8 vs 40.3). Raw default rate falls with age: 11.44% (20–29), 9.59% (30–39), 7.64% (40–49), 6.12% (50–59), 4.92% (60–69); gap 6.53 pp.
- **Intersectionality:** 10 gender × 10-year age-band cells ([20, 30) … [60, 70)). Smallest: `M | 60-69` (7,607 rows, 486 positives, 7,121 negatives); largest: `F | 30-39` (50,119). No cell is sparse, so no minimum cell size is applied. Men have the higher base rate in every band; max−min across cells 8.01 pp (`M | 20-29` 12.52% vs `F | 60-69` 4.51%), wider than the gender-only or age-only gap.


| Age   | F: n   | F: base rate | M: n   | M: base rate |
| ----- | ------ | ------------ | ------ | ------------ |
| 20–29 | 25,649 | 10.62%       | 19,535 | 12.52%       |
| 30–39 | 50,119 | 8.71%        | 32,211 | 10.97%       |
| 40–49 | 49,636 | 6.67%        | 26,963 | 9.42%        |
| 50–59 | 49,350 | 5.10%        | 18,743 | 8.81%        |
| 60–69 | 27,694 | 4.51%        | 7,607  | 6.39%        |


- **Train vs test:** gender, age-band and intersectional shares differ by at most about 1.2 pp.
- **Proxy candidates:** for gender, `OCCUPATION_TYPE` (Cramér's V 0.51), `FLAG_OWN_CAR` (0.35), `ORGANIZATION_TYPE` (0.34); `NAME_EDUCATION_TYPE` is only weakly associated (0.03). For age, `ORGANIZATION_TYPE` (η 0.64) and `NAME_INCOME_TYPE` (η 0.62), both driven by pensioners (mean age 59.7, almost all with `ORGANIZATION_TYPE = XNA` and the `DAYS_EMPLOYED` sentinel); `EXT_SOURCE_1` (Spearman 0.60, 56% missing), `CNT_CHILDREN` (−0.37) and the `DAYS_*` tenure fields. Monitor in the fairness stage; no exclusion decided.
- **Not computed here:** demographic parity / equal opportunity / equalized odds on model predictions, thresholds, or mitigation.



### Short summary

Protected attributes are usable for audit grouping after excluding 4 `XNA` rows. Raw outcome gaps exist by gender and by age and widen for gender × age; all 10 cells are large. Occupation, organization and income type relate to both gender and age, so proxies need later audit; descriptive gaps motivate fairness work but are not model fairness metrics.

---



## 08 — Pre-modelling bias audit

**Scope:** Phase 1, Step 5. Group-level disparities in outcomes, representation, missingness and proxies in the labelled data, with 95% intervals, for gender, age band and the 10 gender × age cells. Repeats the Step 5 parts of `07` and uses the stage-1 artifacts in `artifacts/features/`. Helpers and the auxiliary gender classifier: `analysis/helpers/data_audit.py`.

### Findings

- **Sample:** 307,507 rows; groups match `fairness_groups_train.parquet` exactly. Women are 65.8% of applicants and older; their share rises from 56.8% (20–29) to 78.5% (60–69).
- **Outcomes:** default F 7.00% [6.89, 7.11] vs M 10.14% [9.96, 10.32]; gap F − M −3.14 pp [−3.36, −2.93]; ratio M / F 1.45; repayment ratio M / F 0.966. Age gap 20–29 − 60–69: 6.53 pp [6.16, 6.90]. Cells: max−min 8.01 pp [7.51, 8.56]; Lum et al.'s correction is negligible at this cell size.
- **Stratified:** the gender gap is negative in every age band (−1.9 to −3.7 pp) and every PTI quintile (−2.0 to −4.0 pp). Mantel-Haenszel odds ratio M vs F: 1.50 unstratified, 1.39 by age band, 1.53 by PTI quintile, 1.42 by both.
- **Missingness:** structural and uneven. No car (`OWN_CAR_AGE` missing) 77.8% F vs 43.2% M; `DAYS_EMPLOYED` sentinel / `ORGANIZATION_TYPE = XNA` 22.4% F vs 9.6% M, up to 85% in 60–69; `EXT_SOURCE_1` missing rises with age (37–81%); no bureau record for about 20% of 20–29. Missingness relates to default, and more strongly for men (e.g. no car: +4.2 pp M vs +1.5 pp F).
- **Proxies:** `EXT_SOURCE_1`, `DAYS_EMPLOYED`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE` and `NAME_INCOME_TYPE` are associated with gender or age and with `TARGET`; the strongest predictors (`EXT_SOURCE_2/3`) are weakly group-associated. Gender is predictable from the non-protected engineered features with OOF AUC 0.907 and balanced error rate 0.175, below Feldman et al.'s four-fifths threshold (0.384), so the data admits disparate impact.

### Short summary

The labelled data shows stable gender and age differences in default that are not explained by age composition or PTI, structural missingness concentrated in older applicants and women, and features that jointly encode gender. Observational data, a constructed label and selective labels mean these cannot be attributed to historical bias; they are the baseline for Step 6.

---



## Overall summary (for FE and thesis)



### Dataset structure

- One current application per `SK_ID_CURR`; rich one-to-many history via bureau and previous Home Credit products.
- Train/test application schemas align except `TARGET` (8.1% positive class).
- High coverage for previous apps, installments, and POS/cash; medium for bureau; low for credit cards and bureau monthly balances.



### Not yet covered

- Fairness metrics on OOF predictions and mitigation (pre / in / post), in `src/fairness/`. Feature engineering and the baseline LightGBM scorer are implemented in `src/` ([feature_engineering.md](feature_engineering.md), [credit_scoring.md](credit_scoring.md)).

