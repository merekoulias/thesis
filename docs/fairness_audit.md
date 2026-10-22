# Baseline fairness audit (Phase 1, Step 6)

Out-of-fold (OOF) fairness audit of the four credit scorers trained under the nested-CV plan. Implementation lives in [`src/fairness/`](../src/fairness/). Configuration: [`configs/fairness.yaml`](../configs/fairness.yaml). Methodological choices are recorded in [decisions.md](decisions.md) (rows linked in [Decision mapping](#decision-mapping)).

Mitigation (pre-, in-, post-processing) is **not** part of this package; it will reuse the same audit on `artifacts/runs/<model>/<variant>/` when implemented.

---

## Purpose

Produce reproducible tables, figures and Markdown reports that describe group-wise rates, pre-specified pairwise gaps with uncertainty, calibration and conditional demographic parity diagnostics, comparison with label-implied benchmarks from the data audit, and robustness across hyperparameter ranks 2–5. The code reports numbers and mechanical labels only; it does not draw normative conclusions (see [Scope of the numbers](#scope-of-the-numbers)).

---

## Inputs

| Input | Path (defaults from [`configs/paths.yaml`](../configs/paths.yaml)) | Role |
| ----- | ---------------------------------------------------------------- | ---- |
| OOF predictions | `artifacts/runs/<model>/<variant>/oof_predictions.parquet` | `SK_ID_CURR`, `TARGET`, `y_pred_proba`, `fold` |
| Robustness OOF (ranks 2–5) | `artifacts/runs/<model>/<variant>/robustness/rank_{2..5}/oof_predictions.parquet` | Same schema, per outer-fold candidate rank |
| Fairness groups | `artifacts/features/fairness_groups_train.parquet` | Raw `CODE_GENDER`, `DAYS_BIRTH`, `AGE_BAND`, `GENDER_AGE` |
| Payment-to-income | `data/application_train.csv` (`SK_ID_CURR`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL`) | PTI for conditional demographic parity only (not from `X`) |
| Run metrics | `artifacts/runs/<model>/<variant>/metrics.json` | Pooled OOF ROC AUC, Brier, log loss references for `utility.json` |
| Feature importance | `artifacts/runs/<model>/<variant>/feature_importance.parquet` | Proxy watch list (`mlp` has no file) |

`io.load_audit_frame()` joins OOF (row order preserved), groups and PTI into the audit frame (see [Audit frame](#audit-frame)).

Models audited: `logreg`, `lgbm`, `mlp`, `rf` (registry names in config).

---

## Audit frame

`io.load_audit_frame()` returns one row per audited applicant (307,507 on the full data), **in the row order of the OOF file**:

| Column | Type | Source |
| ------ | ---- | ------ |
| `SK_ID_CURR` | int64 | OOF |
| `y` | int8 0/1 | OOF `TARGET` |
| `score` | float64 in [0, 1] | OOF `y_pred_proba` |
| `fold` | int | OOF `fold` |
| `CODE_GENDER` | str (F/M) | `fairness_groups_train.parquet` |
| `AGE_BAND` | str | idem |
| `GENDER_AGE` | str | idem |
| `PTI` | float64, NaN if unknown | raw CSV: `AMT_ANNUITY / AMT_INCOME_TOTAL` (NaN if income ≤ 0) |
| `PTI_Q` | str `Q1`…`Q5`, NaN if PTI unknown | `pd.qcut(PTI[known], q=5, labels=["Q1", …, "Q5"])` |

Checks (each failure raises `ValueError` naming the file):

1. OOF has exactly `SK_ID_CURR, TARGET, y_pred_proba, fold`; unique `SK_ID_CURR`; no NaN; `TARGET` in {0, 1}; scores in [0, 1].
2. Every OOF `SK_ID_CURR` is found in the groups file; the join neither drops nor duplicates OOF rows (the groups file may contain extra ids).
3. No NaN in the group columns, and every label is one listed in `groupings` of the config. Groups with no rows get NaN rates, never an error.
4. PTI: `data/application_train.csv` is read with `usecols=["SK_ID_CURR", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]` and left-joined. Quintiles are computed **only over the audited rows with known PTI** (307,495 rows; 12 missing). The conditional-parity step (`conditional.py`) compares the cut points and the missing count with `expected_cut_points` (rounded to 3 decimals: 0.104 / 0.144 / 0.186 / 0.247) and `expected_missing` (12) from the config and logs a warning on a mismatch.
5. `GENDER_AGE == CODE_GENDER + " | " + AGE_BAND` for every row.

---

## Coding and threshold

Aligned with [Rate orientation](decisions.md) and [Decision threshold](decisions.md):

- `Y = 1` (`TARGET`) = payment difficulties; `Ŷ = 1` = flagged (credit refused); `Ŷ = 0` = approved.
- Selection metrics use **approval rates** (`Ŷ = 0`). Equal opportunity in this coding is **equal FPR** (wrongful refusal among negatives).
- One **group-blind** threshold per model and per robustness rank:  
  `t = np.quantile(score, 1 − base_rate)` with `base_rate = mean(TARGET)` over audited rows (24,825 / 307,507 ≈ 8.07%).  
  `Ŷ = 1` when `score > t`. On full data this flags exactly 24,825 applicants per model/rank.

---

## Groups and pairs

[Protected attributes and intersectionality](decisions.md), [Age bands](decisions.md), [No minimum cell size](decisions.md):

| Grouping | Levels | Count |
| -------- | ------ | ----- |
| `CODE_GENDER` | F, M | 2 |
| `AGE_BAND` | 20-29 … 60-69 (10-year bands) | 5 |
| `GENDER_AGE` | F \| 20-29 … M \| 60-69 | 10 |

17 groups plus an `ALL` summary row in outputs. Counts are always reported; empty groups get NaN rates.

**Pairwise comparisons** (signed gaps **a − b** as in `configs/fairness.yaml`):

- F − M overall and within each age band (5 bands).
- 20-29 − 60-69 overall and within F and within M.

**Primary comparisons** ([Multiple comparisons](decisions.md)): only **F − M** × **fpr** (EOD) and **F − M** × **fnr** (ΔFNR); no multiplicity correction elsewhere.

**Conditional demographic parity** ([Conditional demographic parity](decisions.md)): pairs F − M and 20-29 − 60-69 within PTI quintiles; 12 rows with unknown PTI excluded from that diagnostic only.

---

## Metric definitions

Rates are defined once in `group_metrics.METRIC_SPECS` ([`src/fairness/group_metrics.py`](../src/fairness/group_metrics.py)). For rows `S` in a group:

| Metric | Subpopulation | Event | Denominator |
| ------ | ------------- | ----- | ----------- |
| `approval_rate` | all | `Ŷ = 0` | `n` |
| `fpr` | `Y = 0` | `Ŷ = 1` | `negatives` |
| `fnr` | `Y = 1` | `Ŷ = 0` | `positives` |
| `tpr` | `Y = 1` | `Ŷ = 1` | `positives` |
| `ppv` | `Ŷ = 1` | `Y = 1` | `flagged` |
| `default_among_approved` | `Ŷ = 0` | `Y = 1` | `approved` |

Denominator 0 → rate and plug-in interval NaN.

**Score-based per-group metrics** (not in `METRIC_SPECS`): `auc` (within-group percentile bootstrap), `brier`, `log_loss` (mean with plug-in SE). Pooled `ALL` AUC must match the run's `oof_roc_auc` in `metrics.json`.

[Fairness metric set](decisions.md) and [Utility metrics](decisions.md) describe how these appear in reporting. Kozodoi et al.'s SP = ½(|ΔFPR| + |ΔFNR|) is **not** computed in the audit (reserved for mitigation trade-off plots).

---

## Intervals and tests

[Fairness uncertainty method](decisions.md), [Baseline fairness audit procedure](decisions.md):

All functions live in [`src/fairness/stats.py`](../src/fairness/stats.py). `z = scipy.stats.norm.ppf(1 − (1 − confidence) / 2)` with `confidence` from the config (0.95 → `z` = 1.959964); `alpha = 1 − confidence`.

| Quantity | Function | Formula |
| -------- | -------- | ------- |
| Group rate | `proportion_ci(events, n, z)` | `r = events / n`; `se = sqrt(r(1 − r) / n)`; interval `(max(0, r − z·se), min(1, r + z·se))` (plug-in, clipped to [0, 1]). `n` is the metric's own denominator ([Metric definitions](#metric-definitions)); `n = 0` → NaN |
| Difference of two rates | `diff_ci(r_a, n_a, r_b, n_b, z)` | `d = r_a − r_b`; `se = sqrt(r_a(1 − r_a)/n_a + r_b(1 − r_b)/n_b)`; interval `d ± z·se` (Wald, not clipped) |
| Disparate impact ratio (approval rate only) | `ratio_ci(r_a, n_a, r_b, n_b, z)` | `q = AR_a / AR_b` (orientation a / b of the pair id); `se_log = sqrt((1 − r_a)/(n_a·r_a) + (1 − r_b)/(n_b·r_b))`; interval `(q·exp(−z·se_log), q·exp(z·se_log))`; NaN if a rate is 0. `gaps.csv` also gives `ratio_lower_over_higher = min(q, 1/q)` (point only), and the reports compare it with `dir_reference_line` (0.8, a reference line, not a legal threshold) |
| Independence test | `g_test(table)` | Hurlin et al.'s likelihood-ratio test, own implementation. Drop all-zero rows and columns of the `k × c` count table; `E = row_tot ⊗ col_tot / N`; `G = 2 Σ O·ln(O/E)` over cells with `O > 0`; `df = (k' − 1)(c' − 1)` on the reduced table; `p = chi2.sf(G, df)`; `df = 0` → `G = 0`, `p` NaN. Equals `scipy.stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")` on tables without zero rows or columns |
| Stratified test | `stratified_g_test(tables)` | `G` and `df` summed over the strata (strata with `df = 0` add nothing); `p = chi2.sf(G_sum, df_sum)` |
| Per-group Brier score and log loss | `mean_ci(values, z)` | mean of the row contributions `(score − y)²` and `−[y ln s + (1 − y) ln(1 − s)]` with `s` clipped to `[1e-15, 1 − 1e-15]`; `se = std(ddof=1) / sqrt(n)`; interval `m ± z·se` |
| Per-group AUC | `bootstrap_auc_ci(y, score, n_boot, seed, alpha)` | point = `roc_auc_score(y, score)`; `n_boot` within-group resamples `idx = rng.integers(0, n, n)`; resamples with one class are NaN and skipped; interval = `np.nanpercentile` at `100·alpha/2` and `100·(1 − alpha/2)`. Empty or one-class group → NaN |
| Worst-group summary over the 10 cells | `worst_group_summary(n, rate, n_boot, seed, alpha)` | Lum et al. (2022). (1) `max − min` of the cell rates with a percentile interval from the within-cell binomial bootstrap `rng.binomial(n, rate, size=(n_boot, k)) / n`. (2) Between-cell variance: uncorrected `var(rate, ddof=1)`; single-corrected = uncorrected − mean of `rate(1 − rate)/n` (floored at 0); double-corrected per bootstrap replicate = `var(y*) − mean(2y*(1 − y*)/n − y*(1 − y*)/n²)` (floored at 0), reported as its percentile interval. Square roots give the matching SDs. `max − min` is biased upwards; the double-corrected SD interval is the check |
| Seeds | `seed_for(base_seed, key)` | `(base_seed · 1,000,003 + zlib.crc32(key)) mod 2³²` with keys `auc\|<model>\|<rank>\|<grouping>\|<group>` and `worst\|<model>\|<rank>\|<metric>`; a new `np.random.default_rng(seed)` per call, so reruns are bit-identical |

Count tables used by the G-tests (all p-values are reported after the gaps and intervals):

| Test | Table |
| ---- | ----- |
| Pairwise gap (`gaps.csv`) | 2 × 2: rows a, b; columns events, non-events **within the metric's subpopulation** (e.g. FPR: refused / approved among negatives); `df = 1` |
| Across the 10 cells (`cell_summary.csv`) | 10 × 2 in the same way; `df = 9` when all cells are non-empty |
| Calibration sufficiency (`calibration_tests.csv`) | per decile bin: groups × (`y = 1`, `y = 0`), stratified over the 10 bins; `df = Σ over bins of (groups present − 1)` (10 / 40 / 90 for gender / age band / cells) |
| Conditional demographic parity (`conditional_parity_tests.csv`) | per PTI quintile: 2 × 2 (a, b) × (approved, refused), stratified over the 5 quintiles; `df = 5` |

[Calibration risk classes](decisions.md): 10 equal-frequency bins of the **pooled** OOF score of each model (edges `np.quantile(score, linspace(0, 1, 11))`, bins `[e_i, e_{i+1})` with the maximum score in the last bin, the same edges for every group); per bin and group the observed default rate with its plug-in interval, the mean score and `calibration_gap = observed − mean score`.

**Comparison with the data** (`comparison.py`):

- Oracle (label-implied) benchmark: approval rate `1 − base_rate` per pair side; `oracle_dir = (1 − p_a) / (1 − p_b)` (F − M ≈ 1.035). `direction` is a mechanical label of the model DIR relative to the oracle DIR around 1: `reversed` (opposite side of 1), `equal`, `wider` or `narrower`; `oracle_in_model_ci` says whether the oracle DIR lies in the model DIR interval.
- Expected signs (Chouldechova 2017): the side with the higher base rate is expected to have the higher FPR and the lower FNR, so `expected_sign_eod = sign(p_a − p_b)` and `expected_sign_dfnr = −expected_sign_eod`; the observed gaps are compared with them.
- Kendall's τ-b (`scipy.stats.kendalltau`) between the cells' base rates and FPRs (Wang et al. §6.1 translated to `Y = 1` = payment difficulties); 10 cells, so low power.
- Proxy watch list: saved importances averaged over folds per feature (`gain` for `lgbm`, `impurity` for `rf`, `|coef|` for `logreg`; none for `mlp`), each feature assigned to the first matching regex group of `proxy_watch_list` (else `other`); per group the share of total importance, the best feature and its rank among all features.

**Robustness ranks 2–5** (`robustness.py`): for each rank its own base-rate threshold, pooled ROC AUC and the F − M gaps of `robustness_metrics` (fpr, fnr, approval_rate) with Wald intervals; no bootstrap.

---

## Procedure

```text
artifacts/runs/<m>/<variant>/[robustness/rank_r/]oof_predictions.parquet ─┐
artifacts/features/fairness_groups_train.parquet                           ├─ io.load_audit_frame()
data/application_train.csv (PTI columns)                                   ─┘
        │
        ▼  AuditFrame: SK_ID_CURR, y, score, fold, CODE_GENDER, AGE_BAND, GENDER_AGE, PTI, PTI_Q
thresholds.base_rate_threshold()  →  Ŷ
        │
        ├─ group_metrics: counts, rates + CIs, AUC bootstrap, Brier/log loss, utility
        ├─ gaps: 9 pairs × gap metrics; cell summaries (Lum)
        ├─ calibration: deciles × groups, sufficiency G-tests
        ├─ conditional: PTI quintiles × pairs, stratified G-tests, cell × quintile table
        ├─ comparison: oracle benchmark, expected signs, Kendall τ, proxy importance
        └─ robustness: ranks 2–5 primary gaps (each rank's own threshold)
        ▼
report.write_outputs()  →  artifacts/fairness/<m>/<variant>/
summary.build_summary() →  artifacts/fairness/summary_<variant>/
```

Modules: see [`src/fairness/`](../src/fairness/) and the plan file `.cursor/plans/fairness-audit.md` section 4.3.

---

## Outputs

Per model: `artifacts/fairness/<model>/<variant>/` ([`configs/paths.yaml`](../configs/paths.yaml) `fairness_dir`).

| File | Columns / keys |
| ---- | -------------- |
| `audit_config.json` | resolved config (`seed`, `n_boot`, `confidence`, groupings, pairs, …), `model`, `models`, `variant`, `input_paths` (repo-relative), `git_commit`, `package_versions`, `started_at`, `finished_at` |
| `threshold.json` | `threshold`, `base_rate`, `n_rows`, `n_flagged`, `n_positives` |
| `utility.json` | `roc_auc`, `fold_roc_auc_mean`, `fold_roc_auc_std`, `fold_roc_auc_min`, `fold_roc_auc_max`, `brier`, `brier_reference`, `log_loss`, `log_loss_reference` (from the run's `metrics.json`), `tpr`, `ppv`, `f1` (at `t`; equal when `n_flagged == n_positives`), `n_flagged` |
| `group_counts.csv` | `grouping, group, n, positives, negatives, flagged, approved, base_rate` (17 groups + one `ALL / all` row) |
| `group_metrics.csv` | long: `grouping, group, metric, value, ci_low, ci_high, denominator` for `approval_rate, fpr, fnr, tpr, ppv, default_among_approved, auc, brier, log_loss` |
| `gaps.csv` | long: `pair_id, metric, rate_a, rate_b, den_a, den_b, diff, diff_ci_low, diff_ci_high, g_stat, df, p_value, ratio, ratio_ci_low, ratio_ci_high, ratio_lower_over_higher, primary` (ratio columns only for `approval_rate`, else NaN; `primary` true only for `F-M` × `fpr` / `fnr`) |
| `cell_summary.csv` | `metric, k_groups, max_min_gap, max_min_ci_low, max_min_ci_high, max_cell, min_cell, var_uncorrected, var_single_corrected, var_double_corrected_ci_low, var_double_corrected_ci_high, sd_uncorrected, sd_single_corrected, sd_double_corrected_ci_low, sd_double_corrected_ci_high, g_stat, df, p_value` |
| `calibration.csv` | `grouping, group, bin, n, events, observed_rate, ci_low, ci_high, mean_score, calibration_gap` (grouping `ALL` included) |
| `calibration_tests.csv` | `grouping, g_stat, df, p_value` |
| `bin_edges.json` | `edges` (the 11 decile edges) |
| `conditional_parity.csv` | `pair_id, quintile, n_a, n_b, approval_rate_a, approval_rate_b, diff, diff_ci_low, diff_ci_high` |
| `conditional_parity_tests.csv` | `pair_id, g_stat, df, p_value, unconditional_diff, unconditional_diff_ci_low, unconditional_diff_ci_high, n_excluded_missing_pti` (the unconditional gap uses all rows) |
| `conditional_parity_cells.csv` | appendix: `group, quintile, n, approved, approval_rate, ci_low, ci_high` for the 10 cells |
| `pti_quintiles.json` | `cut_points` (4), `n_missing` |
| `oracle_benchmark.csv` | `pair_id, oracle_approval_a, oracle_approval_b, oracle_diff, oracle_dir, model_diff, model_dir, model_dir_ci_low, model_dir_ci_high, direction, oracle_in_model_ci` |
| `expected_signs.csv` | `pair_id, base_rate_a, base_rate_b, expected_sign_eod, observed_eod, eod_ci_excludes_0, eod_matches, expected_sign_dfnr, observed_dfnr, dfnr_ci_excludes_0, dfnr_matches` |
| `kendall.json` | `tau`, `p_value`, `n_cells` (cells used), `cells` (base rate and FPR per cell) |
| `proxy_importance.csv` | `feature_group, n_features, importance_share, best_feature, best_feature_rank, importance_type` (absent for `mlp`) |
| `robustness.csv` | `rank, threshold, n_flagged, roc_auc, metric, diff, diff_ci_low, diff_ci_high` (ranks 1–5; rank 1 repeated for reference) |
| `figures/calibration_CODE_GENDER.png`, `figures/calibration_AGE_BAND.png` | observed default rate vs mean score per decile bin, one series per group, 95% error bars, diagonal y = x |
| `figures/cells_error_rates.png` | FPR and FNR of the 10 cells with 95% error bars and the `ALL` value as a dashed line |
| `report.md` | twelve sections: inputs and threshold, utility, primary comparisons, group tables (gender and age bands; cells), pairwise gaps, worst-group summaries, calibration, conditional demographic parity, comparison with the data, robustness ranks, scope of the numbers |
| `audit.log` | one INFO line per step with its duration |

Cross-model `artifacts/fairness/summary_<variant>/`: `utility.csv` (`model, roc_auc, brier, log_loss, tpr, n_flagged`), `primary_gaps.csv` (`model, rank, metric, diff, diff_ci_low, diff_ci_high, p_value`), `figures/primary_gaps.png` (F − M EOD and ΔFNR per model; rank 1 with its interval, ranks 2–5 as hollow markers), `report.md`.

CSV files are written with `index=False`, UTF-8 and unrounded floats; rounding happens only in the Markdown reports (rates and gaps in percentage points with 2 decimals, ratios 3 decimals, AUC 4 decimals, p-values `f"{p:.2e}"`, and `< 1e-300` for a p-value that underflows to 0, which the CSV keeps as `0.0`).

---

## How to run

From repo root with `.venv` activated (after features and model runs exist):

```bash
# Full baseline audit (four models, n_boot=500 from config; ~15 min total)
python -m src.fairness.audit --all --overwrite

# Single model
python -m src.fairness.audit --model lgbm --overwrite

# Smoke / faster bootstrap count
python -m src.fairness.audit --model lgbm --n-boot 20 --out-dir artifacts/fairness_smoke --overwrite

# Cross-model summary (reads existing per-model dirs)
python -m src.fairness.summary

# Optional paths
python -m src.fairness.audit --help
python -m src.fairness.summary --help
```

CLI defaults: `--config configs/fairness.yaml`, `--runs-dir artifacts/runs`, `--features-dir artifacts/features`, `--data-dir data`, `--out-dir artifacts/fairness`.

Tests: `pytest tests/test_fairness_*.py` (synthetic data only).

---

## Results (baseline, rank 1 primary gaps)

Audit run: **2026-10-23** (`python -m src.fairness.audit --all --overwrite`, then `python -m src.fairness.summary`). Full tables: `artifacts/fairness/<model>/baseline/` and `artifacts/fairness/summary_baseline/primary_gaps.csv`.

F − M, rank 1 — EOD (FPR gap, percentage points) and ΔFNR (percentage points). 95% Wald CIs on the difference; p from G-test. Ranks 2–5: min–max of the point estimate across robustness candidates.

| Model | EOD (pp) | 95% CI | p | Ranks 2–5 range (pp) | ΔFNR (pp) | 95% CI | p | Ranks 2–5 range (pp) |
| ----- | -------- | ------ | --- | -------------------- | --------- | ------ | --- | -------------------- |
| logreg | −4.88 | [−5.09, −4.67] | < 1e-300 | −4.91 to −4.86 | 11.22 | [10.05, 12.39] | 2.51e-79 | 11.11 to 11.28 |
| lgbm | −4.43 | [−4.63, −4.22] | < 1e-300 | −4.50 to −4.29 | 10.04 | [8.85, 11.22] | 2.16e-62 | 9.69 to 10.23 |
| mlp | −4.46 | [−4.67, −4.26] | < 1e-300 | −4.74 to −4.19 | 9.56 | [8.40, 10.73] | 1.88e-58 | 8.96 to 10.25 |
| rf | −2.96 | [−3.16, −2.76] | 5.21e-200 | −3.15 to −2.95 | 4.78 | [3.62, 5.93] | 3.61e-16 | 4.53 to 5.19 |

Negative EOD means women's FPR is lower than men's (F − M). Positive ΔFNR means women's FNR is higher than men's.

Data-level base rates (identical across models; from `group_counts.csv`): F 7.00%, M 10.14%, M \| 60-69 6.39% (486 positives / 7,121 negatives).

---

## Scope of the numbers

Fixed paragraph in each per-model `report.md` (section 12), matching `report.scope_text()`:

Intervals condition on the trained model and the decision threshold; labels are selective (only granted loans have an observed `TARGET`); `TARGET` is a proxy for payment difficulties; criteria are observational; p-values are reported after gaps and intervals and are not multiplicity-corrected; only the F-M equal-opportunity difference (EOD, the FPR gap) and the F-M ΔFNR are pre-specified primary comparisons.

---

## Decision mapping

| Topic | decisions.md row |
| ----- | ---------------- |
| Procedure and tests | Baseline fairness audit procedure |
| Metrics and harm | Fairness metric set; Prioritised fairness harm |
| Threshold and coding | Decision threshold; Rate orientation |
| Uncertainty | Fairness uncertainty method |
| Primary vs descriptive | Multiple comparisons |
| CDP | Conditional demographic parity; PTI not a model feature |
| Calibration | Calibration risk classes |
| Cells and τ | Intersectional metric aggregation; Age bands |
| Utility | Utility metrics; Probability quality metrics |
| Robustness ranks | Hyperparameter tuning (candidates 2–5) |
| Mitigation (future) | Mitigation target; Equalized-odds summary (SP for Pareto only) |

---

## Related docs

- [Credit scoring](credit_scoring.md) — OOF inputs
- [Feature engineering](feature_engineering.md) — `fairness_groups_train.parquet`
- [Thesis overview](thesis-overview.md) — workflow stage
- [EDA](EDA.md) — pre-modeling data audit (notebook `08`)
