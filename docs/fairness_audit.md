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

| Quantity | Method |
| -------- | ------ |
| Group rate | Plug-in binomial 95% CI with clipping to [0, 1] |
| Difference of two rates | Wald CI for independent proportions (not clipped) |
| Disparate impact ratio (approval) | Log-scale delta-method CI; `min(q, 1/q)` reported |
| Pairwise / cell / calibration / CDP independence | Hurlin et al. likelihood-ratio **G-test** (`scipy` log-likelihood χ²); p-values after gaps and intervals |
| Worst-group (10 cells) | Max − min with Lum et al. double-corrected variance and within-cell bootstrap (B from config) |
| Per-group AUC | Within-group percentile bootstrap (B from config) |

`z` comes from `confidence` in config (0.95 → 1.959964). Seeds: `seed_for(base_seed, key)` in [`src/fairness/stats.py`](../src/fairness/stats.py).

[Calibration risk classes](decisions.md): 10 equal-frequency bins of the **pooled** OOF score; sufficiency G-test across bins per grouping.

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

| File | Contents |
| ---- | -------- |
| `audit_config.json` | Resolved config, paths, git commit, versions, timestamps |
| `threshold.json` | Base rate, `t`, counts flagged |
| `utility.json` | ROC AUC, fold AUC spread, Brier/log loss vs references, TPR/PPV/F1 at `t` |
| `group_counts.csv` | Per group: `n`, positives, negatives, flagged, approved, base_rate |
| `group_metrics.csv` | Long table of rates and score metrics with CIs |
| `gaps.csv` | All pairwise gaps; `primary` flag on F − M × fpr/fnr |
| `cell_summary.csv` | Worst-group summaries over 10 cells |
| `calibration.csv`, `calibration_tests.csv`, `bin_edges.json` | Decile calibration and sufficiency tests |
| `conditional_parity*.csv`, `pti_quintiles.json` | CDP tables and PTI cut points |
| `oracle_benchmark.csv`, `expected_signs.csv`, `kendall.json`, `proxy_importance.csv` | Data comparison (`mlp`: no proxy file) |
| `robustness.csv` | Ranks 1–5 × primary metrics for F − M |
| `figures/*.png` | Calibration by gender/age, cell FPR/FNR |
| `report.md` | Twelve sections (inputs through scope) |

Cross-model: `artifacts/fairness/summary_<variant>/` — `utility.csv`, `primary_gaps.csv`, `figures/primary_gaps.png`, `report.md`.

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
