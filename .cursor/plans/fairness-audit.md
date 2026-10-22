# Plan: baseline fairness audit of the four credit scorers (Phase 1, Step 6 — audit half)

> **Executor:** Cursor Composer. **Reviewer:** Claude Code (Opus).
> Work through the phases **in order**, each started with the execution prompt in section 6. Do not start phase N+1 until every exit criterion of phase N is met **and** the phase review (section 7) has returned `PASS` or `PASS WITH NITS`.
> If something in this plan is ambiguous or contradicts the code, **stop and ask**. Do not guess.

---

## Progress log (executor fills this in)

**BASE commit** (HEAD of `main` before Phase 0; written in Phase 0): `a9eac5a`

Phase commits are found with `git log --oneline --grep "fairness: phase <N>"`.

| Phase | Title | Status (`todo` / `in progress` / `done`) | Review verdict | Notes |
| ----- | ----- | ---------------------------------------- | -------------- | ----- |
| 0 | Preconditions and baseline snapshot | done | | BASE `a9eac5a`; `git status` clean at start; pytest 154 passed (38 s); black/isort OK; scipy 1.18.0 matplotlib 3.11.0; models plan Phase 7 `done` PASS WITH NITS; inputs: 4× baseline OOF+metrics (+ feature_importance except mlp), 16× robustness OOF rank_2–5, `fairness_groups_train.parquet` 307507 rows, `data/application_train.csv` present. |
| 1 | Config, types and audit input loading | todo | | |
| 2 | Statistics core | todo | | |
| 3 | Threshold, group table and utility | todo | | |
| 4 | Pairwise gaps and cell summaries | todo | | |
| 5 | Calibration, outcome test and conditional demographic parity | todo | | |
| 6 | Comparison with the data and robustness ranks | todo | | |
| 7 | Audit runner, outputs, report, figures, CLI, smoke run | todo | | |
| 8 | Full audit of the four models | todo | | |
| 9 | Documentation | todo | | |

---

## 1. Goal and scope

Implement `src/fairness/` and run the **baseline fairness audit** of Phase 1, Step 6 on the out-of-fold (OOF) predictions of the four credit scorers already trained by `.cursor/plans/multi-model-nested-cv.md`:

| Registry name | Model | OOF input |
| ------------- | ----- | --------- |
| `logreg` | Logistic regression | `artifacts/runs/logreg/baseline/oof_predictions.parquet` |
| `lgbm` | LightGBM | `artifacts/runs/lgbm/baseline/oof_predictions.parquet` |
| `mlp` | MLP | `artifacts/runs/mlp/baseline/oof_predictions.parquet` |
| `rf` | Random forest | `artifacts/runs/rf/baseline/oof_predictions.parquet` |

Plus the robustness runs `artifacts/runs/<model>/baseline/robustness/rank_{2..5}/oof_predictions.parquet`.

The audit is written so that a later **mitigation** plan can run it unchanged on any run directory with the same OOF schema (`artifacts/runs/<model>/<variant>/`).

### In scope

- `src/fairness/` package, `configs/fairness.yaml`, tests `tests/test_fairness_*.py`.
- Per model: group-blind base-rate threshold; utility; group table (17 groups) with intervals; 9 pre-specified pairwise gaps with intervals and likelihood-ratio p-values; worst-group summaries over the 10 cells (Lum et al. corrections); calibration by group with sufficiency tests; outcome test; conditional demographic parity on PTI quintiles; comparison with the data (label-implied benchmark, expected signs, Kendall's τ, proxy watch list); robustness ranks 2–5.
- Written outputs (CSV/JSON), Markdown reports and PNG figures per model, and a cross-model summary.
- Docs (Phase 9).

### Out of scope (do NOT implement)

- **Any mitigation method** (reweighing, ExponentiatedGradient, ThresholdOptimizer, group thresholds, …) → separate plan.
- **Retraining or changing any model**, changing `src/models/`, `configs/models/`, `artifacts/runs/` (read-only inputs).
- **Feature changes** (`src/features/`, `artifacts/features/` are read-only).
- **Fairlearn / AIF360 / any new dependency.**
- **Interpretation** (Step 7): the code produces numbers, tables and figures; it does not write conclusions. The report may state mechanical facts ("interval excludes 0", "wider than the label-implied gap") but no normative judgement.
- Editing `docs/decisions.md` or `docs/open-questions.md` (the reviewer maintains them).
- Notebooks (a presentation notebook may be added later; not in this plan).

---

## 2. Decisions this plan implements (authoritative text in `docs/decisions.md`)

Rows: "Rate orientation", "Decision threshold", "Fairness metric set", "Prioritised fairness harm", "Conditional demographic parity", "Equalized-odds summary", "Utility metrics", "Fairness uncertainty method", "Multiple comparisons", "Calibration risk classes", "Intersectional metric aggregation", "No minimum cell size", "Age bands", "Probability quality metrics", "Baseline fairness audit procedure".

1. **Coding.** `Y = 1` = payment difficulties (`TARGET`). `Ŷ = 1` = flagged = credit refused; `Ŷ = 0` = approved. Selection metrics are reported as **approval rates**. Equal opportunity = equal **FPR**.
2. **Threshold.** One group-blind threshold per model and per rank: `t = np.quantile(scores, 1 − base_rate)`, `Ŷ = 1{score > t}`, base rate = mean of `TARGET` over the audited rows (24,825 / 307,507 = 8.07%).
3. **Groups.** Gender (F, M), age band (20-29 … 60-69), and the 10 gender × age cells (`GENDER_AGE`, e.g. `F | 20-29`). 17 groups in total. No minimum cell size; counts always reported.
4. **Primary comparisons (fixed in advance):** F − M **EOD** (FPR gap) and F − M **ΔFNR**. Everything else is descriptive. No multiplicity correction.
5. **Intervals:** plug-in binomial 95% intervals for group rates with the metric's own denominator; Wald intervals for differences of two independent proportions; DIR on the log scale (delta method); per-group AUC by within-group percentile bootstrap (B = 500); worst-group summaries with Lum et al.'s double-corrected variance and a within-cell bootstrap (B = 500). Intervals condition on the trained model and `t`.
6. **p-values:** Hurlin et al.'s likelihood-ratio (G) test of independence everywhere, reported **after** gaps and intervals.
7. **Calibration:** 10 equal-frequency bins (deciles) of each model's pooled OOF score; observed default rate vs mean score per bin and group with 95% error bars; G-test of sufficiency across the bins. Brier score and log loss per group next to it.
8. **Conditional demographic parity:** PTI = `AMT_ANNUITY / AMT_INCOME_TOTAL` from the **raw** application CSV; quintiles by `pd.qcut(q=5)` on the 307,495 rows with known PTI (cut points ≈ 0.104 / 0.144 / 0.186 / 0.247); the 12 rows without PTI are excluded from this diagnostic only. Pairs F − M and 20-29 − 60-69; per-quintile gaps with intervals plus one stratified G-test per pair; cell × quintile approval rates as an appendix table.
9. **Comparison with the data:** label-implied ("oracle") approval rates `1 − p_g` (men/women ≈ 0.97); Chouldechova's expected signs; Kendall's τ between the cells' base-rate ranking and FPR ranking; proxy watch list in the saved feature importances.
10. **Robustness ranks 2–5:** F − M EOD, ΔFNR and ΔAR as point estimates with Wald intervals (no bootstrap).
11. **Equalized-odds summary SP** = ½(|ΔFPR| + |ΔFNR|) is **not** computed here (it is only for the mitigation trade-off plots).
12. **Reporting:** main text = primary gaps per model, gender results and one cell figure; full tables as artifacts.

---

## 3. Ground rules for the executor

Numbering: sections 1–8 are plan-wide; subsections inside a phase are prefixed with `P` (e.g. `P4.2` = Phase 4, subsection 2).

1. **Do not modify**: anything under `src/data/`, `src/features/`, `src/models/`, `analysis/`, `configs/models/`, `configs/model.yaml`, `artifacts/features/`, `artifacts/runs/`, existing `tests/test_models_*.py` / `test_features_*.py` / `test_data*.py`, `docs/decisions.md`, `docs/open-questions.md`, `.cursor/plans/multi-model-nested-cv.md`. Allowed edits: new files under `src/fairness/` (the existing `src/fairness/__init__.py` may be rewritten), `configs/fairness.yaml`, `configs/paths.yaml` (one added key, Phase 1), new `tests/test_fairness_*.py`, appending fixtures to `tests/conftest.py`, docs in Phase 9.
2. **Style:** black + isort (configured in `pyproject.toml`) before every commit: `black src tests analysis/helpers && isort src tests analysis/helpers`. Google-style docstrings on every public function/class; type hints with `typing` (`Optional[X]`, `Dict`, `List`, `Tuple`, not `X | None`); custom types in `src/fairness/types.py`; no mutable defaults.
3. **No hard-coded settings in `src/`:** seed, B, confidence level, pairs, model list, ranks, bins, PTI checks, watch list come from `configs/fairness.yaml`; paths from `configs/paths.yaml` or CLI arguments. Column names and metric names are module-level `UPPER_CASE` constants.
4. **Dependencies:** only numpy, pandas, scipy, matplotlib, pyyaml, pyarrow, scikit-learn (for `roc_auc_score`, `brier_score_loss`, `log_loss`). `src/fairness/` may import from `src.constants`, `src.models.runs` (loaders), `src.models.metrics` and `src.models.io` (`load_yaml_config`). **`src/` must never import from `analysis/`** (tests may, for cross-checks).
5. **Tests:** tiny synthetic data only (≤ 5,000 rows); never read `data/` or `artifacts/` in tests (exception: Phase 1 tests may load `configs/fairness.yaml`). New fairness tests must run in < 60 s in total.
6. Run from the repo root with `source .venv/bin/activate`.
7. **Git:** work on `main`. One commit per phase, message `fairness: phase N - <short title>`, including the Progress-log update. Review fixes: `fairness: phase N review fixes`. Do not push, amend, rebase, squash, `reset --hard`, `stash`, `clean`.
8. **Randomness:** never use Python's `hash()` for seeds (it is randomized per process). Derive per-task seeds with `seed_for(base_seed, key)` (P2.1).
9. If an exit criterion fails and the fix is not obvious, **stop and report**. Do not loosen tolerances.

### 3.1 Standard checks (part of the exit criteria of every phase)

```bash
source .venv/bin/activate
black --check src tests analysis/helpers
isort --check-only src tests analysis/helpers
pytest -q
```

---

## 4. Target architecture

### 4.1 File tree (new = ★, modified = ✎, read-only input = ·)

```
configs/
  fairness.yaml              ★ audit settings (P1)
  paths.yaml                 ✎ add fairness_dir: artifacts/fairness (P1)
src/fairness/
  __init__.py                ✎ docstring + lazy exports (P7)
  types.py                   ★ constants, dataclasses (P1)
  config.py                  ★ load + validate fairness.yaml (P1)
  io.py                      ★ build the audit frame: OOF + groups + raw PTI (P1)
  stats.py                   ★ intervals, G-tests, Lum summary, bootstraps, seeds (P2)
  thresholds.py              ★ base-rate threshold (P3)
  group_metrics.py           ★ METRIC_SPECS, group counts, group table, utility (P3)
  gaps.py                    ★ pairwise gaps, cell summaries (P4)
  calibration.py             ★ decile bins, calibration table, sufficiency tests (P5)
  conditional.py             ★ PTI quintiles, conditional parity (P5)
  comparison.py              ★ oracle benchmark, expected signs, Kendall, proxy importance (P6)
  robustness.py              ★ ranks 2–5 (P6)
  plots.py                   ★ figures (P7)
  report.py                  ★ write CSV/JSON, Markdown reports (P7)
  audit.py                   ★ run_audit() + CLI python -m src.fairness.audit (P7)
  summary.py                 ★ cross-model summary CLI python -m src.fairness.summary (P7)
tests/
  conftest.py                ✎ append fairness fixtures (P1)
  test_fairness_config.py    ★ (P1)
  test_fairness_stats.py     ★ (P2)
  test_fairness_group_metrics.py ★ (P3)
  test_fairness_gaps.py      ★ (P4)
  test_fairness_calibration.py   ★ (P5)
  test_fairness_conditional.py   ★ (P5)
  test_fairness_comparison.py    ★ (P6)
  test_fairness_audit.py     ★ (P7)
artifacts/ (gitignored)
  runs/<model>/baseline/...  · inputs from the models plan
  features/fairness_groups_train.parquet · input
  fairness/<model>/baseline/ ★ per-model outputs (P8)
  fairness/summary_baseline/ ★ cross-model outputs (P8)
data/application_train.csv   · input (PTI columns only)
```

### 4.2 Data flow

```text
artifacts/runs/<m>/<variant>/[robustness/rank_r/]oof_predictions.parquet ─┐
artifacts/features/fairness_groups_train.parquet (gender, age band, cell)  ├─ io.load_audit_frame()
data/application_train.csv (SK_ID_CURR, AMT_ANNUITY, AMT_INCOME_TOTAL)    ─┘      (join on SK_ID_CURR)
        │
        ▼  AuditFrame: SK_ID_CURR, y, score, fold, CODE_GENDER, AGE_BAND, GENDER_AGE, PTI, PTI_Q
thresholds.base_rate_threshold()  →  yhat
        │
        ├─ group_metrics: counts, rates + CIs, AUC (bootstrap), Brier/log loss, utility
        ├─ gaps: 9 pairs × metrics (Wald, DIR log CI, G-test); cell summaries (Lum)
        ├─ calibration: deciles × groups, sufficiency G-tests
        ├─ conditional: PTI quintiles × pairs, stratified G-tests, cell × quintile table
        ├─ comparison: oracle benchmark, expected signs, Kendall τ, proxy importance
        └─ robustness: ranks 2–5 primary gaps
        ▼
report.write_outputs()  →  artifacts/fairness/<m>/<variant>/  (CSV, JSON, report.md, figures/)
summary  →  artifacts/fairness/summary_<variant>/  (primary gaps across models + ranks, report.md, figure)
```

### 4.3 Module dependencies

| Module | May import from |
| ------ | --------------- |
| `types.py` | stdlib, numpy, pandas |
| `config.py` | `types`, `src.models.io.load_yaml_config` |
| `io.py` | `types`, `src.models.runs`, `src.constants` |
| `stats.py` | numpy, scipy, sklearn.metrics |
| `thresholds.py` | numpy |
| `group_metrics.py` | `types`, `stats`, `src.models.metrics` |
| `gaps.py` | `types`, `stats`, `group_metrics` |
| `calibration.py`, `conditional.py` | `types`, `stats`, `group_metrics` |
| `comparison.py`, `robustness.py` | `types`, `stats`, `gaps`, `group_metrics`, `thresholds` |
| `plots.py`, `report.py` | `types`, matplotlib, pandas |
| `audit.py`, `summary.py` | all of the above |

**Rule:** every metric is defined **once**, in `METRIC_SPECS` (5.3). Rates, gaps, G-tests, worst-group summaries and robustness all look up the subpopulation and event there. No other module re-implements "FPR = …".

---

## 5. Shared reference

### 5.1 `configs/fairness.yaml` (copy exactly)

```yaml
# Fairness audit settings (Phase 1, Step 6). See docs/decisions.md.
seed: 42
n_boot: 500                # Lum et al. within-cell bootstrap and per-group AUC bootstrap
confidence: 0.95           # z = scipy.stats.norm.ppf(0.975) = 1.959964
variant: baseline
models: [logreg, lgbm, mlp, rf]
robustness_ranks: [2, 3, 4, 5]
calibration_bins: 10
dir_reference_line: 0.8

groupings:                 # column in the audit frame -> ordered group labels
  CODE_GENDER: [F, M]
  AGE_BAND: ["20-29", "30-39", "40-49", "50-59", "60-69"]
  GENDER_AGE: ["F | 20-29", "F | 30-39", "F | 40-49", "F | 50-59", "F | 60-69",
               "M | 20-29", "M | 30-39", "M | 40-49", "M | 50-59", "M | 60-69"]

pairs:                     # signed gaps are a - b
  - {id: "F-M",              a: {CODE_GENDER: F},                    b: {CODE_GENDER: M}}
  - {id: "F-M | 20-29",      a: {CODE_GENDER: F, AGE_BAND: "20-29"}, b: {CODE_GENDER: M, AGE_BAND: "20-29"}}
  - {id: "F-M | 30-39",      a: {CODE_GENDER: F, AGE_BAND: "30-39"}, b: {CODE_GENDER: M, AGE_BAND: "30-39"}}
  - {id: "F-M | 40-49",      a: {CODE_GENDER: F, AGE_BAND: "40-49"}, b: {CODE_GENDER: M, AGE_BAND: "40-49"}}
  - {id: "F-M | 50-59",      a: {CODE_GENDER: F, AGE_BAND: "50-59"}, b: {CODE_GENDER: M, AGE_BAND: "50-59"}}
  - {id: "F-M | 60-69",      a: {CODE_GENDER: F, AGE_BAND: "60-69"}, b: {CODE_GENDER: M, AGE_BAND: "60-69"}}
  - {id: "20-29 - 60-69",    a: {AGE_BAND: "20-29"},                 b: {AGE_BAND: "60-69"}}
  - {id: "20-29 - 60-69 | F", a: {CODE_GENDER: F, AGE_BAND: "20-29"}, b: {CODE_GENDER: F, AGE_BAND: "60-69"}}
  - {id: "20-29 - 60-69 | M", a: {CODE_GENDER: M, AGE_BAND: "20-29"}, b: {CODE_GENDER: M, AGE_BAND: "60-69"}}

primary:                   # pre-specified primary comparisons (no multiplicity correction)
  pair: "F-M"
  metrics: [fpr, fnr]

gap_metrics: [approval_rate, fpr, fnr, default_among_approved]
cell_summary_metrics: [approval_rate, fpr, fnr]
robustness_metrics: [fpr, fnr, approval_rate]     # for the primary pair only

conditional_parity:
  pairs: ["F-M", "20-29 - 60-69"]
  n_quantiles: 5
  expected_cut_points: [0.104, 0.144, 0.186, 0.247]   # check only (rounded to 3 decimals)
  expected_missing: 12

proxy_watch_list:          # regex on feature names; first matching group (in this order) wins
  protected: ["^cat__CODE_GENDER_", "^num__DAYS_BIRTH$"]
  ext_source_1: ["^num__EXT_SOURCE_1"]
  job_tenure: ["^num__DAYS_EMPLOYED"]
  occupation: ["^cat__OCCUPATION_TYPE_"]
  employer_type: ["^cat__ORGANIZATION_TYPE_"]
  income_type: ["^cat__NAME_INCOME_TYPE_"]
  missing_flags: ["_MISSING$"]
importance_type:           # which saved importance to use per model; mlp has none
  lgbm: gain
  rf: impurity
  logreg: coef             # absolute value is used
```

Also add to `configs/paths.yaml`: `fairness_dir: artifacts/fairness`.

### 5.2 Audit frame (output of `io.load_audit_frame`)

One row per audited applicant (307,507 on the real data), **in the row order of the OOF file**:

| Column | Type | Source |
| ------ | ---- | ------ |
| `SK_ID_CURR` | int64 | OOF |
| `y` | int8 0/1 | OOF `TARGET` |
| `score` | float64 in [0, 1] | OOF `y_pred_proba` |
| `fold` | int | OOF `fold` |
| `CODE_GENDER` | str (F/M) | `fairness_groups_train.parquet` |
| `AGE_BAND` | str | idem |
| `GENDER_AGE` | str | idem |
| `PTI` | float64, NaN if unknown | raw CSV: `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| `PTI_Q` | str `Q1`…`Q5`, NaN if PTI unknown | `pd.qcut(PTI[known], q=5, labels=["Q1",…,"Q5"])` |

Checks in `load_audit_frame` (each failure → `ValueError` with a clear message):

1. OOF has the 5.7-schema columns of the models plan (`SK_ID_CURR, TARGET, y_pred_proba, fold`), unique `SK_ID_CURR`, no NaN, scores in [0, 1].
2. Every OOF `SK_ID_CURR` is found in the groups file (inner join keeps all OOF rows; the groups file may have more rows only if they are not in the OOF — on the real data both have 307,507).
3. No NaN in the group columns, and every label is one of the labels listed in `groupings` (a small test frame may lack some cells; the real data has all 17 groups — groups with no rows get NaN rates, never an error).
4. PTI: read `data/application_train.csv` with `usecols=["SK_ID_CURR", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]`; left-join onto the frame. Quintiles are computed **only over the audited rows with known PTI**.
5. `GENDER_AGE == CODE_GENDER + " | " + AGE_BAND` for every row.

### 5.3 Metric specifications (`group_metrics.METRIC_SPECS`)

For a set of rows `S` (a group, or a group within a stratum):

| Metric name | Subpopulation of `S` | Event | Denominator name | Formula |
| ----------- | -------------------- | ----- | ---------------- | ------- |
| `approval_rate` | all | `yhat == 0` | `n` | approved / n |
| `fpr` | `y == 0` | `yhat == 1` | `negatives` | wrongful refusals / negatives |
| `fnr` | `y == 1` | `yhat == 0` | `positives` | wrongful approvals / positives |
| `tpr` | `y == 1` | `yhat == 1` | `positives` | 1 − fnr |
| `ppv` | `yhat == 1` | `y == 1` | `flagged` | descriptive only |
| `default_among_approved` | `yhat == 0` | `y == 1` | `approved` | outcome test |

```python
@dataclass(frozen=True)
class MetricSpec:
    name: str
    subpopulation: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (y, yhat) -> bool mask
    event: Callable[[np.ndarray, np.ndarray], np.ndarray]          # (y, yhat) -> bool mask
    denominator: str


METRIC_SPECS: Dict[str, MetricSpec] = {
    "approval_rate": MetricSpec("approval_rate", lambda y, h: np.ones_like(y, bool), lambda y, h: h == 0, "n"),
    "fpr": MetricSpec("fpr", lambda y, h: y == 0, lambda y, h: h == 1, "negatives"),
    "fnr": MetricSpec("fnr", lambda y, h: y == 1, lambda y, h: h == 0, "positives"),
    "tpr": MetricSpec("tpr", lambda y, h: y == 1, lambda y, h: h == 1, "positives"),
    "ppv": MetricSpec("ppv", lambda y, h: h == 1, lambda y, h: y == 1, "flagged"),
    "default_among_approved": MetricSpec("default_among_approved", lambda y, h: h == 0, lambda y, h: y == 1, "approved"),
}
```

(Write the lambdas as small named module-level functions; black will not accept 120-column lines.)

A rate is `events / denominator` with `events = (subpopulation & event).sum()` and `denominator = subpopulation.sum()`. Denominator 0 → rate NaN, interval NaN.

Score-based per-group metrics (not in `METRIC_SPECS`, computed from `y` and `score`): `auc` (bootstrap interval), `brier`, `log_loss` (mean-based interval, 5.4).

### 5.4 Interval and test formulas (`stats.py`)

`z = scipy.stats.norm.ppf(1 − (1 − confidence) / 2)` (1.959964 for 0.95).

| Function | Formula |
| -------- | ------- |
| `proportion_ci(events, n, z)` | `r = events / n`; `se = sqrt(r(1 − r)/n)`; `(r, max(0, r − z·se), min(1, r + z·se))` — same as `analysis.helpers.data_audit.proportion_table` |
| `diff_ci(r_a, n_a, r_b, n_b, z)` | `d = r_a − r_b`; `se = sqrt(r_a(1−r_a)/n_a + r_b(1−r_b)/n_b)`; `(d, d − z·se, d + z·se)` — not clipped |
| `ratio_ci(r_a, n_a, r_b, n_b, z)` | `q = r_a / r_b`; `se_log = sqrt((1−r_a)/(n_a·r_a) + (1−r_b)/(n_b·r_b))`; `(q, q·exp(−z·se_log), q·exp(z·se_log))`; NaN if a rate is 0 |
| `g_test(table)` | `table`: k × c array of counts. Drop all-zero rows and columns. `E = row_tot ⊗ col_tot / N`; `G = 2 Σ O·ln(O/E)` over cells with `O > 0`; `df = (k' − 1)(c' − 1)`; `p = chi2.sf(G, df)`. If `df == 0`: `G = 0, p = NaN`. Must equal `scipy.stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")` on tables without zero rows/columns. |
| `stratified_g_test(tables)` | sum of `G` and of `df` over the strata (each via `g_test`); `p = chi2.sf(G_sum, df_sum)` |
| `mean_ci(values, z)` | `m = mean`; `se = std(ddof=1)/sqrt(n)`; `(m, m − z·se, m + z·se)` — for per-group Brier and log loss (row contributions `(score − y)²` and `−[y ln s + (1−y) ln(1−s)]` with `s` clipped to `[1e-15, 1 − 1e-15]`) |
| `worst_group_summary(n, rate, n_boot, seed, alpha)` | **identical** to `analysis.helpers.data_audit.worst_group_summary` (copy the algorithm, not the import): max − min gap with percentile interval from binomial resampling `rng.binomial(n, rate, size=(n_boot, k)) / n`; uncorrected, single-corrected and double-corrected between-group variance (Lum et al. §4, §5.3) and their square roots |
| `bootstrap_auc_ci(y, score, n_boot, seed, alpha)` | point = `roc_auc_score(y, score)`; for b in range(n_boot): `idx = rng.integers(0, n, n)`; AUC on the resample, skipped (NaN) if it has one class; interval = `np.nanpercentile` at `100·alpha/2` and `100·(1 − alpha/2)` |
| `seed_for(base_seed, key)` | `(base_seed * 1_000_003 + zlib.crc32(key.encode("utf-8"))) % 2**32` |

Keys for `seed_for`: `f"auc|{model}|{rank}|{grouping}|{group}"`, `f"worst|{model}|{rank}|{metric}"`. Same inputs → bit-identical outputs across runs.

### 5.5 Types (`src/fairness/types.py`)

```python
AUDIT_COLUMNS: List[str] = ["SK_ID_CURR", "y", "score", "fold", "CODE_GENDER", "AGE_BAND", "GENDER_AGE", "PTI", "PTI_Q"]
GROUPING_COLUMNS: List[str] = ["CODE_GENDER", "AGE_BAND", "GENDER_AGE"]
PTI_COL = "PTI"
PTI_Q_COL = "PTI_Q"
YHAT_COL = "yhat"


@dataclass(frozen=True)
class PairSpec:
    """Two groups defined by column filters; gaps are a − b."""

    pair_id: str
    a: Dict[str, str]
    b: Dict[str, str]


@dataclass(frozen=True)
class AuditConfig:
    seed: int
    n_boot: int
    confidence: float
    variant: str
    models: List[str]
    robustness_ranks: List[int]
    calibration_bins: int
    dir_reference_line: float
    groupings: Dict[str, List[str]]
    pairs: List[PairSpec]
    primary_pair: str
    primary_metrics: List[str]
    gap_metrics: List[str]
    cell_summary_metrics: List[str]
    robustness_metrics: List[str]
    cdp_pairs: List[str]
    cdp_n_quantiles: int
    cdp_expected_cut_points: List[float]
    cdp_expected_missing: int
    proxy_watch_list: Dict[str, List[str]]
    importance_type: Dict[str, str]

    @property
    def z(self) -> float: ...


@dataclass
class ThresholdResult:
    threshold: float
    base_rate: float
    n_rows: int
    n_flagged: int
    n_positives: int
```

### 5.6 Output contract: `artifacts/fairness/<model>/<variant>/`

| File | Columns / keys |
| ---- | -------------- |
| `audit_config.json` | resolved config, model, variant, input paths, `git_commit`, package versions, `started_at`, `finished_at` |
| `threshold.json` | `ThresholdResult` fields |
| `utility.json` | `roc_auc`, `fold_roc_auc_mean`, `fold_roc_auc_std`, `fold_roc_auc_min`, `fold_roc_auc_max`, `brier`, `brier_reference`, `log_loss`, `log_loss_reference` (from the run's `metrics.json`), `tpr`, `ppv`, `f1` (at `t`; `tpr == ppv == f1` when `n_flagged == n_positives`), `n_flagged` |
| `group_counts.csv` | `grouping, group, n, positives, negatives, flagged, approved, base_rate` (17 rows + one `ALL / all` row) |
| `group_metrics.csv` | long: `grouping, group, metric, value, ci_low, ci_high, denominator` for metrics `approval_rate, fpr, fnr, tpr, ppv, default_among_approved, auc, brier, log_loss` |
| `gaps.csv` | long: `pair_id, metric, rate_a, rate_b, den_a, den_b, diff, diff_ci_low, diff_ci_high, g_stat, df, p_value, ratio, ratio_ci_low, ratio_ci_high, ratio_lower_over_higher, primary` (ratio columns only for `approval_rate`, else NaN; `primary` true only for `F-M` × `fpr`/`fnr`) |
| `cell_summary.csv` | `metric, k_groups, max_min_gap, max_min_ci_low, max_min_ci_high, max_cell, min_cell, var_uncorrected, var_single_corrected, var_double_corrected_ci_low, var_double_corrected_ci_high, sd_uncorrected, sd_single_corrected, sd_double_corrected_ci_low, sd_double_corrected_ci_high, g_stat, df, p_value` |
| `calibration.csv` | `grouping, group, bin, n, events, observed_rate, ci_low, ci_high, mean_score, calibration_gap` (gap = observed − mean score); grouping `ALL` included |
| `calibration_tests.csv` | `grouping, g_stat, df, p_value` (sufficiency across bins) |
| `bin_edges.json` | the 11 decile edges |
| `conditional_parity.csv` | `pair_id, quintile, n_a, n_b, approval_rate_a, approval_rate_b, diff, diff_ci_low, diff_ci_high` |
| `conditional_parity_tests.csv` | `pair_id, g_stat, df, p_value, unconditional_diff, unconditional_diff_ci_low, unconditional_diff_ci_high, n_excluded_missing_pti` |
| `conditional_parity_cells.csv` | appendix: `group, quintile, n, approved, approval_rate, ci_low, ci_high` for the 10 cells |
| `pti_quintiles.json` | cut points (4), `n_missing` |
| `oracle_benchmark.csv` | `pair_id, oracle_approval_a, oracle_approval_b, oracle_diff, oracle_dir, model_diff, model_dir, model_dir_ci_low, model_dir_ci_high, direction, oracle_in_model_ci` |
| `expected_signs.csv` | `pair_id, base_rate_a, base_rate_b, expected_sign_eod, observed_eod, eod_ci_excludes_0, eod_matches, expected_sign_dfnr, observed_dfnr, dfnr_ci_excludes_0, dfnr_matches` |
| `kendall.json` | `tau, p_value, n_cells, cells` (base rate and FPR per cell) |
| `proxy_importance.csv` | `feature_group, n_features, importance_share, best_feature, best_feature_rank, importance_type` (absent for `mlp`) |
| `robustness.csv` | `rank, threshold, n_flagged, roc_auc, metric, diff, diff_ci_low, diff_ci_high` (ranks 1–5; rank 1 repeated for reference) |
| `figures/calibration_CODE_GENDER.png`, `figures/calibration_AGE_BAND.png`, `figures/cells_error_rates.png` | see P7.3 |
| `report.md` | see P7.4 |

Cross-model `artifacts/fairness/summary_<variant>/`: `utility.csv` (one row per model), `primary_gaps.csv` (`model, rank, metric, diff, diff_ci_low, diff_ci_high, p_value`), `figures/primary_gaps.png`, `report.md`.

CSV rules: `index=False`, `encoding="utf-8"`, floats unrounded. JSON: `indent=2`, numpy types converted.

### 5.7 Toy example used in tests (exact values)

```python
# 20 rows. Columns y and yhat given directly (no threshold involved).
F: y    = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
   yhat = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
M: y    = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
   yhat = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
```

| Group | n | pos | neg | flagged | approved | approval_rate | fpr | fnr | tpr | ppv | default_among_approved |
| ----- | - | --- | --- | ------- | -------- | ------------- | --- | --- | --- | --- | ---------------------- |
| F | 10 | 2 | 8 | 3 | 7 | 0.7 | 2/8 = 0.25 | 1/2 = 0.5 | 0.5 | 1/3 | 1/7 |
| M | 10 | 4 | 6 | 4 | 6 | 0.6 | 1/6 | 1/4 = 0.25 | 0.75 | 3/4 | 1/6 |

F − M: Δapproval = 0.1 (se = √0.045 = 0.2121320), EOD = 0.0833333, ΔFNR = 0.25, DIR = 7/6. The G-test for approval uses the 2 × 2 table `[[7, 3], [6, 4]]` (rows F, M; columns approved, refused); check against `scipy.stats.chi2_contingency(..., correction=False, lambda_="log-likelihood")`.

---

## Phase 0 — Preconditions and baseline snapshot

**Goal:** confirm the inputs exist and are final; record BASE.

Steps:

1. `git status --short` → clean except possibly `.cursor/plans/*.md` (plan files). Anything else → stop and ask.
2. Record BASE: `git rev-parse --short HEAD` → Progress log.
3. `source .venv/bin/activate && pytest -q` → record pass count.
4. Confirm `.cursor/plans/multi-model-nested-cv.md` Progress log shows Phase 7 (7b) `done`. If not → stop and ask.
5. For each model in `logreg lgbm mlp rf`: `artifacts/runs/<m>/baseline/oof_predictions.parquet`, `metrics.json`, `feature_importance.parquet` (not for `mlp`) and `robustness/rank_{2..5}/oof_predictions.parquet` exist.
6. `artifacts/features/fairness_groups_train.parquet` exists with 307,507 rows and columns `SK_ID_CURR, CODE_GENDER, DAYS_BIRTH, AGE_YEARS, AGE_BAND, GENDER_AGE`; `data/application_train.csv` exists.
7. `python -c "import scipy, matplotlib; print(scipy.__version__, matplotlib.__version__)"`.
8. Commit the plan: `git add .cursor/plans/fairness-audit.md && git commit -m "fairness: phase 0 - plan"`.

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] BASE recorded; all inputs of steps 5–6 present (paste an `ls` summary in Notes).
- [ ] Plan committed; `git status --short` clean.

---

## Phase 1 — Config, types and audit input loading

**Files:** ★ `configs/fairness.yaml`; ✎ `configs/paths.yaml`; ★ `src/fairness/types.py`; ★ `src/fairness/config.py`; ★ `src/fairness/io.py`; ✎ `tests/conftest.py`; ★ `tests/test_fairness_config.py`.

### P1.1 Config

Create `configs/fairness.yaml` exactly as 5.1 and add `fairness_dir: artifacts/fairness` to `configs/paths.yaml`.

`config.py`:

```python
def default_fairness_config_path() -> Path: ...          # <repo>/configs/fairness.yaml
def load_audit_config(path: PathLike) -> AuditConfig: ...
```

Validation (`ValueError` with the file path in the message):

1. All keys of 5.1 present; no unknown top-level keys.
2. `0 < confidence < 1`; `n_boot >= 1`; `calibration_bins >= 2`; ranks ⊆ {2, 3, 4, 5}.
3. Every pair filter uses only `CODE_GENDER` / `AGE_BAND` and labels listed in `groupings`; pair ids unique; `a != b`.
4. `primary.pair` and every `conditional_parity.pairs` entry is a defined pair id.
5. Every metric in `gap_metrics`, `cell_summary_metrics`, `robustness_metrics`, `primary.metrics` is in `METRIC_SPECS` — in Phase 1 check against the constant `METRIC_NAMES = ("approval_rate", "fpr", "fnr", "tpr", "ppv", "default_among_approved")` in `types.py` (Phase 3 imports it).
6. Every regex in `proxy_watch_list` compiles.

### P1.2 `io.py`

```python
def load_pti(data_dir: PathLike) -> pd.DataFrame:
    """Return SK_ID_CURR and PTI (= AMT_ANNUITY / AMT_INCOME_TOTAL, NaN if either is missing)."""


def assign_pti_quintiles(pti: pd.Series, n_quantiles: int) -> Tuple[pd.Series, List[float]]:
    """pd.qcut on the known values (labels Q1..Qn); returns the labels (NaN where PTI is NaN) and the inner cut points."""


def load_audit_frame(
    run_dir: PathLike,
    rank: int,
    groups_path: PathLike,
    data_dir: PathLike,
    config: AuditConfig,
) -> pd.DataFrame:
    """Build the audit frame of 5.2 with all checks of 5.2."""
```

Use `src.models.runs.load_oof(run_dir, rank)`. Inner cut points = `pti[known].quantile([0.2, 0.4, 0.6, 0.8])` for `n_quantiles = 5` (general: `np.linspace(0, 1, n + 1)[1:-1]`). Rename `TARGET` → `y`, `y_pred_proba` → `score`.

### P1.3 Fixtures — append to `tests/conftest.py`

```python
@pytest.fixture
def toy_audit_frame() -> pd.DataFrame:
    """The 20-row toy example of fairness plan 5.7 with y, yhat, score and group columns."""
    # build F rows then M rows; AGE_BAND = "20-29" for the first 5 rows of each gender, "60-69" for the rest;
    # GENDER_AGE accordingly; score = 0.9 where yhat == 1 else 0.1; PTI = linspace(0.05, 0.4, 20); fold = row % 2


@pytest.fixture
def make_audit_frame():
    """Factory: synthetic audit frame (default 4,000 rows) with all 10 cells, a learnable score and PTI."""
    def _make(n: int = 4000, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        gender = rng.choice(["F", "M"], size=n, p=[0.66, 0.34])
        band = rng.choice(["20-29", "30-39", "40-49", "50-59", "60-69"], size=n)
        logit = -2.4 + 0.4 * (gender == "M") - 0.15 * np.array([int(b[0]) - 2 for b in band])
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
        score = np.clip(1 / (1 + np.exp(-(logit + 1.2 * (y - 0.08) + rng.normal(0, 0.8, n)))), 1e-6, 1 - 1e-6)
        pti = rng.lognormal(-1.8, 0.5, n)
        pti[:3] = np.nan
        frame = pd.DataFrame({"SK_ID_CURR": np.arange(100_000, 100_000 + n), "y": y, "score": score,
                              "fold": np.arange(n) % 5, "CODE_GENDER": gender, "AGE_BAND": band,
                              "GENDER_AGE": [f"{g} | {b}" for g, b in zip(gender, band)], "PTI": pti})
        frame["PTI_Q"], _ = assign_pti_quintiles(frame["PTI"], 5)  # import from src.fairness.io
        return frame
    return _make
```

(Black will reformat; keep the logic.)

### P1.4 Tests — `tests/test_fairness_config.py`

| Test | Asserts |
| ---- | ------- |
| `test_real_config_loads` | `load_audit_config(default_fairness_config_path())`: 9 pairs, primary `F-M` with `[fpr, fnr]`, `z` ≈ 1.959964 (1e-6), 17 groups in `groupings` |
| `test_invalid_configs` (parametrized) | unknown metric; pair with unknown label; duplicate pair id; `primary.pair` undefined; `confidence: 1.0`; bad regex → `ValueError` |
| `test_assign_pti_quintiles` | on `pd.Series([np.nan] + list(range(1, 101)))`: label of NaN is NaN; each quintile has 20 values; 4 cut points equal `Series(range(1,101)).quantile([.2,.4,.6,.8])` |
| `test_load_audit_frame` | write a tiny OOF parquet (schema of the models plan) into `tmp_path/run/oof_predictions.parquet`, a groups parquet and a mini `application_train.csv` (columns `SK_ID_CURR, AMT_ANNUITY, AMT_INCOME_TOTAL` + one irrelevant column) → frame has the 5.2 columns in order, same row order as the OOF, PTI computed, `PTI_Q` assigned |
| `test_load_audit_frame_errors` (parametrized) | OOF id missing in groups; extra group label; duplicate `SK_ID_CURR`; score > 1; `GENDER_AGE` inconsistent → `ValueError` |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `python -c "from src.fairness.config import load_audit_config, default_fairness_config_path as p; c = load_audit_config(p()); print(len(c.pairs), c.primary_pair, round(c.z, 6))"` prints `9 F-M 1.959964`.
- [ ] Real-data spot check (not a test): a short script loads the lgbm rank-1 frame and prints row count 307,507, PTI missing 12, cut points rounded to 3 decimals `[0.104, 0.144, 0.186, 0.247]`. Paste the output in Notes.
- [ ] `git status --short` lists only this phase's files.
- [ ] Commit `fairness: phase 1 - config, types, input loading`.

---

## Phase 2 — Statistics core

**Files:** ★ `src/fairness/stats.py`; ★ `tests/test_fairness_stats.py`.

### P2.1 Functions

Implement every function of 5.4 with these signatures:

```python
def z_value(confidence: float) -> float: ...
def seed_for(base_seed: int, key: str) -> int: ...
def proportion_ci(events: int, n: int, z: float) -> Tuple[float, float, float]: ...
def diff_ci(r_a: float, n_a: int, r_b: float, n_b: int, z: float) -> Tuple[float, float, float]: ...
def ratio_ci(r_a: float, n_a: int, r_b: float, n_b: int, z: float) -> Tuple[float, float, float]: ...
def g_test(table: np.ndarray) -> Tuple[float, int, float]: ...                 # (G, df, p)
def stratified_g_test(tables: Sequence[np.ndarray]) -> Tuple[float, int, float]: ...
def mean_ci(values: np.ndarray, z: float) -> Tuple[float, float, float]: ...
def worst_group_summary(n: Sequence[int], rate: Sequence[float], n_boot: int, seed: int, alpha: float) -> Dict[str, float]: ...
def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int, alpha: float) -> Tuple[float, float, float]: ...
```

`worst_group_summary` returns the same keys as `analysis.helpers.data_audit.worst_group_summary` (`k_groups, max_min_gap, max_min_ci_low, max_min_ci_high, var_uncorrected, var_single_corrected, var_double_corrected_ci_low, var_double_corrected_ci_high, sd_uncorrected, sd_single_corrected, sd_double_corrected_ci_low, sd_double_corrected_ci_high`). Re-implement it (Ground rule 4); do not import from `analysis/`.

Edge cases: `n == 0` → `(nan, nan, nan)`; rates 0 or 1 give `se = 0` (interval collapses; that is the plug-in method, document it in the docstring).

### P2.2 Tests — `tests/test_fairness_stats.py`

| Test | Asserts |
| ---- | ------- |
| `test_z_value` | `z_value(0.95)` = 1.959964 (1e-6) |
| `test_proportion_ci_known` | `proportion_ci(7, 10, z)` = `(0.7, 0.7 − z·√0.021, 0.7 + z·√0.021)`; clipping at 0 and 1 (`proportion_ci(0, 5, z)` = `(0, 0, 0)`) |
| `test_matches_data_audit_helpers` | for random counts, `proportion_ci` equals `analysis.helpers.data_audit.proportion_table` rate/ci; `diff_ci` and `ratio_ci` equal `proportion_gap` diff/ratio intervals; `worst_group_summary` equals the helper's output for the same `(n, rate, n_boot=200, seed=7)` (all keys, `np.isclose` rtol 1e-12) |
| `test_diff_ci_toy` | toy F − M approval: `diff = 0.1`, half-width = z · 0.2121320 (1e-7) |
| `test_ratio_ci_toy` | `ratio = 7/6`; log-scale half-width = z · √(0.3/7 + 0.4/6) |
| `test_g_test_matches_scipy` | for 5 random tables (2×2, 10×2, 3×4) without zeros: `(G, df, p)` equal `chi2_contingency(t, correction=False, lambda_="log-likelihood")` (`statistic`, `dof`, `pvalue`), rtol 1e-10 |
| `test_g_test_zero_rows_and_cols` | a table with an all-zero row and an all-zero column → same result as the table with them removed; a table with one non-zero row → `(0.0, 0, nan)` |
| `test_stratified_g_test` | sum of two independent `g_test` results |
| `test_mean_ci` | known values |
| `test_bootstrap_auc_ci_deterministic` | same seed → identical tuple; point equals `roc_auc_score`; `ci_low ≤ point ≤ ci_high` on a 500-row synthetic sample |
| `test_seed_for_stable` | `seed_for(42, "auc|lgbm|1|CODE_GENDER|F")` is identical across two calls and fits in `[0, 2**32)`; different keys → different seeds |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `grep -n "import analysis\|from analysis" src/fairness/*.py` → nothing.
- [ ] Commit `fairness: phase 2 - statistics core`.

---

## Phase 3 — Threshold, group table and utility

**Files:** ★ `src/fairness/thresholds.py`; ★ `src/fairness/group_metrics.py`; ★ `tests/test_fairness_group_metrics.py`. Config validation in `config.py` now imports `METRIC_SPECS` (replace the Phase 1 constant check).

### P3.1 `thresholds.py`

```python
def base_rate_threshold(score: np.ndarray, y: np.ndarray) -> ThresholdResult:
    """t = np.quantile(score, 1 - mean(y)) (numpy default 'linear'); n_flagged = (score > t).sum()."""


def apply_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    """yhat = (score > threshold).astype(np.int8)."""
```

Log a `WARNING` (module logger) if `n_flagged != n_positives`; do not fail (ties can make an exact match impossible). On the real data all four rank-1 runs flag exactly 24,825.

### P3.2 `group_metrics.py`

```python
METRIC_SPECS: Dict[str, MetricSpec]          # 5.3


def group_masks(frame: pd.DataFrame, config: AuditConfig) -> List[Tuple[str, str, np.ndarray]]:
    """[(grouping, group, bool mask)] for ALL/all and the 17 groups, in config order."""


def rate(y: np.ndarray, yhat: np.ndarray, metric: str) -> Tuple[int, int]:
    """(events, denominator) of a metric on the given rows."""


def group_counts(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame: ...       # 5.6 group_counts.csv
def group_metrics(frame: pd.DataFrame, config: AuditConfig, model: str, rank: int) -> pd.DataFrame: ...  # 5.6 group_metrics.csv
def utility(frame: pd.DataFrame, threshold: ThresholdResult, run_metrics: Dict[str, Any]) -> Dict[str, Any]: ...  # 5.6 utility.json
```

`frame` here = audit frame + `yhat` column. Per group: all `METRIC_SPECS` rates with `proportion_ci`; `auc` via `bootstrap_auc_ci` with `seed_for(config.seed, f"auc|{model}|{rank}|{grouping}|{group}")`; `brier` and `log_loss` via `mean_ci` (5.4). The `ALL / all` row uses the same functions (its AUC equals the run's pooled OOF AUC; assert to 1e-12).

`utility`: `roc_auc` etc. copied from the run's `metrics.json` (`src.models.runs.load_metrics`); `tpr`, `ppv`, `f1 = 2·ppv·tpr/(ppv + tpr)` computed from the frame.

### P3.3 Tests — `tests/test_fairness_group_metrics.py`

| Test | Asserts |
| ---- | ------- |
| `test_threshold_flags_base_rate_share` | on `make_audit_frame()`: `n_flagged == n_positives` (continuous scores); `apply_threshold` flags exactly those rows with `score > t` |
| `test_threshold_with_ties_warns` | scores with heavy ties → warning logged (`caplog`), no exception |
| `test_toy_rates` | on `toy_audit_frame`: every value of the 5.7 table for F and M (exact fractions, 1e-12) |
| `test_toy_counts` | `group_counts`: n/positives/negatives/flagged/approved of 5.7; `ALL` row = sums |
| `test_rates_use_metric_denominator` | `denominator` column equals negatives for `fpr`, positives for `fnr`, approved for `default_among_approved` |
| `test_group_metrics_shape` | on `make_audit_frame()` with `n_boot=50`: 18 groups × 9 metrics rows; all intervals contain the value; no NaN except where a denominator is 0 |
| `test_brier_logloss_all_row` | `ALL` Brier/log loss equal `sklearn` `brier_score_loss` / `log_loss` |
| `test_auc_bootstrap_deterministic` | two calls → identical `group_metrics` frames |
| `test_utility_identity` | when `n_flagged == n_positives`: `tpr == ppv == f1` (1e-12) |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] Real-data spot check: lgbm rank 1 → threshold ≈ 0.2133 and `n_flagged == 24825`; print the gender rows of `group_metrics` (approval_rate, fpr, fnr). Paste in Notes.
- [ ] Commit `fairness: phase 3 - threshold, group table, utility`.

---

## Phase 4 — Pairwise gaps and cell summaries

**Files:** ★ `src/fairness/gaps.py`; ★ `tests/test_fairness_gaps.py`.

### P4.1 Functions

```python
def pair_mask(frame: pd.DataFrame, side: Dict[str, str]) -> np.ndarray: ...


def pair_gaps(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """5.6 gaps.csv for every pair × config.gap_metrics."""


def cell_summary(frame: pd.DataFrame, config: AuditConfig, model: str, rank: int) -> pd.DataFrame:
    """5.6 cell_summary.csv for config.cell_summary_metrics over the GENDER_AGE cells."""
```

For each pair and metric:
- `events_a, den_a = rate(y[mask_a], yhat[mask_a], metric)`, same for b; `rate_a = events_a / den_a`.
- `diff, ci` via `diff_ci`.
- G-test on the 2 × 2 table `[[events_a, den_a − events_a], [events_b, den_b − events_b]]`.
- For `approval_rate` only: `ratio_ci` and `ratio_lower_over_higher = min(ratio, 1/ratio)`.
- `primary = (pair_id == config.primary_pair and metric in config.primary_metrics)`.

Cell summary per metric: per cell `(events, den)` → `rate`; `worst_group_summary(den, rate, n_boot, seed_for(seed, f"worst|{model}|{rank}|{metric}"), alpha)`; `max_cell`/`min_cell` labels; G-test on the 10 × 2 table of (events, den − events).

### P4.2 Tests — `tests/test_fairness_gaps.py`

| Test | Asserts |
| ---- | ------- |
| `test_toy_gaps` | on the toy frame with a one-pair config `F-M`: approval diff 0.1, EOD 0.0833333, ΔFNR 0.25, DIR 7/6, `ratio_lower_over_higher` 6/7; G-test values equal scipy on the same 2 × 2 tables |
| `test_primary_flag` | exactly 2 rows `primary == True` with the real config's pairs on `make_audit_frame()` |
| `test_gap_rows` | 9 pairs × 4 metrics = 36 rows; ratio columns NaN except `approval_rate` |
| `test_sign_convention` | swapping a and b negates `diff` and inverts `ratio` |
| `test_cell_summary` | 3 rows; `max_min_gap` equals max − min of the cell rates computed directly; `max_cell`/`min_cell` correct; deterministic across calls |
| `test_cell_g_test_df` | `df == 9` when all 10 cells are present and have both outcomes |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] Real-data spot check: lgbm rank 1 → print the two primary rows (`F-M` × `fpr`, `fnr`) with intervals. Paste in Notes.
- [ ] Commit `fairness: phase 4 - gaps and cell summaries`.

---

## Phase 5 — Calibration, outcome test and conditional demographic parity

**Files:** ★ `src/fairness/calibration.py`; ★ `src/fairness/conditional.py`; ★ `tests/test_fairness_calibration.py`; ★ `tests/test_fairness_conditional.py`.

(The outcome test, `default_among_approved`, is already in the group table and gap table; nothing new to compute for it.)

### P5.1 `calibration.py`

```python
def decile_bins(score: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """edges = np.quantile(score, np.linspace(0, 1, n_bins + 1)); bin = np.searchsorted(edges[1:-1], score, side='right') (0..n_bins-1)."""


def calibration_table(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame: ...   # 5.6 calibration.csv
def sufficiency_tests(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame: ...   # 5.6 calibration_tests.csv
```

- Bins come from the **pooled** score of the model (all rows), then are applied to every group.
- Per (grouping, group, bin): `n`, `events = y.sum()`, `observed_rate` with `proportion_ci`, `mean_score`, `calibration_gap = observed_rate − mean_score`. Include `grouping = "ALL"`.
- Sufficiency test per grouping: for each bin, a `k × 2` table (groups × [y=1, y=0]); `stratified_g_test` over the bins.

### P5.2 `conditional.py`

```python
def conditional_parity(frame: pd.DataFrame, config: AuditConfig, gaps: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(conditional_parity.csv, conditional_parity_tests.csv) of 5.6."""


def conditional_parity_cells(frame: pd.DataFrame, config: AuditConfig) -> pd.DataFrame: ...   # appendix table
```

- Rows with `PTI_Q` NaN are dropped (count them → `n_excluded_missing_pti`).
- For each pair in `config.cdp_pairs` and each quintile: approval rates of a and b, `diff_ci`.
- Test per pair: one 2 × 2 table per quintile (a/b × approved/refused) → `stratified_g_test` (df = 5 when every quintile has both groups and both outcomes).
- `unconditional_diff` (+ CI) copied from `gaps` (`pair_id`, `metric == "approval_rate"`), i.e. on **all** rows.

### P5.3 Tests

`tests/test_fairness_calibration.py`:

| Test | Asserts |
| ---- | ------- |
| `test_decile_bins_equal_frequency` | 1,000 distinct scores → every bin has 100 rows; bins 0..9 |
| `test_calibration_counts_sum` | per grouping, the sum of `n` over groups and bins equals the number of rows |
| `test_perfectly_calibrated_synthetic` | scores drawn uniform, `y ~ Bernoulli(score)`, 20,000 rows, same for two groups → for `ALL`, `mean_score` lies within `[ci_low, ci_high]` of `observed_rate` in ≥ 9 of 10 bins; sufficiency p-value > 0.01 (seed fixed) |
| `test_sufficiency_detects_group_shift` | group M has `y ~ Bernoulli(min(1, 1.5·score))` → sufficiency p-value < 1e-6 |

`tests/test_fairness_conditional.py`:

| Test | Asserts |
| ---- | ------- |
| `test_quintile_rows` | 2 pairs × 5 quintiles = 10 rows; `n_a + n_b` per quintile matches direct counts |
| `test_missing_pti_excluded` | `make_audit_frame()` has 3 NaN PTI → `n_excluded_missing_pti == 3` |
| `test_stratified_test_df` | `df == 5` on the synthetic frame |
| `test_unconditional_copied` | `unconditional_diff` equals the `gaps` row for the pair's `approval_rate` |
| `test_cells_appendix` | 10 cells × 5 quintiles rows; rates within [0, 1] |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] Real-data spot check: lgbm rank 1 → print `calibration_tests` and `conditional_parity_tests`; confirm `n_excluded_missing_pti == 12`. Paste in Notes.
- [ ] Commit `fairness: phase 5 - calibration and conditional parity`.

---

## Phase 6 — Comparison with the data and robustness ranks

**Files:** ★ `src/fairness/comparison.py`; ★ `src/fairness/robustness.py`; ★ `tests/test_fairness_comparison.py`.

### P6.1 `comparison.py`

```python
def oracle_benchmark(frame: pd.DataFrame, config: AuditConfig, gaps: pd.DataFrame) -> pd.DataFrame: ...
def expected_signs(frame: pd.DataFrame, config: AuditConfig, gaps: pd.DataFrame) -> pd.DataFrame: ...
def kendall_cells(frame: pd.DataFrame, config: AuditConfig) -> Dict[str, Any]: ...
def proxy_importance(importance: Optional[pd.DataFrame], config: AuditConfig, model: str) -> Optional[pd.DataFrame]: ...
```

**Oracle benchmark** (per pair; "approve exactly the applicants who repaid"):
- `oracle_approval_g = 1 − base_rate_g`; `oracle_diff = oracle_a − oracle_b`; `oracle_dir = oracle_a / oracle_b`.
- `model_diff`, `model_dir` and the DIR interval from `gaps` (`approval_rate` rows).
- `direction`: let `dm = model_dir − 1`, `do = oracle_dir − 1`. `"reversed"` if `dm·do < 0`; else `"wider"` if `|dm| > |do|`; else `"narrower"` (`"equal"` if `|dm| == |do|`). These are mechanical labels; interpretation is Step 7.
- `oracle_in_model_ci = model_dir_ci_low ≤ oracle_dir ≤ model_dir_ci_high`.
- Real-data sanity: for `F-M`, `oracle_dir` = (1 − 0.10142)/(1 − 0.06999) ≈ 0.966.

**Expected signs** (Chouldechova 2017: with a calibrated score, the higher-base-rate group has higher FPR and lower FNR):
- `expected_sign_eod = sign(base_rate_a − base_rate_b)`; `expected_sign_dfnr = −expected_sign_eod`.
- `observed_*` from `gaps`; `*_ci_excludes_0`; `*_matches = sign(observed) == expected`.

**Kendall's τ** (Wang et al. §6.1, translated to this coding): over the 10 cells, `scipy.stats.kendalltau(base_rate, fpr)` (tau-b) → `tau`, `p_value`; also store the per-cell values. The report adds the fixed caveat "10 cells: low power".

**Proxy importance** (descriptive; `None` for `mlp`):
- From `artifacts/runs/<m>/baseline/feature_importance.parquet` (`feature, fold, importance_type, importance`), keep `importance_type == config.importance_type[model]`, take `abs(importance)` for `coef`, average over folds per feature.
- Assign each feature to the first matching watch-list group (YAML order); unmatched features → `"other"`.
- Per group: `n_features`, `importance_share = group sum / total`, `best_feature` (largest), `best_feature_rank` (1 = most important over **all** features).

### P6.2 `robustness.py`

```python
def robustness_table(run_dir: PathLike, config: AuditConfig, groups_path: PathLike, data_dir: PathLike) -> pd.DataFrame:
    """5.6 robustness.csv: for ranks [1] + config.robustness_ranks, load the frame, apply that rank's own
    base-rate threshold, and compute config.robustness_metrics for config.primary_pair (diff + Wald CI) and the pooled AUC."""
```

No bootstrap here. Reuse `load_audit_frame`, `base_rate_threshold`, `pair_mask`, `rate`, `diff_ci`.

### P6.3 Tests — `tests/test_fairness_comparison.py`

| Test | Asserts |
| ---- | ------- |
| `test_oracle_toy` | toy frame: oracle approval F = 0.8, M = 0.6 → `oracle_dir` = 4/3; model DIR 7/6 → `direction == "narrower"` |
| `test_direction_labels` | hand-made gaps rows produce `wider`, `narrower`, `reversed` |
| `test_expected_signs_toy` | base rate F 0.2 < M 0.4 → expected EOD sign −1, observed EOD +0.0833 → `eod_matches` False; ΔFNR expected +1, observed +0.25 → True |
| `test_kendall_perfect` | cells where FPR is a monotone increasing function of base rate → `tau == 1` |
| `test_proxy_importance` | tiny importance frame with features `cat__CODE_GENDER_F`, `num__EXT_SOURCE_1`, `num__EXT_SOURCE_1_MISSING`, `num__X`: EXT_SOURCE_1 missing flag goes to `ext_source_1` (first match); shares sum to 1 incl. `other`; `coef` uses absolute values; `mlp` → `None` |
| `test_robustness_table` | write OOF files for ranks 1–5 (synthetic) into `tmp_path/run/...` (rank r under `robustness/rank_r/`), groups parquet and mini CSV → 5 ranks × 3 metrics rows; each rank's own threshold; rank 1 equals `pair_gaps` for the same metric |

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] Real-data spot check: lgbm → `oracle_benchmark` row `F-M` (oracle DIR ≈ 0.966) and `robustness_table` (5 ranks). Paste in Notes.
- [ ] Commit `fairness: phase 6 - data comparison and robustness`.

---

## Phase 7 — Audit runner, outputs, report, figures, CLI, smoke run

**Files:** ★ `src/fairness/audit.py`; ★ `src/fairness/report.py`; ★ `src/fairness/plots.py`; ★ `src/fairness/summary.py`; ✎ `src/fairness/__init__.py`; ★ `tests/test_fairness_audit.py`.

### P7.1 `audit.py`

```python
def run_audit(
    model: str,
    config: AuditConfig,
    runs_dir: PathLike,
    features_dir: PathLike,
    data_dir: PathLike,
    out_dir: PathLike,
) -> Path:
    """Audit rank 1 of <runs_dir>/<model>/<config.variant>, plus robustness; write 5.6 outputs; return the output dir."""


def main(argv: Optional[List[str]] = None) -> None: ...
```

CLI:

```
python -m src.fairness.audit (--model {logreg,lgbm,mlp,rf} | --all)
    [--config configs/fairness.yaml] [--runs-dir artifacts/runs]
    [--features-dir artifacts/features] [--data-dir data]
    [--out-dir artifacts/fairness] [--n-boot N] [--overwrite]
```

- Default paths from `configs/paths.yaml` (`runs_dir`, `features_dir`, `data_dir`, `fairness_dir`).
- `--n-boot` overrides `config.n_boot` (smoke runs only); the value used is written to `audit_config.json`.
- Existing `<out_dir>/<model>/<variant>/` without `--overwrite` → exit with an error.
- Order inside `run_audit`: load frame (rank 1) → threshold → yhat → group counts/metrics → utility → gaps → cell summary → calibration (+ tests) → conditional parity (+ tests, cells) → comparison (oracle, signs, Kendall, proxies) → robustness → figures → report. Log one INFO line per step with its duration.
- Logging: `logging` to stdout and `<out>/audit.log`.

### P7.2 `report.py`

```python
def write_outputs(results: Dict[str, Any], out_dir: Path) -> None: ...     # CSV/JSON of 5.6
def write_model_report(results: Dict[str, Any], out_dir: Path, config: AuditConfig) -> None: ...  # report.md
def write_summary_report(rows: Dict[str, pd.DataFrame], out_dir: Path) -> None: ...
```

Formatting in the Markdown reports only (CSV stays unrounded): rates and gaps in percentage points with 2 decimals (`7.83%`, `−0.42 pp [−0.61, −0.23]`); ratios 3 decimals; p-values `f"{p:.2e}"`; AUC 4 decimals.

### P7.3 Figures (`plots.py`, matplotlib, `Agg` backend, 150 dpi, default colour cycle, one colour per group kept consistent within a figure)

1. `calibration_<grouping>.png` for `CODE_GENDER` and `AGE_BAND`: x = mean score per bin, y = observed default rate, 95% error bars, one series per group, dashed diagonal y = x, axis labels "Mean predicted risk (bin)" / "Observed default rate", title `"<model>: calibration by <grouping> (deciles of pooled score)"`.
2. `cells_error_rates.png`: two panels side by side (FPR, FNR); x = the 10 cells in config order; point + 95% error bar per cell; horizontal dashed line at the `ALL` value; y in %.
3. Summary `primary_gaps.png`: one row per model; F − M EOD (left panel) and ΔFNR (right panel); rank 1 as a filled marker with its CI, ranks 2–5 as small hollow markers; vertical line at 0; x in pp.

### P7.4 `report.md` sections (per model)

1. Inputs and threshold (model, variant, rows, base rate, `t`, flagged)
2. Utility (AUC + fold spread, TPR = PPV = F1 at `t`, Brier / log loss vs references)
3. **Primary comparisons** (F − M EOD and ΔFNR with CI and p-value)
4. Group table — gender and age bands (approval rate, FPR, FNR, default among approved, AUC, Brier; counts)
5. Group table — the 10 cells (same columns)
6. Pairwise gaps (9 pairs; ΔAR, DIR, EOD, ΔFNR, Δ default among approved)
7. Worst-group summaries (with the fixed caveat: "max − min is biased upwards; see the double-corrected SD interval")
8. Calibration (figure links, sufficiency tests)
9. Conditional demographic parity (per-quintile gaps, tests, unconditional gap, excluded rows)
10. Comparison with the data (oracle benchmark, expected signs, Kendall τ with the low-power caveat, proxy watch list)
11. Robustness ranks (table)
12. Scope of the numbers (fixed text): intervals condition on the trained model and `t`; selective labels (only granted loans have `TARGET`); `TARGET` is a proxy; observational criteria; p-values follow gaps and are not multiplicity-corrected; only F − M EOD and ΔFNR are primary.

Summary `report.md`: utility table (4 models), primary gaps table (rank 1 with CI and p; ranks 2–5 range), figure link.

### P7.5 `summary.py`

```
python -m src.fairness.summary [--out-dir artifacts/fairness] [--variant baseline]
```

Reads every `<out>/<model>/<variant>/` that exists, writes `<out>/summary_<variant>/` (5.6).

### P7.6 Tests — `tests/test_fairness_audit.py`

| Test | Asserts |
| ---- | ------- |
| `test_run_audit_end_to_end` | synthetic inputs in `tmp_path` (OOF ranks 1–5 from `make_audit_frame` scores, `metrics.json` with the keys `utility` needs, a tiny `feature_importance.parquet`, groups parquet, mini CSV), `n_boot=20` → every file of 5.6 exists; `report.md` contains the 12 section headings |
| `test_cli_refuses_existing_output` | second run without `--overwrite` → `SystemExit` non-zero |
| `test_mlp_without_importance` | model `mlp` with no importance file → `proxy_importance.csv` absent, report says "not available for mlp" |
| `test_summary` | two audited fake models → `summary_baseline/primary_gaps.csv` has rows for both models and ranks 1–5 |
| `test_audit_deterministic` | two runs → identical `group_metrics.csv` and `cell_summary.csv` |

### P7.7 Smoke run on real data

```bash
python -m src.fairness.audit --model lgbm --n-boot 20 --out-dir artifacts/fairness_smoke --overwrite
python -m src.fairness.summary --out-dir artifacts/fairness_smoke
```

**Exit criteria**

- [ ] Standard checks (3.1) pass; fairness tests < 60 s in total.
- [ ] Smoke run finishes (< 5 min); `artifacts/fairness_smoke/lgbm/baseline/` has every file of 5.6; open the three figures and `report.md` and confirm they render (describe them in Notes).
- [ ] `group_counts.csv` totals: n 307,507, positives 24,825, flagged 24,825.
- [ ] Commit `fairness: phase 7 - audit runner, report, figures`.

---

## Phase 8 — Full audit of the four models

No code changes. Runtime: a few minutes per model (dominated by the AUC bootstraps).

```bash
python -m src.fairness.audit --all --overwrite
python -m src.fairness.summary
```

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] `artifacts/fairness/{logreg,lgbm,mlp,rf}/baseline/` each have every file of 5.6 (`proxy_importance.csv` not for `mlp`); `audit_config.json` shows `n_boot: 500`.
- [ ] For every model: `threshold.json` `n_flagged == 24825`; `group_counts.csv` `ALL` row n 307,507 / positives 24,825; `utility.json` `roc_auc` equals the run's `metrics.json` `oof_roc_auc` (1e-12) and equals the `ALL` AUC in `group_metrics.csv`.
- [ ] `conditional_parity_tests.csv` `n_excluded_missing_pti == 12`; `pti_quintiles.json` cut points round to `[0.104, 0.144, 0.186, 0.247]`.
- [ ] Cross-check with Step 5 (data-level numbers that must not depend on the model): base rates in `group_counts.csv` equal the Step 2/5 notes (F 7.00%, M 10.14%, M 60-69 6.39% with 486 positives / 7,121 negatives).
- [ ] `robustness.csv` has ranks 1–5 for every model.
- [ ] `artifacts/fairness/summary_baseline/` exists; paste the summary primary-gaps table into Notes.
- [ ] Commit the Progress-log update: `fairness: phase 8 - full audit`.

---

## Phase 9 — Documentation

**Files:** ★ `docs/fairness_audit.md`; ✎ `CLAUDE.md`; ✎ `src/README.md`; ✎ `docs/thesis-overview.md`; ✎ `docs/credit_scoring.md` (only its "Downstream: fairness" section). No code changes.

1. **`docs/fairness_audit.md`:** purpose; inputs; coding and threshold; groups and pairs; metric definitions (5.3) and interval/test formulas (5.4); procedure (4.2 diagram); outputs (5.6); how to run; results table from Phase 8 (summary primary gaps with date); scope of the numbers (P7.4 section 12). Link the decision rows.
2. **`CLAUDE.md`:** Commands — add `python -m src.fairness.audit --all` and `python -m src.fairness.summary`. Architecture — replace the `src/fairness/` "not yet implemented" bullet with the implemented audit (inputs `artifacts/runs/<model>/<variant>/oof_predictions.parquet` + `fairness_groups_train.parquet` + raw PTI; outputs `artifacts/fairness/`; mitigation still to come). If the models plan's Phase 9 has not yet updated other parts of `CLAUDE.md`, touch only these two places.
3. **`src/README.md`:** `fairness/` row and *Status*.
4. **`docs/thesis-overview.md`:** fairness row → "Baseline audit done (Step 6); mitigation TODO".
5. **`docs/credit_scoring.md`:** "Downstream: fairness" → link `docs/fairness_audit.md`; remove any `artifacts/models` path there.
6. Do **not** edit `docs/decisions.md` / `docs/open-questions.md`.

**Exit criteria**

- [ ] Standard checks (3.1) pass.
- [ ] Every command in the changed docs runs as written (`--help` at least).
- [ ] `git diff <BASE> --name-only | xargs grep -n "/Users/"` → nothing.
- [ ] Commit `fairness: phase 9 - documentation`.

---

## 6. Execution prompts (for Cursor Composer)

### 6.1 Per-phase execution prompt (template)

Copy, replace `<N>` and `<TITLE>`, and append the phase notes from 6.2.

```text
Execute Phase <N> ("<TITLE>") of .cursor/plans/fairness-audit.md, and only that phase.

1. Read first, in full: sections 1–5 of the plan (Goal and scope, Decisions, Ground rules
   including 3.1, Target architecture, Shared reference) and Phase <N>. Then read every
   existing file you will import from or edit in this phase.
2. Check the Progress log: the previous phase must be `done` with review verdict PASS or
   PASS WITH NITS (Phase 0 has no predecessor). If not, stop and tell me.
3. Set Phase <N> to `in progress` in the Progress log.
4. Implement exactly what Phase <N> specifies: file names, public signatures, constants,
   config values, output columns and test names as written; reuse the code given in the
   plan. Touch only the files this phase names. Do not implement anything from later
   phases. Do not add dependencies. Never import from analysis/ in src/.
5. If the plan is ambiguous, contradicts the code, or an exit criterion fails and the fix
   is not obvious: stop and ask me. Never loosen tolerances, never skip, weaken or delete
   tests to make them pass. Never write interpretations or conclusions into code or
   reports beyond the mechanical labels the plan defines.
6. When the code is written, from the repo root:
   a. source .venv/bin/activate
   b. black src tests analysis/helpers && isort src tests analysis/helpers
   c. run this phase's tests, then the full suite: pytest -q
   d. go through every exit criterion of Phase <N> one by one, run its command and keep
      the output (real-data spot checks are short scripts run in the terminal, not tests).
7. Update the Progress log row of Phase <N>: status `done`, Notes with the key evidence.
   Then `git status --short` (only this phase's files, plus the plan), `git add` exactly
   those files, and commit "fairness: phase <N> - <short title>".
8. Reply with: files changed; an exit-criteria table (criterion → met / not met →
   evidence); deviations from the plan and why; open questions. End with
   "Phase <N> ready for review." Then stop. Do not start the next phase.
```

### 6.2 Phase notes (append to the template)

| Phase | Notes to append |
| ----- | --------------- |
| 0 | No code. Only checks and the plan commit. |
| 1 | `config.py` checks metric names against `METRIC_NAMES` in `types.py` until Phase 3. The real-data spot check reads `data/application_train.csv` (only three columns). |
| 2 | `worst_group_summary` must reproduce `analysis.helpers.data_audit.worst_group_summary` exactly (same RNG calls in the same order); the cross-check test may import from `analysis`, `src/` must not. |
| 3 | `METRIC_SPECS` is the single definition of every rate; later phases must use it. The `ALL` AUC must equal the run's pooled OOF AUC. |
| 4 | Signs are always a − b as written in the pair id. Only `F-M` × `fpr`/`fnr` are primary. |
| 5 | Bins come from the pooled score of the model, never per group. PTI comes from the raw CSV, never from `X`. |
| 6 | The `direction` labels are mechanical; do not add any interpretation text. |
| 7 | Figures: matplotlib only (no seaborn), `Agg` backend. Open the smoke figures and describe them in Notes. |
| 8 | No code changes. If a check fails, stop and report; do not patch code in this phase. |
| 9 | Documentation only. Use the Phase 8 summary table from the Notes for the results section. |

### 6.3 Fix-after-review prompt

```text
The review of Phase <N> of .cursor/plans/fairness-audit.md returned <FAIL / PASS WITH NITS>.
Findings:
<paste the findings>

Fix exactly these findings: all blockers and majors, and the minors and nits unless I
marked them "skip". Change nothing else. Then run black + isort, the standard checks (3.1)
and every exit criterion of Phase <N> again, update the Notes of Phase <N>, and commit
"fairness: phase <N> review fixes". Reply with a table: finding → what you changed
(file:line) → evidence it is fixed. Then stop.
```

---

## 7. Review prompts (executed by Claude Code)

### 7.1 Per-phase review prompt (template)

Copy, replace `<N>` and `<TITLE>`, and append the phase-specific checklist from 7.2.

```text
Review Phase <N> ("<TITLE>") of .cursor/plans/fairness-audit.md.

Context: MSc thesis repo (bias in ML credit scoring). The plan was executed by a smaller
model. Read, in this order: the plan's sections 1–5 and Phase <N>; docs/decisions.md rows
listed in plan section 2; the vault notes
/Users/c4t/c4t-vault/1-Projects/Uni/thesis/Notes/writing-notes/phase-1/2-mathematical-setup.md,
3-fairness-question-and-criteria.md, 4-fairness-metrics.md (for the definitions); CLAUDE.md.

Do:
1. Find the phase commit(s) with `git log --oneline --grep "fairness: phase <N>"` and
   inspect them (`git show --stat`, `git show`). Check `git status` for leftovers.
2. Verify every exit criterion of Phase <N> yourself (run the commands and the real-data
   spot checks; do not trust the Progress log).
3. Check plan conformance: file names, signatures, config values, output columns, and the
   out-of-scope list (no mitigation, no model or feature changes, no new dependencies,
   no imports from analysis/ in src/, no interpretation text).
4. Check correctness against the definitions: label and decision orientation (Y = 1
   payment difficulties, Ŷ = 1 refused, approval = Ŷ = 0, EOD = FPR gap), metric
   denominators, sign convention a − b, interval formulas, G-test degrees of freedom,
   bootstrap determinism, threshold rule, PTI from the raw CSV, bins from the pooled score.
5. Check style: standard checks (plan 3.1); docstrings, typing, no hard-coded settings.
6. Run the phase-specific checks below.

Report:
- Verdict: PASS / PASS WITH NITS / FAIL.
- Findings ranked by severity (blocker / major / minor / nit), each with file:line, what is
  wrong, why it matters, and the concrete fix.
- Exit-criteria table: criterion → met / not met → evidence.
Do not change code unless I ask. If the verdict is PASS or PASS WITH NITS, write the
verdict into the plan's Progress log.
```

### 7.2 Phase-specific checklists

**Phase 1**
- Config validation covers every rule of P1.1; error messages name the file.
- Audit frame keeps the OOF row order; join cannot silently drop or duplicate rows; PTI from the CSV with only three columns read; quintiles on known PTI of the audited rows, matching notebook 08 (`pd.qcut(q=5)`).

**Phase 2**
- `g_test` drops zero rows/columns before computing df; equals scipy's log-likelihood statistic; `stratified_g_test` sums G and df.
- `worst_group_summary` reproduces the helper bit-for-bit (run the cross-check test yourself; read both implementations).
- No use of `hash()`; `seed_for` deterministic.

**Phase 3**
- `METRIC_SPECS` subpopulations and events match 5.3 (check FNR = approved among positives, default among approved = positives among approved).
- Threshold: `np.quantile` default method, strict `>`; real data flags 24,825 for lgbm.
- Brier/log loss per group use row contributions with clipping; `ALL` AUC equals the run AUC.

**Phase 4**
- 2 × 2 tables use (events, denominator − events) of the metric's subpopulation, not of the whole group.
- DIR only for approval rate; `ratio_lower_over_higher` correct.
- Cell summary uses the metric's denominator per cell (positives for FNR, negatives for FPR).

**Phase 5**
- Deciles from the pooled score; sufficiency test df = Σ over bins of (groups present − 1).
- Conditional parity excludes exactly the NaN-PTI rows; unconditional gap taken from all rows; stratified df = 5 on real data.

**Phase 6**
- Oracle uses `1 − base_rate` of each side of the pair; real F − M oracle DIR ≈ 0.966.
- Expected signs follow base rates (higher base rate → higher FPR, lower FNR).
- Proxy grouping uses first-match order; `coef` absolute values; ranks over all features.
- Robustness recomputes the threshold per rank.

**Phase 7**
- All 5.6 files and columns; Markdown formatting only in reports; figures readable (open them).
- CLI guards; `--n-boot` recorded; logging per step.
- Report sections 1–12 present; section 12 text present verbatim in meaning; no interpretation beyond mechanical labels.

**Phase 8**
- Re-run the Phase 8 checks with a script.
- Sanity of results: primary gaps per model with CIs; compare signs with Chouldechova's prediction (men and younger applicants expected higher FPR, lower FNR); compare the F − M DIR with the 0.966 oracle; spread of primary gaps across ranks 2–5 vs the rank-1 CI width; calibration p-values; any figure that looks wrong.
- Report a results summary for the thesis author (no interpretation beyond what the numbers show; flag surprises for Step 7).

**Phase 9**
- Docs match behaviour and commands; no stale "not yet implemented" for the audit; no absolute paths.
- Then propose updates to `docs/decisions.md` (status of the audit rows) and `docs/open-questions.md` (do not apply without the user).

### 7.3 Final review prompt (whole implementation)

```text
Final review of the whole implementation of .cursor/plans/fairness-audit.md
(commits on main from BASE, recorded in the Progress log, to HEAD).

Read the full plan, docs/decisions.md (rows in plan section 2), the vault notes for
Phase 1 Steps 2–5 (writing-notes/phase-1/), CLAUDE.md and docs/fairness_audit.md.

Do:
1. `git log --oneline <BASE>..HEAD` and `git diff <BASE>..HEAD --stat`; confirm one commit
   per phase (+ documented fix commits) and no file changed outside Ground rule 1.
2. Run the standard checks (plan 3.1) and `pytest -q --durations=15`.
3. Re-run `python -m src.fairness.audit --model lgbm --out-dir <scratch> --overwrite` and
   compare its CSVs with artifacts/fairness/lgbm/baseline/ (must be identical).
4. Methodology audit against the decisions and the vault notes: orientation, threshold,
   denominators, intervals, tests, primary comparisons, cells, calibration bins, PTI
   quintiles and exclusion, oracle benchmark, expected signs, Kendall translation,
   robustness thresholds, no SP, no mitigation, no interpretation.
5. Reusability audit: can a mitigation plan audit artifacts/runs/<model>/<variant>/
   by only passing a different variant? Point out anything model- or baseline-specific.
6. Results audit for the thesis author: per model, primary gaps with CIs and p-values,
   direction relative to the oracle benchmark, expected-sign matches, calibration and
   sufficiency, conditional parity, robustness spread, proxy shares; list surprises and
   anything Step 7 must explain. Plain language, no conclusions beyond the numbers.
7. Code quality: duplication (single METRIC_SPECS), dead code, docstrings/types,
   performance hot spots.

Report:
- Overall verdict: DONE / DONE WITH FOLLOW-UPS / NOT DONE.
- Findings ranked by severity with file:line and concrete fixes.
- The results summary (point 6).
- Proposed updates to docs/decisions.md and docs/open-questions.md (do not apply them
  without the user's confirmation), and which numbers in the vault notes for Step 4
  (legacy t ≈ 0.209, FP = FN = 16,839, half-widths) need refreshing.
```

---

## 8. Appendix

### 8.1 Why these design choices (do not change them)

- **One `METRIC_SPECS`:** every later number (rates, gaps, tests, worst-group, robustness, conditional parity) uses the same subpopulation and event, so a denominator mistake cannot exist in one table and not another.
- **Own implementation, no Fairlearn:** the decisions need metric-specific intervals, Lum et al.'s corrections and likelihood-ratio tests, which the libraries do not provide; a second dependency would only give point estimates that the tests already check against hand values.
- **G-test everywhere:** Hurlin et al.'s likelihood-ratio test for statistical parity reduces to the χ² test of independence; using the likelihood-ratio (G) form for every test keeps one test family (pairs, cells, deciles, PTI strata).
- **Bins from the pooled score:** a risk class must mean the same score range in every group, otherwise the calibration comparison compares different risks.
- **PTI from the raw CSV:** `X` holds a median-imputed `AMT_ANNUITY`; the 12 applications without it must be excluded, not given an invented PTI.
- **Robustness without bootstrap:** the question is whether the gap moves between near-equal models, which point estimates and Wald intervals answer; B = 500 bootstraps for 16 extra runs add runtime without changing that answer.
- **Variant in every path:** the mitigation plan will audit `artifacts/runs/<model>/<variant>/` with the same code.

### 8.2 Troubleshooting

| Symptom | Likely cause | Action |
| ------- | ------------ | ------ |
| `n_flagged` ≠ 24,825 | `>=` instead of `>`, or a non-default `np.quantile` method | use the 5.4 / P3.1 definitions |
| `ALL` AUC ≠ run AUC | rows reordered or dropped in the join | keep the OOF order; check the join |
| PTI cut points differ from 0.104 / 0.144 / 0.186 / 0.247 | quintiles on all CSV rows (incl. XNA) or on rows with NaN | compute on the audited rows with known PTI |
| G-test df too large | zero rows/columns not dropped | see `g_test` |
| Bootstrap results differ between runs | `hash()` or a shared RNG across tasks | `seed_for` per task, new `default_rng` per call |
| `worst_group_summary` differs from the helper | different RNG call order or shape | `rng.binomial(n, rate, size=(n_boot, k))` exactly as the helper |
| Figures empty or overlapping | wrong backend or missing `tight_layout` | `matplotlib.use("Agg")`; `fig.tight_layout()` |
