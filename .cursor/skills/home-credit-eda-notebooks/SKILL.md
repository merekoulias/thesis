---
name: home-credit-eda-notebooks
description: Implements Home Credit EDA notebooks 02 through 06 using the thesis repo baseline. Use when the user asks to create, write, implement, or revise `02_bureau_eda`, `03_previous_applications_eda`, `04_installments_eda`, `05_pos_cash_eda`, or `06_credit_card_eda`.
disable-model-invocation: true
---

# Home Credit EDA Notebooks

## Scope

Use this skill to implement table-specific EDA notebooks in `analysis/notebooks/`:

- `02_bureau_eda.ipynb`: `bureau.csv` + `bureau_balance.csv`
- `03_previous_applications_eda.ipynb`: `previous_application.csv`
- `04_installments_eda.ipynb`: `installments_payments.csv`
- `05_pos_cash_eda.ipynb`: `POS_CASH_balance.csv`
- `06_credit_card_eda.ipynb`: `credit_card_balance.csv`

If the user does not specify which notebook to work on, ask before editing.

## Required Context

Before implementing, read:

- `docs/decisions.md`
- `.cursor/rules/python-data-analyst.mdc`
- `analysis/notebooks/00_dataset_overview.ipynb`
- `analysis/notebooks/01_application_eda.ipynb`
- `analysis/notebooks/02_bureau_eda.ipynb`
- `analysis/notebooks/03_previous_applications_eda.ipynb`
- `analysis/notebooks/04_installments_eda.ipynb`

Use `00_dataset_overview.ipynb`, `01_application_eda.ipynb` and `02_bureau_eda.ipynb` only as style and output references. Do not use old implementation plans as the source of truth, because the user has changed notebook outputs manually.

Use only repo-relative paths such as `data/bureau.csv` and `analysis/notebooks/02_bureau_eda.ipynb`. Never expose local absolute paths in notebook code, markdown, summaries, errors, or final responses.

## Notebook Baseline

Every EDA notebook must include at least:

- Purpose: 2-3 sentences explaining which table(s) are analyzed and what questions the notebook answers.
- Load and shape: `read_csv`, rows x columns, primary key(s), and grain.
- Schema: dtypes and a note/link to `data/HomeCredit_columns_description.csv` for ambiguous columns.
- Missingness: count and percent missing per column, with columns above a configurable high-missingness threshold flagged.
- Cardinality: unique values for categorical columns.
- Numeric summaries: `describe()`, min/max, and obvious scale or range checks for numeric columns.
- Key integrity: duplicate checks on index keys and unexpected nulls in ID columns.
- Sample rows: `.head()` and optionally one odd/example row when it explains a finding.
- 1-3 plots: distributions or comparisons for the most important columns in that table.
- Short findings: bullet points covering what was learned and what matters for feature engineering, modeling, joins, leakage, or fairness.

Each section title must use `## N. Title` and be followed immediately by 1-2 simple sentences explaining the purpose of that section.

## Exclusions

Do not include:

- Final credit scoring model training.
- Null-importance feature selection.
- Fairness metrics or mitigation.
- Kaggle submission logic.

Protected attributes such as `CODE_GENDER` and `NAME_EDUCATION_TYPE` are audit/grouping variables only. Do not propose them as model features.

## Analyst Style

Make notebooks look like real analyst work:

- Code cells should compute tables and charts.
- Markdown cells should explain interpretation after the evidence is shown.
- Do not print interpretive terminal text such as "this means..." or "identified sentinel..." from code cells.
- Use `display()` for dataframes/tables and keep stdout for minimal operational information only when useful.
- Prefer focused, readable pandas code with descriptive variable names.
- Use vectorized pandas/numpy operations where practical.
- Keep plots labeled with clear titles, axes, and readable sizing.
- For ambiguous column meanings, reference `data/HomeCredit_columns_description.csv`.
- When printing more than one value, table, etc., print also a title or label to explain what is being printed.
- Do not recompute information from code cell to code cell. If you compute missigness in one code cell, do not do that again in the next one. 

## Domain-Specific Work

In addition to the baseline, add sections that fit the table:

- For relationship tables, check join keys and coverage against `application_train.csv` / `application_test.csv`.
- For monthly history tables, inspect time ranges such as `MONTHS_BALANCE` and per-application sequence lengths.
- For payment tables, inspect payment timing, payment amounts, missing amounts, and over/under-payment patterns.
- For status tables, inspect status distributions and rare statuses.
- For previous applications, inspect contract status, decision outcomes, product/application types, and temporal fields.
- When joins are affected, note which key (`SK_ID_CURR`, `SK_ID_PREV`, or `SK_ID_BUREAU`) is involved and what that implies for later feature aggregation.
- Check for possible leakage: columns that directly encode future outcomes, post-application events, or target-like statuses should be flagged for later feature-engineering review, not used here for modeling.
- Check possible fairness/proxy relevance when table features relate to region, occupation, income, education, gender, or socioeconomic status.

## Execution Workflow

After writing or substantially revising a notebook:

1. Execute it from the repo root, for example:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace analysis/notebooks/02_bureau_eda.ipynb
   ```

2. Fix execution errors and re-run until the notebook completes.
3. Do extra research when joins, engineered features, protected attributes, leakage, fairness, or any useful/interesting finding needs more evidence.
4. Keep outputs clean and reproducible. If notebook JSON becomes invalid because of stale outputs, clear outputs and re-execute.

## Summary

End each notebook with `## Summary of findings` and concise bullets. Include:

- What the table grain and key integrity checks showed.
- Main missingness/cardinality/anomaly findings.
- Important plot or distribution findings.
- Join/coverage implications.
- Feature engineering implications.
- Leakage or fairness/proxy cautions when relevant.

Keep the summary descriptive and evidence-based. Do not overstate causality.
