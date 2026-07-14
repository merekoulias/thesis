# Sync thesis decisions — examples

## Example 1: Resolve open question → new accepted decision

**User:** "Let's use fairlearn for fairness metrics and mitigation."

**Actions:**

1. Add row to `docs/decisions.md`:

| Date | Name | Decision | Status |
|------|------|----------|--------|
| 2026-07 | Fairness library | Use **fairlearn** for fairness metrics and pre/in/post-processing with sklearn-compatible LightGBM. | accepted |

2. Remove from `docs/open-questions.md` under Fairness methodology:
   - `Fairness library: Confirm fairlearn vs AIF360 vs other.`

3. Update `thesis-context.mdc` Fairness line: replace "prefer fairlearn unless changed" with "use fairlearn".

---

## Example 2: Update existing open decision → accepted

**User:** "Confirm the default fairness metrics list from decisions."

**Actions:**

1. Find row **Default fairness metrics** (status `open`).
2. Keep Decision text; change Status to `accepted`.
3. Remove **Final fairness metric set** bullet from open-questions.md.

Do **not** change closed rows.

---

## Example 3: Mark implemented decision as closed

**User:** "Baseline re-run on Mac is done and documented in experiments.md."

**Actions:**

1. If a decision row exists for the baseline run policy, set Status to `closed`.
2. Remove **Re-run baseline on new Mac** from open-questions.md Infrastructure section.
3. Do not edit other closed rows (e.g. Repository language, No Kaggle submission).

---

## Example 4: New open question from discussion

**User:** "I'm not sure whether to merge rare education categories yet."

**Actions:**

1. Do **not** add to decisions.md (unresolved).
2. Ensure open-questions.md Intersectionality section has:
   - `**Education category handling:** ...`

Only add a decisions row when the user picks an approach.

---

## Example 5: User corrects a decision

**User:** "Change classifier decision: we'll use 10-fold CV instead of 5-fold."

**Actions:**

1. Find **Classifier and evaluation metric** (must not be `closed`).
2. Update Decision cell: `10-fold` instead of `5-fold`.
3. Keep Status as `open` unless user says it's final.

---

## Anti-patterns

| Bad | Good |
|-----|------|
| Set status `closed` without user saying work is done | `accepted` until implemented, then `closed` |
| Edit a `closed` row | Add a new decision if requirements truly change (rare; confirm with user) |
| Leave resolved bullets in open-questions.md | Remove or narrow them |
| Duplicate table rows for the same Name | Update the existing row |
