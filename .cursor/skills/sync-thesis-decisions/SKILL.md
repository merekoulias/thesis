---
name: sync-thesis-decisions
description: >-
  Updates docs/decisions.md and docs/open-questions.md to keep thesis design
  choices in sync. Use when the user resolves an open question, records a new
  decision, changes decision status (accepted/open/closed), or asks to sync
  documentation after a discussion about thesis scope, methodology, or repo
  conventions.
disable-model-invocation: true
---

# Sync thesis decisions

Keep `docs/decisions.md` and `docs/open-questions.md` aligned. Read both files before editing.

## When to use

- User resolves or changes a design choice
- User says "record this decision", "update decisions", "sync docs"
- End of session: propose doc updates from decisions made in chat (confirm with user first)

## Do not

- Invent or finalize decisions the user has not stated
- Change rows with status **closed** (final state)
- Remove the Statuses legend or table headers from `decisions.md`

## Status lifecycle

| Status | Use when |
|--------|----------|
| **open** | Under discussion; not final |
| **accepted** | Settled in principle; may not be implemented yet |
| **closed** | Settled, implemented, and documented; **do not edit again** |

Typical flow: `open` → `accepted` → `closed` (when work is done). Skip `accepted` only if the user closes in one step.

## Workflow

Copy this checklist and track progress:

```
- [ ] 1. Read docs/decisions.md and docs/open-questions.md
- [ ] 2. Identify what changed (new / update / resolve)
- [ ] 3. Edit decisions.md
- [ ] 4. Edit open-questions.md
- [ ] 5. Update thesis-context.mdc if needed (accepted/closed only)
- [ ] 6. Summarize changes to user
```

### Step 1 — Classify the change

| Type | Action |
|------|--------|
| **New decision** | Add table row |
| **Resolve open question** | Update or add decision row; remove matching bullet from open-questions.md |
| **Revise decision** | Edit Decision column and/or Status on existing row (not if closed) |
| **Reopen** | Change status from accepted → open; optionally add bullet back to open-questions.md |
| **New open question** | Add bullet under the right section in open-questions.md |

### Step 2 — Edit `docs/decisions.md`

Table columns: **Date | Name | Decision | Status**

Rules:

- **Date:** `YYYY-MM` (e.g. `2026-07`). Use the month the decision was made or confirmed.
- **Name:** Short title; match existing style (sentence case, no trailing period).
- **Decision:** One cell; concise but complete. Include constraints and file references where relevant.
- **Sort:** Newest date first; within the same month, newest entry at the top.
- **Duplicate names:** Update the existing row instead of adding a second row with the same Name.

### Step 3 — Edit `docs/open-questions.md`

- Remove the bullet when fully resolved and recorded in decisions.md.
- If partially resolved, narrow the bullet to what remains open.
- New bullets: place under the existing section (Advisor, Fairness methodology, Intersectionality, Modeling, Infrastructure, Written thesis).
- Keep format: `- **Topic label:** description`

### Step 4 — Update `.cursor/rules/thesis-context.mdc` (optional)

Only when a change is **accepted** or **closed** and affects always-on context (dataset, protected attributes, metrics defaults, repo conventions). Keep the rule under ~30 lines; link to `docs/decisions.md` for detail.

## Examples

See [examples.md](examples.md).

## Related files

- `docs/thesis-overview.md` — update RQs or pipeline description only if the user explicitly changes scope
- `docs/experiments.md` — not in scope for this skill (use a separate experiment-log skill)
