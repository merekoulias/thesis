---
name: thesis-asset
description: >-
  Creates, updates, removes or audits a table, figure, diagram or other visual
  asset for the thesis or the presentation, and freezes it in thesis_assets/
  with its manifest entry, Greek caption, provenance and AI-use log row. Use
  when the user asks to make, draw, plot, render, change, restyle, move or
  remove a thesis figure/table/diagram, or to check that the thesis assets are
  up to date before sending a draft.
argument-hint: "<what to make> | update <id or description>: <change> | remove <id> | audit"
---

# Thesis asset

One skill for every kind of asset. The steps that all kinds share are in this file. The house style is in `style.md`. Each kind has its own file under `kinds/`, which you read only when you need it.

Modes, chosen from `$ARGUMENTS`:

| Call | Mode |
|---|---|
| `/thesis-asset <description>` | **create** |
| `/thesis-asset update <id or description>: <change>` | **update** |
| `/thesis-asset remove <id>` | **remove** |
| `/thesis-asset audit` | **audit** |

If the user asks to change an asset in plain words without the keyword (for example "make the ROC arrow shorter" or "in the decision diagram rename Μοντέλο"), treat it as **update**.

## Step 0: load context (every mode)

1. `learnings.md`, next to this file. Its entries **override** the defaults here and in `style.md`.
2. `style.md`, then the `kinds/<kind>.md` file the request needs: `plot.md`, `diagram.md` or `table.md`.
3. `thesis_assets/README.md` and `thesis_assets/manifest.yaml`.
4. For terms and notation: the vault's `text/glossary.md` (if it exists), the chapter's guide in `text/guides/` (its "Figures and tables" and "Terms" sections), and rule 7 of `.cursor/rules/thesis-writing.mdc`.

## Files you may touch

Change **only** these. Anything else needs the user's explicit OK in this conversation.

- `thesis_assets/render/<chapter-or-topic>.py`: the render function for this asset, plus shared helpers in `render/style.py` if a fix belongs there.
- `thesis_assets/diagrams/<id>.drawio`
- `thesis_assets/figures/<id>.*` and `thesis_assets/tables/<id>.*`, **only through `build.py`**, never by hand
- `thesis_assets/manifest.yaml`, `numbers.yaml` and `provenance.json` (the last only through `build.py`)
- `docs/ai-use.md`: one row, new or extended
- vault `text/glossary.md`: new term pairs only
- `learnings.md` in this skill

Never edit `thesis.docx` or any chapter `.docx`, anything in `src/` or `artifacts/`, or another asset's render code. If the user's prompt says "don't change other files", it means files outside this list. The bookkeeping above is still required, because `CLAUDE.md` requires it.

## Create

1. **Parse the request.** Work out:
   - the kind: `figure` (a plot), `diagram` or `table`
   - the content
   - what the reader should see: the point of the asset
   - the chapter or section
   - `used_in`: thesis, slides or both
   - whether it is **data-driven** (reads `artifacts/`) or **illustrative** (no data)

   If the chapter, or what the reader should see, is missing and the chapter guide doesn't give it, ask. Don't guess. Anything the user specified overrides `style.md` for this asset only.
2. **Check the content before drawing.**
   - Notation must match the thesis: X, Y, R, t, Ŷ (Ŷ = 1 means refused), A, G, and the decision rule Ŷ = 1{R > t}, as in `src/fairness/thresholds.py` and the Ch. 2 guide.
   - Symbols the text doesn't define yet must be flagged.
   - Greek terms must match the glossary.
   - The point must be visible in the asset.

   Raise problems first, in one short list, and suggest a fix. Proceed with the user's version unless the problem is a factual error.
3. **Find the source.**
   - **Data-driven:** find the artifact file and the columns that hold the data. Show the user the path. If no artifact holds what is needed, **stop**: name the pipeline step (`src/...`) that would have to produce it. A render function never computes a metric.
   - **Illustrative:** record the parameters in the manifest `notes` (e.g. the formula), and add "Ενδεικτικ…" or "υποθετικά δεδομένα" to the caption.
4. **Pick a stable id.** It is a descriptive slug (`roc-illustration`, `fpr-gap-gender-age`), never a number like `fig-2-2`. Check that it is not already in the manifest.
5. **Make it**, following `kinds/<kind>.md`:
   - a render function in `thesis_assets/render/<chNN or topic>.py`, or
   - `thesis_assets/diagrams/<id>.drawio`, or
   - a copy of an artifact (`copy`) when the artifact already meets `style.md`. It rarely does, because artifact figures are in English.
6. **Add the manifest entry:** `id`, `kind`, `chapter`, `caption`, `used_in`, `source`, `notes`.
   - The caption is Greek, has no "Σχήμα N:" prefix (Word numbers figures), and says what the reader should see. It must explain any encoding the figure has no legend for, such as "διακεκομμένη γραμμή = …".
   - If the asset shows headline numbers that the text will quote, add them to `numbers.yaml`.
7. **Build and check:** `python -m thesis_assets.build --only <id>`, then `python -m thesis_assets.build --check`. Both must pass. Fix and repeat until they do.
8. **Look at the output.** Read the PNG, or open the HTML for tables. Check for overlapping or clipped text, text crossing lines, wrong glyphs (Ŷ, −, ≈, ∈), the decimal comma, width ≤ 16 cm and black-and-white legibility. Fix and rebuild until clean, at most three rounds. Then report anything still not right.
9. **Glossary:** add any new Greek term pairs the asset introduces to `text/glossary.md` (create it if missing) and list them in your reply.
10. **Log it** in `docs/ai-use.md`, section 3.1 for the code and 3.2 for the caption. AI role `generated`, plus the date, the location (`thesis_assets/...`), the model and the extent.
11. **Reply** with:
    - the PNG (show it)
    - the paths of the SVG/PNG or CSV/HTML
    - the caption
    - one sentence the text could use to point to the asset and say what to see in it (a suggestion; the user writes the text)
    - how to insert it in Word, from `style.md`
    - glossary additions
    - anything you flagged in step 2

## Update

The id never changes on an update. If the change makes it a different asset, it's a **create** plus a **remove**.

1. **Find the asset.** Use the id or match the description against the manifest's ids, captions and notes. If two assets match, ask which one.
2. **Find out what changed and who changed it:**
   - **The user asks for a change** (content, style, labels, data): edit the source, i.e. the render function, the `.drawio` file, or the `copy` path / `inputs`.
   - **The user edited the source themselves** (for example the `.drawio` file in the draw.io app): don't touch the source. Rebuild it.
   - **`--check` says the asset is stale** (an artifact was regenerated, or the code or `.drawio` file changed): show what changed (which input), rebuild, and compare the old and new PNG. If numbers in `numbers.yaml` changed, list them: the text must change too.
   - **The caption or chapter changed in Word:** update the manifest only, with no rebuild.
3. **Keep the old PNG** for comparison: copy it to the scratchpad before rebuilding.
4. **Do steps 7–10 of Create.** For the AI-use log, extend the asset's existing row (date range, what changed) rather than adding a new one, unless the AI role changed.
5. **Reply** with the old and new PNG side by side (or a short diff for tables), what changed, and the Word step: in Word, right-click the figure → *Change Picture* → *From a File* → pick the new `.svg`. This keeps the size, position and caption. For tables, paste the new HTML table over the old one. If the caption changed, give the new caption to paste.

## Remove

1. Confirm with the user. Show the entry and where it is used (`chapter`, `used_in`).
2. Delete the manifest entry, the source (render function, if no other asset uses it, or the `.drawio` file) and the outputs. Remove its `numbers.yaml` entries.
3. Run `python -m thesis_assets.build --check`, which drops the provenance entry on the next build. It must pass.
4. Log the removal in `docs/ai-use.md` if AI made the asset. Remind the user to delete the figure and its cross-references in Word.

## Audit

Run this before a draft goes to the supervisor.

1. `python -m thesis_assets.build --check`: report the errors and warnings.
2. Read `thesis.docx` and the chapter `.docx` files in the vault's `text/` with `pandoc <file> -t plain`. Collect every "Σχήμα N:" and "Πίνακας N:" caption. Match them to manifest captions (normalise whitespace). Report:
   - captions in Word with no manifest entry: an asset not frozen
   - manifest entries with `used_in: thesis` whose caption is not in Word: the asset was removed, or its caption was edited in one place only
   - captions that differ slightly between Word and the manifest: show both
3. For each `numbers.yaml` entry, check that the value appears in the text, written with a Greek decimal comma.
4. List the `_TODO_` rows still open in `docs/ai-use.md`.
5. Report as a checklist. Fix nothing without asking.

## Learn

When the user corrects a style choice, a term or a layout habit that will recur, add a short dated entry to `learnings.md`. One-off content changes don't count. Merge or replace entries; never keep two that contradict each other.
