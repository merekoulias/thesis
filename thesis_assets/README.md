# Thesis assets

Frozen copies of every table, figure and diagram used in the thesis (`thesis.docx` in the vault) and the presentation, plus a register of key numbers quoted in the text. This folder is tracked by git. `artifacts/` is gitignored and gets overwritten, so the version that went into the thesis lives here, with a record of where it came from.

To create, change, remove or audit assets, use the Claude Code skill **`/thesis-asset`** (`.claude/skills/thesis-asset/`). It follows the conventions below and keeps the manifest, the glossary and `docs/ai-use.md` in step.

## Layout

```
thesis_assets/
  README.md         # this file
  manifest.yaml     # one entry per table/figure/diagram (hand-edited)
  numbers.yaml      # key in-text numbers and their sources (hand-edited)
  provenance.json   # written by build.py: commit, input and output hashes
  build.py          # builds and checks everything below
  types.py
  render/           # render functions (one module per chapter) + style.py
  diagrams/         # <id>.drawio sources (diagrams.net)
  figures/          # <id>.svg + <id>.png (300 dpi), built
  tables/           # <id>.csv (exact) + <id>.html (for Word), built
```

## Commands (from the repo root)

```bash
python -m thesis_assets.build                  # build every asset in the manifest
python -m thesis_assets.build --only ID [ID]   # build some assets
python -m thesis_assets.build --check          # validate; exit 1 on any error
```

Diagrams need the draw.io desktop app (`brew install --cask drawio`), or `DRAWIO_BIN` pointing at its CLI.

`--check` reports:
- **errors:** an asset never built; an output missing or edited by hand; an asset **stale** because one of its inputs changed since the build (an artifact, the `.drawio` file, or the render code including `render/style.py`); a file in `figures/`, `tables/` or `diagrams/` with no manifest entry; a number in `numbers.yaml` that no longer matches its source.
- **warnings:** an input or number source missing on this machine. This is normal on a fresh clone, because `artifacts/` is not in git.

## Conventions

- **Ids are stable descriptive slugs** (`roc-illustration`), never numbers like "Figure 2.2". Thesis numbering changes whenever a section moves. The id is the file-name stem, and `chapter` records where the asset currently sits. An update never changes the id.
- **`caption` is the caption text as in Word**, in Greek, without the "Σχήμα N:" / "Πίνακας N:" prefix. If you change it in Word, change it here too. `/thesis-asset audit` compares the two.
- **Never edit files in `figures/` or `tables/` by hand.** Change the source and rebuild. `--check` flags hand edits.
- **Three kinds of source:**
  - `copy`: a pipeline artifact that is already fit for the thesis is copied as is. This is rare, because artifact figures are in English.
  - `render: module:function`: a Python function in `render/` that writes `<out_stem>.<ext>` and returns the paths. Data-driven renders **read artifacts only and never recompute a metric**, and list what they read in `inputs`. Illustrative renders read nothing and say so in the caption.
  - `drawio`: a `.drawio` file in `diagrams/` (`kind: diagram` only), exported by the draw.io CLI. You can edit it in the draw.io app, then rebuild.
- **House style** (full version in `.claude/skills/thesis-asset/style.md`, code in `render/style.py`):
  - Times New Roman 11 pt, black and greys, no title, Greek decimal comma, real Unicode symbols (Ŷ, −, ≈).
  - Drawn at the final size, at most 16.0 cm wide (the text width of `thesis.docx`), and inserted in Word at 100 %.
  - Figures as SVG (vector, for Word) plus a 300-dpi PNG (slides). Tables as an exact CSV plus an HTML table to paste into Word as a native table.
- **`used_in`** lists `thesis` and/or `slides`. An asset made only for the slides still gets an entry.
- **Retiring an asset:** delete its entry, its source and its files, then rebuild. Or use `/thesis-asset remove <id>`.

## Updating an asset

| What changed | What to do |
|---|---|
| You want a different look or content | `/thesis-asset update <id>: <change>`, or edit the source yourself and run `--only <id>` |
| You edited the `.drawio` file in the app | `python -m thesis_assets.build --only <id>` |
| `--check` says *stale* (artifacts regenerated, code changed) | rebuild with `--only <id>`, compare the PNG, and fix the text if a registered number changed |
| Caption or section changed in Word | update `caption` / `chapter` in the manifest (no rebuild) |

In Word, replace a figure with right-click → *Change Picture* → *From a File*. This keeps its size, position and caption.

## Numbers register

`numbers.yaml` lists the numbers the text leans on: headline AUCs, the main gaps and the counts behind them. Each entry points to a JSON key or a CSV row/column in `artifacts/`. `--check` re-reads each source and compares it at the number of decimals used in the text. Two optional source flags make the source comparable with how the text writes it: `percent: true` multiplies by 100 (the text says 8,07%, the artifact holds 0.0807), and `complement: true` takes 1 − x first (e.g. the share of applicants without payment difficulties, from the base rate). This enforces the rule in `.cursor/rules/thesis-writing.mdc` that every number comes from `artifacts/`. You don't need to register every number, only the ones a rerun could change.
