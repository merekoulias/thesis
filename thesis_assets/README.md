# Thesis assets

Frozen copies of every table and figure used in the thesis (`thesis.docx` in the vault) and the presentation, plus a register of key numbers quoted in the text. This folder is tracked by git. `artifacts/` is gitignored and gets overwritten, so the version that went into the thesis lives here, with a record of where it came from.

## Layout

```
thesis_assets/
  README.md         # this file
  manifest.yaml     # one entry per table/figure (hand-edited)
  numbers.yaml      # key in-text numbers and their sources (hand-edited)
  provenance.json   # written by build.py: commit, input and output hashes
  build.py          # builds and checks everything below
  types.py
  render/           # render functions for thesis-only figures/tables
  figures/          # <id>.pdf / <id>.png
  tables/           # <id>.csv
```

## Commands (from the repo root)

```bash
python -m thesis_assets.build                  # build every asset in the manifest
python -m thesis_assets.build --only ID [ID]   # build some assets
python -m thesis_assets.build --check          # validate; exit 1 on any error
```

`--check` reports:
- **errors:** an asset never built; an output missing or edited by hand; an asset **stale** because one of its inputs changed since the build; a file in `figures/` or `tables/` with no manifest entry; a number in `numbers.yaml` that no longer matches its source.
- **warnings:** an input or number source missing on this machine. This is normal on a fresh clone, because `artifacts/` is not in git.

## Conventions

- **Ids are stable descriptive slugs** (`primary-gaps-baseline`), never numbers like "Figure 4.2". Thesis numbering changes whenever a section moves. The id is the file-name stem, and `chapter` records where the asset currently sits.
- **`caption` is the caption exactly as it appears in the thesis**, in Greek. If you change it in Word, change it here too.
- **Never edit files in `figures/` or `tables/` by hand.** Change the source or the render function and rebuild. `--check` flags hand edits.
- **Two ways to make an asset:**
  - `copy`: the pipeline artifact is already fit for the thesis. It is copied as is.
  - `render`: the thesis needs its own version: Greek labels, print size, vector format, a subset of rows. Put the function in `render/<chapter-or-topic>.py`. It has the signature `fn(out_stem: Path) -> List[Path]` and writes `<out_stem>.<ext>`. It **reads artifacts only and never recomputes a metric**, so thesis numbers stay identical to the audit outputs. List every artifact it reads in `inputs`.
- **Formats:** figures as `.pdf` (vector, for the thesis) plus `.png` at 300 dpi (for slides). Tables as `.csv` with full precision. The thesis shows rounded values, so the CSV keeps the exact numbers behind them.
- **Fonts:** matplotlib's default DejaVu Sans covers Greek. If you switch to Times New Roman or Calibri to match the body text (see `text/writting-rules.md`), check that Greek glyphs render.
- **`used_in`** lists `thesis` and/or `slides`. An asset made only for the slides still gets an entry.
- **Retiring an asset:** delete its entry and its files, then rebuild. The provenance entry is dropped on the next build.

## Numbers register

`numbers.yaml` lists the numbers the text leans on: headline AUCs, the main gaps and the counts behind them. Each entry points to a JSON key or a CSV row/column in `artifacts/`. `--check` re-reads each source and compares it at the number of decimals used in the text. This enforces the rule in `.cursor/rules/thesis-writing.mdc` that every number comes from `artifacts/`. You don't need to register every number, only the ones a rerun could change.
