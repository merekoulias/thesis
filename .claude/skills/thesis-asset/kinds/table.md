# Kind: table (`kind: table`)

For any table in the thesis: results, model summaries, notation tables, comparison tables. Tables go into Word as **native Word tables**, not images.

## Files and build
- A render function in `thesis_assets/render/ch<N>.py` that builds a `pandas.DataFrame` exactly as it should appear (Greek column headers, Greek row labels, rows in order), then returns `save_table(df, out_stem, decimals=...)`.
- `save_table` writes:
  - `tables/<id>.csv`: full precision, the exact numbers behind the thesis. Values stay numeric here; don't round.
  - `tables/<id>.html`: rounded, Greek decimal comma, Times New Roman 11 pt, booktabs-style rules (top, header, bottom). This is the file you paste into Word.
- Manifest: `kind: table`, `source: {render: thesis_assets.render.chN:<fn>, inputs: [<artifacts read>]}`. A `copy` of an artifact CSV is allowed only if its headers are already Greek and it needs no reshaping, which is rare.
- **Text-only tables** (e.g. "the four models: flexibility, interpretability, hyperparameters") are tables too. Build the DataFrame from literals in the render function, and keep facts that come from the repo (e.g. hyperparameter names from `configs/models/*.yaml`) read from those files and listed in `inputs`.

## Rules
- **Decimals:** choose per column, using the precision the text quotes: rates usually 3, gaps 3, ratios 2, counts 0 with the Greek thousands dot. Pass `formatters={"col": fn}` for the exceptions. For CIs, put "0,045 [0,041, 0,049]" in one cell, or use separate columns if the text compares them.
- **Units in headers**, not in cells: "FPR (%)" or "FPR". Never mix % and proportions in one table.
- **No empty cells without a reason.** Write "—" and explain it in a note under the table if needed.
- **Width:** a table must fit in 16 cm at 11 pt. If it has more than about 7 numeric columns, split it or transpose it.
- **Numbers the text quotes** go into `numbers.yaml` with the CSV row/column as their source.

## Check
- Open the HTML (or read the CSV) after the build. Check the headers, the row order, the rounding and that every number has a decimal comma.
- Reply with the table as markdown (rounded, Greek) so the user can review it in the chat, plus the HTML path and the Word steps from `style.md`.
