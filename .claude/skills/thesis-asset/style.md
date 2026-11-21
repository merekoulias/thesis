# House style (all assets)

These defaults apply unless the user's request or `learnings.md` says otherwise. The code version is in `thesis_assets/render/style.py`, so use it rather than restating the values.

## Page and size
- The text width is **16.0 cm**: A4 with 3 cm left and 2 cm right margins in `thesis.docx`. No asset may be wider. `save_figure` refuses wider figures.
- **Draw at the final printed size** and insert in Word at **100 %**. Then 11 pt in the figure is 11 pt on paper. Never scale a figure in Word to make it fit; redraw it.
- Typical sizes: full width 16 cm; a single square plot about 8–10 cm; two plots side by side 16 cm in total.

## Text
- **Times New Roman 11 pt** for all text: labels, ticks, annotations, table cells. Don't go smaller (the department rules say at least 11 pt).
- **No title** in the figure. The caption lives in Word.
- **Greek decimal comma** in every number: `0,5`, `1,0`, `AUC ≈ 0,80`. Use `greek_number` / `greek_formatter`. Use a real minus sign (−, U+2212).
- **Real Unicode characters**, never ASCII stand-ins: Ŷ (U+0176), ≈, ∈, ≤, ≥, −, ×, Φ, ⁻¹.
- **Variables** (X, Y, R, t, Ŷ, A, G, g) are italic where the tool allows it: `$t$` mathtext in matplotlib. draw.io plain-text labels can't mix italic and upright, so variables stay upright there (see `kinds/diagram.md`).
- Terms come from the vault's `text/glossary.md`. One Greek term per concept, and the same term as in the text.

## Colour and lines
- **Black and greys only**, in this order: `BLACK`, `DARK_GREY`, `MID_GREY`, `LIGHT_GREY`. Fills use `FILL_GREY`. Everything must stay readable in black-and-white print.
- Tell series apart with line style or marker first (solid, dashed, dotted; ●, ▲, ■) and shade second.
- **Dashed means "not observed / reference / counterfactual"** (the chance diagonal, unobserved outcomes). Keep that meaning across figures.
- Line widths: data 1.2 pt, reference lines 0.9 pt, arrows 0.8 pt. Spines left and bottom only.

## Output files
- Figures and diagrams: **SVG** (vector, for Word; text stored as paths so Word can't substitute fonts) plus a **300-dpi PNG** (slides and preview).
- Tables: **CSV** with full precision (the exact numbers behind the thesis) plus **HTML** (rounded, decimal comma, Times New Roman 11 pt) to paste into Word as a native table.
- Every file is named `<id>.<ext>` and is written by `python -m thesis_assets.build`.

## Inserting in Word
- Figure: *Insert → Pictures → This Device → `<id>.svg`*, size 100 %. Then *References → Insert Caption → Σχήμα*, and paste the manifest caption.
- Update: right-click the figure → *Change Picture → From a File* (keeps size, position and caption).
- Table: open `<id>.html` in a browser, copy the table and paste it into Word. Then *Insert Caption → Πίνακας* **above** the table.

## Captions
- Greek, without the "Σχήμα N:" / "Πίνακας N:" prefix (Word adds it).
- The first sentence says what the asset is. The rest says what the reader should see, plus any encoding the asset has no legend for (dashed lines, shading).
- Illustrative assets say so: "Ενδεικτική …", "υποθετικά δεδομένα".
- Data-driven assets name the data and the sample if the text doesn't: "Home Credit, n = 307.507 αιτήσεις". Greek thousands separator is a dot.
