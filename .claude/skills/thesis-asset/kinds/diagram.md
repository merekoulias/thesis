# Kind: diagram (`kind: diagram`, diagrams.net / draw.io)

For flowcharts, pipelines, schematics and concept maps: boxes and arrows, no data. Example: `thesis_assets/diagrams/score-decision-outcome.drawio`.

## Files and build
- Source: `thesis_assets/diagrams/<id>.drawio`. It is uncompressed XML: write it directly and keep it readable, so the user can also open and edit it in the draw.io app.
- Manifest: `kind: diagram`, `source: {drawio: thesis_assets/diagrams/<id>.drawio}`.
- `build.py` exports it with the draw.io desktop CLI (`/Applications/draw.io.app`, or `$DRAWIO_BIN`) to `figures/<id>.svg` and a 300-dpi PNG, with the light theme and a 4 px border. The CLI prints harmless `mach_port_rendezvous` errors on macOS; ignore them.
- If the user edits the `.drawio` file in the app, `--check` reports the asset as stale. Rebuild it, and don't rewrite their layout.

## Units
draw.io and Word both treat 1 px as 1/96 inch.
- Max diagram width: **16 cm = 605 px**, including the 8 px of border, so the content should be **≤ 595 px**.
- **11 pt = 14.67 px** → `fontSize=14.67`.
- Rough text width in Times New Roman at 14.67 px: about 6.6 px per character. Break lines before a label reaches the box width.

## Cell styles

```
box:     rounded=0;whiteSpace=nowrap;html=0;fillColor=#FFFFFF;strokeColor=#000000;fontColor=#000000;fontFamily=Times New Roman;fontSize=14.67;
diamond: rhombus;<same as box without rounded>
dashed:  add dashed=1;
edge:    endArrow=block;endFill=1;endSize=6;html=0;strokeColor=#000000;   (+ fontColor/fontFamily/fontSize when it has a label)
```

- **`html=0` always.** HTML labels export as `<foreignObject>`, which Word can't render. Plain labels export as SVG `<text>`. Break lines with `&#xa;` in `value`. Escape `>` as `&gt;` and `<` as `&lt;`.
- Plain labels can't mix italic and upright, so variables stay upright in diagrams. Say so in the reply if the plot figures use italic variables.
- Edge labels such as "Ναι" / "Όχι": set the `value` on the edge, and add `<mxPoint x="16" y="0" as="offset"/>` inside its geometry to move the label off the line.
- Pin edge ends with `exitX/exitY/entryX/entryY` when the default routing picks the wrong side (for example diamond top/bottom).
- Give every cell a readable `id` (`score`, `rule`, `e-rule-refuse`) so later updates can find it.

## Layout
- Flow left to right unless the user asks otherwise. Branch vertically from a decision (diamond), with the "yes" branch up and the "no" branch down.
- Keep boxes on one row the same height, and leave 20–25 px between boxes.
- Align the centres of boxes connected by straight arrows, so the arrows are straight.
- No title box. No legend unless the caption can't carry the encoding.
- After the build, read the PNG. Check for text overflowing a box (the commonest problem; fix it with a line break or a wider box), crossing arrows, and the width in cm (PNG width / 300 × 2.54 ≤ 16).
