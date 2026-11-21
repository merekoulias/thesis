# Kind: plot (`kind: figure`, matplotlib)

For charts, curves, bar plots, heatmaps and anything else drawn from numbers or a formula. Example: `thesis_assets/render/ch2.py:roc_illustration`.

## Where the code goes
- `thesis_assets/render/ch<N>.py`, one module per chapter, or `render/<topic>.py` if several chapters share it. A function per asset, named after its id with underscores (`roc-illustration` → `roc_illustration`).
- Manifest: `source: {render: thesis_assets.render.ch2:roc_illustration, inputs: [<artifact paths read>]}`. `build.py` hashes the module and `style.py` itself, so you don't need to list them in `inputs`.
- Repo code style: Black (88), isort, type hints with `typing`, Google docstrings. Run `black` and `isort` on the file.

## Function template

```python
def <asset_id>(out_stem: Path) -> List[Path]:
    """<One line: what the figure shows>.

    Args:
        out_stem: Output path without suffix.

    Returns:
        The SVG and PNG paths.
    """
    apply_style()
    data = pd.read_csv(REPO_ROOT / "artifacts/...")   # data-driven only
    fig, ax = plt.subplots(figsize=(cm(16.0), cm(7.0)), layout="constrained")
    ...
    ax.xaxis.set_major_formatter(greek_formatter(1))
    return save_figure(fig, out_stem)
```

- Get `REPO_ROOT` from `thesis_assets.build`, so the paths don't depend on the working directory.
- Use `layout="constrained"` and **no** `bbox_inches="tight"`: the figure must keep its width.
- Put module-level constants for the parameters of illustrative curves (e.g. `ROC_BINORMAL_A = 1.19`), with a docstring line that says what they give (AUC ≈ 0.80).

## Rules
- **Read, don't compute.** Data-driven plots read numbers from `artifacts/` (CSV/JSON/parquet) and only reshape them: filter, sort, pivot, relabel. Rounding is for display only. If a number isn't in an artifact, stop (SKILL.md step 3).
- **Greek labels** for the axes, legend and annotations. Map English column or group labels to Greek in a dict at the top of the module (e.g. `{"F": "Γυναίκες", "M": "Άνδρες"}`), using the glossary terms.
- **Groups and cells:** use the order the fairness audit uses (gender F, M; age bands from youngest to oldest). Show `n` where it matters, and mark cells with small `n`.
- **Uncertainty:** if the artifact has CIs, draw them as error bars or bands. Don't drop them silently.
- **Annotations** never cross data lines or other text. Put labels in empty space and use a thin arrow if needed. Check the PNG.
- **Legends:** prefer direct labels on lines over a legend box. If you need a legend, keep it frameless and inside the free area.

## Common fixes after viewing the PNG
- Text touching a line: move it about 0.03–0.05 axis units perpendicular to the line. With `rotation_mode="anchor"`, alignment happens before rotation.
- Long Greek labels: break them onto two lines with `\n`, or widen the figure up to 16 cm.
- `findfont` warnings: every `mathtext.*` slot must point to Times New Roman, which `apply_style` does. Don't introduce other families.
