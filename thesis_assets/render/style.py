"""House style for thesis figures and tables (see the ``thesis-asset`` skill).

Figures are drawn at their final printed size, so they must be inserted in Word at
100 %. Text is Times New Roman 11 pt, colours are black and greys, decimals use
the Greek decimal comma, and there is no title (the caption lives in Word).
"""

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

TEXT_WIDTH_CM = 16.0
"""Text width of ``thesis.docx``: A4 minus 3 cm left and 2 cm right margins."""

FONT_FAMILY = "Times New Roman"
FONT_SIZE_PT = 11.0
PNG_DPI = 300

BLACK = "#000000"
DARK_GREY = "#555555"
MID_GREY = "#8C8C8C"
LIGHT_GREY = "#D9D9D9"
FILL_GREY = "#EDEDED"
GREYS: Sequence[str] = (BLACK, DARK_GREY, MID_GREY, LIGHT_GREY)
"""Series colours, in order. Prefer line style / marker over colour to tell
series apart (the thesis may be printed in black and white)."""

_MINUS = "−"


def cm(value: float) -> float:
    """Convert centimetres to inches (matplotlib figure sizes are in inches)."""
    return value / 2.54


def apply_style() -> None:
    """Set matplotlib ``rcParams`` to the thesis house style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY],
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "mathtext.sf": FONT_FAMILY,
            "mathtext.tt": FONT_FAMILY,
            "mathtext.cal": f"{FONT_FAMILY}:italic",
            "axes.edgecolor": BLACK,
            "axes.labelcolor": BLACK,
            "axes.linewidth": 0.8,
            "axes.prop_cycle": matplotlib.cycler(color=list(GREYS)),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "text.color": BLACK,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "axes.unicode_minus": True,
            "svg.fonttype": "path",
            "svg.hashsalt": "thesis-assets",
            "figure.dpi": 100,
            "savefig.dpi": PNG_DPI,
        }
    )


def greek_number(value: float, decimals: int) -> str:
    """Format a number with a Greek decimal comma and a real minus sign.

    Args:
        value: The number.
        decimals: Digits after the decimal comma.

    Returns:
        E.g. ``greek_number(-0.25, 2) == "−0,25"``.
    """
    text = f"{value:.{decimals}f}".replace(".", ",")
    return text.replace("-", _MINUS)


def greek_formatter(decimals: int) -> FuncFormatter:
    """Tick formatter that writes ``decimals`` digits with a decimal comma."""
    return FuncFormatter(lambda x, _pos: greek_number(x, decimals))


def save_figure(fig: Figure, out_stem: Path) -> List[Path]:
    """Save ``fig`` as ``<out_stem>.svg`` (text as paths) and a 300-dpi PNG.

    The figure keeps the size it was created with (no tight bounding box), so
    text stays 11 pt when the figure is inserted at 100 %.

    Args:
        fig: The finished figure.
        out_stem: Output path without suffix (given by ``build.py``).

    Returns:
        The SVG and PNG paths.
    """
    width_cm = fig.get_figwidth() * 2.54
    if width_cm > TEXT_WIDTH_CM + 1e-6:
        raise ValueError(f"figure is {width_cm:.1f} cm wide; max {TEXT_WIDTH_CM} cm")
    svg = out_stem.with_suffix(".svg")
    png = out_stem.with_suffix(".png")
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(png, format="png", dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    return [svg, png]


def save_table(
    table: pd.DataFrame,
    out_stem: Path,
    decimals: int,
    formatters: Optional[dict] = None,
) -> List[Path]:
    """Save a table as an exact CSV plus a Word-ready HTML table.

    The CSV keeps full precision (the numbers behind the thesis). The HTML is
    rounded, uses the Greek decimal comma and Times New Roman 11 pt; open it in
    a browser and paste it into Word to get a native Word table.

    Args:
        table: Rows and columns exactly as they should appear (Greek headers).
        out_stem: Output path without suffix (given by ``build.py``).
        decimals: Default decimals for float columns in the HTML.
        formatters: Optional column → ``Callable[[value], str]`` overrides.

    Returns:
        The CSV and HTML paths.
    """
    csv_path = out_stem.with_suffix(".csv")
    html_path = out_stem.with_suffix(".html")
    table.to_csv(csv_path, index=False)
    fmt: dict = {}
    for col in table.columns:
        if formatters and col in formatters:
            fmt[col] = formatters[col]
        elif pd.api.types.is_float_dtype(table[col]):
            fmt[col] = _float_formatter(decimals)
    body = table.to_html(index=False, formatters=fmt, border=0, escape=True)
    style = (
        f"table{{border-collapse:collapse;font-family:'{FONT_FAMILY}';"
        f"font-size:{FONT_SIZE_PT:g}pt}}"
        "th,td{padding:2pt 6pt;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "thead tr{border-top:1pt solid #000;border-bottom:0.5pt solid #000}"
        "tbody tr:last-child{border-bottom:1pt solid #000}"
    )
    html_path.write_text(
        f'<!doctype html><html lang="el"><meta charset="utf-8">'
        f"<style>{style}</style>{body}</html>\n",
        encoding="utf-8",
    )
    return [csv_path, html_path]


def _float_formatter(decimals: int) -> Callable[[float], str]:
    """Formatter for float cells: Greek decimal comma, empty for NaN."""
    return lambda v: "" if pd.isna(v) else greek_number(float(v), decimals)
