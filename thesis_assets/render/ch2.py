"""Render functions for Chapter 2 (credit scoring and machine learning)."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from thesis_assets.render.style import (
    BLACK,
    DARK_GREY,
    FILL_GREY,
    apply_style,
    cm,
    greek_formatter,
    save_figure,
)

ROC_BINORMAL_A = 1.19
"""Binormal separation; AUC = Φ(a / √2) ≈ 0.80 for equal variances."""
ROC_POINT_FPR = 0.40
"""FPR of the marked threshold point."""


def _binormal_tpr(fpr: np.ndarray, a: float) -> np.ndarray:
    """TPR of the equal-variance binormal ROC curve, ``Φ(a + Φ⁻¹(FPR))``."""
    return norm.cdf(a + norm.ppf(fpr))


def roc_illustration(out_stem: Path) -> List[Path]:
    """Illustrative ROC curve (no data): AUC, chance line and a threshold point.

    Args:
        out_stem: Output path without suffix.

    Returns:
        The SVG and PNG paths.
    """
    apply_style()
    a = ROC_BINORMAL_A
    fpr = np.linspace(0.0, 1.0, 1001)
    tpr = _binormal_tpr(fpr, a)

    fig, ax = plt.subplots(figsize=(cm(9.0), cm(9.0)), layout="constrained")
    ax.fill_between(fpr, tpr, color=FILL_GREY, linewidth=0)
    ax.plot(fpr, tpr, color=BLACK)
    ax.plot([0, 1], [0, 1], color=DARK_GREY, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.text(0.70, 0.30, "AUC", ha="center", va="center")
    ax.text(
        0.68,
        0.58,
        "Τυχαία ταξινόμηση (AUC = 0,5)",
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="center",
        color=DARK_GREY,
    )

    x0 = ROC_POINT_FPR
    y0 = float(_binormal_tpr(np.array([x0]), a)[0])
    ax.plot([x0], [y0], marker="o", markersize=4, color=BLACK)
    ax.annotate(
        "κατώφλι $t$",
        xy=(x0, y0),
        xytext=(6, -4),
        textcoords="offset points",
        ha="left",
        va="top",
    )

    # Short arrow following the curve towards (0, 0), just above-left of it.
    seg = np.linspace(x0 - 0.05, 0.17, 40)
    path_x = seg - 0.04
    path_y = _binormal_tpr(seg, a) + 0.05
    ax.plot(path_x[:-1], path_y[:-1], color=BLACK, linewidth=0.8)
    ax.annotate(
        "",
        xy=(path_x[-1], path_y[-1]),
        xytext=(path_x[-3], path_y[-3]),
        arrowprops={"arrowstyle": "-|>", "color": BLACK, "linewidth": 0.8},
    )
    ax.text(path_x[0], path_y[0] + 0.02, "μεγαλύτερο $t$", ha="right", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(greek_formatter(1))
    ax.yaxis.set_major_formatter(greek_formatter(1))
    ax.set_xlabel("Ποσοστό ψευδώς θετικών (FPR)")
    ax.set_ylabel("Ποσοστό αληθώς θετικών (TPR)")
    return save_figure(fig, out_stem)
