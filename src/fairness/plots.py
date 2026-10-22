"""Matplotlib figures for the fairness audit."""

from pathlib import Path
from typing import Dict, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fairness.types import GENDER_AGE_COL, AuditConfig

PathLike = Union[str, Path]

FIG_DPI = 150


def _group_colors(labels: List[str]) -> Dict[str, tuple]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {label: cycle[i % len(cycle)] for i, label in enumerate(labels)}


def plot_calibration(
    calibration: pd.DataFrame,
    grouping: str,
    model: str,
    out_path: PathLike,
) -> None:
    """Calibration figure: mean score vs observed default rate by decile bin.

    Args:
        calibration: ``calibration.csv`` content.
        grouping: ``CODE_GENDER`` or ``AGE_BAND``.
        model: Model name for the title.
        out_path: PNG output path.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = calibration[calibration["grouping"] == grouping]
    groups = sorted(sub["group"].unique())
    colors = _group_colors(groups)
    fig, ax = plt.subplots(figsize=(7, 5))
    for group in groups:
        gdf = sub[sub["group"] == group].sort_values("bin")
        x = gdf["mean_score"].to_numpy()
        y = gdf["observed_rate"].to_numpy()
        yerr = np.vstack([y - gdf["ci_low"].to_numpy(), gdf["ci_high"].to_numpy() - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            capsize=3,
            label=group,
            color=colors[group],
        )
    lims = [0.0, 1.0]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk (bin)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"{model}: calibration by {grouping} (deciles of pooled score)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_cells_error_rates(
    group_metrics: pd.DataFrame,
    config: AuditConfig,
    model: str,
    out_path: PathLike,
) -> None:
    """Side-by-side FPR and FNR by gender × age cell with ALL reference lines.

    Args:
        group_metrics: ``group_metrics.csv`` content.
        config: Audit settings (cell order).
        model: Model name for the title.
        out_path: PNG output path.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = config.groupings[GENDER_AGE_COL]
    colors = _group_colors(cells)

    def series(metric: str) -> pd.DataFrame:
        sub = group_metrics[
            (group_metrics["grouping"] == GENDER_AGE_COL)
            & (group_metrics["metric"] == metric)
        ].set_index("group")
        return sub.reindex(cells)

    fpr = series("fpr")
    fnr = series("fnr")
    all_fpr = group_metrics[
        (group_metrics["grouping"] == "ALL") & (group_metrics["metric"] == "fpr")
    ].iloc[0]
    all_fnr = group_metrics[
        (group_metrics["grouping"] == "ALL") & (group_metrics["metric"] == "fnr")
    ].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    x = np.arange(len(cells))
    for ax, metric_df, all_row, title in (
        (axes[0], fpr, all_fpr, "FPR"),
        (axes[1], fnr, all_fnr, "FNR"),
    ):
        y = 100.0 * metric_df["value"].to_numpy()
        yerr = np.vstack(
            [
                y - 100.0 * metric_df["ci_low"].to_numpy(),
                100.0 * metric_df["ci_high"].to_numpy() - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=3,
            color=colors[cells[0]],
            ecolor="gray",
        )
        for i, cell in enumerate(cells):
            ax.scatter(i, y[i], color=colors[cell], zorder=3)
        ax.axhline(100.0 * all_row["value"], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{model}: {title} by cell")
        ax.set_ylabel("Rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_primary_gaps(primary_gaps: pd.DataFrame, out_path: PathLike) -> None:
    """Cross-model primary gaps: rank 1 filled, ranks 2–5 hollow.

    Args:
        primary_gaps: ``primary_gaps.csv`` content.
        out_path: PNG output path.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["fpr", "fnr"]
    titles = {"fpr": "F − M EOD (FPR gap)", "fnr": "F − M ΔFNR"}
    models = sorted(primary_gaps["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, len(models) * 0.6)), sharey=True)
    y_pos = np.arange(len(models))

    for ax, metric in zip(axes, metrics):
        sub = primary_gaps[primary_gaps["metric"] == metric]
        for i, model in enumerate(models):
            msub = sub[sub["model"] == model]
            for _, row in msub.iterrows():
                rank = int(row["rank"])
                x = 100.0 * row["diff"]
                xerr = np.array(
                    [
                        [100.0 * (row["diff"] - row["diff_ci_low"])],
                        [100.0 * (row["diff_ci_high"] - row["diff"])],
                    ]
                )
                if rank == 1:
                    ax.errorbar(
                        x,
                        i,
                        xerr=xerr,
                        fmt="o",
                        capsize=3,
                        color="black",
                        markersize=8,
                    )
                else:
                    ax.errorbar(
                        x,
                        i,
                        xerr=xerr,
                        fmt="o",
                        mfc="white",
                        mec="black",
                        capsize=2,
                        markersize=5,
                    )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel("Gap (pp)")
        ax.set_title(titles[metric])
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
