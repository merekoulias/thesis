"""Matplotlib figures for the fairness audit."""

from pathlib import Path
from typing import Dict, Sequence, Set, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fairness.types import (
    ALL_GROUP,
    ALL_GROUPING,
    FNR_METRIC,
    FPR_METRIC,
    GENDER_AGE_COL,
    METRIC_LABELS,
    AuditConfig,
)

PathLike = Union[str, Path]

FIG_DPI = 150
AXIS_MARGIN = 1.05


def _group_colors(labels: Sequence[str]) -> Dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {label: cycle[i % len(cycle)] for i, label in enumerate(labels)}


def plot_calibration(
    calibration: pd.DataFrame,
    grouping: str,
    model: str,
    out_path: PathLike,
    config: AuditConfig,
) -> None:
    """Calibration figure: mean score vs observed default rate by decile bin.

    Args:
        calibration: ``calibration.csv`` content.
        grouping: ``CODE_GENDER`` or ``AGE_BAND``.
        model: Model name for the title.
        out_path: PNG output path.
        config: Audit settings (group order).
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = calibration[calibration["grouping"] == grouping]
    groups = [g for g in config.groupings[grouping] if g in set(sub["group"])]
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
    upper = AXIS_MARGIN * float(
        np.nanmax(sub[["mean_score", "observed_rate", "ci_high"]].to_numpy())
    )
    ax.plot([0.0, upper], [0.0, upper], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
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

    fpr = series(FPR_METRIC)
    fnr = series(FNR_METRIC)

    def pooled(metric: str) -> pd.Series:
        return group_metrics[
            (group_metrics["grouping"] == ALL_GROUPING)
            & (group_metrics["group"] == ALL_GROUP)
            & (group_metrics["metric"] == metric)
        ].iloc[0]

    all_fpr = pooled(FPR_METRIC)
    all_fnr = pooled(FNR_METRIC)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    x = np.arange(len(cells))
    for ax, metric_df, all_row, title in (
        (axes[0], fpr, all_fpr, FPR_METRIC.upper()),
        (axes[1], fnr, all_fnr, FNR_METRIC.upper()),
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


def plot_primary_gaps(
    primary_gaps: pd.DataFrame,
    out_path: PathLike,
    pair_id: str,
    metrics: Sequence[str],
) -> None:
    """Cross-model primary gaps: rank 1 filled, ranks 2–5 hollow.

    Args:
        primary_gaps: ``primary_gaps.csv`` content.
        out_path: PNG output path.
        pair_id: Pair whose gaps are shown (panel titles).
        metrics: Metrics to draw, one panel each.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(primary_gaps["model"]))
    ranks = sorted(primary_gaps["rank"].unique())
    offsets = dict(zip(ranks, np.linspace(-0.25, 0.25, len(ranks))))
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5 * len(metrics), max(3.0, len(models) * 1.2)),
        sharey=True,
        squeeze=False,
    )
    y_pos = np.arange(len(models))

    for ax, metric in zip(axes[0], metrics):
        sub = primary_gaps[primary_gaps["metric"] == metric]
        drawn: Set[bool] = set()
        for i, model in enumerate(models):
            for _, row in sub[sub["model"] == model].iterrows():
                rank = int(row["rank"])
                first = rank == 1
                label = None
                if first not in drawn:
                    label = "Rank 1" if first else "Ranks 2–5"
                    drawn.add(first)
                xerr = np.array(
                    [
                        [100.0 * (row["diff"] - row["diff_ci_low"])],
                        [100.0 * (row["diff_ci_high"] - row["diff"])],
                    ]
                )
                ax.errorbar(
                    100.0 * row["diff"],
                    i + offsets[rank],
                    xerr=xerr,
                    fmt="o",
                    mfc="black" if first else "white",
                    mec="black",
                    color="black" if first else "gray",
                    ecolor="black" if first else "gray",
                    capsize=3 if first else 2,
                    markersize=8 if first else 5,
                    label=label,
                )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel("Gap (pp)")
        ax.set_title(f"{pair_id} {METRIC_LABELS.get(metric, metric)}")
    axes[0][0].set_ylim(len(models) - 0.5, -0.5)
    axes[0][0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
