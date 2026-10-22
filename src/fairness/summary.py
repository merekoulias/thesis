"""Cross-model summary over completed fairness audits."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.fairness.audit import default_paths
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.plots import plot_primary_gaps
from src.fairness.report import write_summary_report
from src.fairness.types import AuditConfig


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_primary_gaps(model_dir: Path, config: AuditConfig) -> pd.DataFrame:
    """Build primary-gap rows for ranks 1–5 from one model output directory."""
    gaps_path = model_dir / "gaps.csv"
    robust_path = model_dir / "robustness.csv"
    if not gaps_path.exists() or not robust_path.exists():
        return pd.DataFrame()

    gaps = pd.read_csv(gaps_path)
    robust = pd.read_csv(robust_path)
    model = model_dir.parent.name
    rows: List[Dict[str, Any]] = []

    for metric in config.primary_metrics:
        g_row = gaps[
            (gaps["pair_id"] == config.primary_pair) & (gaps["metric"] == metric)
        ]
        if not g_row.empty:
            r = g_row.iloc[0]
            rows.append(
                {
                    "model": model,
                    "rank": 1,
                    "metric": metric,
                    "diff": r["diff"],
                    "diff_ci_low": r["diff_ci_low"],
                    "diff_ci_high": r["diff_ci_high"],
                    "p_value": r["p_value"],
                }
            )
        for rank in config.robustness_ranks:
            r_row = robust[(robust["rank"] == rank) & (robust["metric"] == metric)]
            if r_row.empty:
                continue
            r = r_row.iloc[0]
            rows.append(
                {
                    "model": model,
                    "rank": rank,
                    "metric": metric,
                    "diff": r["diff"],
                    "diff_ci_low": r["diff_ci_low"],
                    "diff_ci_high": r["diff_ci_high"],
                    "p_value": float("nan"),
                }
            )

    return pd.DataFrame(rows)


def build_summary(
    out_dir: Path, variant: str, config: Optional[AuditConfig] = None
) -> Path:
    """Aggregate per-model audits into ``summary_<variant>/``.

    Args:
        out_dir: Fairness output root.
        variant: Run variant name (e.g. ``baseline``).
        config: Audit settings; ``configs/fairness.yaml`` when omitted.

    Returns:
        Path to the summary directory.
    """
    if config is None:
        config = load_audit_config(default_fairness_config_path())
    summary_dir = out_dir / f"summary_{variant}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    utility_rows: List[Dict[str, Any]] = []
    gap_frames: List[pd.DataFrame] = []

    for model in config.models:
        model_dir = out_dir / model / variant
        util_path = model_dir / "utility.json"
        if not util_path.exists():
            continue
        util = _read_json(util_path)
        utility_rows.append(
            {
                "model": model,
                "roc_auc": util["roc_auc"],
                "brier": util["brier"],
                "log_loss": util["log_loss"],
                "tpr": util["tpr"],
                "n_flagged": util["n_flagged"],
            }
        )
        gap_frames.append(collect_primary_gaps(model_dir, config))

    utility = pd.DataFrame(utility_rows)
    primary_gaps = (
        pd.concat(gap_frames, ignore_index=True) if gap_frames else pd.DataFrame()
    )

    utility.to_csv(summary_dir / "utility.csv", index=False, encoding="utf-8")
    primary_gaps.to_csv(summary_dir / "primary_gaps.csv", index=False, encoding="utf-8")
    if not primary_gaps.empty:
        plot_primary_gaps(
            primary_gaps,
            summary_dir / "figures" / "primary_gaps.png",
            config.primary_pair,
            config.primary_metrics,
        )
    write_summary_report(
        {"utility": utility, "primary_gaps": primary_gaps}, summary_dir, config
    )
    return summary_dir


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Cross-model fairness audit summary")
    parser.add_argument("--config", type=Path, default=default_fairness_config_path())
    parser.add_argument("--out-dir", type=Path, default=paths["fairness_dir"])
    parser.add_argument(
        "--variant", default=None, help="Run variant (default: config variant)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for ``python -m src.fairness.summary``."""
    args = _parse_args(argv)
    config = load_audit_config(args.config)
    build_summary(args.out_dir, args.variant or config.variant, config)


if __name__ == "__main__":
    main()
