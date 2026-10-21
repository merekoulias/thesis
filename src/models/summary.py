"""CLI: summarize nested-CV runs into a CSV and Markdown table."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.runs import DEFAULT_VARIANT, METRICS_FILE, ROBUSTNESS_DIR

SUMMARY_COLUMNS = [
    "model",
    "rank",
    "oof_roc_auc",
    "std_fold_roc_auc",
    "oof_brier",
    "brier_reference",
    "oof_log_loss",
    "log_loss_reference",
    "mean_roc_auc_gap",
    "mean_selection_optimism",
    "runtime_hours",
]


def _row_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": metrics["model"],
        "rank": metrics["rank"],
        "oof_roc_auc": metrics["oof_roc_auc"],
        "std_fold_roc_auc": metrics["std_fold_roc_auc"],
        "oof_brier": metrics["oof_brier"],
        "brier_reference": metrics["brier_reference"],
        "oof_log_loss": metrics["oof_log_loss"],
        "log_loss_reference": metrics["log_loss_reference"],
        "mean_roc_auc_gap": metrics["mean_roc_auc_gap"],
        "mean_selection_optimism": metrics["mean_selection_optimism"],
        "runtime_hours": float(metrics["runtime_seconds"]) / 3600.0,
    }


def collect_summary_rows(
    runs_dir: Path, variant: str = DEFAULT_VARIANT
) -> List[Dict[str, Any]]:
    """Collect summary rows from all model run directories under ``runs_dir``.

    Args:
        runs_dir: Root runs directory.
        variant: Variant subdirectory name (default ``baseline``).

    Returns:
        Rows sorted by model name then rank.
    """
    rows: List[Dict[str, Any]] = []
    root = Path(runs_dir)
    if not root.is_dir():
        return rows

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run_dir = model_dir / variant
        if not run_dir.is_dir():
            continue
        main_metrics = run_dir / METRICS_FILE
        if main_metrics.exists():
            with main_metrics.open(encoding="utf-8") as f:
                rows.append(_row_from_metrics(json.load(f)))
        robustness = run_dir / ROBUSTNESS_DIR
        if robustness.is_dir():
            for rank_dir in sorted(robustness.glob("rank_*")):
                path = rank_dir / METRICS_FILE
                if path.exists():
                    with path.open(encoding="utf-8") as f:
                        rows.append(_row_from_metrics(json.load(f)))

    rows.sort(key=lambda r: (r["model"], r["rank"]))
    return rows


def _format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| model | rank | oof_roc_auc | std_fold_roc_auc | oof_brier | "
        "brier_reference | oof_log_loss | log_loss_reference | "
        "mean_roc_auc_gap | mean_selection_optimism | runtime_hours |"
    )
    sep = (
        "| ----- | ---- | ----------- | ---------------- | --------- | "
        "--------------- | ------------ | ------------------ | "
        "---------------- | ----------------------- | ------------- |"
    )
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {rank} | {oof_roc_auc:.4f} | {std_fold_roc_auc:.4f} | "
            "{oof_brier:.4f} | {brier_reference:.4f} | {oof_log_loss:.4f} | "
            "{log_loss_reference:.4f} | {mean_roc_auc_gap:.4f} | "
            "{mean_selection_optimism:.4f} | {runtime_hours:.4f} |".format(**row)
        )
    return "\n".join(lines)


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write summary rows to a CSV file.

    Args:
        rows: Summary rows from ``collect_summary_rows``.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in SUMMARY_COLUMNS:
                if key not in ("model", "rank") and isinstance(out[key], float):
                    out[key] = f"{out[key]:.4f}"
            writer.writerow({k: out[k] for k in SUMMARY_COLUMNS})


def main(argv: Optional[List[str]] = None) -> None:
    """Write and print a summary of all runs under ``runs_dir``.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description="Summarize nested-CV run metrics.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        help="Run variant subdirectory name.",
    )
    args = parser.parse_args(argv)

    rows = collect_summary_rows(args.runs_dir, args.variant)
    out_path = Path(args.runs_dir) / f"summary_{args.variant}.csv"
    write_summary_csv(rows, out_path)
    print(_format_markdown_table(rows))


if __name__ == "__main__":
    main()
