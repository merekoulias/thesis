"""Write audit artifacts (CSV/JSON) and Markdown reports."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.fairness.types import AuditConfig, PairSpec, ThresholdResult

PathLike = Union[str, Path]

MINUS = "\u2212"

SCOPE_SECTION_TEXT = (
    "Intervals condition on the trained model and the decision threshold; "
    "labels are selective (only granted loans have a observed `TARGET`); "
    "`TARGET` is a proxy for payment difficulties; criteria are observational; "
    "p-values are reported after gaps and intervals and are not multiplicity-corrected; "
    "only F − M equalized odds (FPR gap) and F − M ΔFNR are pre-specified primary comparisons."
)

WORST_GROUP_CAVEAT = (
    "Max − min across cells is biased upwards; see the double-corrected SD interval."
)

KENDALL_CAVEAT = "Kendall τ uses only ten gender × age cells and has limited power."


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, PairSpec):
        return {"id": obj.pair_id, "a": obj.a, "b": obj.b}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, allow_nan=False)
        f.write("\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def format_rate(value: float) -> str:
    """Format a proportion as a percentage with two decimals."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    text = f"{100.0 * float(value):.2f}%"
    if text.startswith("-"):
        return MINUS + text[1:]
    return text


def format_pp(value: float, ci_low: float, ci_high: float) -> str:
    """Format a gap and Wald interval in percentage points."""
    if any(np.isnan(x) for x in (value, ci_low, ci_high)):
        return "nan"

    def pp(x: float) -> str:
        text = f"{100.0 * x:.2f}"
        if text.startswith("-"):
            return MINUS + text[1:]
        return text

    return f"{pp(value)} pp [{pp(ci_low)}, {pp(ci_high)}]"


def format_ratio(value: float) -> str:
    """Format a ratio with three decimals."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def format_auc(value: float) -> str:
    """Format AUC with four decimals."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.4f}"


def format_p_value(p: float) -> str:
    """Format a p-value for Markdown; underflowed zero becomes ``< 1e-300``."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "nan"
    p = float(p)
    if p == 0.0:
        return "< 1e-300"
    return f"{p:.2e}"


def audit_config_payload(
    config: AuditConfig,
    model: str,
    input_paths: Dict[str, str],
    git_commit: Optional[str],
    versions: Dict[str, str],
    started_at: datetime,
    finished_at: datetime,
) -> Dict[str, Any]:
    """Build the ``audit_config.json`` document."""
    return {
        "model": model,
        "variant": config.variant,
        "seed": config.seed,
        "n_boot": config.n_boot,
        "confidence": config.confidence,
        "robustness_ranks": list(config.robustness_ranks),
        "calibration_bins": config.calibration_bins,
        "dir_reference_line": config.dir_reference_line,
        "groupings": config.groupings,
        "pairs": [_to_jsonable(p) for p in config.pairs],
        "primary_pair": config.primary_pair,
        "primary_metrics": list(config.primary_metrics),
        "gap_metrics": list(config.gap_metrics),
        "cell_summary_metrics": list(config.cell_summary_metrics),
        "robustness_metrics": list(config.robustness_metrics),
        "conditional_parity": {
            "pairs": list(config.cdp_pairs),
            "n_quantiles": config.cdp_n_quantiles,
            "expected_cut_points": list(config.cdp_expected_cut_points),
            "expected_missing": config.cdp_expected_missing,
        },
        "proxy_watch_list": config.proxy_watch_list,
        "importance_type": config.importance_type,
        "input_paths": input_paths,
        "git_commit": git_commit,
        "package_versions": versions,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def write_outputs(results: Dict[str, Any], out_dir: Path) -> None:
    """Write CSV and JSON artifacts of plan 5.6 (figures and report are separate).

    Args:
        results: Audit result bundle produced by ``run_audit``.
        out_dir: Model output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_json(out / "audit_config.json", results["audit_config"])
    _write_json(out / "threshold.json", asdict(results["threshold"]))
    _write_json(out / "utility.json", results["utility"])
    _write_json(out / "bin_edges.json", results["bin_edges"])
    _write_json(out / "pti_quintiles.json", results["pti_quintiles"])
    _write_json(out / "kendall.json", results["kendall"])

    _write_csv(out / "group_counts.csv", results["group_counts"])
    _write_csv(out / "group_metrics.csv", results["group_metrics"])
    _write_csv(out / "gaps.csv", results["gaps"])
    _write_csv(out / "cell_summary.csv", results["cell_summary"])
    _write_csv(out / "calibration.csv", results["calibration"])
    _write_csv(out / "calibration_tests.csv", results["calibration_tests"])
    _write_csv(out / "conditional_parity.csv", results["conditional_parity"])
    _write_csv(
        out / "conditional_parity_tests.csv", results["conditional_parity_tests"]
    )
    _write_csv(
        out / "conditional_parity_cells.csv", results["conditional_parity_cells"]
    )
    _write_csv(out / "oracle_benchmark.csv", results["oracle_benchmark"])
    _write_csv(out / "expected_signs.csv", results["expected_signs"])
    _write_csv(out / "robustness.csv", results["robustness"])

    proxy = results.get("proxy_importance")
    if proxy is not None:
        _write_csv(out / "proxy_importance.csv", proxy)


def _metric_lookup(
    metrics: pd.DataFrame, grouping: str, group: str
) -> Dict[str, Dict[str, float]]:
    sub = metrics[(metrics["grouping"] == grouping) & (metrics["group"] == group)]
    out: Dict[str, Dict[str, float]] = {}
    for _, row in sub.iterrows():
        out[str(row["metric"])] = {
            "value": float(row["value"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "denominator": float(row["denominator"]),
        }
    return out


def _primary_gap_lines(gaps: pd.DataFrame, config: AuditConfig) -> List[str]:
    lines: List[str] = []
    for metric in config.primary_metrics:
        row = gaps[
            (gaps["pair_id"] == config.primary_pair) & (gaps["metric"] == metric)
        ]
        if row.empty:
            lines.append(f"- {metric}: no row")
            continue
        r = row.iloc[0]
        label = "EOD (FPR gap)" if metric == "fpr" else "ΔFNR"
        lines.append(
            f"- {label}: {format_pp(r['diff'], r['diff_ci_low'], r['diff_ci_high'])}; "
            f"p = {format_p_value(r['p_value'])}"
        )
    return lines


def write_model_report(
    results: Dict[str, Any], out_dir: Path, config: AuditConfig
) -> None:
    """Write ``report.md`` for one audited model (plan P7.4).

    Args:
        results: Audit result bundle.
        out_dir: Model output directory.
        config: Audit configuration.
    """
    out = Path(out_dir)
    model = results["model"]
    thr: ThresholdResult = results["threshold"]
    utility = results["utility"]
    gaps = results["gaps"]
    metrics = results["group_metrics"]
    counts = results["group_counts"]
    cell_summary = results["cell_summary"]
    cal_tests = results["calibration_tests"]
    cdp = results["conditional_parity"]
    cdp_tests = results["conditional_parity_tests"]
    oracle = results["oracle_benchmark"]
    signs = results["expected_signs"]
    kendall = results["kendall"]
    robust = results["robustness"]
    proxy = results.get("proxy_importance")

    lines: List[str] = [
        f"# Fairness audit: {model} ({config.variant})",
        "",
        "## 1. Inputs and threshold",
        "",
        f"- Model: `{model}`; variant: `{config.variant}`",
        f"- Rows: {thr.n_rows:,}; base rate: {format_rate(thr.base_rate)}",
        f"- Threshold `t`: {thr.threshold:.6f}; flagged: {thr.n_flagged:,}",
        "",
        "## 2. Utility",
        "",
        f"- ROC AUC: {format_auc(utility['roc_auc'])} "
        f"(fold mean {format_auc(utility['fold_roc_auc_mean'])}, "
        f"std {format_auc(utility['fold_roc_auc_std'])}, "
        f"min {format_auc(utility['fold_roc_auc_min'])}, "
        f"max {format_auc(utility['fold_roc_auc_max'])})",
        f"- At `t`: TPR = {format_rate(utility['tpr'])}, "
        f"PPV = {format_rate(utility['ppv'])}, F1 = {format_rate(utility['f1'])}",
        f"- Brier: {utility['brier']:.6f} (reference {utility['brier_reference']:.6f})",
        f"- Log loss: {utility['log_loss']:.6f} "
        f"(reference {utility['log_loss_reference']:.6f})",
        "",
        "## 3. Primary comparisons",
        "",
        *(_primary_gap_lines(gaps, config)),
        "",
        "## 4. Group table — gender and age bands",
        "",
    ]

    for grouping in ("CODE_GENDER", "AGE_BAND"):
        lines.append(f"### {grouping}")
        lines.append("")
        grp_counts = counts[counts["grouping"] == grouping]
        for _, crow in grp_counts.iterrows():
            g = crow["group"]
            m = _metric_lookup(metrics, grouping, g)
            lines.append(
                f"- **{g}** (n = {int(crow['n']):,}, pos = {int(crow['positives']):,}): "
                f"approval {format_rate(m.get('approval_rate', {}).get('value', float('nan')))}, "
                f"FPR {format_rate(m.get('fpr', {}).get('value', float('nan')))}, "
                f"FNR {format_rate(m.get('fnr', {}).get('value', float('nan')))}, "
                f"default among approved "
                f"{format_rate(m.get('default_among_approved', {}).get('value', float('nan')))}, "
                f"AUC {format_auc(m.get('auc', {}).get('value', float('nan')))}, "
                f"Brier {m.get('brier', {}).get('value', float('nan')):.4f}"
            )
        lines.append("")

    lines.extend(["## 5. Group table — the 10 cells", ""])
    cell_counts = counts[counts["grouping"] == "GENDER_AGE"]
    for _, crow in cell_counts.iterrows():
        g = crow["group"]
        m = _metric_lookup(metrics, "GENDER_AGE", g)
        lines.append(
            f"- **{g}** (n = {int(crow['n']):,}): "
            f"approval {format_rate(m.get('approval_rate', {}).get('value', float('nan')))}, "
            f"FPR {format_rate(m.get('fpr', {}).get('value', float('nan')))}, "
            f"FNR {format_rate(m.get('fnr', {}).get('value', float('nan')))}, "
            f"default among approved "
            f"{format_rate(m.get('default_among_approved', {}).get('value', float('nan')))}, "
            f"AUC {format_auc(m.get('auc', {}).get('value', float('nan')))}, "
            f"Brier {m.get('brier', {}).get('value', float('nan')):.4f}"
        )
    lines.append("")

    lines.extend(["## 6. Pairwise gaps", ""])
    for _, row in gaps.iterrows():
        metric = row["metric"]
        extra = ""
        if metric == "approval_rate" and not np.isnan(row["ratio"]):
            extra = f"; DIR {format_ratio(row['ratio'])}"
        lines.append(
            f"- {row['pair_id']} / {metric}: "
            f"{format_pp(row['diff'], row['diff_ci_low'], row['diff_ci_high'])}{extra}"
        )
    lines.append("")

    lines.extend(["## 7. Worst-group summaries", "", WORST_GROUP_CAVEAT, ""])
    for _, row in cell_summary.iterrows():
        lines.append(
            f"- {row['metric']}: max − min {format_pp(row['max_min_gap'], row['max_min_ci_low'], row['max_min_ci_high'])} "
            f"({row['max_cell']} vs {row['min_cell']}); p = {format_p_value(row['p_value'])}"
        )
    lines.append("")

    lines.extend(
        [
            "## 8. Calibration",
            "",
            "- Figures: "
            "`figures/calibration_CODE_GENDER.png`, "
            "`figures/calibration_AGE_BAND.png`",
            "",
        ]
    )
    for _, row in cal_tests.iterrows():
        lines.append(
            f"- {row['grouping']}: G = {row['g_stat']:.2f}, df = {int(row['df'])}, "
            f"p = {format_p_value(row['p_value'])}"
        )
    lines.append("")

    lines.extend(["## 9. Conditional demographic parity", ""])
    for _, row in cdp_tests.iterrows():
        lines.append(
            f"- {row['pair_id']}: stratified G = {row['g_stat']:.2f}, df = {int(row['df'])}, "
            f"p = {format_p_value(row['p_value'])}; "
            f"unconditional Δapproval {format_pp(row['unconditional_diff'], row['unconditional_diff_ci_low'], row['unconditional_diff_ci_high'])}; "
            f"excluded (missing PTI): {int(row['n_excluded_missing_pti'])}"
        )
    lines.append("")
    for _, row in cdp.iterrows():
        lines.append(
            f"- {row['pair_id']} / {row['quintile']}: "
            f"{format_pp(row['diff'], row['diff_ci_low'], row['diff_ci_high'])}"
        )
    lines.append("")

    lines.extend(["## 10. Comparison with the data", ""])
    fm_oracle = oracle[oracle["pair_id"] == config.primary_pair]
    if not fm_oracle.empty:
        o = fm_oracle.iloc[0]
        lines.append(
            f"- Oracle benchmark ({config.primary_pair}): oracle DIR {format_ratio(o['oracle_dir'])}, "
            f"model DIR {format_ratio(o['model_dir'])}, direction `{o['direction']}`"
        )
    lines.append(
        f"- Kendall τ: {kendall['tau']:.4f} (p = {format_p_value(kendall['p_value'])}). {KENDALL_CAVEAT}"
    )
    lines.append("")
    if proxy is not None and not proxy.empty:
        for _, row in proxy.iterrows():
            lines.append(
                f"- Proxy `{row['feature_group']}`: share {row['importance_share']:.3f}, "
                f"best `{row['best_feature']}` (rank {int(row['best_feature_rank'])})"
            )
    else:
        lines.append(f"- Proxy watch list: not available for {model}")
    lines.append("")

    lines.extend(["## 11. Robustness ranks", ""])
    for rank in sorted(robust["rank"].unique()):
        sub = robust[robust["rank"] == rank]
        parts = [
            f"{r['metric']} {format_pp(r['diff'], r['diff_ci_low'], r['diff_ci_high'])}"
            for _, r in sub.iterrows()
        ]
        lines.append(f"- Rank {int(rank)}: " + "; ".join(parts))
    lines.append("")

    lines.extend(["## 12. Scope of the numbers", "", SCOPE_SECTION_TEXT, ""])

    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame_to_markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in frame.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + body)


def write_summary_report(rows: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write the cross-model ``report.md`` under ``summary_<variant>/``.

    Args:
        rows: ``utility`` and ``primary_gaps`` DataFrames.
        out_dir: Summary output directory.
    """
    out = Path(out_dir)
    utility = rows["utility"]
    gaps = rows["primary_gaps"]

    lines = [
        "# Baseline fairness audit — cross-model summary",
        "",
        "## Utility",
        "",
        _frame_to_markdown_table(utility),
        "",
        "## Primary gaps (F − M)",
        "",
        _frame_to_markdown_table(gaps),
        "",
        "## Figure",
        "",
        "See `figures/primary_gaps.png`.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
