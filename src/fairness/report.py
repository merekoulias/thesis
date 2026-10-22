"""Write audit artifacts (CSV/JSON) and Markdown reports."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.constants import AGE_BAND_COL
from src.fairness.types import (
    APPROVAL_RATE_METRIC,
    AUC_METRIC,
    BRIER_METRIC,
    DEFAULT_AMONG_APPROVED_METRIC,
    FNR_METRIC,
    FPR_METRIC,
    GENDER_AGE_COL,
    GENDER_COL,
    METRIC_LABELS,
    AuditConfig,
    PairSpec,
    ThresholdResult,
)

PathLike = Union[str, Path]

MINUS = "\u2212"

WORST_GROUP_CAVEAT = (
    "Max − min across cells is biased upwards; see the double-corrected SD interval."
)

KENDALL_CAVEAT = "Kendall τ uses only ten gender × age cells and has limited power."

GROUP_TABLE_HEADERS: List[str] = [
    "Group",
    "n",
    "Positives",
    "Negatives",
    "Flagged",
    "Approval rate",
    "FPR",
    "FNR",
    "Default among approved",
    "AUC",
    "Brier",
]


def scope_text(config: AuditConfig) -> str:
    """Return the fixed "scope of the numbers" paragraph (report section 12).

    Args:
        config: Audit configuration (names the primary pair).

    Returns:
        Paragraph text.
    """
    pair = config.primary_pair
    return (
        "Intervals condition on the trained model and the decision threshold; "
        "labels are selective (only granted loans have an observed `TARGET`); "
        "`TARGET` is a proxy for payment difficulties; criteria are observational; "
        "p-values are reported after gaps and intervals and are not "
        "multiplicity-corrected; only the "
        f"{pair} equal-opportunity difference (EOD, the FPR gap) and the "
        f"{pair} ΔFNR are pre-specified primary comparisons."
    )


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


def _is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _signed(text: str) -> str:
    return MINUS + text[1:] if text.startswith("-") else text


def _pp_num(value: float) -> str:
    return _signed(f"{100.0 * float(value):.2f}")


def format_rate(value: float) -> str:
    """Format a proportion as a percentage with two decimals."""
    if _is_missing(value):
        return "nan"
    return _signed(f"{100.0 * float(value):.2f}%")


def format_rate_ci(value: float, ci_low: float, ci_high: float) -> str:
    """Format a rate with its interval, e.g. ``7.83% [7.52, 8.14]``."""
    if any(_is_missing(x) for x in (value, ci_low, ci_high)):
        return format_rate(value)
    return f"{format_rate(value)} [{_pp_num(ci_low)}, {_pp_num(ci_high)}]"


def format_pp(value: float, ci_low: float, ci_high: float) -> str:
    """Format a gap and Wald interval in percentage points."""
    if any(_is_missing(x) for x in (value, ci_low, ci_high)):
        return "nan"
    return f"{_pp_num(value)} pp [{_pp_num(ci_low)}, {_pp_num(ci_high)}]"


def format_pp_point(value: float) -> str:
    """Format a point estimate in percentage points."""
    if _is_missing(value):
        return "nan"
    return f"{_pp_num(value)} pp"


def format_ratio(value: float) -> str:
    """Format a ratio with three decimals."""
    if _is_missing(value):
        return "nan"
    return f"{float(value):.3f}"


def format_ratio_ci(value: float, ci_low: float, ci_high: float) -> str:
    """Format a ratio with its interval, three decimals."""
    if any(_is_missing(x) for x in (value, ci_low, ci_high)):
        return format_ratio(value)
    return f"{format_ratio(value)} [{format_ratio(ci_low)}, {format_ratio(ci_high)}]"


def format_auc(value: float) -> str:
    """Format AUC with four decimals."""
    if _is_missing(value):
        return "nan"
    return f"{float(value):.4f}"


def format_auc_ci(value: float, ci_low: float, ci_high: float) -> str:
    """Format AUC with its interval, four decimals."""
    if any(_is_missing(x) for x in (value, ci_low, ci_high)):
        return format_auc(value)
    return f"{format_auc(value)} [{format_auc(ci_low)}, {format_auc(ci_high)}]"


def format_p_value(p: float) -> str:
    """Format a p-value for Markdown; underflowed zero becomes ``< 1e-300``."""
    if _is_missing(p):
        return "nan"
    p = float(p)
    if p == 0.0:
        return "< 1e-300"
    return f"{p:.2e}"


def _p_text(p: float) -> str:
    text = format_p_value(p)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def _fmt_bool(value: Any) -> str:
    if _is_missing(value):
        return "nan"
    return "yes" if bool(value) else "no"


def _fmt_sign(value: Any) -> str:
    if _is_missing(value):
        return "nan"
    sign = int(value)
    return {1: "+1", -1: MINUS + "1"}.get(sign, str(sign))


def _fmt_float(value: Any, digits: int) -> str:
    return "nan" if _is_missing(value) else f"{float(value):.{digits}f}"


def _fmt_int(value: Any) -> str:
    return "nan" if _is_missing(value) else f"{int(value):,}"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    """Return GitHub-flavoured Markdown table lines (``|`` in cells is escaped)."""
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return lines


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
        "models": list(config.models),
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
    """Write CSV and JSON artifacts of plan 5.6 (figures, report, config are separate).

    Args:
        results: Audit result bundle produced by ``run_audit``.
        out_dir: Model output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

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


def write_audit_config(payload: Dict[str, Any], out_dir: Path) -> None:
    """Write ``audit_config.json`` (called last so ``finished_at`` is final).

    Args:
        payload: Output of ``audit_config_payload``.
        out_dir: Model output directory.
    """
    _write_json(Path(out_dir) / "audit_config.json", payload)


def _primary_gap_lines(gaps: pd.DataFrame, config: AuditConfig) -> List[str]:
    lines: List[str] = []
    for metric in config.primary_metrics:
        row = gaps[
            (gaps["pair_id"] == config.primary_pair) & (gaps["metric"] == metric)
        ]
        label = METRIC_LABELS.get(metric, metric)
        if row.empty:
            lines.append(f"- {label}: no row")
            continue
        r = row.iloc[0]
        lines.append(
            f"- {config.primary_pair} {label}: "
            f"{format_pp(r['diff'], r['diff_ci_low'], r['diff_ci_high'])}; "
            f"{_p_text(r['p_value'])}"
        )
    return lines


def _group_table(
    grouping: str,
    groups: Sequence[str],
    counts: pd.DataFrame,
    metrics: pd.DataFrame,
) -> List[str]:
    cnt = counts[counts["grouping"] == grouping].set_index("group")
    met = metrics[metrics["grouping"] == grouping].set_index(["group", "metric"])

    def rate_cell(group: str, metric: str) -> str:
        if (group, metric) not in met.index:
            return "nan"
        r = met.loc[(group, metric)]
        return format_rate_ci(r["value"], r["ci_low"], r["ci_high"])

    def auc_cell(group: str) -> str:
        if (group, AUC_METRIC) not in met.index:
            return "nan"
        r = met.loc[(group, AUC_METRIC)]
        return format_auc_ci(r["value"], r["ci_low"], r["ci_high"])

    def brier_cell(group: str) -> str:
        if (group, BRIER_METRIC) not in met.index:
            return "nan"
        return _fmt_float(met.loc[(group, BRIER_METRIC)]["value"], 4)

    rows = []
    for g in groups:
        if g not in cnt.index:
            continue
        c = cnt.loc[g]
        rows.append(
            [
                g,
                _fmt_int(c["n"]),
                _fmt_int(c["positives"]),
                _fmt_int(c["negatives"]),
                _fmt_int(c["flagged"]),
                rate_cell(g, APPROVAL_RATE_METRIC),
                rate_cell(g, FPR_METRIC),
                rate_cell(g, FNR_METRIC),
                rate_cell(g, DEFAULT_AMONG_APPROVED_METRIC),
                auc_cell(g),
                brier_cell(g),
            ]
        )
    return _md_table(GROUP_TABLE_HEADERS, rows)


def _gaps_table(gaps: pd.DataFrame, config: AuditConfig) -> List[str]:
    rows = []
    for _, r in gaps.iterrows():
        metric = r["metric"]
        is_ratio = metric == APPROVAL_RATE_METRIC and not _is_missing(r["ratio"])
        rows.append(
            [
                r["pair_id"],
                METRIC_LABELS.get(metric, metric),
                format_rate(r["rate_a"]),
                format_rate(r["rate_b"]),
                format_pp(r["diff"], r["diff_ci_low"], r["diff_ci_high"]),
                format_p_value(r["p_value"]),
                (
                    format_ratio_ci(r["ratio"], r["ratio_ci_low"], r["ratio_ci_high"])
                    if is_ratio
                    else ""
                ),
                (
                    _fmt_bool(r["ratio_lower_over_higher"] < config.dir_reference_line)
                    if is_ratio
                    else ""
                ),
            ]
        )
    headers = [
        "Pair (a − b)",
        "Metric",
        "Rate a",
        "Rate b",
        "Gap [95% CI]",
        "p",
        "DIR [95% CI]",
        "DIR below reference",
    ]
    return _md_table(headers, rows)


def _worst_group_table(cell_summary: pd.DataFrame) -> List[str]:
    rows = []
    for _, r in cell_summary.iterrows():
        rows.append(
            [
                METRIC_LABELS.get(r["metric"], r["metric"]).replace("Δ ", ""),
                format_pp(r["max_min_gap"], r["max_min_ci_low"], r["max_min_ci_high"]),
                r["max_cell"],
                r["min_cell"],
                format_pp_point(r["sd_uncorrected"]),
                format_pp_point(r["sd_single_corrected"]),
                f"[{_pp_num(r['sd_double_corrected_ci_low'])}, "
                f"{_pp_num(r['sd_double_corrected_ci_high'])}] pp",
                f"{_fmt_float(r['g_stat'], 2)} ({_fmt_int(r['df'])})",
                format_p_value(r["p_value"]),
            ]
        )
    headers = [
        "Metric",
        "Max − min [95% CI]",
        "Max cell",
        "Min cell",
        "SD uncorrected",
        "SD single-corrected",
        "SD double-corrected [95% CI]",
        "G (df)",
        "p",
    ]
    return _md_table(headers, rows)


def _test_table(tests: pd.DataFrame, label_col: str) -> List[str]:
    rows = [
        [
            r[label_col],
            _fmt_float(r["g_stat"], 2),
            _fmt_int(r["df"]),
            format_p_value(r["p_value"]),
        ]
        for _, r in tests.iterrows()
    ]
    return _md_table([label_col, "G", "df", "p"], rows)


def _cdp_tables(cdp: pd.DataFrame, cdp_tests: pd.DataFrame) -> List[str]:
    test_rows = [
        [
            r["pair_id"],
            _fmt_float(r["g_stat"], 2),
            _fmt_int(r["df"]),
            format_p_value(r["p_value"]),
            format_pp(
                r["unconditional_diff"],
                r["unconditional_diff_ci_low"],
                r["unconditional_diff_ci_high"],
            ),
            _fmt_int(r["n_excluded_missing_pti"]),
        ]
        for _, r in cdp_tests.iterrows()
    ]
    quintile_rows = [
        [
            r["pair_id"],
            r["quintile"],
            _fmt_int(r["n_a"]),
            _fmt_int(r["n_b"]),
            format_rate(r["approval_rate_a"]),
            format_rate(r["approval_rate_b"]),
            format_pp(r["diff"], r["diff_ci_low"], r["diff_ci_high"]),
        ]
        for _, r in cdp.iterrows()
    ]
    return [
        *_md_table(
            [
                "Pair (a − b)",
                "Stratified G",
                "df",
                "p",
                "Unconditional Δ approval [95% CI]",
                "Rows excluded (missing PTI)",
            ],
            test_rows,
        ),
        "",
        *_md_table(
            [
                "Pair (a − b)",
                "PTI quintile",
                "n a",
                "n b",
                "Approval a",
                "Approval b",
                "Δ approval [95% CI]",
            ],
            quintile_rows,
        ),
    ]


def _oracle_table(oracle: pd.DataFrame) -> List[str]:
    rows = [
        [
            r["pair_id"],
            format_rate(r["oracle_approval_a"]),
            format_rate(r["oracle_approval_b"]),
            format_ratio(r["oracle_dir"]),
            format_ratio_ci(
                r["model_dir"], r["model_dir_ci_low"], r["model_dir_ci_high"]
            ),
            "nan" if _is_missing(r["direction"]) else f"`{r['direction']}`",
            _fmt_bool(r["oracle_in_model_ci"]),
        ]
        for _, r in oracle.iterrows()
    ]
    headers = [
        "Pair (a − b)",
        "Oracle approval a",
        "Oracle approval b",
        "Oracle DIR",
        "Model DIR [95% CI]",
        "Model vs oracle",
        "Oracle DIR inside model CI",
    ]
    return _md_table(headers, rows)


def _signs_table(signs: pd.DataFrame) -> List[str]:
    rows = [
        [
            r["pair_id"],
            format_rate(r["base_rate_a"]),
            format_rate(r["base_rate_b"]),
            _fmt_sign(r["expected_sign_eod"]),
            format_pp_point(r["observed_eod"]),
            _fmt_bool(r["eod_ci_excludes_0"]),
            _fmt_bool(r["eod_matches"]),
            _fmt_sign(r["expected_sign_dfnr"]),
            format_pp_point(r["observed_dfnr"]),
            _fmt_bool(r["dfnr_ci_excludes_0"]),
            _fmt_bool(r["dfnr_matches"]),
        ]
        for _, r in signs.iterrows()
    ]
    headers = [
        "Pair (a − b)",
        "Base rate a",
        "Base rate b",
        "Expected EOD sign",
        "Observed EOD",
        "CI excludes 0",
        "Sign matches",
        "Expected ΔFNR sign",
        "Observed ΔFNR",
        "CI excludes 0",
        "Sign matches",
    ]
    return _md_table(headers, rows)


def _proxy_table(proxy: pd.DataFrame) -> List[str]:
    rows = [
        [
            r["feature_group"],
            _fmt_int(r["n_features"]),
            _fmt_float(r["importance_share"], 3),
            f"`{r['best_feature']}`",
            _fmt_int(r["best_feature_rank"]),
        ]
        for _, r in proxy.iterrows()
    ]
    return _md_table(
        ["Feature group", "Features", "Importance share", "Best feature", "Rank"], rows
    )


def _robustness_table(robust: pd.DataFrame, config: AuditConfig) -> List[str]:
    metrics = list(config.robustness_metrics)
    rows = []
    for rank in sorted(robust["rank"].unique()):
        sub = robust[robust["rank"] == rank]
        first = sub.iloc[0]
        cells = []
        for metric in metrics:
            m = sub[sub["metric"] == metric]
            cells.append(
                "nan"
                if m.empty
                else format_pp(
                    m.iloc[0]["diff"],
                    m.iloc[0]["diff_ci_low"],
                    m.iloc[0]["diff_ci_high"],
                )
            )
        rows.append(
            [
                int(rank),
                _fmt_float(first["threshold"], 6),
                _fmt_int(first["n_flagged"]),
                format_auc(first["roc_auc"]),
                *cells,
            ]
        )
    headers = [
        "Rank",
        "Threshold t",
        "Flagged",
        "ROC AUC",
        *[f"{config.primary_pair} {METRIC_LABELS.get(m, m)} [95% CI]" for m in metrics],
    ]
    return _md_table(headers, rows)


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
    counts = results["group_counts"]
    metrics = results["group_metrics"]
    kendall = results["kendall"]
    proxy = results.get("proxy_importance")
    groupings = config.groupings

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
        *_primary_gap_lines(results["gaps"], config),
        "",
        "## 4. Group table — gender and age bands",
        "",
    ]
    for grouping in (GENDER_COL, AGE_BAND_COL):
        lines.extend([f"### {grouping}", ""])
        lines.extend(_group_table(grouping, groupings[grouping], counts, metrics))
        lines.append("")

    lines.extend(["## 5. Group table — the 10 cells", ""])
    lines.extend(
        _group_table(GENDER_AGE_COL, groupings[GENDER_AGE_COL], counts, metrics)
    )
    lines.append("")

    lines.extend(
        [
            "## 6. Pairwise gaps",
            "",
            f"DIR reference line: {config.dir_reference_line:.2f} (a reference, not a "
            'legal threshold); "DIR below reference" compares min(DIR, 1/DIR) with it.',
            "",
            *_gaps_table(results["gaps"], config),
            "",
            "## 7. Worst-group summaries",
            "",
            WORST_GROUP_CAVEAT,
            "",
            *_worst_group_table(results["cell_summary"]),
            "",
            "## 8. Calibration",
            "",
            "- Figures: "
            + ", ".join(
                f"`figures/calibration_{g}.png`" for g in (GENDER_COL, AGE_BAND_COL)
            ),
            "",
            *_test_table(results["calibration_tests"], "grouping"),
            "",
            "## 9. Conditional demographic parity",
            "",
            *_cdp_tables(
                results["conditional_parity"], results["conditional_parity_tests"]
            ),
            "",
            "## 10. Comparison with the data",
            "",
            "### Oracle benchmark",
            "",
            *_oracle_table(results["oracle_benchmark"]),
            "",
            "### Expected signs",
            "",
            *_signs_table(results["expected_signs"]),
            "",
            "### Kendall τ",
            "",
            f"- τ = {kendall['tau']:.4f} ({_p_text(kendall['p_value'])}). "
            f"{KENDALL_CAVEAT}",
            "",
            "### Proxy watch list",
            "",
        ]
    )
    if proxy is not None and not proxy.empty:
        lines.extend(_proxy_table(proxy))
    else:
        lines.append(f"- Proxy watch list: not available for {model}")
    lines.extend(
        [
            "",
            "## 11. Robustness ranks",
            "",
            *_robustness_table(results["robustness"], config),
            "",
            "## 12. Scope of the numbers",
            "",
            scope_text(config),
            "",
        ]
    )

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_report(
    rows: Dict[str, pd.DataFrame], out_dir: Path, config: AuditConfig
) -> None:
    """Write the cross-model ``report.md`` under ``summary_<variant>/``.

    Args:
        rows: ``utility`` and ``primary_gaps`` DataFrames.
        out_dir: Summary output directory.
        config: Audit configuration (primary pair and metrics).
    """
    out = Path(out_dir)
    utility = rows["utility"]
    gaps = rows["primary_gaps"]
    pair = config.primary_pair

    utility_rows = [
        [
            r["model"],
            format_auc(r["roc_auc"]),
            _fmt_float(r["brier"], 6),
            _fmt_float(r["log_loss"], 6),
            format_rate(r["tpr"]),
            _fmt_int(r["n_flagged"]),
        ]
        for _, r in utility.iterrows()
    ]
    gap_rows = []
    if not gaps.empty:
        for model in gaps["model"].unique():
            for metric in config.primary_metrics:
                sub = gaps[(gaps["model"] == model) & (gaps["metric"] == metric)]
                first = sub[sub["rank"] == 1]
                others = sub[sub["rank"] != 1]
                if first.empty:
                    continue
                r = first.iloc[0]
                spread = (
                    f"{_pp_num(others['diff'].min())} to {_pp_num(others['diff'].max())} pp"
                    if not others.empty
                    else "nan"
                )
                gap_rows.append(
                    [
                        model,
                        METRIC_LABELS.get(metric, metric),
                        format_pp(r["diff"], r["diff_ci_low"], r["diff_ci_high"]),
                        format_p_value(r["p_value"]),
                        spread,
                    ]
                )

    lines = [
        "# Baseline fairness audit — cross-model summary",
        "",
        "## Utility",
        "",
        *_md_table(
            ["Model", "ROC AUC", "Brier", "Log loss", "TPR at t", "Flagged"],
            utility_rows,
        ),
        "",
        f"## Primary gaps ({pair})",
        "",
        *_md_table(
            [
                "Model",
                "Metric",
                "Rank 1 gap [95% CI]",
                "p (rank 1)",
                "Ranks 2–5 (min to max)",
            ],
            gap_rows,
        ),
        "",
        "## Figure",
        "",
        "See `figures/primary_gaps.png`.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
