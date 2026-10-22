"""Run the baseline fairness audit for one model and write artifacts."""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

import numpy as np
import pandas as pd

from src.constants import AGE_BAND_COL
from src.fairness.calibration import calibration_table, decile_bins, sufficiency_tests
from src.fairness.comparison import (
    expected_signs,
    kendall_cells,
    oracle_benchmark,
    proxy_importance,
)
from src.fairness.conditional import conditional_parity, conditional_parity_cells
from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.gaps import cell_summary, pair_gaps
from src.fairness.group_metrics import group_counts, group_metrics, utility
from src.fairness.io import PathLike, assign_pti_quintiles, load_audit_frame
from src.fairness.plots import plot_calibration, plot_cells_error_rates
from src.fairness.report import (
    audit_config_payload,
    write_audit_config,
    write_model_report,
    write_outputs,
)
from src.fairness.robustness import robustness_table
from src.fairness.thresholds import apply_threshold, base_rate_threshold
from src.fairness.types import (
    ALL_GROUP,
    ALL_GROUPING,
    AUC_METRIC,
    GENDER_COL,
    PTI_COL,
    PTI_Q_COL,
    SCORE_COL,
    Y_COL,
    YHAT_COL,
    AuditConfig,
)
from src.models.io import load_yaml_config
from src.models.runs import (
    IMPORTANCE_FILE,
    METRICS_FILE,
    OOF_FILE,
    load_metrics,
    run_dir_for,
)

T = TypeVar("T")

PACKAGE_LOGGER = "src.fairness"

# Not __name__: under ``python -m`` that is "__main__", outside PACKAGE_LOGGER.
logger = logging.getLogger(f"{PACKAGE_LOGGER}.audit")

FAIRNESS_GROUPS_FILE = "fairness_groups_train.parquet"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _display_path(path: Path) -> str:
    """Return ``path`` relative to the repo root when inside it, else absolute."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_repo_root()))
    except ValueError:
        return str(resolved)


def default_paths() -> Dict[str, Path]:
    """Load default path settings from ``configs/paths.yaml``."""
    raw = load_yaml_config(_repo_root() / "configs" / "paths.yaml")
    root = _repo_root()
    return {
        "runs_dir": root / raw["runs_dir"],
        "features_dir": root / raw["features_dir"],
        "data_dir": root / raw["data_dir"],
        "fairness_dir": root / raw["fairness_dir"],
    }


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_versions() -> Dict[str, str]:
    import matplotlib
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
    }


@contextmanager
def _audit_logging(out_dir: Path) -> Iterator[None]:
    """Log the package's INFO records to stdout and ``<out_dir>/audit.log``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    pkg = logging.getLogger(PACKAGE_LOGGER)
    previous_level = pkg.level
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out_dir / "audit.log", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(fmt)
        pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)
    try:
        yield
    finally:
        pkg.setLevel(previous_level)
        for handler in handlers:
            pkg.removeHandler(handler)
            handler.close()


def _timed_step(name: str, func: Callable[[], T]) -> T:
    start = time.perf_counter()
    result = func()
    logger.info("%s completed in %.1fs", name, time.perf_counter() - start)
    return result


def _model_out_dir(out_dir: PathLike, model: str, variant: str) -> Path:
    return Path(out_dir) / model / variant


def check_output_free(out_dir: PathLike, model: str, variant: str) -> Path:
    """Return the model output directory; raise if it already holds files.

    Args:
        out_dir: Fairness output root.
        model: Registry model name.
        variant: Run variant name.

    Returns:
        Path to ``<out_dir>/<model>/<variant>/``.

    Raises:
        FileExistsError: If the directory exists and is not empty.
    """
    path = _model_out_dir(out_dir, model, variant)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory {path} already exists; pass --overwrite to replace it."
        )
    return path


def _assert_all_auc_matches(
    metrics_frame: pd.DataFrame, run_auc: float, tol: float = 1e-12
) -> None:
    row = metrics_frame[
        (metrics_frame["grouping"] == ALL_GROUPING)
        & (metrics_frame["group"] == ALL_GROUP)
        & (metrics_frame["metric"] == AUC_METRIC)
    ]
    if row.empty:
        raise ValueError("ALL / all / auc row missing from group_metrics")
    auc = float(row.iloc[0]["value"])
    if not np.isclose(auc, run_auc, rtol=0, atol=tol):
        raise ValueError(f"ALL AUC {auc} != run oof_roc_auc {run_auc}")


def run_audit(
    model: str,
    config: AuditConfig,
    runs_dir: PathLike,
    features_dir: PathLike,
    data_dir: PathLike,
    out_dir: PathLike,
    overwrite: bool = False,
) -> Path:
    """Audit rank 1 plus robustness ranks; write plan 5.6 outputs.

    Args:
        model: Registry model name.
        config: Audit configuration (``n_boot`` may already be overridden).
        runs_dir: Root of model run directories.
        features_dir: Feature artifacts directory.
        data_dir: Raw CSV directory for PTI.
        out_dir: Fairness output root.
        overwrite: Replace an existing model output directory (removed first).

    Returns:
        Path to ``<out_dir>/<model>/<variant>/``.

    Raises:
        FileExistsError: If the output directory is not empty and ``overwrite``
            is false.
    """
    oof_path = run_dir_for(runs_dir, model, config.variant) / OOF_FILE
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    if overwrite:
        out_path = _model_out_dir(out_dir, model, config.variant)
        if out_path.exists():
            shutil.rmtree(out_path)
    out_path = check_output_free(out_dir, model, config.variant)
    with _audit_logging(out_path):
        return _run_audit(model, config, runs_dir, features_dir, data_dir, out_path)


def _run_audit(
    model: str,
    config: AuditConfig,
    runs_dir: PathLike,
    features_dir: PathLike,
    data_dir: PathLike,
    out_path: Path,
) -> Path:
    started_at = datetime.now(timezone.utc)
    run_dir = run_dir_for(runs_dir, model, config.variant)
    groups_path = Path(features_dir) / FAIRNESS_GROUPS_FILE

    input_paths = {
        "run_dir": _display_path(run_dir),
        "groups_path": _display_path(groups_path),
        "data_dir": _display_path(Path(data_dir)),
        "oof_rank_1": _display_path(run_dir / OOF_FILE),
        "metrics_rank_1": _display_path(run_dir / METRICS_FILE),
    }

    frame = _timed_step(
        "load_audit_frame",
        lambda: load_audit_frame(run_dir, 1, groups_path, data_dir, config),
    )
    run_metrics = load_metrics(run_dir, 1)

    thr = _timed_step(
        "threshold",
        lambda: base_rate_threshold(
            frame[SCORE_COL].to_numpy(), frame[Y_COL].to_numpy()
        ),
    )
    frame[YHAT_COL] = apply_threshold(frame[SCORE_COL].to_numpy(), thr.threshold)

    counts = _timed_step("group_counts", lambda: group_counts(frame, config))
    metrics_df = _timed_step(
        "group_metrics",
        lambda: group_metrics(frame, config, model, 1),
    )
    _assert_all_auc_matches(metrics_df, float(run_metrics["oof_roc_auc"]))
    util = _timed_step("utility", lambda: utility(frame, thr, run_metrics))
    gaps = _timed_step("pair_gaps", lambda: pair_gaps(frame, config))
    cells = _timed_step("cell_summary", lambda: cell_summary(frame, config, model, 1))
    calibration = _timed_step(
        "calibration_table", lambda: calibration_table(frame, config)
    )
    cal_tests = _timed_step(
        "calibration_tests", lambda: sufficiency_tests(frame, config)
    )
    edges, _ = decile_bins(frame[SCORE_COL].to_numpy(), config.calibration_bins)
    bin_edges = {"edges": edges.tolist()}

    cond_parity, cond_tests = _timed_step(
        "conditional_parity",
        lambda: conditional_parity(frame, config, gaps),
    )
    cond_cells = _timed_step(
        "conditional_parity_cells", lambda: conditional_parity_cells(frame, config)
    )
    _, cut_points = assign_pti_quintiles(frame[PTI_COL], config.cdp_n_quantiles)
    pti_meta = {
        "cut_points": [float(x) for x in cut_points],
        "n_missing": int(frame[PTI_Q_COL].isna().sum()),
    }

    oracle = _timed_step(
        "oracle_benchmark",
        lambda: oracle_benchmark(frame, config, gaps),
    )
    signs = _timed_step("expected_signs", lambda: expected_signs(frame, config, gaps))
    kendall = _timed_step("kendall_cells", lambda: kendall_cells(frame, config))

    importance_path = run_dir / IMPORTANCE_FILE
    importance = pd.read_parquet(importance_path) if importance_path.exists() else None
    proxy = _timed_step(
        "proxy_importance",
        lambda: proxy_importance(importance, config, model),
    )

    robust = _timed_step(
        "robustness_table",
        lambda: robustness_table(run_dir, config, groups_path, data_dir),
    )

    results: Dict[str, Any] = {
        "model": model,
        "threshold": thr,
        "utility": util,
        "group_counts": counts,
        "group_metrics": metrics_df,
        "gaps": gaps,
        "cell_summary": cells,
        "calibration": calibration,
        "calibration_tests": cal_tests,
        "bin_edges": bin_edges,
        "conditional_parity": cond_parity,
        "conditional_parity_tests": cond_tests,
        "conditional_parity_cells": cond_cells,
        "pti_quintiles": pti_meta,
        "oracle_benchmark": oracle,
        "expected_signs": signs,
        "kendall": kendall,
        "proxy_importance": proxy,
        "robustness": robust,
    }

    _timed_step("write_outputs", lambda: write_outputs(results, out_path))

    fig_dir = out_path / "figures"
    _timed_step(
        "figures",
        lambda: (
            plot_calibration(
                calibration,
                GENDER_COL,
                model,
                fig_dir / f"calibration_{GENDER_COL}.png",
                config,
            ),
            plot_calibration(
                calibration,
                AGE_BAND_COL,
                model,
                fig_dir / f"calibration_{AGE_BAND_COL}.png",
                config,
            ),
            plot_cells_error_rates(
                metrics_df, config, model, fig_dir / "cells_error_rates.png"
            ),
        ),
    )

    _timed_step("report", lambda: write_model_report(results, out_path, config))

    write_audit_config(
        audit_config_payload(
            config,
            model,
            input_paths,
            _git_commit(),
            _package_versions(),
            started_at,
            datetime.now(timezone.utc),
        ),
        out_path,
    )
    return out_path


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Run the baseline fairness audit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Registry model name (must be in config)")
    group.add_argument("--all", action="store_true", help="Audit every model in config")
    parser.add_argument("--config", type=Path, default=default_fairness_config_path())
    parser.add_argument("--runs-dir", type=Path, default=paths["runs_dir"])
    parser.add_argument("--features-dir", type=Path, default=paths["features_dir"])
    parser.add_argument("--data-dir", type=Path, default=paths["data_dir"])
    parser.add_argument("--out-dir", type=Path, default=paths["fairness_dir"])
    parser.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help="Override bootstrap replicates (smoke runs)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for ``python -m src.fairness.audit``."""
    args = _parse_args(argv)
    config = load_audit_config(args.config)
    if args.n_boot is not None:
        config = replace(config, n_boot=args.n_boot)

    models = list(config.models) if args.all else [args.model]
    unknown = [m for m in models if m not in config.models]
    if unknown:
        sys.exit(f"error: model(s) {unknown} not in config models {config.models}")
    if not args.overwrite:
        try:
            for model in models:
                check_output_free(args.out_dir, model, config.variant)
        except FileExistsError as exc:
            sys.exit(f"error: {exc}")
    for model in models:
        run_audit(
            model,
            config,
            args.runs_dir,
            args.features_dir,
            args.data_dir,
            args.out_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
