"""Build the fairness audit frame from OOF predictions, groups, and raw PTI."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from src.constants import (
    AGE_BAND_COL,
    INTERSECTION_COL,
    SK_ID_CURR,
    TABLE_FILES,
    TARGET_COL,
)
from src.fairness.types import (
    ANNUITY_COL,
    AUDIT_COLUMNS,
    FOLD_COL,
    GENDER_COL,
    GROUPING_COLUMNS,
    INCOME_COL,
    OOF_COLUMNS,
    OOF_SCORE_COL,
    PTI_COL,
    PTI_Q_COL,
    SCORE_COL,
    Y_COL,
    AuditConfig,
    pti_quintile_labels,
)
from src.models.runs import load_oof

PathLike = Union[str, Path]

_GROUPS_COLUMNS: List[str] = [SK_ID_CURR] + GROUPING_COLUMNS


def load_pti(data_dir: PathLike) -> pd.DataFrame:
    """Return ``SK_ID_CURR`` and PTI (= ``AMT_ANNUITY / AMT_INCOME_TOTAL``).

    PTI is NaN when the annuity is missing or the income is missing or not positive.

    Args:
        data_dir: Directory containing ``application_train.csv``.

    Returns:
        Frame with columns ``SK_ID_CURR`` and ``PTI``.
    """
    csv_path = Path(data_dir) / TABLE_FILES["application_train"]
    app = pd.read_csv(csv_path, usecols=[SK_ID_CURR, ANNUITY_COL, INCOME_COL])
    income = app[INCOME_COL]
    pti = (app[ANNUITY_COL] / income).where(income > 0)
    return pd.DataFrame({SK_ID_CURR: app[SK_ID_CURR], PTI_COL: pti})


def assign_pti_quintiles(
    pti: pd.Series, n_quantiles: int
) -> Tuple[pd.Series, List[float]]:
    """Assign PTI quintile labels via ``pd.qcut`` on known values.

    Args:
        pti: Payment-to-income ratios (NaN where unknown).
        n_quantiles: Number of quantile bins (e.g. 5).

    Returns:
        Tuple of label series (NaN where PTI is NaN) and inner cut points on known PTI.
    """
    labels = pti_quintile_labels(n_quantiles)
    out = pd.Series(np.nan, index=pti.index, dtype=object)
    known = pti.notna()
    if known.any():
        probs = np.linspace(0.0, 1.0, n_quantiles + 1)[1:-1]
        cut_points = pti[known].quantile(probs).tolist()
        out.loc[known] = pd.qcut(
            pti[known],
            q=n_quantiles,
            labels=labels,
        )
    else:
        cut_points = []
    return out, cut_points


def _validate_oof(oof: pd.DataFrame, source: str) -> None:
    missing = [c for c in OOF_COLUMNS if c not in oof.columns]
    if missing:
        raise ValueError(f"{source}: OOF missing columns {missing}")
    extra = [c for c in oof.columns if c not in OOF_COLUMNS]
    if extra:
        raise ValueError(f"{source}: OOF has unexpected columns {extra}")
    if oof[SK_ID_CURR].duplicated().any():
        raise ValueError(f"{source}: duplicate SK_ID_CURR in OOF")
    if oof[OOF_COLUMNS].isna().any().any():
        raise ValueError(f"{source}: NaN in OOF columns")
    if not oof[TARGET_COL].isin([0, 1]).all():
        raise ValueError(f"{source}: {TARGET_COL} must be 0 or 1")
    scores = oof[OOF_SCORE_COL].to_numpy()
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError(f"{source}: {OOF_SCORE_COL} must lie in [0, 1]")


def _validate_group_labels(
    frame: pd.DataFrame, config: AuditConfig, path: Path
) -> None:
    for col in GROUPING_COLUMNS:
        allowed = set(config.groupings[col])
        values = frame[col]
        if values.isna().any():
            raise ValueError(f"{path}: NaN in group column {col!r}")
        bad = set(values.unique()) - allowed
        if bad:
            raise ValueError(f"{path}: unknown labels in {col!r}: {sorted(bad)}")


def load_audit_frame(
    run_dir: PathLike,
    rank: int,
    groups_path: PathLike,
    data_dir: PathLike,
    config: AuditConfig,
) -> pd.DataFrame:
    """Build the audit frame of plan 5.2 with all checks of 5.2.

    Args:
        run_dir: Model run directory (``artifacts/runs/<model>/<variant>``).
        rank: OOF rank (1 = top-level, >= 2 under robustness/).
        groups_path: Parquet with fairness grouping columns.
        data_dir: Raw CSV directory for PTI.
        config: Loaded audit configuration (group label allow-list).

    Returns:
        Audit frame with columns :data:`AUDIT_COLUMNS`, OOF row order preserved.

    Raises:
        ValueError: If the OOF file, the groups file or the PTI source violates
            the checks of plan 5.2 (the message names the offending file).
    """
    run_path = Path(run_dir)
    groups_file = Path(groups_path)
    oof = load_oof(run_path, rank)
    _validate_oof(oof, f"{run_path} (rank {rank})")

    groups = pd.read_parquet(groups_file, columns=_GROUPS_COLUMNS)
    n_oof = len(oof)
    try:
        merged = oof.merge(groups, on=SK_ID_CURR, how="inner", validate="many_to_one")
    except pd.errors.MergeError as exc:
        raise ValueError(f"{groups_file}: duplicate SK_ID_CURR ({exc})") from exc
    if len(merged) != n_oof:
        raise ValueError(
            f"{groups_file}: not every OOF SK_ID_CURR found in groups "
            f"(OOF {n_oof} rows, joined {len(merged)})"
        )

    _validate_group_labels(merged, config, groups_file)

    expected_gender_age = (
        merged[GENDER_COL].astype(str) + " | " + merged[AGE_BAND_COL].astype(str)
    )
    if not expected_gender_age.equals(merged[INTERSECTION_COL].astype(str)):
        raise ValueError(
            f"{groups_file}: GENDER_AGE must equal CODE_GENDER + ' | ' + AGE_BAND"
        )

    pti_frame = load_pti(data_dir)
    pti_source = Path(data_dir) / TABLE_FILES["application_train"]
    try:
        merged = merged.merge(
            pti_frame, on=SK_ID_CURR, how="left", validate="many_to_one"
        )
    except pd.errors.MergeError as exc:
        raise ValueError(f"{pti_source}: duplicate SK_ID_CURR ({exc})") from exc

    pti_q, _ = assign_pti_quintiles(merged[PTI_COL], config.cdp_n_quantiles)
    merged[PTI_Q_COL] = pti_q

    frame = pd.DataFrame(
        {
            SK_ID_CURR: merged[SK_ID_CURR].astype(np.int64),
            Y_COL: merged[TARGET_COL].astype(np.int8),
            SCORE_COL: merged[OOF_SCORE_COL].astype(np.float64),
            FOLD_COL: merged[FOLD_COL],
            GENDER_COL: merged[GENDER_COL],
            AGE_BAND_COL: merged[AGE_BAND_COL],
            INTERSECTION_COL: merged[INTERSECTION_COL],
            PTI_COL: merged[PTI_COL],
            PTI_Q_COL: merged[PTI_Q_COL],
        }
    )
    return frame[AUDIT_COLUMNS]
