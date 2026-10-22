"""Load and validate ``configs/fairness.yaml``."""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Union

from src.fairness.group_metrics import METRIC_SPECS
from src.fairness.types import GROUPING_COLUMNS, AuditConfig, PairSpec
from src.models.io import load_yaml_config

PathLike = Union[str, Path]

_FAIRNESS_TOP_KEYS: Set[str] = {
    "seed",
    "n_boot",
    "confidence",
    "variant",
    "models",
    "robustness_ranks",
    "calibration_bins",
    "dir_reference_line",
    "groupings",
    "pairs",
    "primary",
    "gap_metrics",
    "cell_summary_metrics",
    "robustness_metrics",
    "conditional_parity",
    "proxy_watch_list",
    "importance_type",
}

_PAIR_FILTER_COLUMNS: Set[str] = {"CODE_GENDER", "AGE_BAND"}

_CONDITIONAL_PARITY_KEYS: Set[str] = {
    "pairs",
    "n_quantiles",
    "expected_cut_points",
    "expected_missing",
}

_PRIMARY_KEYS: Set[str] = {"pair", "metrics"}

_PAIR_KEYS: Set[str] = {"id", "a", "b"}

_ALLOWED_ROBUSTNESS_RANKS: Set[int] = {2, 3, 4, 5}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_fairness_config_path() -> Path:
    """Return ``<repo>/configs/fairness.yaml``."""
    return _repo_root() / "configs" / "fairness.yaml"


def _require_keys(
    data: Dict[str, Any], expected: Set[str], path: Path, label: str
) -> None:
    missing = expected - set(data.keys())
    if missing:
        raise ValueError(f"{path}: {label} missing keys: {sorted(missing)}")
    extra = set(data.keys()) - expected
    if extra:
        raise ValueError(f"{path}: {label} unknown keys: {sorted(extra)}")


def _cast(value: Any, to_type: Callable[[Any], Any], path: Path, name: str) -> Any:
    """Convert ``value`` with ``to_type``; raise ``ValueError`` naming ``path``."""
    try:
        return to_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {name} has invalid value {value!r}") from exc


def _cast_list(
    values: Any, to_type: Callable[[Any], Any], path: Path, name: str
) -> List[Any]:
    """Convert every element of a list; raise ``ValueError`` naming ``path``."""
    if not isinstance(values, list):
        raise ValueError(f"{path}: {name} must be a list")
    return [_cast(v, to_type, path, name) for v in values]


def _parse_pair_filters(
    raw: Dict[str, Any], path: Path, pair_id: str, side: str
) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: pair {pair_id!r} {side} must be a mapping, got {type(raw)}"
        )
    if not raw:
        raise ValueError(f"{path}: pair {pair_id!r} {side} must not be empty")
    unknown_cols = set(raw.keys()) - _PAIR_FILTER_COLUMNS
    if unknown_cols:
        raise ValueError(
            f"{path}: pair {pair_id!r} {side} uses unknown columns: {sorted(unknown_cols)}"
        )
    return {str(k): str(v) for k, v in raw.items()}


def _validate_pair_labels(
    filters: Dict[str, str],
    groupings: Dict[str, List[str]],
    path: Path,
    pair_id: str,
    side: str,
) -> None:
    for col, label in filters.items():
        allowed = groupings.get(col)
        if allowed is None:
            raise ValueError(
                f"{path}: pair {pair_id!r} references unknown grouping {col!r}"
            )
        if label not in allowed:
            raise ValueError(
                f"{path}: pair {pair_id!r} {side} label {label!r} "
                f"not in groupings[{col!r}]"
            )


def _validate_metrics(metrics: List[str], path: Path, field_name: str) -> None:
    allowed = set(METRIC_SPECS)
    for name in metrics:
        if name not in allowed:
            raise ValueError(f"{path}: unknown metric {name!r} in {field_name}")


def load_audit_config(path: PathLike) -> AuditConfig:
    """Load and validate fairness audit configuration.

    Args:
        path: Path to ``fairness.yaml``.

    Returns:
        Parsed :class:`AuditConfig`.

    Raises:
        ValueError: On schema or consistency violations (message includes ``path``).
    """
    config_path = Path(path)
    raw = load_yaml_config(config_path)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: root must be a mapping")

    _require_keys(raw, _FAIRNESS_TOP_KEYS, config_path, "top level")

    confidence = _cast(raw["confidence"], float, config_path, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"{config_path}: confidence must satisfy 0 < confidence < 1")

    n_boot = _cast(raw["n_boot"], int, config_path, "n_boot")
    if n_boot < 1:
        raise ValueError(f"{config_path}: n_boot must be >= 1")

    calibration_bins = _cast(
        raw["calibration_bins"], int, config_path, "calibration_bins"
    )
    if calibration_bins < 2:
        raise ValueError(f"{config_path}: calibration_bins must be >= 2")

    robustness_ranks = _cast_list(
        raw["robustness_ranks"], int, config_path, "robustness_ranks"
    )
    bad_ranks = [r for r in robustness_ranks if r not in _ALLOWED_ROBUSTNESS_RANKS]
    if bad_ranks:
        raise ValueError(
            f"{config_path}: robustness_ranks must be subset of {{2,3,4,5}}, "
            f"got invalid {bad_ranks}"
        )

    groupings_raw = raw["groupings"]
    if not isinstance(groupings_raw, dict):
        raise ValueError(f"{config_path}: groupings must be a mapping")
    if set(groupings_raw) != set(GROUPING_COLUMNS):
        raise ValueError(
            f"{config_path}: groupings keys must be exactly {GROUPING_COLUMNS}, "
            f"got {sorted(groupings_raw)}"
        )
    groupings: Dict[str, List[str]] = {
        str(k): _cast_list(vals, str, config_path, f"groupings[{k!r}]")
        for k, vals in groupings_raw.items()
    }

    pairs_raw = raw["pairs"]
    if not isinstance(pairs_raw, list):
        raise ValueError(f"{config_path}: pairs must be a list")
    pairs: List[PairSpec] = []
    pair_ids: Set[str] = set()
    for entry in pairs_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{config_path}: each pair entry must be a mapping")
        _require_keys(entry, _PAIR_KEYS, config_path, f"pair {entry.get('id')!r}")
        pair_id = str(entry["id"])
        if pair_id in pair_ids:
            raise ValueError(f"{config_path}: duplicate pair id {pair_id!r}")
        pair_ids.add(pair_id)
        a = _parse_pair_filters(entry["a"], config_path, pair_id, "a")
        b = _parse_pair_filters(entry["b"], config_path, pair_id, "b")
        if a == b:
            raise ValueError(f"{config_path}: pair {pair_id!r} has identical a and b")
        _validate_pair_labels(a, groupings, config_path, pair_id, "a")
        _validate_pair_labels(b, groupings, config_path, pair_id, "b")
        pairs.append(PairSpec(pair_id=pair_id, a=a, b=b))

    primary_raw = raw["primary"]
    if not isinstance(primary_raw, dict):
        raise ValueError(f"{config_path}: primary must be a mapping")
    _require_keys(primary_raw, _PRIMARY_KEYS, config_path, "primary")
    primary_pair = str(primary_raw["pair"])
    if primary_pair not in pair_ids:
        raise ValueError(
            f"{config_path}: primary.pair {primary_pair!r} is not a defined pair id"
        )
    primary_metrics = _cast_list(
        primary_raw["metrics"], str, config_path, "primary.metrics"
    )
    _validate_metrics(primary_metrics, config_path, "primary.metrics")

    gap_metrics = _cast_list(raw["gap_metrics"], str, config_path, "gap_metrics")
    cell_summary_metrics = _cast_list(
        raw["cell_summary_metrics"], str, config_path, "cell_summary_metrics"
    )
    robustness_metrics = _cast_list(
        raw["robustness_metrics"], str, config_path, "robustness_metrics"
    )
    _validate_metrics(gap_metrics, config_path, "gap_metrics")
    _validate_metrics(cell_summary_metrics, config_path, "cell_summary_metrics")
    _validate_metrics(robustness_metrics, config_path, "robustness_metrics")

    cdp_raw = raw["conditional_parity"]
    if not isinstance(cdp_raw, dict):
        raise ValueError(f"{config_path}: conditional_parity must be a mapping")
    _require_keys(cdp_raw, _CONDITIONAL_PARITY_KEYS, config_path, "conditional_parity")
    cdp_pairs = _cast_list(
        cdp_raw["pairs"], str, config_path, "conditional_parity.pairs"
    )
    for pid in cdp_pairs:
        if pid not in pair_ids:
            raise ValueError(
                f"{config_path}: conditional_parity.pairs entry {pid!r} "
                f"is not a defined pair id"
            )

    proxy_watch_list_raw = raw["proxy_watch_list"]
    if not isinstance(proxy_watch_list_raw, dict):
        raise ValueError(f"{config_path}: proxy_watch_list must be a mapping")
    proxy_watch_list: Dict[str, List[str]] = {}
    for group_name, patterns in proxy_watch_list_raw.items():
        if not isinstance(patterns, list):
            raise ValueError(
                f"{config_path}: proxy_watch_list[{group_name!r}] must be a list"
            )
        compiled_patterns: List[str] = []
        for pat in patterns:
            pat_str = str(pat)
            try:
                re.compile(pat_str)
            except re.error as exc:
                raise ValueError(
                    f"{config_path}: invalid regex in proxy_watch_list[{group_name!r}]: "
                    f"{pat_str!r} ({exc})"
                ) from exc
            compiled_patterns.append(pat_str)
        proxy_watch_list[str(group_name)] = compiled_patterns

    importance_type_raw = raw["importance_type"]
    if not isinstance(importance_type_raw, dict):
        raise ValueError(f"{config_path}: importance_type must be a mapping")
    importance_type = {str(k): str(v) for k, v in importance_type_raw.items()}

    return AuditConfig(
        seed=_cast(raw["seed"], int, config_path, "seed"),
        n_boot=n_boot,
        confidence=confidence,
        variant=str(raw["variant"]),
        models=_cast_list(raw["models"], str, config_path, "models"),
        robustness_ranks=robustness_ranks,
        calibration_bins=calibration_bins,
        dir_reference_line=_cast(
            raw["dir_reference_line"], float, config_path, "dir_reference_line"
        ),
        groupings=groupings,
        pairs=pairs,
        primary_pair=primary_pair,
        primary_metrics=primary_metrics,
        gap_metrics=gap_metrics,
        cell_summary_metrics=cell_summary_metrics,
        robustness_metrics=robustness_metrics,
        cdp_pairs=cdp_pairs,
        cdp_n_quantiles=_cast(
            cdp_raw["n_quantiles"], int, config_path, "conditional_parity.n_quantiles"
        ),
        cdp_expected_cut_points=_cast_list(
            cdp_raw["expected_cut_points"],
            float,
            config_path,
            "conditional_parity.expected_cut_points",
        ),
        cdp_expected_missing=_cast(
            cdp_raw["expected_missing"],
            int,
            config_path,
            "conditional_parity.expected_missing",
        ),
        proxy_watch_list=proxy_watch_list,
        importance_type=importance_type,
    )
