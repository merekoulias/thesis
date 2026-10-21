"""Hyperparameter candidate sampling for nested cross-validation."""

import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np

SUPPORTED_DISTRIBUTIONS: Tuple[str, ...] = (
    "uniform",
    "loguniform",
    "int_uniform",
    "int_loguniform",
    "choice",
)

_DIST_CHOICES = ", ".join(sorted(SUPPORTED_DISTRIBUTIONS))


def _is_int_not_bool(value: Any) -> bool:
    return type(value) is int


def _round_sig6(value: float) -> float:
    return float(f"{value:.6g}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_distribution(name: str, spec: Dict[str, Any]) -> None:
    """Raise ValueError if a search-space entry is malformed (see plan 5.4).

    Args:
        name: Parameter name (for error messages).
        spec: Distribution specification with a ``dist`` key.

    Raises:
        ValueError: If the specification is invalid.
    """
    prefix = f"search_space['{name}']"
    if "dist" not in spec:
        raise ValueError(f"{prefix}: missing 'dist'; expected one of ({_DIST_CHOICES})")
    dist = spec["dist"]
    if dist not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"{prefix}: unknown dist '{dist}'; expected one of ({_DIST_CHOICES})"
        )

    allowed_keys = {"dist", "values"} if dist == "choice" else {"dist", "low", "high"}
    unexpected = sorted(set(spec) - allowed_keys)
    if unexpected:
        raise ValueError(
            f"{prefix}: {dist} does not accept keys {unexpected}; "
            f"allowed keys are {sorted(allowed_keys)}"
        )

    if dist == "choice":
        if "values" not in spec:
            raise ValueError(f"{prefix}: choice requires 'values'")
        values = spec["values"]
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"{prefix}: choice needs a non-empty list 'values'")
        return

    if "low" not in spec or "high" not in spec:
        raise ValueError(f"{prefix}: {dist} requires 'low' and 'high'")
    low, high = spec["low"], spec["high"]

    if dist in ("int_uniform", "int_loguniform"):
        if not _is_int_not_bool(low) or not _is_int_not_bool(high):
            raise ValueError(
                f"{prefix}: {dist} low and high must be int, "
                f"got low={low!r}, high={high!r}"
            )
    elif not _is_number(low) or not _is_number(high):
        raise ValueError(
            f"{prefix}: {dist} low and high must be numbers, "
            f"got low={low!r}, high={high!r}"
        )

    if dist == "loguniform" and not (0 < low < high):
        raise ValueError(
            f"{prefix}: loguniform needs 0 < low < high, got low={low}, high={high}"
        )
    if dist == "int_loguniform" and not (1 <= low < high):
        raise ValueError(
            f"{prefix}: int_loguniform needs 1 <= low < high, "
            f"got low={low}, high={high}"
        )
    if dist in ("uniform", "int_uniform") and not low < high:
        raise ValueError(
            f"{prefix}: {dist} needs low < high, got low={low}, high={high}"
        )


def _draw_parameter(name: str, spec: Dict[str, Any], rng: np.random.Generator) -> Any:
    """Sample one parameter value from a validated distribution spec."""
    dist = spec["dist"]
    if dist == "uniform":
        low, high = spec["low"], spec["high"]
        return _round_sig6(float(rng.uniform(low, high)))
    if dist == "loguniform":
        low, high = spec["low"], spec["high"]
        return _round_sig6(float(math.exp(rng.uniform(math.log(low), math.log(high)))))
    if dist == "int_uniform":
        low, high = spec["low"], spec["high"]
        return int(rng.integers(low, high, endpoint=True))
    if dist == "int_loguniform":
        low, high = spec["low"], spec["high"]
        raw = math.exp(rng.uniform(math.log(low), math.log(high)))
        return int(round(raw))
    if dist == "choice":
        values = spec["values"]
        idx = int(rng.integers(len(values)))
        return copy.deepcopy(values[idx])
    raise ValueError(
        f"search_space['{name}']: unknown dist '{dist}'; "
        f"expected one of ({_DIST_CHOICES})"
    )


def sample_candidates(
    default_params: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
    n_configs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return n_configs candidates: [deepcopy(default_params), draw_1, ..., draw_{n-1}].

    One rng = np.random.default_rng(seed) is created once and used for all draws.
    Within each draw, parameters are sampled in sorted(search_space) order.

    Args:
        default_params: Candidate 0 (copied, not aliased).
        search_space: Per-parameter distribution specs.
        n_configs: Total number of candidates including the default.
        seed: RNG seed for draws 1..n_configs-1.

    Returns:
        List of parameter dicts.

    Raises:
        ValueError: If n_configs < 1.
    """
    if n_configs < 1:
        raise ValueError(f"n_configs must be >= 1, got {n_configs}")

    for param_name, spec in search_space.items():
        validate_distribution(param_name, spec)

    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = [copy.deepcopy(default_params)]
    sorted_keys = sorted(search_space.keys())

    for _ in range(n_configs - 1):
        draw: Dict[str, Any] = {}
        for key in sorted_keys:
            draw[key] = _draw_parameter(key, search_space[key], rng)
        candidates.append(draw)

    return candidates
