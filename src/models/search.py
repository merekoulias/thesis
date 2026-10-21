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


def validate_distribution(name: str, spec: Dict[str, Any]) -> None:
    """Raise ValueError if a search-space entry is malformed (see plan 5.4).

    Args:
        name: Parameter name (for error messages).
        spec: Distribution specification with a ``dist`` key.

    Raises:
        ValueError: If the specification is invalid.
    """
    if "dist" not in spec:
        raise ValueError(
            f"search_space['{name}']: missing 'dist'; "
            f"expected one of ({_DIST_CHOICES})"
        )
    dist = spec["dist"]
    if dist not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"search_space['{name}']: unknown dist '{dist}'; "
            f"expected one of ({_DIST_CHOICES})"
        )

    if dist == "uniform":
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"search_space['{name}']: uniform requires 'low' and 'high'"
            )
        low, high = spec["low"], spec["high"]
        if not low < high:
            raise ValueError(
                f"search_space['{name}']: uniform needs low < high, "
                f"got low={low}, high={high}"
            )
    elif dist == "loguniform":
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"search_space['{name}']: loguniform requires 'low' and 'high'"
            )
        low, high = spec["low"], spec["high"]
        if not (0 < low < high):
            raise ValueError(
                f"search_space['{name}']: loguniform needs 0 < low < high, "
                f"got low={low}, high={high}"
            )
    elif dist == "int_uniform":
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"search_space['{name}']: int_uniform requires 'low' and 'high'"
            )
        low, high = spec["low"], spec["high"]
        if not _is_int_not_bool(low) or not _is_int_not_bool(high):
            raise ValueError(
                f"search_space['{name}']: int_uniform low and high must be int, "
                f"got low={low!r}, high={high!r}"
            )
        if not low < high:
            raise ValueError(
                f"search_space['{name}']: int_uniform needs low < high, "
                f"got low={low}, high={high}"
            )
    elif dist == "int_loguniform":
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"search_space['{name}']: int_loguniform requires 'low' and 'high'"
            )
        low, high = spec["low"], spec["high"]
        if not _is_int_not_bool(low) or not _is_int_not_bool(high):
            raise ValueError(
                f"search_space['{name}']: int_loguniform low and high must be int, "
                f"got low={low!r}, high={high!r}"
            )
        if not (1 <= low < high):
            raise ValueError(
                f"search_space['{name}']: int_loguniform needs 1 <= low < high, "
                f"got low={low}, high={high}"
            )
    elif dist == "choice":
        if "values" not in spec:
            raise ValueError(f"search_space['{name}']: choice requires 'values'")
        values = spec["values"]
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"search_space['{name}']: choice needs a non-empty list 'values'"
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
