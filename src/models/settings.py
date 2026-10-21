"""Load and validate nested-CV and per-model YAML configuration."""

from pathlib import Path
from typing import Any, Dict, Set, Tuple, Union

from src.models.io import load_yaml_config
from src.models.search import validate_distribution
from src.models.types import CVSettings, ModelConfig

PathLike = Union[str, Path]

KNOWN_MODELS: Tuple[str, ...] = ("lgbm", "rf", "logreg", "mlp")
EARLY_STOPPING_MODELS: Tuple[str, ...] = ("lgbm",)

_ALLOWED_MODEL_TOP_KEYS: Set[str] = {
    "name",
    "early_stopping_rounds",
    "fixed_params",
    "default_params",
    "search_space",
}

_CV_KEYS: Tuple[str, ...] = (
    "seed",
    "n_outer_folds",
    "n_inner_folds",
    "n_configs",
    "top_k",
)

_FORBIDDEN_PARAM_KEYS: Set[str] = {"random_state", "seed"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cv_config_path() -> Path:
    """Return <repo>/configs/models/cv.yaml."""
    return _repo_root() / "configs" / "models" / "cv.yaml"


def default_model_config_path(model_name: str) -> Path:
    """Return <repo>/configs/models/<model_name>.yaml.

    Args:
        model_name: Registry name, which is also the YAML file stem.

    Returns:
        Path resolved from this file's location.
    """
    return _repo_root() / "configs" / "models" / f"{model_name}.yaml"


def load_cv_settings(path: PathLike) -> CVSettings:
    """Load configs/models/cv.yaml. Validates: all five keys present and int.

    Also requires n_outer_folds >= 2; n_inner_folds >= 2; n_configs >= 1;
    1 <= top_k <= n_configs.

    Args:
        path: Path to a CV settings YAML file.

    Returns:
        Parsed CVSettings.

    Raises:
        ValueError: On missing keys, wrong types, or invalid values.
    """
    config_path = Path(path)
    raw = load_yaml_config(config_path)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping at top level")

    missing = [k for k in _CV_KEYS if k not in raw]
    if missing:
        raise ValueError(f"{config_path}: missing required keys: {missing}")

    extra = set(raw.keys()) - set(_CV_KEYS)
    if extra:
        raise ValueError(f"{config_path}: unexpected keys: {sorted(extra)}")

    values: Dict[str, Any] = {}
    for key in _CV_KEYS:
        val = raw[key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(
                f"{config_path}: {key} must be an int, "
                f"got {val!r} ({type(val).__name__})"
            )
        values[key] = val

    if values["n_outer_folds"] < 2:
        raise ValueError(
            f"{config_path}: n_outer_folds must be >= 2, got {values['n_outer_folds']}"
        )
    if values["n_inner_folds"] < 2:
        raise ValueError(
            f"{config_path}: n_inner_folds must be >= 2, got {values['n_inner_folds']}"
        )
    if values["n_configs"] < 1:
        raise ValueError(
            f"{config_path}: n_configs must be >= 1, got {values['n_configs']}"
        )
    if not (1 <= values["top_k"] <= values["n_configs"]):
        raise ValueError(
            f"{config_path}: top_k must satisfy 1 <= top_k <= n_configs, "
            f"got top_k={values['top_k']}, n_configs={values['n_configs']}"
        )

    return CVSettings(
        seed=values["seed"],
        n_outer_folds=values["n_outer_folds"],
        n_inner_folds=values["n_inner_folds"],
        n_configs=values["n_configs"],
        top_k=values["top_k"],
    )


def _check_no_forbidden_keys(
    mapping: Dict[str, Any], label: str, config_path: Path
) -> None:
    for key in mapping:
        if key in _FORBIDDEN_PARAM_KEYS:
            raise ValueError(f"{config_path}: {label} must not contain '{key}'")


def load_model_config(path: PathLike) -> ModelConfig:
    """Load configs/models/<name>.yaml and validate it (plan 5.3/5.4).

    Args:
        path: Path to a model YAML file.

    Returns:
        Parsed ModelConfig.

    Raises:
        ValueError: On schema or consistency violations.
    """
    config_path = Path(path)
    raw = load_yaml_config(config_path)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping at top level")

    unknown_top = set(raw.keys()) - _ALLOWED_MODEL_TOP_KEYS
    if unknown_top:
        raise ValueError(
            f"{config_path}: unexpected top-level keys: {sorted(unknown_top)}"
        )

    required = {"name", "fixed_params", "default_params", "search_space"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"{config_path}: missing required keys: {sorted(missing)}")

    name = raw["name"]
    if not isinstance(name, str):
        raise ValueError(f"{config_path}: name must be a string, got {name!r}")

    stem = config_path.stem
    if name not in KNOWN_MODELS:
        raise ValueError(
            f"{config_path}: unknown model name '{name}'; "
            f"expected one of {KNOWN_MODELS}"
        )
    if name != stem:
        raise ValueError(f"{config_path}: name '{name}' must match file stem '{stem}'")

    fixed_params = raw["fixed_params"]
    default_params = raw["default_params"]
    search_space = raw["search_space"]
    if not isinstance(fixed_params, dict):
        raise ValueError(f"{config_path}: fixed_params must be a mapping")
    if not isinstance(default_params, dict):
        raise ValueError(f"{config_path}: default_params must be a mapping")
    if not isinstance(search_space, dict):
        raise ValueError(f"{config_path}: search_space must be a mapping")

    if set(default_params) != set(search_space):
        raise ValueError(
            f"{config_path}: default_params keys must equal search_space keys; "
            f"default_only={set(default_params) - set(search_space)}, "
            f"search_only={set(search_space) - set(default_params)}"
        )

    overlap = set(fixed_params) & set(search_space)
    if overlap:
        raise ValueError(
            f"{config_path}: fixed_params and search_space must not share keys: "
            f"{sorted(overlap)}"
        )

    _check_no_forbidden_keys(fixed_params, "fixed_params", config_path)
    _check_no_forbidden_keys(default_params, "default_params", config_path)
    for param_name, spec in search_space.items():
        if param_name in _FORBIDDEN_PARAM_KEYS:
            raise ValueError(
                f"{config_path}: search_space must not contain '{param_name}'"
            )
        if not isinstance(spec, dict):
            raise ValueError(
                f"{config_path}: search_space['{param_name}'] must be a mapping"
            )
        try:
            validate_distribution(param_name, spec)
        except ValueError as exc:
            raise ValueError(f"{config_path}: {exc}") from exc

    early_stopping_rounds = raw.get("early_stopping_rounds")
    if name in EARLY_STOPPING_MODELS:
        if not isinstance(early_stopping_rounds, int) or isinstance(
            early_stopping_rounds, bool
        ):
            raise ValueError(
                f"{config_path}: {name} requires early_stopping_rounds as int >= 1, "
                f"got {early_stopping_rounds!r}"
            )
        if early_stopping_rounds < 1:
            raise ValueError(
                f"{config_path}: early_stopping_rounds must be >= 1, "
                f"got {early_stopping_rounds}"
            )
    else:
        if early_stopping_rounds is not None:
            raise ValueError(
                f"{config_path}: early_stopping_rounds must be absent or null for "
                f"{name}, got {early_stopping_rounds!r}"
            )
        early_stopping_rounds = None

    return ModelConfig(
        name=name,
        fixed_params=fixed_params,
        default_params=default_params,
        search_space=search_space,
        early_stopping_rounds=early_stopping_rounds,
    )
