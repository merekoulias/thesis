"""Tests for model YAML settings and hyperparameter candidate sampling."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.models.search import sample_candidates
from src.models.settings import (
    default_cv_config_path,
    default_model_config_path,
    load_cv_settings,
    load_model_config,
)
from src.models.types import CVSettings

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO_ROOT / "configs" / "models"


def _write_model_yaml(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def test_real_cv_configs_load() -> None:
    cv = load_cv_settings(_CONFIGS / "cv.yaml")
    assert cv == CVSettings(42, 5, 3, 20, 5)

    smoke = load_cv_settings(_CONFIGS / "cv_smoke.yaml")
    assert smoke == CVSettings(42, 3, 2, 3, 2)


def test_real_model_configs_load() -> None:
    for model_name in ("lgbm", "rf", "logreg", "mlp"):
        cfg = load_model_config(_CONFIGS / f"{model_name}.yaml")
        assert cfg.name == model_name

    lgbm = load_model_config(_CONFIGS / "lgbm.yaml")
    assert lgbm.early_stopping_rounds == 50

    for model_name in ("rf", "logreg", "mlp"):
        cfg = load_model_config(_CONFIGS / f"{model_name}.yaml")
        assert cfg.early_stopping_rounds is None


def test_sample_candidates_first_is_default() -> None:
    default = {"C": 1.0, "x": 2, "h": [128, 64]}
    space = {"C": {"dist": "loguniform", "low": 1e-4, "high": 10.0}}
    cands = sample_candidates(default, space, 3, 42)
    assert cands[0] == default
    cands[0]["C"] = 99.0
    cands[0]["h"].append(32)
    assert default["C"] == 1.0
    assert default["h"] == [128, 64]


def test_sample_candidates_deterministic() -> None:
    space = {
        "a": {"dist": "uniform", "low": 0.0, "high": 1.0},
        "b": {"dist": "int_uniform", "low": 1, "high": 10},
    }
    default = {"a": 0.5, "b": 5}
    first = sample_candidates(default, space, 5, 42)
    second = sample_candidates(default, space, 5, 42)
    third = sample_candidates(default, space, 5, 43)
    assert first == second
    assert first != third


def test_sample_candidates_key_order_irrelevant() -> None:
    default = {"a": 1.0, "b": 2.0, "c": 3.0}
    space_ab = {
        "a": {"dist": "uniform", "low": 0.0, "high": 1.0},
        "b": {"dist": "uniform", "low": 0.0, "high": 1.0},
        "c": {"dist": "uniform", "low": 0.0, "high": 1.0},
    }
    space_ba = {
        "c": space_ab["c"],
        "b": space_ab["b"],
        "a": space_ab["a"],
    }
    c1 = sample_candidates(default, space_ab, 4, 7)
    c2 = sample_candidates(default, space_ba, 4, 7)
    assert c1 == c2


def test_sample_candidates_types_and_bounds() -> None:
    space = {
        "u": {"dist": "uniform", "low": 0.2, "high": 0.8},
        "lu": {"dist": "loguniform", "low": 0.01, "high": 100.0},
        "iu": {"dist": "int_uniform", "low": 3, "high": 9},
        "ilu": {"dist": "int_loguniform", "low": 2, "high": 200},
        "ch": {"dist": "choice", "values": ["a", "b", None, [1, 2]]},
    }
    default = {"u": 0.5, "lu": 1.0, "iu": 5, "ilu": 10, "ch": "a"}
    cands = sample_candidates(default, space, 200, 123)
    choice_values = space["ch"]["values"]

    for cand in cands[1:]:
        assert 0.2 <= cand["u"] <= 0.8
        assert isinstance(cand["u"], float)
        assert cand["lu"] > 0
        assert 0.01 <= cand["lu"] <= 100.0
        assert isinstance(cand["lu"], float)
        assert type(cand["iu"]) is int
        assert 3 <= cand["iu"] <= 9
        assert type(cand["ilu"]) is int
        assert 2 <= cand["ilu"] <= 200
        assert cand["ch"] in choice_values


def test_sample_candidates_json_serializable() -> None:
    for model_name in ("lgbm", "rf", "logreg", "mlp"):
        cfg = load_model_config(_CONFIGS / f"{model_name}.yaml")
        cands = sample_candidates(cfg.default_params, cfg.search_space, 20, 42)
        json.dumps(cands)


@pytest.mark.parametrize(
    "name,data,match",
    [
        (
            "bad_dist",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs"},
                "default_params": {"C": 1.0},
                "search_space": {
                    "C": {"dist": "gaussian", "low": 0.1, "high": 1.0},
                },
            },
            "unknown dist",
        ),
        (
            "key_mismatch",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs"},
                "default_params": {"C": 1.0, "extra": 1},
                "search_space": {
                    "C": {"dist": "loguniform", "low": 1e-4, "high": 10.0},
                },
            },
            "default_params keys",
        ),
        (
            "random_state_fixed",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs", "random_state": 0},
                "default_params": {"C": 1.0},
                "search_space": {
                    "C": {"dist": "loguniform", "low": 1e-4, "high": 10.0},
                },
            },
            "random_state",
        ),
        (
            "fixed_search_overlap",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs", "C": 1.0},
                "default_params": {"C": 1.0},
                "search_space": {
                    "C": {"dist": "loguniform", "low": 1e-4, "high": 10.0},
                },
            },
            "must not share keys",
        ),
        (
            "lgbm_no_es",
            {
                "name": "lgbm",
                "fixed_params": {"objective": "binary"},
                "default_params": {"num_leaves": 31},
                "search_space": {
                    "num_leaves": {"dist": "int_uniform", "low": 10, "high": 50},
                },
            },
            "early_stopping_rounds",
        ),
        (
            "rf_with_es",
            {
                "name": "rf",
                "early_stopping_rounds": 10,
                "fixed_params": {"n_estimators": 10},
                "default_params": {"max_depth": None},
                "search_space": {
                    "max_depth": {
                        "dist": "choice",
                        "values": [None, 5],
                    },
                },
            },
            "early_stopping_rounds",
        ),
        (
            "stem_mismatch",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs"},
                "default_params": {"C": 1.0},
                "search_space": {
                    "C": {"dist": "loguniform", "low": 1e-4, "high": 10.0},
                },
            },
            "file stem",
        ),
        (
            "loguniform_zero",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs"},
                "default_params": {"C": 1.0},
                "search_space": {
                    "C": {"dist": "loguniform", "low": 0.0, "high": 10.0},
                },
            },
            "loguniform needs 0 < low < high",
        ),
        (
            "string_bound",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs"},
                "default_params": {"C": 1.0},
                "search_space": {
                    "C": {"dist": "loguniform", "low": "1e-4", "high": 10.0},
                },
            },
            "must be numbers",
        ),
        (
            "unexpected_spec_key",
            {
                "name": "logreg",
                "early_stopping_rounds": None,
                "fixed_params": {"solver": "lbfgs"},
                "default_params": {"C": 1.0},
                "search_space": {
                    "C": {
                        "dist": "uniform",
                        "low": 0.1,
                        "high": 1.0,
                        "values": [1],
                    },
                },
            },
            "does not accept keys",
        ),
    ],
)
def test_invalid_model_configs(
    tmp_path: Path, name: str, data: dict, match: str
) -> None:
    yaml_name = "wrong_stem" if name == "stem_mismatch" else data["name"]
    path = _write_model_yaml(tmp_path, yaml_name, data)
    with pytest.raises(ValueError, match=match) as excinfo:
        load_model_config(path)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"top_k": 25, "n_configs": 20}, "top_k"),
        ({"n_inner_folds": 1}, "n_inner_folds"),
        ({"seed": 42, "n_outer_folds": 5, "n_inner_folds": 3, "top_k": 2}, "missing"),
    ],
)
def test_invalid_cv_settings(tmp_path: Path, overrides: dict, match: str) -> None:
    base = {
        "seed": 42,
        "n_outer_folds": 5,
        "n_inner_folds": 3,
        "n_configs": 20,
        "top_k": 5,
    }
    if match == "missing":
        data = overrides
    else:
        data = {**base, **overrides}
    path = tmp_path / "cv.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    with pytest.raises(ValueError, match=match):
        load_cv_settings(path)


def test_default_paths_resolve_to_repo_configs() -> None:
    assert default_cv_config_path() == _CONFIGS / "cv.yaml"
    assert default_model_config_path("mlp") == _CONFIGS / "mlp.yaml"
