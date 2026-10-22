"""Credit scoring models (LightGBM, random forest, logistic regression, MLP)."""

from typing import Any

__all__ = [
    "run_nested_cv",
    "get_model_spec",
    "MODEL_REGISTRY",
    "load_oof",
    "load_selected_params",
    "CVSettings",
    "ModelConfig",
    "NestedCVResult",
    "CVResult",
    "TrainingData",
]


def __getattr__(name: str) -> Any:
    if name == "run_nested_cv":
        from src.models.nested import run_nested_cv

        return run_nested_cv
    if name == "get_model_spec":
        from src.models.registry import get_model_spec

        return get_model_spec
    if name == "MODEL_REGISTRY":
        from src.models.registry import MODEL_REGISTRY

        return MODEL_REGISTRY
    if name == "load_oof":
        from src.models.runs import load_oof

        return load_oof
    if name == "load_selected_params":
        from src.models.runs import load_selected_params

        return load_selected_params
    if name == "CVSettings":
        from src.models.types import CVSettings

        return CVSettings
    if name == "ModelConfig":
        from src.models.types import ModelConfig

        return ModelConfig
    if name == "NestedCVResult":
        from src.models.types import NestedCVResult

        return NestedCVResult
    if name == "CVResult":
        from src.models.types import CVResult

        return CVResult
    if name == "TrainingData":
        from src.models.types import TrainingData

        return TrainingData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
