"""Credit scoring models (LightGBM)."""

__all__ = [
    "train_credit_scorer",
    "save_model_artifacts",
    "ModelArtifacts",
    "CVResult",
    "TrainingData",
]


def __getattr__(name: str):
    if name == "train_credit_scorer":
        from src.models.pipeline import train_credit_scorer

        return train_credit_scorer
    if name == "save_model_artifacts":
        from src.models.pipeline import save_model_artifacts

        return save_model_artifacts
    if name == "ModelArtifacts":
        from src.models.types import ModelArtifacts

        return ModelArtifacts
    if name == "CVResult":
        from src.models.types import CVResult

        return CVResult
    if name == "TrainingData":
        from src.models.types import TrainingData

        return TrainingData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
