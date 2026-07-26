"""Feature engineering: clean, aggregate, encode to applicant matrix."""

from src.constants import PROTECTED_ATTRIBUTES, TARGET_COL

__all__ = ["PROTECTED_ATTRIBUTES", "TARGET_COL", "build_features"]


def __getattr__(name: str):
    if name == "build_features":
        from src.features.pipeline import build_features

        return build_features
    if name == "FeatureArtifacts":
        from src.features.pipeline import FeatureArtifacts

        return FeatureArtifacts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
