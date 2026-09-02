"""Orchestrate LightGBM credit-scoring training and artifact persistence."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

from src.models.io import load_training_data, load_yaml_config
from src.models.training import train_lgbm_cv
from src.models.types import CVResult, ModelArtifacts, TrainingData

PathLike = Union[str, Path]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "model.yaml"


def train_credit_scorer(
    features_dir: PathLike = "artifacts/features",
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[PathLike] = None,
) -> ModelArtifacts:
    """Load features and train LightGBM with stratified CV.

    Args:
        features_dir: Directory with feature parquet artifacts.
        config: Optional pre-loaded model config dict.
        config_path: Path to model.yaml (used when config is None).

    Returns:
        ModelArtifacts with CV results and training data.
    """
    if config is None:
        path = Path(config_path) if config_path else _default_config_path()
        config = load_yaml_config(path)

    training_data = load_training_data(features_dir)
    cv_result = train_lgbm_cv(training_data, config, return_models=True)
    return ModelArtifacts(cv_result=cv_result, training_data=training_data)


def _metrics_dict(artifacts: ModelArtifacts) -> Dict[str, Any]:
    """Build serializable metrics summary."""
    cv = artifacts.cv_result
    data = artifacts.training_data
    return {
        "n_samples": int(len(data.X)),
        "n_features": int(len(data.feature_names)),
        "n_folds": len(cv.fold_results),
        "fold_roc_aucs": cv.fold_roc_aucs,
        "mean_fold_roc_auc": cv.mean_fold_roc_auc,
        "std_fold_roc_auc": cv.std_fold_roc_auc,
        "oof_roc_auc": cv.oof_roc_auc,
        "best_iterations": [fr.best_iteration for fr in cv.fold_results],
        "config": cv.config,
    }


def save_model_artifacts(
    artifacts: ModelArtifacts,
    models_dir: PathLike,
) -> Path:
    """Persist OOF predictions, metrics, ROC, models, and importance.

    Args:
        artifacts: Output of train_credit_scorer.
        models_dir: Destination directory for model artifacts.

    Returns:
        Path to the artifacts directory.
    """
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = artifacts.cv_result
    config = cv.config
    outputs_cfg = config.get("outputs", {})

    cv.oof_predictions.to_parquet(out_dir / "oof_predictions.parquet", index=False)

    metrics = _metrics_dict(artifacts)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "roc_curve.json").open("w", encoding="utf-8") as f:
        json.dump(cv.roc_curve, f, indent=2)

    if outputs_cfg.get("save_roc_plot", True):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            cv.roc_curve["fpr"],
            cv.roc_curve["tpr"],
            label=f"OOF AUC = {cv.oof_roc_auc:.4f}",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Out-of-fold ROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=120)
        plt.close(fig)

    if outputs_cfg.get("save_fold_models", True):
        models_subdir = out_dir / "fold_models"
        models_subdir.mkdir(exist_ok=True)
        for fold_idx, model in enumerate(cv.models):
            model.booster_.save_model(str(models_subdir / f"fold_{fold_idx}.txt"))

    if outputs_cfg.get("save_feature_importance", True):
        if cv.feature_importance_gain is not None:
            cv.feature_importance_gain.to_parquet(
                out_dir / "feature_importance_gain.parquet", index=False
            )
        if cv.feature_importance_split is not None:
            cv.feature_importance_split.to_parquet(
                out_dir / "feature_importance_split.parquet", index=False
            )

    fold_details = [asdict(fr) for fr in cv.fold_results]
    # Convert numpy arrays to lists for JSON serialization
    for detail in fold_details:
        detail["train_indices"] = detail["train_indices"].tolist()
        detail["val_indices"] = detail["val_indices"].tolist()
    with (out_dir / "fold_details.json").open("w", encoding="utf-8") as f:
        json.dump(fold_details, f, indent=2)

    return out_dir


def main() -> None:
    """CLI entrypoint: train LightGBM and save model artifacts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train LightGBM credit scorer with stratified CV"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="artifacts/features",
        help="Directory with feature parquet artifacts",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="artifacts/models",
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model.yaml (default: configs/model.yaml)",
    )
    args = parser.parse_args()

    config_path = args.config
    artifacts = train_credit_scorer(
        features_dir=args.features_dir,
        config_path=config_path,
    )
    out = save_model_artifacts(artifacts, args.models_dir)
    cv = artifacts.cv_result
    print(
        f"Wrote model artifacts to {out} | "
        f"OOF AUC={cv.oof_roc_auc:.4f} | "
        f"fold AUCs={[round(a, 4) for a in cv.fold_roc_aucs]}"
    )


if __name__ == "__main__":
    main()
