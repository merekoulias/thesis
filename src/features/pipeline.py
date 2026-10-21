"""Orchestrate feature engineering into model-ready matrices."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.data.loader import HomeCreditData, load_home_credit_data
from src.features.application import clean_application, extract_fairness_groups
from src.features.bureau import aggregate_bureau
from src.features.credit_card import aggregate_credit_card
from src.features.encoding import FeatureEncoder, add_missing_indicators
from src.features.installments import aggregate_installments
from src.features.pos_cash import aggregate_pos_cash
from src.features.previous import aggregate_previous
from src.features.types import SK_ID_CURR, FeatureMatrixBundle

PathLike = Union[str, Path]


@dataclass
class FeatureArtifacts:
    """In-memory feature engineering outputs."""

    bundle: FeatureMatrixBundle
    encoder: FeatureEncoder


def _aggregate_all_children(data: HomeCreditData) -> dict:
    """Compute all client-level child aggregates once."""
    return {
        "bureau": aggregate_bureau(data.bureau, data.bureau_balance),
        "previous": aggregate_previous(data.previous_application),
        "installments": aggregate_installments(data.installments_payments),
        "pos_cash": aggregate_pos_cash(data.pos_cash),
        "credit_card": aggregate_credit_card(data.credit_card),
    }


def _left_join_features(
    base: pd.DataFrame, features: pd.DataFrame, history_flag: str
) -> pd.DataFrame:
    """Left-join client features and fill missing-history indicator."""
    out = base.merge(features, on=SK_ID_CURR, how="left")
    if history_flag in out.columns:
        out[history_flag] = out[history_flag].fillna(0).astype("int8")
    return out


def _build_client_matrix(app: pd.DataFrame, child_features: dict) -> pd.DataFrame:
    """Join all child-table aggregates onto the application base."""
    matrix = app.copy()
    joins = (
        ("bureau", "HAS_BUREAU_HISTORY"),
        ("previous", "HAS_PREV_APP_HISTORY"),
        ("installments", "HAS_INSTALLMENT_HISTORY"),
        ("pos_cash", "HAS_POS_CASH_HISTORY"),
        ("credit_card", "HAS_CREDIT_CARD_HISTORY"),
    )
    for key, flag in joins:
        matrix = _left_join_features(matrix, child_features[key], flag)

    for flag in (
        "HAS_BUREAU_HISTORY",
        "HAS_PREV_APP_HISTORY",
        "HAS_INSTALLMENT_HISTORY",
        "HAS_POS_CASH_HISTORY",
        "HAS_CREDIT_CARD_HISTORY",
    ):
        if flag in matrix.columns:
            matrix[flag] = matrix[flag].fillna(0).astype("int8")

    return matrix


def build_features(
    data: Optional[HomeCreditData] = None,
    data_dir: PathLike = "data",
) -> FeatureArtifacts:
    """Build train/test model-ready feature matrices.

    Args:
        data: Optional preloaded HomeCreditData. If None, loads from disk.
        data_dir: CSV directory used when data is None.

    Returns:
        FeatureArtifacts with X/y, ids, fairness groups, and encoder.
    """
    if data is None:
        data = load_home_credit_data(data_dir=data_dir)

    train_app, test_app, y_train, report = clean_application(
        data.application_train,
        data.application_test,
        xna_dropped=data.xna_gender_rows_dropped,
    )

    fairness_train = extract_fairness_groups(train_app)
    fairness_test = extract_fairness_groups(test_app)

    ids_train = train_app[SK_ID_CURR].copy()
    ids_test = test_app[SK_ID_CURR].copy()

    child_features = _aggregate_all_children(data)
    train_matrix = _build_client_matrix(train_app, child_features)
    test_matrix = _build_client_matrix(test_app, child_features)

    # Align columns after joins (child one-hots may differ by split)
    all_cols = sorted(set(train_matrix.columns).union(test_matrix.columns))
    # Keep SK_ID_CURR first for readability
    ordered = [SK_ID_CURR] + [c for c in all_cols if c != SK_ID_CURR]
    train_matrix = train_matrix.reindex(columns=ordered)
    test_matrix = test_matrix.reindex(columns=ordered)

    train_matrix, test_matrix, _ = add_missing_indicators(train_matrix, test_matrix)

    feature_cols = [c for c in train_matrix.columns if c != SK_ID_CURR]
    X_train_raw = train_matrix[feature_cols]
    X_test_raw = test_matrix[feature_cols]

    encoder = FeatureEncoder()
    X_train = encoder.fit_transform(X_train_raw)
    X_test = encoder.transform(X_test_raw)

    # Align index to applicant order
    X_train.index = ids_train.index
    X_test.index = ids_test.index
    y_train.index = ids_train.index

    bundle = FeatureMatrixBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        ids_train=ids_train,
        ids_test=ids_test,
        fairness_groups_train=fairness_train,
        fairness_groups_test=fairness_test,
        feature_names=list(X_train.columns),
        numeric_source_columns=list(encoder.numeric_columns_),
        categorical_source_columns=list(encoder.categorical_columns_),
        cleaning_report=report,
    )
    return FeatureArtifacts(bundle=bundle, encoder=encoder)


def save_feature_artifacts(
    artifacts: FeatureArtifacts, artifacts_dir: PathLike
) -> Path:
    """Persist feature matrices and fairness groups to parquet/csv.

    Args:
        artifacts: Output of build_features.
        artifacts_dir: Destination directory.

    Returns:
        Path to the artifacts directory.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = artifacts.bundle

    bundle.X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    bundle.X_test.to_parquet(out_dir / "X_test.parquet", index=False)
    bundle.y_train.to_frame("TARGET").to_parquet(
        out_dir / "y_train.parquet", index=False
    )
    bundle.ids_train.to_frame(SK_ID_CURR).to_parquet(
        out_dir / "ids_train.parquet", index=False
    )
    bundle.ids_test.to_frame(SK_ID_CURR).to_parquet(
        out_dir / "ids_test.parquet", index=False
    )
    bundle.fairness_groups_train.to_parquet(
        out_dir / "fairness_groups_train.parquet", index=False
    )
    if bundle.fairness_groups_test is not None:
        bundle.fairness_groups_test.to_parquet(
            out_dir / "fairness_groups_test.parquet", index=False
        )

    meta = {
        "n_features": len(bundle.feature_names),
        "n_train": int(len(bundle.X_train)),
        "n_test": int(len(bundle.X_test)),
        "xna_dropped": int(bundle.cleaning_report.xna_gender_rows_dropped),
        "building_cols_dropped": (bundle.cleaning_report.building_columns_dropped),
        "numeric_source_columns": bundle.numeric_source_columns,
        "categorical_source_columns": bundle.categorical_source_columns,
        "feature_names": bundle.feature_names,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def main() -> None:
    """CLI entrypoint: build and save feature artifacts."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Home Credit feature matrices")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with Home Credit CSVs",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/features",
        help="Output directory for engineered features",
    )
    args = parser.parse_args()

    artifacts = build_features(data_dir=args.data_dir)
    path = save_feature_artifacts(artifacts, args.artifacts_dir)
    bundle = artifacts.bundle
    print(
        f"Wrote features to {path} | "
        f"train={bundle.X_train.shape} test={bundle.X_test.shape} | "
        f"XNA dropped={bundle.cleaning_report.xna_gender_rows_dropped}"
    )


if __name__ == "__main__":
    main()
