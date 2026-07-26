"""Tests for end-to-end feature matrix contracts and encoding."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import HomeCreditData, load_home_credit_data
from src.features.pipeline import build_features, save_feature_artifacts
from src.features.types import INTERSECTION_COL, PROTECTED_ATTRIBUTES, TARGET_COL


def _drop_xna(data: HomeCreditData) -> HomeCreditData:
    train = data.application_train[
        data.application_train["CODE_GENDER"] != "XNA"
    ].copy()
    return HomeCreditData(
        application_train=train,
        application_test=data.application_test.copy(),
        bureau=data.bureau,
        bureau_balance=data.bureau_balance,
        previous_application=data.previous_application,
        installments_payments=data.installments_payments,
        pos_cash=data.pos_cash,
        credit_card=data.credit_card,
        xna_gender_rows_dropped=1,
    )


def test_build_features_contract(tiny_home_credit_data: HomeCreditData) -> None:
    data = _drop_xna(tiny_home_credit_data)
    artifacts = build_features(data=data)
    bundle = artifacts.bundle

    assert len(bundle.X_train) == len(bundle.y_train)
    assert len(bundle.X_train) == len(bundle.ids_train)
    assert len(bundle.X_test) == len(bundle.ids_test)
    assert bundle.ids_train.is_unique
    assert bundle.ids_test.is_unique

    # One row per applicant after XNA drop
    assert len(bundle.X_train) == 3
    assert len(bundle.X_test) == 2

    assert TARGET_COL not in bundle.X_train.columns
    assert TARGET_COL not in bundle.X_test.columns
    assert list(bundle.X_train.columns) == list(bundle.X_test.columns)

    # Model-ready: no object/string columns
    assert not any(
        pd.api.types.is_object_dtype(bundle.X_train[c])
        or (
            hasattr(pd.api.types, "is_string_dtype")
            and pd.api.types.is_string_dtype(bundle.X_train[c])
        )
        for c in bundle.X_train.columns
    )
    assert np.isfinite(bundle.X_train.to_numpy(dtype=float)).all()

    # Protected attributes encoded into X
    gender_cols = [
        c for c in bundle.X_train.columns if "CODE_GENDER" in c
    ]
    edu_cols = [
        c for c in bundle.X_train.columns if "NAME_EDUCATION_TYPE" in c
    ]
    assert gender_cols, "Expected encoded CODE_GENDER columns in X"
    assert edu_cols, "Expected encoded NAME_EDUCATION_TYPE columns in X"

    # Raw fairness groups retained
    for col in PROTECTED_ATTRIBUTES + [INTERSECTION_COL]:
        assert col in bundle.fairness_groups_train.columns
    assert len(bundle.fairness_groups_train) == len(bundle.X_train)


def test_build_features_from_csv_dir(fixture_csv_dir: Path) -> None:
    data = load_home_credit_data(fixture_csv_dir)
    artifacts = build_features(data=data)
    assert artifacts.bundle.cleaning_report.xna_gender_rows_dropped == 1
    assert len(artifacts.bundle.X_train) == 3


def test_save_feature_artifacts(
    tiny_home_credit_data: HomeCreditData, tmp_path: Path
) -> None:
    artifacts = build_features(data=_drop_xna(tiny_home_credit_data))
    out = save_feature_artifacts(artifacts, tmp_path / "features")
    assert (out / "X_train.parquet").exists()
    assert (out / "y_train.parquet").exists()
    assert (out / "fairness_groups_train.parquet").exists()
    assert (out / "metadata.json").exists()
