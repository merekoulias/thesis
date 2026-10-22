"""Tests for fairness config loading and audit frame I/O."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.fairness.config import default_fairness_config_path, load_audit_config
from src.fairness.io import assign_pti_quintiles, load_audit_frame, load_pti
from src.fairness.types import AUDIT_COLUMNS


def test_real_config_loads() -> None:
    config = load_audit_config(default_fairness_config_path())
    assert len(config.pairs) == 9
    assert config.primary_pair == "F-M"
    assert config.primary_metrics == ["fpr", "fnr"]
    assert abs(config.z - 1.959964) < 1e-6
    n_groups = sum(len(labels) for labels in config.groupings.values())
    assert n_groups == 17


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


@pytest.fixture
def fairness_yaml_template(tmp_path: Path) -> Path:
    data = _load_yaml(default_fairness_config_path())
    path = tmp_path / "fairness.yaml"
    _write_yaml(path, data)
    return path


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d.update({"gap_metrics": ["approval_rate", "not_a_metric"]}),
        lambda d: d["pairs"][0]["a"].update({"CODE_GENDER": "Z"}),
        lambda d: d["pairs"].append(deepcopy(d["pairs"][0])),
        lambda d: d["primary"].update({"pair": "NO-SUCH-PAIR"}),
        lambda d: d.update({"confidence": 1.0}),
        lambda d: d["proxy_watch_list"]["protected"].append("[invalid"),
        lambda d: d["pairs"][0].pop("id"),
        lambda d: d["pairs"][0].update({"a": {}}),
        lambda d: d["pairs"][0].update({"note": "x"}),
        lambda d: d["groupings"].pop("GENDER_AGE"),
        lambda d: d.update({"confidence": "abc"}),
    ],
    ids=[
        "unknown_metric",
        "unknown_label",
        "duplicate_pair_id",
        "undefined_primary_pair",
        "confidence_one",
        "bad_regex",
        "pair_without_id",
        "empty_pair_filter",
        "unknown_pair_key",
        "missing_grouping",
        "non_numeric_confidence",
    ],
)
def test_invalid_configs(fairness_yaml_template: Path, mutator) -> None:
    data = _load_yaml(fairness_yaml_template)
    mutator(data)
    _write_yaml(fairness_yaml_template, data)
    with pytest.raises(ValueError, match="fairness.yaml"):
        load_audit_config(fairness_yaml_template)


def test_assign_pti_quintiles() -> None:
    series = pd.Series([np.nan] + list(range(1, 101)), dtype=float)
    labels, cut_points = assign_pti_quintiles(series, 5)
    assert pd.isna(labels.iloc[0])
    for q in range(1, 6):
        assert (labels == f"Q{q}").sum() == 20
    expected = pd.Series(range(1, 101), dtype=float).quantile([0.2, 0.4, 0.6, 0.8])
    assert len(cut_points) == 4
    for got, exp in zip(cut_points, expected):
        assert np.isclose(got, exp)


def test_load_audit_frame(tmp_path: Path) -> None:
    config = load_audit_config(default_fairness_config_path())
    ids = [105, 101, 110, 103, 108, 102, 109, 104, 107, 106]
    oof = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "y_pred_proba": np.linspace(0.1, 0.9, 10),
            "fold": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    oof.to_parquet(run_dir / "oof_predictions.parquet", index=False)

    bands = ["20-29", "30-39", "40-49", "50-59", "60-69"] * 2
    genders = ["F", "M"] * 5
    groups = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "CODE_GENDER": genders,
            "AGE_BAND": bands,
            "GENDER_AGE": [f"{g} | {b}" for g, b in zip(genders, bands)],
        }
    )
    # Different row order (plus an extra id) than the OOF: the frame must keep the OOF order.
    groups = pd.concat(
        [groups, groups.iloc[:1].assign(SK_ID_CURR=999)], ignore_index=True
    ).sort_values("SK_ID_CURR")
    groups_path = tmp_path / "groups.parquet"
    groups.to_parquet(groups_path, index=False)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    app = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "AMT_ANNUITY": np.linspace(5000, 50000, 10),
            "AMT_INCOME_TOTAL": [100000.0] * 10,
            "IGNORE_ME": ["x"] * 10,
        }
    ).sample(frac=1, random_state=0)
    app.to_csv(data_dir / "application_train.csv", index=False)

    frame = load_audit_frame(run_dir, 1, groups_path, data_dir, config)
    assert list(frame.columns) == AUDIT_COLUMNS
    assert frame["SK_ID_CURR"].tolist() == ids
    assert frame["y"].tolist() == oof["TARGET"].tolist()
    expected_cells = groups.set_index("SK_ID_CURR").loc[ids, "GENDER_AGE"].tolist()
    assert frame["GENDER_AGE"].tolist() == expected_cells
    expected_pti = (app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]).set_axis(
        app["SK_ID_CURR"]
    )
    assert np.allclose(frame["PTI"].to_numpy(), expected_pti.loc[ids].to_numpy())
    assert frame["PTI_Q"].notna().all()


@pytest.mark.parametrize(
    "setup",
    [
        "missing_group_id",
        "extra_group_label",
        "duplicate_oof_id",
        "score_above_one",
        "gender_age_mismatch",
        "target_not_binary",
        "duplicate_group_id",
        "duplicate_csv_id",
    ],
)
def test_load_audit_frame_errors(tmp_path: Path, setup: str) -> None:
    config = load_audit_config(default_fairness_config_path())
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    oof = pd.DataFrame(
        {
            "SK_ID_CURR": [101, 102],
            "TARGET": [0, 1],
            "y_pred_proba": [0.2, 0.8],
            "fold": [0, 1],
        }
    )
    groups = pd.DataFrame(
        {
            "SK_ID_CURR": [101, 102],
            "CODE_GENDER": ["F", "M"],
            "AGE_BAND": ["20-29", "30-39"],
            "GENDER_AGE": ["F | 20-29", "M | 30-39"],
        }
    )

    if setup == "missing_group_id":
        groups = groups.iloc[:1]
    elif setup == "extra_group_label":
        groups.loc[0, "AGE_BAND"] = "99-99"
        groups.loc[0, "GENDER_AGE"] = "F | 99-99"
    elif setup == "duplicate_oof_id":
        oof = pd.concat([oof, oof.iloc[:1]], ignore_index=True)
    elif setup == "score_above_one":
        oof.loc[0, "y_pred_proba"] = 1.5
    elif setup == "gender_age_mismatch":
        groups.loc[0, "GENDER_AGE"] = "F | 30-39"
    elif setup == "target_not_binary":
        oof.loc[0, "TARGET"] = 2
    elif setup == "duplicate_group_id":
        groups = pd.concat([groups, groups.iloc[:1]], ignore_index=True)

    oof.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    groups_path = tmp_path / "groups.parquet"
    groups.to_parquet(groups_path, index=False)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    app = pd.DataFrame(
        {
            "SK_ID_CURR": [101, 102],
            "AMT_ANNUITY": [10000.0, 20000.0],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0],
        }
    )
    if setup == "duplicate_csv_id":
        app = pd.concat([app, app.iloc[:1]], ignore_index=True)
    app.to_csv(data_dir / "application_train.csv", index=False)

    with pytest.raises(ValueError):
        load_audit_frame(run_dir, 1, groups_path, data_dir, config)


def test_load_pti_non_positive_income_is_missing(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "AMT_ANNUITY": [10.0, 20.0, np.nan],
            "AMT_INCOME_TOTAL": [0.0, 100.0, 100.0],
        }
    ).to_csv(tmp_path / "application_train.csv", index=False)
    pti = load_pti(tmp_path)
    assert pti["PTI"].isna().tolist() == [True, False, True]
    assert np.isclose(pti["PTI"].iloc[1], 0.2)
