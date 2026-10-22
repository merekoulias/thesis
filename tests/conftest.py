"""Shared fixtures for feature-engineering tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import HomeCreditData


@pytest.fixture
def tiny_home_credit_data() -> HomeCreditData:
    """Minimal multi-table fixture for FE contract tests."""
    application_train = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "CODE_GENDER": ["F", "M", "F", "XNA"],
            "DAYS_BIRTH": [-9000, -12000, -16000, -20000],
            "NAME_EDUCATION_TYPE": [
                "Secondary / secondary special",
                "Higher education",
                "Secondary / secondary special",
                "Lower secondary",
            ],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Cash loans",
                "Revolving loans",
                "Cash loans",
            ],
            "FLAG_OWN_CAR": ["N", "Y", "N", "Y"],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0, np.nan, 150000.0],
            "DAYS_EMPLOYED": [-1000, 365243, -500, -200],
            "OWN_CAR_AGE": [np.nan, 3.0, np.nan, 10.0],
            "EXT_SOURCE_1": [0.2, np.nan, 0.5, np.nan],
            "EXT_SOURCE_2": [0.3, 0.4, np.nan, 0.6],
            "EXT_SOURCE_3": [np.nan, np.nan, 0.1, 0.2],
            "APARTMENTS_AVG": [0.1, np.nan, 0.2, np.nan],
            "APARTMENTS_MEDI": [0.1, np.nan, 0.2, np.nan],
            "APARTMENTS_MODE": [0.1, np.nan, 0.2, np.nan],
            "HOUSETYPE_MODE": ["block of flats", np.nan, "terraced house", np.nan],
        }
    )
    application_test = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 11],
            "CODE_GENDER": ["M", "F"],
            "DAYS_BIRTH": [-11000, -22000],
            "NAME_EDUCATION_TYPE": [
                "Higher education",
                "Secondary / secondary special",
            ],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
            "FLAG_OWN_CAR": ["Y", "N"],
            "AMT_INCOME_TOTAL": [120000.0, np.nan],
            "DAYS_EMPLOYED": [365243, -300],
            "OWN_CAR_AGE": [5.0, np.nan],
            "EXT_SOURCE_1": [np.nan, 0.4],
            "EXT_SOURCE_2": [0.5, np.nan],
            "EXT_SOURCE_3": [0.3, np.nan],
            "APARTMENTS_AVG": [np.nan, 0.3],
            "APARTMENTS_MEDI": [np.nan, 0.3],
            "APARTMENTS_MODE": [np.nan, 0.3],
            "HOUSETYPE_MODE": [np.nan, "block of flats"],
        }
    )

    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 10],
            "SK_ID_BUREAU": [100, 101, 102, 103],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed"],
            "CREDIT_CURRENCY": ["currency 1"] * 4,
            "DAYS_CREDIT": [-100, -200, -50, -30],
            "CREDIT_DAY_OVERDUE": [0, 0, 5, 0],
            "DAYS_CREDIT_ENDDATE": [-10, 100, 20, 5],
            "DAYS_ENDDATE_FACT": [np.nan, -5, np.nan, -1],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, np.nan, 10.0, np.nan],
            "CNT_CREDIT_PROLONG": [0, 0, 1, 0],
            "AMT_CREDIT_SUM": [1000.0, 2000.0, 1500.0, 800.0],
            "AMT_CREDIT_SUM_DEBT": [100.0, -50.0, 200.0, 0.0],
            "AMT_CREDIT_SUM_LIMIT": [500.0, -10.0, 0.0, 100.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 5.0, 0.0],
            "CREDIT_TYPE": [
                "Consumer credit",
                "Credit card",
                "Consumer credit",
                "Microloan",
            ],
            "DAYS_CREDIT_UPDATE": [-10, -20, -5, -2],
            "AMT_ANNUITY": [np.nan, 50.0, 40.0, np.nan],
        }
    )
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [100, 100, 101, 102],
            "MONTHS_BALANCE": [-1, -2, -1, -1],
            "STATUS": ["0", "1", "C", "0"],
        }
    )
    previous_application = pd.DataFrame(
        {
            "SK_ID_PREV": [1000, 1001, 1002, 1003],
            "SK_ID_CURR": [1, 1, 2, 10],
            "NAME_CONTRACT_TYPE": [
                "Consumer loans",
                "Cash loans",
                "Consumer loans",
                "Revolving loans",
            ],
            "AMT_ANNUITY": [100.0, 200.0, 150.0, 80.0],
            "AMT_APPLICATION": [1000.0, 0.0, 2000.0, 500.0],
            "AMT_CREDIT": [900.0, 1500.0, 1800.0, 500.0],
            "AMT_DOWN_PAYMENT": [100.0, np.nan, 200.0, np.nan],
            "AMT_GOODS_PRICE": [1000.0, np.nan, 2000.0, np.nan],
            "NAME_CONTRACT_STATUS": [
                "Approved",
                "Refused",
                "Approved",
                "Approved",
            ],
            "DAYS_DECISION": [-100, -200, -50, -20],
            "NAME_CLIENT_TYPE": ["Repeater", "New", "Repeater", "New"],
            "NAME_PRODUCT_TYPE": ["x-sell", "x-sell", "walk-in", "x-sell"],
            "CHANNEL_TYPE": [
                "Country-wide",
                "Credit and cash offices",
                "Country-wide",
                "AP+ (Cash loan)",
            ],
            "SELLERPLACE_AREA": [100, -1, 50, 10],
            "CNT_PAYMENT": [12, 6, 24, 0],
            "DAYS_FIRST_DRAWING": [365243, -10, 365243, -5],
            "DAYS_FIRST_DUE": [-90, -180, -40, 365243],
            "DAYS_LAST_DUE_1ST_VERSION": [-10, -20, 0, 365243],
            "DAYS_LAST_DUE": [-10, 365243, 0, 365243],
            "DAYS_TERMINATION": [-5, 365243, 5, 365243],
            "RATE_INTEREST_PRIMARY": [np.nan, np.nan, 0.1, np.nan],
            "RATE_INTEREST_PRIVILEGED": [np.nan, np.nan, np.nan, np.nan],
            "RATE_DOWN_PAYMENT": [0.1, np.nan, 0.2, np.nan],
        }
    )
    installments_payments = pd.DataFrame(
        {
            "SK_ID_PREV": [1000, 1000, 1000, 1002],
            "SK_ID_CURR": [1, 1, 1, 2],
            "NUM_INSTALMENT_VERSION": [1, 1, 1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 1, 2, 1],
            "DAYS_INSTALMENT": [-90, -90, -60, -40],
            "DAYS_ENTRY_PAYMENT": [-88, -87, -50, np.nan],
            "AMT_INSTALMENT": [100.0, 100.0, 100.0, 150.0],
            "AMT_PAYMENT": [40.0, 60.0, 100.0, np.nan],
        }
    )
    pos_cash = pd.DataFrame(
        {
            "SK_ID_PREV": [1000, 1000, 1002],
            "SK_ID_CURR": [1, 1, 2],
            "MONTHS_BALANCE": [-1, -2, -1],
            "CNT_INSTALMENT": [12, 12, 24],
            "CNT_INSTALMENT_FUTURE": [10, 11, 20],
            "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed"],
            "SK_DPD": [0, 5, 0],
            "SK_DPD_DEF": [0, 0, 0],
        }
    )
    credit_card = pd.DataFrame(
        {
            "SK_ID_PREV": [1003, 1003],
            "SK_ID_CURR": [10, 10],
            "MONTHS_BALANCE": [-1, -2],
            "AMT_BALANCE": [100.0, 0.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [500.0, 500.0],
            "AMT_DRAWINGS_CURRENT": [50.0, 0.0],
            "AMT_INST_MIN_REGULARITY": [20.0, 20.0],
            "AMT_PAYMENT_CURRENT": [10.0, 25.0],
            "AMT_TOTAL_RECEIVABLE": [100.0, 0.0],
            "AMT_RECIVABLE": [100.0, 0.0],
            "AMT_RECEIVABLE_PRINCIPAL": [90.0, 0.0],
            "SK_DPD": [0, 2],
            "SK_DPD_DEF": [0, 0],
            "NAME_CONTRACT_STATUS": ["Active", "Active"],
        }
    )

    # Drop XNA outside loader for fixture consistency with pipeline path
    # that expects already-validated frames; tests for loader cover XNA.
    return HomeCreditData(
        application_train=application_train,
        application_test=application_test,
        bureau=bureau,
        bureau_balance=bureau_balance,
        previous_application=previous_application,
        installments_payments=installments_payments,
        pos_cash=pos_cash,
        credit_card=credit_card,
        xna_gender_rows_dropped=0,
    )


@pytest.fixture
def fixture_csv_dir(tiny_home_credit_data: HomeCreditData, tmp_path: Path) -> Path:
    """Write tiny fixtures to a temporary CSV directory."""
    data = tiny_home_credit_data
    mapping = {
        "application_train.csv": data.application_train,
        "application_test.csv": data.application_test,
        "bureau.csv": data.bureau,
        "bureau_balance.csv": data.bureau_balance,
        "previous_application.csv": data.previous_application,
        "installments_payments.csv": data.installments_payments,
        "POS_CASH_balance.csv": data.pos_cash,
        "credit_card_balance.csv": data.credit_card,
    }
    for name, frame in mapping.items():
        frame.to_csv(tmp_path / name, index=False)
    return tmp_path


@pytest.fixture
def make_training_data():
    """Factory for small synthetic TrainingData with a learnable signal."""
    from src.constants import SK_ID_CURR, TARGET_COL
    from src.models.types import TrainingData

    def _make(n_samples: int = 400, n_features: int = 8, seed: int = 0) -> TrainingData:
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            rng.normal(size=(n_samples, n_features)),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        X["cat__CODE_GENDER_F"] = (rng.random(n_samples) < 0.66).astype(float)
        X["cat__CODE_GENDER_M"] = 1.0 - X["cat__CODE_GENDER_F"]
        X["num__DAYS_BIRTH"] = -rng.integers(7500, 25000, size=n_samples).astype(float)
        logits = -1.5 + 1.5 * X["feat_0"] - 1.0 * X["feat_1"]
        y = (rng.random(n_samples) < 1.0 / (1.0 + np.exp(-logits))).astype(np.int8)
        ids = pd.Series(np.arange(1, n_samples + 1), name=SK_ID_CURR)
        return TrainingData(
            X=X, y=pd.Series(y, name=TARGET_COL), ids=ids, feature_names=list(X.columns)
        )

    return _make


@pytest.fixture
def tiny_model_configs():
    """Fast ModelConfig per registry model (for unit tests only)."""
    from src.models.types import ModelConfig

    return {
        "lgbm": ModelConfig(
            name="lgbm",
            fixed_params={
                "objective": "binary",
                "n_estimators": 40,
                "learning_rate": 0.1,
                "verbosity": -1,
                "n_jobs": 1,
                "subsample_freq": 1,
            },
            default_params={"num_leaves": 7, "subsample": 0.8},
            search_space={
                "num_leaves": {"dist": "int_uniform", "low": 4, "high": 15},
                "subsample": {"dist": "uniform", "low": 0.6, "high": 1.0},
            },
            early_stopping_rounds=5,
        ),
        "rf": ModelConfig(
            name="rf",
            fixed_params={"n_estimators": 15, "n_jobs": 1},
            default_params={"min_samples_leaf": 5},
            search_space={
                "min_samples_leaf": {"dist": "int_loguniform", "low": 2, "high": 30}
            },
        ),
        "logreg": ModelConfig(
            name="logreg",
            fixed_params={"max_iter": 500},
            default_params={"C": 1.0},
            search_space={"C": {"dist": "loguniform", "low": 0.01, "high": 10.0}},
        ),
        "mlp": ModelConfig(
            name="mlp",
            fixed_params={"early_stopping": False},
            default_params={"hidden_layer_sizes": [8], "alpha": 0.1, "max_iter": 5},
            search_space={
                "hidden_layer_sizes": {"dist": "choice", "values": [[4], [8]]},
                "alpha": {"dist": "loguniform", "low": 0.001, "high": 1.0},
                "max_iter": {"dist": "int_uniform", "low": 3, "high": 8},
            },
        ),
    }


@pytest.fixture
def toy_audit_frame() -> pd.DataFrame:
    """The 20-row toy example of fairness plan 5.7 with y, yhat, score and group columns."""
    f_y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    f_yhat = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    m_y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    m_yhat = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    y = np.array(f_y + m_y, dtype=np.int8)
    yhat = np.array(f_yhat + m_yhat, dtype=np.int8)
    n = len(y)
    gender = ["F"] * 10 + ["M"] * 10
    band = ["20-29"] * 5 + ["60-69"] * 5 + ["20-29"] * 5 + ["60-69"] * 5
    score = np.where(yhat == 1, 0.9, 0.1)
    pti = np.linspace(0.05, 0.4, n)
    from src.fairness.io import assign_pti_quintiles

    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(1, n + 1, dtype=np.int64),
            "y": y,
            "yhat": yhat.astype(np.int8),
            "score": score.astype(np.float64),
            "fold": np.arange(n) % 2,
            "CODE_GENDER": gender,
            "AGE_BAND": band,
            "GENDER_AGE": [f"{g} | {b}" for g, b in zip(gender, band)],
            "PTI": pti.astype(np.float64),
        }
    )
    frame["PTI_Q"], _ = assign_pti_quintiles(frame["PTI"], 5)
    return frame


@pytest.fixture
def make_audit_frame():
    """Factory: synthetic audit frame (default 4,000 rows) with all 10 cells."""
    from src.fairness.io import assign_pti_quintiles

    def _make(n: int = 4000, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        gender = rng.choice(["F", "M"], size=n, p=[0.66, 0.34])
        band = rng.choice(["20-29", "30-39", "40-49", "50-59", "60-69"], size=n)
        logit = (
            -2.4
            + 0.4 * (gender == "M")
            - 0.15 * np.array([int(b[0]) - 2 for b in band])
        )
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
        score = np.clip(
            1 / (1 + np.exp(-(logit + 1.2 * (y - 0.08) + rng.normal(0, 0.8, n)))),
            1e-6,
            1 - 1e-6,
        )
        pti = rng.lognormal(-1.8, 0.5, n)
        pti[:3] = np.nan
        frame = pd.DataFrame(
            {
                "SK_ID_CURR": np.arange(100_000, 100_000 + n),
                "y": y,
                "score": score,
                "fold": np.arange(n) % 5,
                "CODE_GENDER": gender,
                "AGE_BAND": band,
                "GENDER_AGE": [f"{g} | {b}" for g, b in zip(gender, band)],
                "PTI": pti,
            }
        )
        frame["PTI_Q"], _ = assign_pti_quintiles(frame["PTI"], 5)
        return frame

    return _make
