"""Shared schema constants used by data loading and feature engineering."""

from typing import Dict, List

SK_ID_CURR = "SK_ID_CURR"
SK_ID_PREV = "SK_ID_PREV"
SK_ID_BUREAU = "SK_ID_BUREAU"

TARGET_COL = "TARGET"
PROTECTED_ATTRIBUTES: List[str] = ["CODE_GENDER", "DAYS_BIRTH"]

# Fairness grouping: gender x 10-year age bands (see docs/decisions.md)
AGE_YEARS_COL = "AGE_YEARS"
AGE_BAND_COL = "AGE_BAND"
INTERSECTION_COL = "GENDER_AGE"
DAYS_PER_YEAR = 365.25
# Left-closed band edges in years: [20, 30), [30, 40), ..., [60, 70)
AGE_BAND_EDGES: List[int] = [20, 30, 40, 50, 60, 70]

# Home Credit date sentinel (unemployed / not applicable)
DAYS_SENTINEL = 365243

STRUCTURAL_CAT_PLACEHOLDERS = ("XNA", "XAP")

TABLE_FILES: Dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "credit_card": "credit_card_balance.csv",
}

REQUIRED_KEYS: Dict[str, List[str]] = {
    "application_train": [SK_ID_CURR, TARGET_COL, "CODE_GENDER"],
    "application_test": [SK_ID_CURR, "CODE_GENDER"],
    "bureau": [SK_ID_CURR, SK_ID_BUREAU],
    "bureau_balance": [SK_ID_BUREAU],
    "previous_application": [SK_ID_CURR, SK_ID_PREV],
    "installments_payments": [SK_ID_CURR, SK_ID_PREV],
    "pos_cash": [SK_ID_CURR, SK_ID_PREV],
    "credit_card": [SK_ID_CURR, SK_ID_PREV],
}

MISSING_TOKEN = "__MISSING__"
