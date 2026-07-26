"""Shared schema constants used by data loading and feature engineering."""

from typing import Dict, List

SK_ID_CURR = "SK_ID_CURR"
SK_ID_PREV = "SK_ID_PREV"
SK_ID_BUREAU = "SK_ID_BUREAU"

TARGET_COL = "TARGET"
PROTECTED_ATTRIBUTES: List[str] = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]
INTERSECTION_COL = "GENDER_EDUCATION"

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
