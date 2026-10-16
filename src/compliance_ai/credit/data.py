"""Load, validate and enrich the Statlog German Credit data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from compliance_ai.config import load_config, resolve

EXPECTED_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount", "savings",
    "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age",
    "other_installment_plans", "housing", "number_credits", "job",
    "people_liable", "telephone", "foreign_worker", "credit_risk",
]

NUMERIC_RANGES = {
    "duration": (1, 120),      # months
    "amount": (1, 100_000),    # Deutsche Mark
    "age": (18, 100),
    "installment_rate": (1, 4),
    "present_residence": (1, 4),
    "number_credits": (1, 10),
    "people_liable": (1, 5),
}


# Allowed values of each categorical column, as they appear in the training data
# (Statlog German Credit, corrected codebook by Grömping 2019).
CATEGORIES = {
    "status": ["no checking account", "... < 100 DM", "0 <= ... < 200 DM",
               "... >= 200 DM / salary for at least 1 year"],
    "credit_history": ["no credits taken/all credits paid back duly",
                       "all credits at this bank paid back duly",
                       "existing credits paid back duly till now",
                       "delay in paying off in the past",
                       "critical account/other credits existing"],
    "purpose": ["car (new)", "car (used)", "furniture/equipment", "radio/television",
                "domestic appliances", "repairs", "education", "retraining",
                "business", "others"],
    "savings": ["unknown/no savings account", "... < 100 DM", "100 <= ... < 500 DM",
                "500 <= ... < 1000 DM", "... >= 1000 DM"],
    "employment_duration": ["unemployed", "... < 1 year", "1 <= ... < 4 years",
                            "4 <= ... < 7 years", "... >= 7 years"],
    "personal_status_sex": ["male : divorced/separated", "female : divorced/separated/married",
                            "male : single", "male : married/widowed"],
    "other_debtors": ["none", "co-applicant", "guarantor"],
    "property": ["unknown/no property", "car or other",
                 "building society savings agreement/life insurance", "real estate"],
    "other_installment_plans": ["bank", "stores", "none"],
    "housing": ["for free", "rent", "own"],
    "job": ["unemployed/unskilled - non-resident", "unskilled - resident",
            "skilled employee/official",
            "management/self-employed/highly qualified employee/officer"],
    "telephone": ["no", "yes"],
    "foreign_worker": ["yes", "no"],
}


class DataValidationError(ValueError):
    """Raised when the input data does not match the expected schema."""


def validate(df: pd.DataFrame, require_target: bool = True, check_categories: bool = True) -> None:
    """Raise DataValidationError if columns, missing values, ranges or categories are wrong.

    check_categories=False lets drift monitoring report new categories instead of rejecting them.
    """
    expected = EXPECTED_COLUMNS if require_target else EXPECTED_COLUMNS[:-1]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if df[expected].isna().any().any():
        cols = df[expected].columns[df[expected].isna().any()].tolist()
        raise DataValidationError(f"missing values in: {cols}")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        bad = ~df[col].between(lo, hi)
        if bad.any():
            raise DataValidationError(f"{col}: {int(bad.sum())} value(s) outside [{lo}, {hi}]")
    for col, allowed in (CATEGORIES.items() if check_categories else ()):
        bad = ~df[col].isin(allowed)
        if bad.any():
            examples = sorted(map(str, df.loc[bad, col].unique()))[:3]
            raise DataValidationError(f"{col}: unknown value(s) {examples}; allowed: {allowed}")
    if require_target and not set(df["credit_risk"].unique()) <= {0, 1}:
        raise DataValidationError("credit_risk must be 0 (bad) or 1 (good)")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive audit attributes and domain features. Does not modify the input."""
    out = df.copy()
    # Audit-only attributes (never model inputs, see config.yaml).
    out["sex"] = np.where(out["personal_status_sex"].str.startswith("female"), "female", "male")
    out["age_group"] = np.where(out["age"] < 25, "under_25", "25_plus")
    # Domain features.
    out["amount_per_month"] = out["amount"] / out["duration"]
    out["log_amount"] = np.log1p(out["amount"])
    out["has_checking_account"] = (out["status"] != "no checking account").astype(int)
    out["no_savings"] = out["savings"].str.contains("unknown|no savings", case=False).astype(int)
    out["delayed_before"] = out["credit_history"].str.contains("delay", case=False).astype(int)
    return out


def load_credit_data(path: str | None = None) -> pd.DataFrame:
    cfg = load_config()["credit"]
    df = pd.read_csv(resolve(path or cfg["data_path"]))
    validate(df)
    return add_features(df)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) model inputs, excluding target and protected attributes."""
    cfg = load_config()["credit"]
    excluded = {cfg["target"], *cfg["protected"], *cfg["drop_from_features"]}
    cols = [c for c in df.columns if c not in excluded]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical
