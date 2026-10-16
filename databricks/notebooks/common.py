"""Shared rules for the Databricks job: data quality, cost, credit score, drift and fairness.

Plain pandas/numpy so it runs on serverless compute without extra installs, and so it can be
unit-tested locally (tests/test_databricks_common.py checks it against the main package).
The notebooks in this folder import it with `from common import ...`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "credit_risk"                          # 1 = repaid, 0 = defaulted (source coding)
COST_FALSE_APPROVAL, COST_FALSE_REJECTION = 5, 1

# Same ranges as compliance_ai.credit.data.NUMERIC_RANGES.
RANGES = {
    "duration": (1, 120), "amount": (1, 100_000), "age": (18, 100),
    "installment_rate": (1, 4), "present_residence": (1, 4),
    "number_credits": (1, 10), "people_liable": (1, 5),
}

# Same allowed values as compliance_ai.credit.data.CATEGORIES.
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

# Collected for the fairness audit only, never model inputs.
AUDIT_ONLY = ["personal_status_sex", "foreign_worker", "age", "sex", "age_group"]
NOT_FEATURES = {TARGET, "default", "applicant_id", "_ingested_at", "_source_file", "_row_hash",
                *AUDIT_ONLY}

# Drift thresholds, as in compliance_ai.monitoring.
PSI_MONITOR, PSI_ALERT = 0.10, 0.25
MAX_QUARANTINE_SHARE = 0.05


# ---------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Audit attributes and domain features; pandas twin of the Spark code in 02_silver."""
    out = df.copy()
    out["sex"] = np.where(out["personal_status_sex"].str.startswith("female"), "female", "male")
    out["age_group"] = np.where(out["age"] < 25, "under_25", "25_plus")
    out["amount_per_month"] = out["amount"] / out["duration"]
    out["has_checking_account"] = (out["status"] != "no checking account").astype(int)
    out["no_savings"] = out["savings"].str.lower().str.contains("unknown|no savings").astype(int)
    out["delayed_before"] = out["credit_history"].str.lower().str.contains("delay").astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c not in NOT_FEATURES]
    numeric = [c for c in features if df[c].dtype.kind in "if"]
    return numeric, [c for c in features if c not in numeric]


# ---------------------------------------------------------------- decisions
def expected_cost(y_default: np.ndarray, p_default: np.ndarray, threshold: float) -> float:
    """Average cost per applicant: approving a defaulter costs 5, rejecting a good payer costs 1."""
    y = np.asarray(y_default)
    reject = np.asarray(p_default) >= threshold
    false_approval = np.sum(~reject & (y == 1))
    false_rejection = np.sum(reject & (y == 0))
    return float((COST_FALSE_APPROVAL * false_approval + COST_FALSE_REJECTION * false_rejection) / len(y))


def best_threshold(y_default: np.ndarray, p_default: np.ndarray) -> tuple[float, float]:
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    costs = [expected_cost(y_default, p_default, t) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def score_fields(p_default: np.ndarray, threshold: float) -> pd.DataFrame:
    """Credit score 0-100, PASS/FAIL and the human-review flag, as in compliance_ai.credit.predict."""
    p = np.asarray(p_default, dtype=float)
    passed = p < threshold
    return pd.DataFrame({
        "p_default": np.round(p, 4),
        "credit_score": np.round(100 * (1 - p), 1),
        "pass_mark": round(100 * (1 - threshold), 1),
        "result": np.where(passed, "PASS", "FAIL"),
        "requires_human_review": ~passed,       # GDPR Art. 22: no solely automated rejection
    })


# ---------------------------------------------------------------- drift
def psi(expected_shares, actual_shares, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected_shares, dtype=float), eps, None)
    a = np.clip(np.asarray(actual_shares, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI with decile bins from the reference (numeric) or category shares (categorical)."""
    if reference.dtype.kind in "if":
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(reference, bins=edges)[0]
        cur = np.histogram(current.astype(float), bins=edges)[0]
        return psi(ref / ref.sum(), cur / max(cur.sum(), 1))
    cats = sorted(set(reference.astype(str)) | set(current.astype(str)))
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    return psi([ref.get(c, 0.0) for c in cats], [cur.get(c, 0.0) for c in cats])


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        value = feature_psi(reference[col], current[col])
        status = "alert" if value >= PSI_ALERT else "monitor" if value >= PSI_MONITOR else "stable"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- fairness
def fairness_table(scores: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """Approval rate per group and disparate impact (group rate / best group rate; < 0.8 fails)."""
    rows = []
    approved = scores["result"].eq("PASS")
    for attribute in attributes:
        rates = approved.groupby(scores[attribute]).mean()
        sizes = scores[attribute].value_counts()
        for group, rate in rates.items():
            di = rate / rates.max() if rates.max() > 0 else np.nan
            rows.append({"attribute": attribute, "group": str(group), "applicants": int(sizes[group]),
                         "approval_rate": round(float(rate), 3), "disparate_impact": round(float(di), 3),
                         "four_fifths_rule": "pass" if di >= 0.8 else "FAIL"})
    return pd.DataFrame(rows)
