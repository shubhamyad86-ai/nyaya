"""Score new applicants with the trained model bundle and explain each decision."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from compliance_ai.config import load_config, resolve
from compliance_ai.credit import explain
from compliance_ai.credit.data import add_features, validate

# Plain-language names for the model inputs, used in reason codes and the scorer page.
FEATURE_LABELS = {
    "status": "Checking account balance",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "amount": "Loan amount (DM)",
    "savings": "Savings",
    "employment_duration": "Time in current job",
    "installment_rate": "Instalment as share of income (1-4)",
    "other_debtors": "Co-applicant or guarantor",
    "present_residence": "Years at current address (1-4)",
    "property": "Most valuable property",
    "other_installment_plans": "Other instalment plans",
    "housing": "Housing",
    "number_credits": "Credits at this bank",
    "job": "Job",
    "people_liable": "People financially dependent",
    "telephone": "Telephone registered",
    "amount_per_month": "Repayment per month (DM)",
    "has_checking_account": "Has a checking account",
    "no_savings": "No savings",
    "delayed_before": "Paid late in the past",
}


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    cfg = load_config()["credit"]
    model_path = Path(path) if path else resolve(cfg["model_dir"]) / "credit_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - run `python -m compliance_ai.credit.train` first"
        )
    return joblib.load(model_path)


def score(applicants: pd.DataFrame | list[dict], top_reasons: int = 4) -> list[dict]:
    """Return P(default), decision, a 0-100 credit score with PASS/FAIL, and reason codes.

    credit_score = 100 x (1 - P(default)). The pass mark is the same cut-off expressed as a score:
    100 x (1 - threshold), where the threshold minimises expected cost (a wrongly approved bad loan
    costs 5x a wrongly rejected good one). FAIL never means an automatic rejection: under GDPR
    Article 22 the application goes to a human underwriter.
    """
    df = pd.DataFrame(applicants) if not isinstance(applicants, pd.DataFrame) else applicants.copy()
    validate(df, require_target=False)
    df = add_features(df)

    bundle = load_bundle()
    X = df[bundle["numeric"] + bundle["categorical"]]
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    shap_df = explain.shap_values(bundle["pipeline"], X, bundle["background"])

    results = []
    threshold = float(bundle["threshold"])
    for i, idx in enumerate(X.index):
        passed = bool(p[i] < threshold)
        reasons = explain.reason_codes(shap_df.loc[idx], X.loc[idx], top_n=top_reasons)
        for r in reasons:
            r["label"] = FEATURE_LABELS.get(r["feature"], r["feature"].replace("_", " "))
        results.append({
            "p_default": round(float(p[i]), 4),
            "decision": "approve" if passed else "refer_or_reject",
            "threshold": threshold,
            "credit_score": round(100 * (1 - float(p[i])), 1),
            "pass_mark": round(100 * (1 - threshold), 1),
            "result": "PASS" if passed else "FAIL",
            "requires_human_review": not passed,
            "model": bundle["model_name"],
            "reasons": reasons,
        })
    return results
