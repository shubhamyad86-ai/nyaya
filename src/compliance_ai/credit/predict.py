"""Score new applicants with the trained model bundle and explain each decision."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from compliance_ai.config import load_config, resolve
from compliance_ai.credit import explain
from compliance_ai.credit.data import add_features, validate


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
    """Return P(default), decision and reason codes for each applicant."""
    df = pd.DataFrame(applicants) if not isinstance(applicants, pd.DataFrame) else applicants.copy()
    validate(df, require_target=False)
    df = add_features(df)

    bundle = load_bundle()
    X = df[bundle["numeric"] + bundle["categorical"]]
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    shap_df = explain.shap_values(bundle["pipeline"], X, bundle["background"])

    results = []
    for i, idx in enumerate(X.index):
        results.append({
            "p_default": round(float(p[i]), 4),
            "decision": "refer_or_reject" if p[i] >= bundle["threshold"] else "approve",
            "threshold": bundle["threshold"],
            "model": bundle["model_name"],
            "reasons": explain.reason_codes(shap_df.loc[idx], X.loc[idx], top_n=top_reasons),
        })
    return results
