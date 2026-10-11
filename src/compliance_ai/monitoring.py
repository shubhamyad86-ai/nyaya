"""Data and prediction drift monitoring with the Population Stability Index (PSI).

At training time a reference profile is saved: decile bins for each numeric feature,
category shares for each categorical feature, and the distribution of the model's
scores. A batch of new applicants is compared against it.

PSI = sum((actual% - expected%) * ln(actual% / expected%)), the standard credit-risk
stability measure. Common reading: < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant
shift - investigate, and consider retraining before trusting the model on this population.

PSI alone over-reacts on small batches: with 30-60 applicants, a feature with ten
categories reaches PSI > 0.25 by chance alone (random samples of the training data were
flagged 70-100% of the time). So each feature also gets a chi-square goodness-of-fit
p-value, and a shift counts only when it is both large (PSI) and unlikely to be noise
(p < 0.01 for "significant_shift", p < 0.05 for "monitor", Bonferroni-corrected for the
number of features checked at once).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from compliance_ai.config import load_config, resolve

EPS = 1e-4
STABLE, SIGNIFICANT = 0.10, 0.25
P_SIGNIFICANT, P_MONITOR = 0.01, 0.05
MIN_BATCH = 30


def _status(psi: float, p_value: float = 0.0, n_tests: int = 1) -> str:
    """Bonferroni: with ~20 features tested at once, each must clear p < alpha / n_tests."""
    if psi >= SIGNIFICANT and p_value < P_SIGNIFICANT / n_tests:
        return "significant_shift"
    if psi >= STABLE and p_value < P_MONITOR / n_tests:
        return "monitor"
    return "stable"


def chi_square_p(expected_shares, counts) -> float:
    """Goodness-of-fit p-value: how likely these counts are if nothing changed."""
    e = np.clip(np.asarray(expected_shares, dtype=float), EPS, None)
    e = e / e.sum()
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    if n == 0 or len(c) < 2:
        return 1.0
    stat = float(np.sum((c - n * e) ** 2 / (n * e)))
    return float(chi2.sf(stat, len(c) - 1))


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_profile(values: pd.Series, bins: int = 10) -> dict:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(values, bins=edges)[0]
    return {"type": "numeric", "edges": [float(x) for x in edges], "shares": (counts / counts.sum()).tolist(),
            "mean": float(values.mean())}


def _categorical_profile(values: pd.Series) -> dict:
    shares = values.astype(str).value_counts(normalize=True)
    return {"type": "categorical", "shares": {k: float(v) for k, v in shares.items()}}


def build_profile(X: pd.DataFrame, scores: np.ndarray | None = None) -> dict:
    profile = {"n_reference": len(X), "features": {}}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            profile["features"][col] = _numeric_profile(X[col])
        else:
            profile["features"][col] = _categorical_profile(X[col])
    if scores is not None:
        profile["prediction"] = _numeric_profile(pd.Series(scores))
    return profile


def _feature_psi(ref: dict, values: pd.Series) -> tuple[float, float, dict]:
    """Return (PSI, chi-square p-value, detail) for one feature."""
    if ref["type"] == "numeric":
        edges = np.array([-np.inf if e == float("-inf") else np.inf if e == float("inf") else e
                          for e in ref["edges"]])
        counts = np.histogram(values.astype(float), bins=edges)[0]
        actual = counts / max(counts.sum(), 1)
        return (psi(ref["shares"], actual), chi_square_p(ref["shares"], counts),
                {"reference_mean": round(ref["mean"], 3), "current_mean": round(float(values.mean()), 3)})
    cats = sorted(set(ref["shares"]) | set(values.astype(str)))
    current = values.astype(str).value_counts(normalize=True)
    expected = [ref["shares"].get(c, 0.0) for c in cats]
    actual = [float(current.get(c, 0.0)) for c in cats]
    counts = [int(values.astype(str).eq(c).sum()) for c in cats]
    unseen = [c for c in cats if c not in ref["shares"]]
    top = max(cats, key=lambda c: abs(float(current.get(c, 0.0)) - ref["shares"].get(c, 0.0)))
    detail = {"largest_change": top,
              "share_reference": round(ref["shares"].get(top, 0.0), 3),
              "share_current": round(float(current.get(top, 0.0)), 3)}
    if unseen:
        detail["unseen_categories"] = unseen
    return psi(expected, actual), chi_square_p(expected, counts), detail


def drift_report(profile: dict, X: pd.DataFrame, scores: np.ndarray | None = None) -> dict:
    if len(X) < MIN_BATCH:
        raise ValueError(f"need at least {MIN_BATCH} applicants for a reliable drift check, got {len(X)}")
    features = []
    checked = [c for c in profile["features"] if c in X.columns]
    n_tests = len(checked) + (1 if scores is not None and "prediction" in profile else 0)
    for col in checked:
        ref = profile["features"][col]
        value, p_value, detail = _feature_psi(ref, X[col])
        features.append({"feature": col, "psi": round(value, 4), "p_value": round(p_value, 4),
                         "status": _status(value, p_value, n_tests), **detail})
    features.sort(key=lambda f: f["psi"], reverse=True)

    report = {"n_current": len(X), "n_reference": profile["n_reference"], "features": features}
    if scores is not None and "prediction" in profile:
        value, p_value, detail = _feature_psi(profile["prediction"], pd.Series(scores))
        report["prediction"] = {"psi": round(value, 4), "p_value": round(p_value, 4),
                                "status": _status(value, p_value, n_tests), **detail}

    shifted = [f["feature"] for f in features if f["status"] == "significant_shift"]
    watch = [f["feature"] for f in features if f["status"] == "monitor"]
    pred_status = report.get("prediction", {}).get("status", "stable")
    if shifted or pred_status == "significant_shift":
        overall = "alert"
    elif watch or pred_status == "monitor":
        overall = "watch"
    else:
        overall = "ok"
    report["overall"] = overall
    if overall == "ok":
        report["summary"] = "No meaningful drift."
    elif overall == "alert":
        report["summary"] = (f"Significant shift in: {', '.join(shifted) or 'model scores'}. "
                             "Review before relying on the model.")
    else:
        report["summary"] = f"Moderate change in: {', '.join(watch) or 'model scores'}. Keep monitoring."
    return report


def profile_path() -> Path:
    return resolve(load_config()["credit"]["model_dir"]) / "reference_profile.json"


def save_profile(profile: dict, path: Path | None = None) -> None:
    (path or profile_path()).write_text(json.dumps(profile))


def check_drift(applicants: pd.DataFrame | list[dict]) -> dict:
    """Score a batch with the trained model and compare it with the training population."""
    from compliance_ai.credit.data import add_features, validate
    from compliance_ai.credit.predict import load_bundle

    path = profile_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m compliance_ai.credit.train` first")
    profile = json.loads(path.read_text())
    df = pd.DataFrame(applicants) if not isinstance(applicants, pd.DataFrame) else applicants.copy()
    validate(df, require_target=False)
    df = add_features(df)
    bundle = load_bundle()
    X = df[bundle["numeric"] + bundle["categorical"]]
    scores = bundle["pipeline"].predict_proba(X)[:, 1]
    return drift_report(profile, X, scores)
