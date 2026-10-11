"""SHAP explanations: global feature importance and per-applicant reason codes.

GDPR Article 22 and the EU AI Act both expect meaningful information about
the logic behind automated credit decisions. Reason codes turn the model's
SHAP values into the top factors that pushed one applicant's risk up or down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _explainer(pipeline: Pipeline, background: np.ndarray):
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)


def _positive_class(values) -> np.ndarray:
    """Normalise SHAP output to an (n_samples, n_features) array for the default class."""
    vals = values.values if hasattr(values, "values") else values
    vals = np.asarray(vals)
    if vals.ndim == 3:          # (n, features, classes)
        vals = vals[:, :, 1]
    return vals


def _group_one_hot(shap_df: pd.DataFrame, original_columns: list[str]) -> pd.DataFrame:
    """Sum one-hot SHAP columns back onto their source feature (e.g. purpose_car -> purpose)."""
    mapping = {}
    for col in shap_df.columns:
        matches = [c for c in original_columns if col == c or col.startswith(c + "_")]
        mapping[col] = max(matches, key=len) if matches else col
    return shap_df.T.groupby(mapping).sum().T


def _xgboost_contributions(model, Xt) -> np.ndarray | None:
    """Exact TreeSHAP values computed by XGBoost itself (pred_contribs).

    Used instead of shap.TreeExplainer for XGBoost: XGBoost 3.1+ stores base_score as a
    vector ("[5E-1]"), which older SHAP releases cannot parse ("could not convert string to
    float"). XGBoost's own implementation always matches the installed XGBoost version.
    """
    if type(model).__module__.split(".")[0] != "xgboost":
        return None
    import xgboost as xgb

    contribs = model.get_booster().predict(xgb.DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True)
    return np.asarray(contribs)[:, :-1]          # last column is the bias term


def shap_values(pipeline: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (log-odds of default) per original feature, one row per applicant."""
    prep = pipeline.named_steps["prep"]
    Xt = prep.transform(X)
    vals = _xgboost_contributions(pipeline.named_steps["model"], Xt)
    if vals is None:
        explainer = _explainer(pipeline, prep.transform(background))
        vals = _positive_class(explainer(Xt) if not isinstance(explainer, shap.TreeExplainer)
                               else explainer.shap_values(Xt))
    raw = pd.DataFrame(vals, columns=prep.get_feature_names_out(), index=X.index)
    return _group_one_hot(raw, list(X.columns))


def global_importance(shap_df: pd.DataFrame) -> pd.Series:
    return shap_df.abs().mean().sort_values(ascending=False)


def reason_codes(shap_row: pd.Series, applicant: pd.Series, top_n: int = 4) -> list[dict]:
    """Top factors for one applicant, strongest first. Positive impact = raises default risk."""
    top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)[:top_n]
    return [
        {
            "feature": feat,
            "value": applicant.get(feat).item() if hasattr(applicant.get(feat), "item")
            else applicant.get(feat),
            "impact": round(float(v), 4),
            "direction": "increases risk" if v > 0 else "decreases risk",
        }
        for feat, v in top.items()
    ]
