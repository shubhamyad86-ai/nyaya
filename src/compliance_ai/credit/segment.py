"""Unsupervised analysis: applicant segments (k-means) and unusual applications (Isolation Forest)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEGMENT_FEATURES = ["duration", "amount", "age", "installment_rate", "amount_per_month", "number_credits"]


def segment(df: pd.DataFrame, seed: int, k_range=range(2, 7)) -> tuple[pd.Series, dict]:
    X = StandardScaler().fit_transform(df[SEGMENT_FEATURES])
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(X)
    labels = pd.Series(labels, index=df.index, name="segment")

    profile = (
        df.assign(segment=labels, default=1 - df["credit_risk"])
        .groupby("segment")
        .agg(n=("default", "size"), default_rate=("default", "mean"),
             median_amount=("amount", "median"), median_duration=("duration", "median"),
             median_age=("age", "median"))
        .round(3)
    )
    return labels, {
        "silhouette_by_k": scores,
        "best_k": best_k,
        "profile": profile.reset_index().to_dict("records"),
    }


def anomalies(df: pd.DataFrame, seed: int, contamination: float = 0.03) -> tuple[pd.Series, dict]:
    X = StandardScaler().fit_transform(df[SEGMENT_FEATURES])
    iso = IsolationForest(contamination=contamination, n_estimators=300, random_state=seed)
    flag = pd.Series(iso.fit_predict(X) == -1, index=df.index, name="anomaly")
    default = 1 - df["credit_risk"]
    return flag, {
        "n_flagged": int(flag.sum()),
        "default_rate_flagged": float(default[flag].mean()),
        "default_rate_not_flagged": float(default[~flag].mean()),
        "median_amount_flagged": float(np.median(df.loc[flag, "amount"])),
    }
