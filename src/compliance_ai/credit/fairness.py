"""Fairness audit on protected attributes the model never sees.

For each group we report the approval rate, the approval rate among applicants
who actually repaid (equal opportunity) and among those who defaulted.
The disparate impact ratio compares each group's approval rate with the most
favoured group; values below 0.8 fail the common "four-fifths" rule of thumb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_metrics(y_default, p_default, groups, threshold: float) -> pd.DataFrame:
    df = pd.DataFrame({
        "y_default": np.asarray(y_default),
        "approved": (np.asarray(p_default) < threshold).astype(int),
        "group": np.asarray(groups),
    })
    rows = []
    for name, g in df.groupby("group"):
        good = g[g.y_default == 0]
        bad = g[g.y_default == 1]
        rows.append({
            "group": name,
            "n": len(g),
            "actual_default_rate": g.y_default.mean(),
            "approval_rate": g.approved.mean(),
            "approval_rate_if_repaid": good.approved.mean() if len(good) else np.nan,
            "approval_rate_if_defaulted": bad.approved.mean() if len(bad) else np.nan,
        })
    out = pd.DataFrame(rows).set_index("group")
    out["disparate_impact_ratio"] = out["approval_rate"] / out["approval_rate"].max()
    return out


def fairness_summary(table: pd.DataFrame) -> dict:
    return {
        "demographic_parity_difference": float(table.approval_rate.max() - table.approval_rate.min()),
        "equal_opportunity_difference": float(
            table.approval_rate_if_repaid.max() - table.approval_rate_if_repaid.min()
        ),
        "min_disparate_impact_ratio": float(table.disparate_impact_ratio.min()),
        "passes_four_fifths_rule": bool(table.disparate_impact_ratio.min() >= 0.8),
    }


def audit(y_default, p_default, frame: pd.DataFrame, attributes: list[str], threshold: float) -> dict:
    results = {}
    for attr in attributes:
        table = group_metrics(y_default, p_default, frame[attr].values, threshold)
        results[attr] = {"groups": table.round(4).reset_index().to_dict("records"),
                         **fairness_summary(table)}
    return results
