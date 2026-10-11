"""Business-cost metrics and decision-threshold selection.

Positive class = default (bad credit). A model "rejects" an applicant when
P(default) >= threshold. Costs follow the Statlog cost matrix:
approving a future defaulter costs 5, rejecting a good applicant costs 1.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_cost(y_default, p_default, threshold: float, c_fa: float, c_fr: float) -> float:
    """Average cost per applicant at a threshold."""
    y = np.asarray(y_default)
    reject = np.asarray(p_default) >= threshold
    false_approvals = np.sum((~reject) & (y == 1))   # defaulter approved
    false_rejections = np.sum(reject & (y == 0))     # good applicant rejected
    return float((c_fa * false_approvals + c_fr * false_rejections) / len(y))


def best_threshold(y_default, p_default, c_fa: float, c_fr: float) -> tuple[float, float]:
    """Grid-search the threshold that minimises expected cost. Returns (threshold, cost)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [expected_cost(y_default, p_default, t, c_fa, c_fr) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def classification_report(y_default, p_default, threshold: float, c_fa: float, c_fr: float) -> dict:
    y = np.asarray(y_default)
    pred = (np.asarray(p_default) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p_default),
        "pr_auc": average_precision_score(y, p_default),
        "brier": brier_score_loss(y, p_default),
        "threshold": threshold,
        "expected_cost": expected_cost(y, p_default, threshold, c_fa, c_fr),
        "recall_defaulters": recall_score(y, pred, zero_division=0),
        "precision_defaulters": precision_score(y, pred, zero_division=0),
        "f1_defaulters": f1_score(y, pred, zero_division=0),
        "approval_rate": float(1 - pred.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
