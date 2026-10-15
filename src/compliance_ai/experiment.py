"""A/B testing toolkit: assignment, sample size, significance tests and guardrail checks.

In production an A/B test of a credit model works like this:

1. Each applicant is assigned to a variant by hashing a stable ID (`assign_variant`). The same
   applicant always lands in the same group, and the split is random with respect to risk.
2. Variant A (champion) or B (challenger) makes the decision; the variant is logged with it.
3. Before the test, `sample_size_per_arm` says how many applicants each arm needs.
4. During the test, `srm_check` catches a broken split (sample ratio mismatch).
5. After the outcome window (for loans: until defaults are observed), compare the arms with
   `two_proportion_test` (rates such as approval or default) and `bootstrap_diff` (means such as
   cost per applicant). Offline, on the same applicants, `mcnemar` compares paired decisions.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import binomtest, chi2, norm


def assign_variant(unit_id: str | int, experiment: str, share_b: float = 0.5) -> str:
    """Deterministic, uniform assignment: hash(experiment + id) -> bucket in [0, 1)."""
    digest = hashlib.sha256(f"{experiment}:{unit_id}".encode()).hexdigest()
    bucket = int(digest[:15], 16) / 16**15
    return "B" if bucket < share_b else "A"


def sample_size_per_arm(p_base: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Units per arm to detect an absolute change `mde` in a rate (two-sided z-test)."""
    p_new = p_base + mde
    if not 0 < p_base < 1 or not 0 < p_new < 1:
        raise ValueError("p_base and p_base + mde must lie between 0 and 1")
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    p_bar = (p_base + p_new) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * math.sqrt(p_base * (1 - p_base) + p_new * (1 - p_new))) ** 2
    return math.ceil(num / mde**2)


def two_proportion_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Pooled two-sided z-test for p_B - p_A, with an unpooled confidence interval."""
    if min(n_a, n_b) == 0:
        raise ValueError("both arms need at least one unit")
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled if se_pooled > 0 else 0.0
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    half = norm.ppf(1 - alpha / 2) * se
    return {"rate_a": p_a, "rate_b": p_b, "diff": p_b - p_a, "z": z,
            "p_value": float(2 * norm.sf(abs(z))),
            "ci_low": p_b - p_a - half, "ci_high": p_b - p_a + half}


def bootstrap_diff(values_a, values_b, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap CI and two-sided p-value for mean(B) - mean(A)."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    rng = np.random.default_rng(seed)
    diffs = (rng.choice(b, (n_boot, len(b))).mean(1) - rng.choice(a, (n_boot, len(a))).mean(1))
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "ci_low": float(np.quantile(diffs, alpha / 2)),
            "ci_high": float(np.quantile(diffs, 1 - alpha / 2)), "p_value": float(min(p, 1.0))}


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar test: do two models make different numbers of mistakes on the SAME units?"""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else binomtest(only_b, n, 0.5).pvalue
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def srm_check(n_a: int, n_b: int, share_b: float = 0.5, alpha: float = 0.001) -> dict:
    """Sample ratio mismatch: a chi-square test that the split matches the planned share."""
    total = n_a + n_b
    exp_a, exp_b = total * (1 - share_b), total * share_b
    stat = (n_a - exp_a) ** 2 / exp_a + (n_b - exp_b) ** 2 / exp_b
    p = float(chi2.sf(stat, 1))
    return {"n_a": n_a, "n_b": n_b, "p_value": p, "mismatch": p < alpha}
