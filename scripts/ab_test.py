"""Champion vs challenger A/B test for the credit model.

    python scripts/ab_test.py                                   # challenger: logistic regression
    python scripts/ab_test.py --challenger random_forest
    python scripts/ab_test.py --challenger xgboost --challenger-threshold 0.5   # same model, new threshold

The champion is the production model in models/credit_model.joblib. The challenger is trained on
the same training split, and its threshold is chosen by business cost on out-of-fold predictions,
exactly as in training. Both are then tested on the 200 held-out applicants that neither has seen.

The experiment has three parts:
1. Plan:       how many applicants a live test needs (power analysis).
2. Randomised: each test applicant is hashed into arm A or B, as in production, and the arms
               are compared on approval rate, default rate among approvals and cost per applicant.
3. Paired:     both models decide on ALL test applicants (an offline "shadow" comparison), which
               needs far fewer applicants because every applicant is their own control.

Writes reports/ab_test.md and logs the run to MLflow (experiment "credit-ab-test").
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from compliance_ai.config import load_config, resolve
from compliance_ai.credit.data import feature_columns, load_credit_data
from compliance_ai.credit.metrics import best_threshold
from compliance_ai.credit.models import build_pipeline, candidate_models
from compliance_ai.credit.predict import load_bundle
from compliance_ai.experiment import (
    assign_variant,
    bootstrap_diff,
    mcnemar,
    sample_size_per_arm,
    srm_check,
    two_proportion_test,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "credit-model-ab-1"


def per_applicant_cost(y_default, approved, c_fa, c_fr) -> np.ndarray:
    """5 for approving a defaulter, 1 for rejecting a good applicant, 0 otherwise."""
    y, a = np.asarray(y_default), np.asarray(approved, bool)
    return np.where(a & (y == 1), c_fa, 0.0) + np.where(~a & (y == 0), c_fr, 0.0)


def verdict(p: float, alpha: float = 0.05) -> str:
    return "significant" if p < alpha else "not significant"


def main() -> None:
    parser = argparse.ArgumentParser(description="Champion vs challenger A/B test")
    parser.add_argument("--challenger", default="logistic_regression",
                        choices=list(candidate_models(0)))
    parser.add_argument("--challenger-threshold", type=float, default=None,
                        help="fix the challenger's threshold instead of choosing it by cost")
    parser.add_argument("--mde", type=float, default=0.05,
                        help="smallest change in approval rate worth detecting (absolute)")
    args = parser.parse_args()

    cfg_all = load_config()
    cfg, seed = cfg_all["credit"], cfg_all["seed"]
    c_fa, c_fr = cfg["cost_false_approval"], cfg["cost_false_rejection"]

    # Same split as training, so the test applicants are unseen by both models.
    df = load_credit_data()
    numeric, categorical = feature_columns(df)
    X = df[numeric + categorical]
    y = (1 - df[cfg["target"]]).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    bundle = load_bundle()
    cols = bundle["numeric"] + bundle["categorical"]
    p_a = bundle["pipeline"].predict_proba(X_te[cols])[:, 1]
    thr_a, name_a = bundle["threshold"], bundle["model_name"]

    pipe_b = build_pipeline(clone(candidate_models(seed)[args.challenger]), numeric, categorical)
    if args.challenger_threshold is None:
        cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
        oof = cross_val_predict(pipe_b, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr_b, _ = best_threshold(y_tr, oof, c_fa, c_fr)
    else:
        thr_b = args.challenger_threshold
    pipe_b.fit(X_tr, y_tr)
    p_b = pipe_b.predict_proba(X_te)[:, 1]

    approve_a, approve_b = p_a < thr_a, p_b < thr_b
    label_a, label_b = f"{name_a} @ {thr_a:.2f}", f"{args.challenger} @ {thr_b:.2f}"
    print(f"Champion (A)   {label_a}\nChallenger (B) {label_b}\nTest applicants: {len(y_te)}\n")

    # 1. Plan
    base_rate = float(approve_a.mean())
    n_needed = sample_size_per_arm(base_rate, -args.mde if base_rate - args.mde > 0 else args.mde)

    # 2. Randomised split, as it would run live
    ids = X_te.index.to_numpy()
    arm = np.array([assign_variant(i, EXPERIMENT) for i in ids])
    in_a, in_b = arm == "A", arm == "B"
    srm = srm_check(int(in_a.sum()), int(in_b.sum()))
    approved = np.where(in_a, approve_a, approve_b)
    cost = per_applicant_cost(y_te, approved, c_fa, c_fr)

    appr = two_proportion_test(int(approved[in_a].sum()), int(in_a.sum()),
                               int(approved[in_b].sum()), int(in_b.sum()))
    dflt_a, dflt_b = y_te[in_a & approved], y_te[in_b & approved]
    dflt = (two_proportion_test(int(dflt_a.sum()), len(dflt_a), int(dflt_b.sum()), len(dflt_b))
            if len(dflt_a) and len(dflt_b) else None)
    cost_rand = bootstrap_diff(cost[in_a], cost[in_b], seed=seed)

    # 3. Paired comparison on every test applicant
    cost_a_all = per_applicant_cost(y_te, approve_a, c_fa, c_fr)
    cost_b_all = per_applicant_cost(y_te, approve_b, c_fa, c_fr)
    paired_diff = cost_b_all - cost_a_all
    paired = bootstrap_diff(np.zeros_like(paired_diff), paired_diff, seed=seed)
    correct_a = approve_a == (y_te == 0)
    correct_b = approve_b == (y_te == 0)
    mc = mcnemar(correct_a, correct_b)

    lines = [
        "# A/B test: champion vs challenger credit model", "",
        f"- **Champion (A):** {label_a}", f"- **Challenger (B):** {label_b}",
        f"- **Test applicants:** {len(y_te)} held-out applicants, unseen by both models", "",
        "## 1. Plan: power analysis", "",
        f"Champion approval rate: {base_rate:.1%}. To detect a change of {args.mde:.0%} points "
        f"with 80% power at alpha = 0.05, a live test needs **{n_needed:,} applicants per arm** "
        f"({2 * n_needed:,} in total).", "",
        "## 2. Randomised A/B split (how it runs live)", "",
        f"Applicants were hashed into arms with `assign_variant` (experiment `{EXPERIMENT}`): "
        f"A = {srm['n_a']}, B = {srm['n_b']}. Sample-ratio check p = {srm['p_value']:.3f} "
        f"({'MISMATCH' if srm['mismatch'] else 'split is fine'}).", "",
        "| Metric | A | B | Difference (B - A) | 95% CI | p-value | Result |",
        "|---|---|---|---|---|---|---|",
        f"| Approval rate | {appr['rate_a']:.1%} | {appr['rate_b']:.1%} | {appr['diff']:+.1%} | "
        f"{appr['ci_low']:+.1%} to {appr['ci_high']:+.1%} | {appr['p_value']:.3f} | "
        f"{verdict(appr['p_value'])} |",
    ]
    if dflt:
        lines.append(
            f"| Default rate among approved | {dflt['rate_a']:.1%} | {dflt['rate_b']:.1%} | "
            f"{dflt['diff']:+.1%} | {dflt['ci_low']:+.1%} to {dflt['ci_high']:+.1%} | "
            f"{dflt['p_value']:.3f} | {verdict(dflt['p_value'])} |")
    lines += [
        f"| Cost per applicant | {cost_rand['mean_a']:.3f} | {cost_rand['mean_b']:.3f} | "
        f"{cost_rand['diff']:+.3f} | {cost_rand['ci_low']:+.3f} to {cost_rand['ci_high']:+.3f} | "
        f"{cost_rand['p_value']:.3f} | {verdict(cost_rand['p_value'])} |", "",
        f"Each arm has about {len(y_te) // 2} applicants against the {n_needed:,} the plan asks for, "
        "so this split is underpowered: a non-significant result here means 'not enough data', "
        "not 'no difference'.", "",
        "## 3. Paired comparison (both models on every test applicant)", "",
        f"- Cost per applicant: A {cost_a_all.mean():.3f}, B {cost_b_all.mean():.3f}; difference "
        f"{paired['mean_b']:+.3f} (95% CI {paired['ci_low']:+.3f} to {paired['ci_high']:+.3f}, "
        f"p = {paired['p_value']:.3f}, {verdict(paired['p_value'])}).",
        f"- Decisions that disagree: only A right on {mc['only_a_correct']}, only B right on "
        f"{mc['only_b_correct']} (McNemar p = {mc['p_value']:.3f}, {verdict(mc['p_value'])}).",
        f"- Approval rate: A {approve_a.mean():.1%}, B {approve_b.mean():.1%}.", "",
        "## Decision", "",
    ]
    if paired["p_value"] < 0.05 and paired["mean_b"] < 0:
        lines.append("The challenger costs significantly less. Next step: a live A/B test sized as in "
                     "part 1, with default rates monitored as a guardrail.")
    elif paired["p_value"] < 0.05:
        lines.append("The challenger costs significantly more. Keep the champion.")
    else:
        lines.append("No significant difference in cost. Keep the champion: switching models "
                     "carries risk and needs evidence, and this test does not provide it.")

    report = resolve(cfg["report_dir"]) / "ab_test.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("credit-ab-test")
    with mlflow.start_run(run_name=f"{name_a}-vs-{args.challenger}"):
        mlflow.log_params({"champion": name_a, "champion_threshold": thr_a,
                           "challenger": args.challenger, "challenger_threshold": thr_b,
                           "mde": args.mde, "n_test": len(y_te)})
        mlflow.log_metrics({"n_needed_per_arm": n_needed, "approval_diff": appr["diff"],
                            "approval_p": appr["p_value"], "cost_diff_paired": paired["mean_b"],
                            "cost_p_paired": paired["p_value"], "mcnemar_p": mc["p_value"]})
        mlflow.log_artifact(str(report))

    print(pd.DataFrame({"A (champion)": [approve_a.mean(), cost_a_all.mean()],
                        "B (challenger)": [approve_b.mean(), cost_b_all.mean()]},
                       index=["approval rate", "cost per applicant"]).round(3))
    print(f"\nPaired cost difference {paired['mean_b']:+.3f}, p = {paired['p_value']:.3f}; "
          f"live test needs {n_needed:,} applicants per arm.\nReport: {report}")


if __name__ == "__main__":
    main()
