"""Check a CSV of new loan applicants: score, explain, audit fairness, check drift and flag
the decisions that the GDPR says need a human.

    python scripts/check_csv.py path/to/applicants.csv
    python scripts/check_csv.py path/to/applicants.csv --memos 5   # also write law memos for 5 referrals

The CSV needs the 20 German Credit columns (see data/raw/german_credit.csv). A `credit_risk`
column is optional; if present (1 = repaid, 0 = defaulted) the report also shows accuracy.

Writes reports/<name>_scored.csv (one row per applicant) and reports/<name>_report.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from compliance_ai.config import resolve
from compliance_ai.credit.data import DataValidationError, add_features
from compliance_ai.credit.fairness import fairness_summary, group_metrics
from compliance_ai.credit.predict import score
from compliance_ai.monitoring import MIN_BATCH, check_drift

PROTECTED = {"age_group": "Age (under 25 vs 25+)", "sex": "Sex", "foreign_worker": "Foreign worker"}


def _fmt_reasons(reasons: list[dict]) -> str:
    return "; ".join(f"{r['feature']}={r['value']} ({r['direction']})" for r in reasons)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and audit a CSV of loan applicants")
    parser.add_argument("csv", help="CSV with the German Credit columns")
    parser.add_argument("--memos", type=int, default=0,
                        help="write a GDPR review memo for the first N referred applicants")
    args = parser.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} applicants from {path.name}")

    try:
        results = score(df)
    except DataValidationError as err:
        print(f"The CSV does not match the expected format: {err}")
        return 1

    out = df.copy()
    out["p_default"] = [r["p_default"] for r in results]
    out["decision"] = [r["decision"] for r in results]
    out["requires_human_review"] = out["decision"] != "approve"
    out["top_reasons"] = [_fmt_reasons(r["reasons"]) for r in results]
    threshold = results[0]["threshold"]
    model = results[0]["model"]

    lines = [f"# Applicant check: {path.name}", "",
             f"Model: **{model}**, threshold {threshold:.3f} on P(default). {len(out)} applicants.", ""]

    n_ref = int(out["requires_human_review"].sum())
    lines += ["## Decisions", "",
              f"- Approved: {len(out) - n_ref} ({(len(out) - n_ref) / len(out):.0%})",
              f"- Referred or rejected: {n_ref} ({n_ref / len(out):.0%})", "",
              "Under GDPR Article 22, every referred or rejected applicant is flagged for a human "
              "reviewer (`requires_human_review = True`), and Articles 13-15 require meaningful "
              "information about the logic involved, which the `top_reasons` column provides.", ""]

    if "credit_risk" in out.columns:
        y_default = 1 - out["credit_risk"]
        predicted = out["requires_human_review"].astype(int)
        caught = (predicted[y_default == 1] == 1).mean() if (y_default == 1).any() else float("nan")
        lines += ["## Accuracy (labels found in the CSV)", "",
                  f"- Accuracy: {(predicted == y_default).mean():.1%}",
                  f"- Defaulters caught: {caught:.1%}", ""]
    else:
        y_default = pd.Series(0, index=out.index)

    feats = add_features(df)
    lines += ["## Fairness audit", "",
              "Approval rate per group and the disparate impact ratio (below 0.8 fails the "
              "four-fifths rule). The model never sees these attributes directly.", "",
              "| Attribute | Group | n | Approval rate | Disparate impact |", "|---|---|---|---|---|"]
    verdicts = []
    for col, label in PROTECTED.items():
        if col not in feats.columns or feats[col].nunique() < 2:
            continue
        table = group_metrics(y_default, out["p_default"], feats[col], threshold)
        for group, row in table.iterrows():
            lines.append(f"| {label} | {group} | {int(row.n)} | {row.approval_rate:.0%} | "
                         f"{row.disparate_impact_ratio:.2f} |")
        s = fairness_summary(table)
        small = int(table.n.min()) < 20
        verdicts.append(f"- {label}: {'passes' if s['passes_four_fifths_rule'] else 'FAILS'} "
                        f"the four-fifths rule (lowest ratio {s['min_disparate_impact_ratio']:.2f})"
                        + (f" - caution, smallest group has only {int(table.n.min())} applicants"
                           if small else ""))
    lines += ["", *verdicts, ""]

    lines += ["## Drift against the training data", ""]
    if len(out) >= MIN_BATCH:
        drift = check_drift(df)
        lines += [f"**{drift['overall'].upper()}**: {drift['summary']}", "",
                  "| Feature | PSI | Status |", "|---|---|---|"]
        lines += [f"| {f['feature']} | {f['psi']:.3f} | {f['status']} |" for f in drift["features"][:8]]
        lines.append("")
    else:
        lines += [f"Skipped: the drift check needs at least {MIN_BATCH} applicants.", ""]

    if args.memos:
        from compliance_ai.review import review
        lines += ["## Review memos (GDPR)", ""]
        referred = out[out["requires_human_review"]].head(args.memos)
        for i, (idx, _) in enumerate(referred.iterrows(), 1):
            memo = review(df.loc[idx].to_dict())
            lines += [f"**Applicant row {idx}** ({memo['memo_mode']} memo)", "", memo["memo"], ""]
            print(f"Memo {i}/{len(referred)} written")

    reports = resolve("reports")
    reports.mkdir(exist_ok=True)
    scored = reports / f"{path.stem}_scored.csv"
    report = reports / f"{path.stem}_report.md"
    out.to_csv(scored, index=False)
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nApproved {len(out) - n_ref}, referred {n_ref} (flagged for human review under GDPR Art. 22)")
    print(*verdicts, sep="\n")
    print(f"\nScored applicants: {scored}\nReport:            {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
