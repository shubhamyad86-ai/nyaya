"""Check a file of loan applicants from the command line: CSV, Excel or PDF, up to 3,000 rows.

    python scripts/check_csv.py path/to/applicants.csv
    python scripts/check_csv.py applicants.pdf --jurisdictions EU          # EU rules only
    python scripts/check_csv.py applicants.xlsx --memos 5                  # + law memos for 5 referrals

Same engine as the upload page (http://127.0.0.1:8000/batch): every applicant gets a credit score
with PASS or FAIL and the main reasons; the batch gets a fairness audit, a drift check and the EU
and India policy checks. A `credit_risk` column (1 = repaid, 0 = defaulted) is optional; if present,
the report also shows accuracy.

Writes reports/<name>_scored.csv, reports/<name>_rows_to_fix.csv (if any) and reports/<name>_report.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from compliance_ai.batch import BatchError, check_file
from compliance_ai.config import resolve

ICON = {"pass": "PASS", "warn": "REVIEW", "fail": "FAIL", "info": "INFO"}


def report_markdown(r: dict) -> str:
    s = r["summary"]
    lines = [f"# Applicant check: {r['file']['name']}", "",
             f"Batch `{r['batch_id']}`, checked {r['checked_at']}. Model **{r['model']['name']}**, "
             f"pass mark {r['model']['pass_mark']} (credit score = 100 x (1 - P(default))).", "",
             f"- Applicants in file: {s['rows']:,}",
             f"- PASS: {s['passed']:,} | FAIL (to human review): {s['failed']:,} | "
             f"invalid rows: {s['invalid']:,}",
             f"- Approval rate: {s['approval_rate']:.1%}" if s["approval_rate"] is not None else "", ""]
    for p in r["policies"]:
        lines += [f"## {p['name']} rules: {ICON[p['overall']]}", "",
                  f"PASS -> {p['actions']['PASS']}. FAIL -> {p['actions']['FAIL']}.", "",
                  "| Result | Check | Finding | Law |", "|---|---|---|---|"]
        lines += [f"| {ICON[c['status']]} | {c['title']} | {c['finding']} | {c['law']} |"
                  for c in p["checks"]]
        for n in p["notices"]:
            lines += [f"> {n}", ""]
    lines += ["## Fairness audit", "", "| Attribute | Group | n | Approval rate | Ratio |",
              "|---|---|---|---|---|"]
    for a in r["fairness"]:
        lines += [f"| {a['label']} | {g['group']} | {g['applicants']} | {g['approval_rate']:.0%} | "
                  f"{g['disparate_impact']} |" for g in a["groups"]]
    lines += ["", "## Drift", ""]
    if r["drift"]:
        lines += [f"**{r['drift']['overall'].upper()}**: {r['drift']['summary']}", ""]
    else:
        lines += [f"Skipped: needs at least {r['drift_min_rows']} valid applicants.", ""]
    if r["accuracy"]:
        a = r["accuracy"]
        lines += ["## Accuracy (outcomes found in the file)", "",
                  f"- Accuracy {a['accuracy']:.1%}, defaulters caught {a['defaulters_caught']:.1%}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and audit a file of loan applicants")
    parser.add_argument("file", help="CSV, Excel (.xlsx) or PDF with the German Credit columns")
    parser.add_argument("--jurisdictions", default="EU,IN", help="EU, IN or EU,IN (default)")
    parser.add_argument("--memos", type=int, default=0,
                        help="write a GDPR review memo for the first N referred applicants")
    args = parser.parse_args()

    path = Path(args.file)
    try:
        r = check_file(path.read_bytes(), path.name, args.jurisdictions.split(","))
    except (BatchError, ValueError) as err:
        print(f"Could not check {path.name}: {err}")
        return 1

    rows = []
    for a in r["applicants"]:
        row = {"row": a["row"], "reference": a["reference"], "credit_score": a["credit_score"],
               "result": a["result"], "p_default": a["p_default"],
               "requires_human_review": a["requires_human_review"],
               "top_reasons": "; ".join(f"{x['label']}={x['value']} ({x['direction']})"
                                        for x in a["reasons"])}
        row.update({f"action_{code}": text for code, text in a["actions"].items()})
        rows.append(row)

    md = report_markdown(r)
    if args.memos:
        from compliance_ai.review import review

        source = pd.read_csv(path) if path.suffix.lower() == ".csv" else None
        referred = [a for a in r["applicants"] if a["result"] == "FAIL"][: args.memos]
        md += "\n## Review memos (GDPR)\n"
        for a in referred:
            if source is None:
                md += "\nMemos are available for CSV files only.\n"
                break
            memo = review(source.iloc[a["row"] - 2].drop(labels=["credit_risk"], errors="ignore").to_dict())
            md += f"\n**Row {a['row']}** ({memo['memo_mode']} memo)\n\n{memo['memo']}\n"

    reports = resolve("reports")
    reports.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(reports / f"{path.stem}_scored.csv", index=False)
    if r["invalid_rows"]:
        pd.DataFrame(r["invalid_rows"]).to_csv(reports / f"{path.stem}_rows_to_fix.csv", index=False)
    (reports / f"{path.stem}_report.md").write_text(md, encoding="utf-8")

    s = r["summary"]
    print(f"{s['rows']:,} applicants: {s['passed']:,} PASS, {s['failed']:,} FAIL (human review), "
          f"{s['invalid']:,} invalid rows")
    for p in r["policies"]:
        n = p["counts"]
        print(f"{p['name']}: {ICON[p['overall']]} "
              f"({n['pass']} passed, {n['warn']} to review, {n['fail']} failed)")
        for c in p["checks"]:
            if c["status"] in ("warn", "fail"):
                print(f"  {ICON[c['status']]}: {c['title']} - {c['finding']}")
    print(f"\nResults: reports/{path.stem}_scored.csv\nReport:  reports/{path.stem}_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
