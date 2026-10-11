"""Drift monitoring demo on real applicant groups -> reports/drift_report.md (+ .json).

    python scripts/drift_demo.py

Three batches are compared with the training population:
1. the held-out test set          - same population, should be stable
2. applicants under 25            - a new customer segment (e.g. a student-loan campaign)
3. the largest 25% of loans       - a product change towards bigger loans
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compliance_ai.config import load_config, resolve  # noqa: E402
from compliance_ai.monitoring import check_drift  # noqa: E402


def main() -> None:
    cfg = load_config()
    raw_path = resolve(cfg["credit"]["data_path"])
    import pandas as pd

    raw = pd.read_csv(raw_path)
    y = 1 - raw["credit_risk"]
    _, test = train_test_split(raw, test_size=0.2, stratify=y, random_state=cfg["seed"])
    batches = {
        "Held-out test set (same population)": test,
        "Applicants under 25 (new segment)": raw[raw.age < 25],
        "Largest 25% of loans (product change)": raw[raw.amount >= raw.amount.quantile(0.75)],
    }
    results, lines = {}, ["# Drift monitoring report", "",
                          "PSI per feature against the training population. "
                          "< 0.10 stable · 0.10-0.25 monitor · > 0.25 significant shift. A shift also "
                          "needs a chi-square p-value below 0.01 (Bonferroni-corrected across features), so "
                          "small batches are not flagged for random noise.", ""]
    for name, batch in batches.items():
        rep = check_drift(batch.drop(columns="credit_risk"))
        results[name] = rep
        lines += [f"## {name}", "",
                  f"**{rep['overall'].upper()}** - {rep['summary']} "
                  f"({rep['n_current']} applicants; model-score PSI {rep['prediction']['psi']:.2f}, "
                  f"mean P(default) {rep['prediction']['reference_mean']:.2f} -> "
                  f"{rep['prediction']['current_mean']:.2f})", "",
                  "| Feature | PSI | p-value | Status |", "|---|---|---|---|"]
        lines += [f"| {f['feature']} | {f['psi']:.3f} | {f['p_value']:.2g} | {f['status']} |"
                  for f in rep["features"][:5]]
        lines.append("")
        print(f"{name}: {rep['overall']} - {rep['summary']}")

    out = resolve(cfg["credit"]["report_dir"])
    (out / "drift_report.json").write_text(json.dumps(results, indent=2))
    (out / "drift_report.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'drift_report.md'}")


if __name__ == "__main__":
    main()
