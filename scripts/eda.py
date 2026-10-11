"""Exploratory data analysis of the credit data -> reports/eda.md + figures.

    python scripts/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compliance_ai.config import load_config, resolve  # noqa: E402
from compliance_ai.credit.data import load_credit_data  # noqa: E402


def default_rate_by(df, col, min_n=30):
    t = df.groupby(col)["default"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "default_rate"})
    return t[t.n >= min_n].sort_values("default_rate", ascending=False)


def main() -> None:
    out = resolve(load_config()["credit"]["report_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = load_credit_data()
    df["default"] = 1 - df["credit_risk"]

    # Figure 1: amount and duration by outcome.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, unit in [(axes[0], "amount", "DM"), (axes[1], "duration", "months")]:
        data = [df.loc[df.default == 0, col], df.loc[df.default == 1, col]]
        ax.boxplot(data, tick_labels=["repaid", "defaulted"], showfliers=False)
        ax.set(title=f"Loan {col} by outcome", ylabel=f"{col} ({unit})")
    fig.tight_layout()
    fig.savefig(out / "eda_amount_duration.png", dpi=150)
    plt.close(fig)

    # Figure 2: default rate by checking-account status.
    status = default_rate_by(df, "status")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(status.index, status.default_rate)
    ax.axvline(df.default.mean(), ls="--", color="grey", label=f"overall {df.default.mean():.0%}")
    ax.set(xlabel="Default rate", title="Default rate by checking-account status")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eda_status.png", dpi=150)
    plt.close(fig)

    numeric = ["duration", "amount", "age", "installment_rate", "amount_per_month", "default"]
    corr = df[numeric].corr()["default"]
    lines = [
        "# Exploratory data analysis - German Credit",
        "",
        f"{len(df):,} applicants, {df.default.mean():.0%} defaulted, no missing values, "
        f"{df.shape[1] - 1} columns after feature engineering.",
        "",
        "## Key findings",
        "",
        f"- Defaulters borrowed more (median {df.loc[df.default == 1, 'amount'].median():,.0f} DM vs "
        f"{df.loc[df.default == 0, 'amount'].median():,.0f} DM) for longer "
        f"(median {df.loc[df.default == 1, 'duration'].median():.0f} vs "
        f"{df.loc[df.default == 0, 'duration'].median():.0f} months).",
        f"- Duration has the strongest linear link with default (r = {corr['duration']:.2f}); "
        f"age is weakly negative (r = {corr['age']:.2f}).",
        f"- Checking-account status is a strong separator: default rates range from "
        f"{status.default_rate.min():.0%} to {status.default_rate.max():.0%}.",
        f"- Applicants under 25 default more often ({df.loc[df.age < 25, 'default'].mean():.0%} vs "
        f"{df.loc[df.age >= 25, 'default'].mean():.0%}), so age-related disparities must be "
        "audited, not assumed away.",
        "",
        "![Amount and duration](eda_amount_duration.png)",
        "",
        "![Status](eda_status.png)",
        "",
        "## Default rate by category (groups with at least 30 applicants)",
        "",
    ]
    for col in ["status", "credit_history", "savings", "purpose", "housing"]:
        t = default_rate_by(df, col)
        lines += [f"### {col}", "", "| value | n | default rate |", "|---|---|---|"]
        lines += [f"| {idx} | {int(r.n)} | {r.default_rate:.0%} |" for idx, r in t.iterrows()]
        lines.append("")
    lines += [
        "## Data caveat",
        "",
        "Category labels come from the decoded CSV mirror. Groemping (2019) showed that the original "
        "Statlog codebook mislabels several categories (e.g. credit history and loan purpose), which "
        "explains counter-intuitive patterns such as 'critical account' lowering risk. The numeric "
        "values are the real survey data; treat category names as approximate.",
    ]
    (out / "eda.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'eda.md'}")


if __name__ == "__main__":
    main()
