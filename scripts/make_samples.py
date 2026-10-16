"""Make sample upload files for the batch check page: a 3,000-row CSV, an Excel file, a PDF table
and a PDF of application forms.

    python scripts/make_samples.py          # writes data/samples/

The rows are drawn from the real German Credit data with small random changes to the amount and
duration, so they look realistic without being copies. A few rows get deliberate errors so the
"rows to correct" report has something to show.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/samples"


def make_applicants(n: int, seed: int = 7, errors: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.read_csv(ROOT / "data/raw/german_credit.csv").drop(columns="credit_risk")
    df = base.sample(n, replace=n > len(base), random_state=seed).reset_index(drop=True)
    df["amount"] = (df["amount"] * rng.uniform(0.85, 1.15, n)).round(-1).clip(250, 20000).astype(int)
    df["duration"] = (df["duration"] + rng.choice([-3, 0, 0, 3], n)).clip(4, 72)
    df.insert(0, "applicant_id", [f"APP-{i:05d}" for i in range(1, n + 1)])
    df = df.astype(object)
    mistakes = [("amount", -100), ("purpose", "yacht"), ("age", ""), ("duration", 500)] * 10
    for i, (col, bad) in zip(rng.choice(n, errors, replace=False), mistakes[:errors], strict=True):
        df.at[i, col] = bad
    return df


def pdf_table(df: pd.DataFrame, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

    doc = SimpleDocTemplate(str(path), pagesize=landscape(A3), leftMargin=14, rightMargin=14,
                            topMargin=18, bottomMargin=18, title="Loan applications")
    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 5),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF8"))]))
    doc.build([table])


def pdf_forms(df: pd.DataFrame, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for _, row in df.iterrows():
        y = 800
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Loan application")
        c.setFont("Helvetica", 10)
        for col, value in row.items():
            y -= 18
            c.drawString(60, y, f"{col}: {value}")
        c.showPage()
    c.save()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    big = make_applicants(3000, errors=12)
    big.to_csv(OUT / "applicants_3000.csv", index=False)
    small = make_applicants(300, seed=11, errors=4)
    small.to_excel(OUT / "applicants_300.xlsx", index=False)
    pdf_table(make_applicants(120, seed=21, errors=2), OUT / "applicants_120_table.pdf")
    pdf_forms(make_applicants(10, seed=31), OUT / "applications_10_forms.pdf")
    for p in sorted(OUT.iterdir()):
        print(f"{p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
