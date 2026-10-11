"""Download the real datasets used by this project.

Sources
-------
1. Statlog German Credit data (UCI Machine Learning Repository, Hofmann 1994),
   1,000 real loan applicants, in the decoded CSV mirror by selva86/datasets.
2. GDPR - Regulation (EU) 2016/679, official English text as published on
   EUR-Lex, mirrored in the GDPRtEXT research project (coolharsh55/GDPRtEXT).
3. EU AI Act - Regulation (EU) 2024/1689, official Official Journal HTML from EUR-Lex,
   mirrored in jsiddamsetty/aiactbrowser (pinned commit). Credit scoring is a
   high-risk AI use under Annex III, point 5(b).
4. Optional (--extra): the AI Act PDF straight from EUR-Lex (not used by default).

Usage:  python scripts/download_data.py [--extra]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE = {
    "data/raw/german_credit.csv":
        "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv",
    "data/regulations/gdpr_regulation_2016_679.html":
        "https://raw.githubusercontent.com/coolharsh55/GDPRtEXT/master/scripts/GDPR_en.html",
    "data/regulations/eu_ai_act_2024_1689.html":
        "https://raw.githubusercontent.com/jsiddamsetty/aiactbrowser/"
        "0f14b821e20a643624b6267dbd63fb4b9bee6b57/build/source-oj.html",
}

EXTRA = {
    "data/extra/eu_ai_act_2024_1689.pdf":
        "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
}


def download(rel_path: str, url: str) -> None:
    target = ROOT / rel_path
    if target.exists() and target.stat().st_size > 0:
        print(f"  exists  {rel_path}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "compliance-ai/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        target.write_bytes(resp.read())
    print(f"  saved   {rel_path} ({target.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extra", action="store_true", help="also fetch the EU AI Act PDF from EUR-Lex")
    args = parser.parse_args()

    files = dict(CORE)
    if args.extra:
        files.update(EXTRA)

    failed = 0
    for rel_path, url in files.items():
        try:
            download(rel_path, url)
        except Exception as exc:  # network errors should not hide the others
            failed += 1
            print(f"  FAILED  {rel_path}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
