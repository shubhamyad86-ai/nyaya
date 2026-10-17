"""Put together the folder to upload to a Hugging Face Space.

    python deploy/hf_space/build.py        ->  dist/hf_space/

Upload the *contents* of dist/hf_space/ to a new Docker Space (see docs/HF_SPACE.md).
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = ROOT / "dist" / "hf_space"

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
shutil.copy(HERE / "Dockerfile", OUT / "Dockerfile")
shutil.copy(HERE / "requirements.txt", OUT / "requirements.txt")
shutil.copy(HERE / "SPACE_README.md", OUT / "README.md")
shutil.copy(ROOT / "config.yaml", OUT / "config.yaml")
shutil.copytree(ROOT / "src", OUT / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
(OUT / "data" / "raw").mkdir(parents=True)
shutil.copy(ROOT / "data" / "raw" / "german_credit.csv", OUT / "data" / "raw")

files = [p for p in OUT.rglob("*") if p.is_file()]
print(f"{OUT}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
