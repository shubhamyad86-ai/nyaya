"""Shared fixtures. Tests use the real data in data/ (run scripts/download_data.py first)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from compliance_ai.config import ROOT

CREDIT_CSV = ROOT / "data/raw/german_credit.csv"
GDPR_HTML = ROOT / "data/regulations/gdpr_regulation_2016_679.html"
MODEL = ROOT / "models/credit_model.joblib"


def require(path: Path, hint: str) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} missing - {hint}")


@pytest.fixture(scope="session")
def raw_credit() -> pd.DataFrame:
    require(CREDIT_CSV, "run scripts/download_data.py")
    return pd.read_csv(CREDIT_CSV)


@pytest.fixture(scope="session")
def applicant(raw_credit) -> dict:
    row = raw_credit.drop(columns="credit_risk").iloc[1].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


@pytest.fixture(scope="session")
def tfidf_index(tmp_path_factory) -> Path:
    require(GDPR_HTML, "run scripts/download_data.py")
    from compliance_ai.rag.store import build_index

    index_dir = tmp_path_factory.mktemp("rag_index")
    build_index(backend="tfidf", index_dir=index_dir)
    return index_dir
