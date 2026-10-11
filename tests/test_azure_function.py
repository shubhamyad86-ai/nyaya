"""Build the Azure Functions zip and call its handlers from the unzipped copy (no Azure needed)."""
from __future__ import annotations

import importlib
import json
import sys
import zipfile

import pytest

from compliance_ai.config import ROOT
from tests.conftest import MODEL, require

func = pytest.importorskip("azure.functions")


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    require(MODEL, "run python -m compliance_ai.credit.train")
    sys.path.insert(0, str(ROOT / "deploy/azure_function"))
    build = importlib.import_module("build")
    out = tmp_path_factory.mktemp("azure")
    zip_path = out / "compliance_ai_function.zip"
    assert build.main(zip_path) == 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert all("\\" not in n for n in names)            # Linux-safe paths
        zf.extractall(out / "pkg")
    sys.path.insert(0, str(out / "pkg"))
    yield out / "pkg", names
    sys.path.remove(str(out / "pkg"))
    sys.path.remove(str(ROOT / "deploy/azure_function"))


def call(handler, method: str, body=None):
    req = func.HttpRequest(method=method, url="/api/x", headers={"Content-Type": "application/json"},
                           body=json.dumps(body).encode() if body is not None else b"")
    resp = handler.build().get_user_function()(req)
    return resp.status_code, json.loads(resp.get_body())


def test_package_contents_and_pins(package):
    pkg, names = package
    for needed in ["function_app.py", "host.json", "config.yaml", "requirements.txt", ".python_version",
                   "compliance_ai/credit/predict.py", "models/credit_model.joblib",
                   "models/reference_profile.json"]:
        assert needed in names
    reqs = (pkg / "requirements.txt").read_text()
    assert "scikit-learn==" in reqs and "pandas==" in reqs     # pinned to the training versions


def test_handlers(package):
    pkg, _ = package
    fa = importlib.import_module("function_app")
    applicant = json.loads((pkg / "sample_applicant.json").read_text())
    batch = json.loads((pkg / "sample_batch.json").read_text())

    assert call(fa.health, "GET")[1]["status"] == "ok"
    status, body = call(fa.score_applicant, "POST", applicant)
    assert status == 200 and 0 <= body["p_default"] <= 1
    assert call(fa.score_applicant, "POST", [applicant])[0] == 400
    assert call(fa.score_applicant, "POST", {**applicant, "age": 5})[0] == 422
    status, body = call(fa.drift, "POST", batch)
    assert status == 200 and body["overall"] in {"ok", "watch", "alert"}
    assert call(fa.drift, "POST", {"applicants": batch["applicants"][:5]})[0] == 422
