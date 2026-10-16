"""Batch check: reading CSV/Excel/PDF, row validation, EU and India policy checks, audit log, API."""
import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from compliance_ai.api import app
from compliance_ai.batch import MAX_ROWS, BatchError, check_file, read_table, template_csv, validate_rows
from compliance_ai.policies import evaluate, load_packs
from compliance_ai.web import sample_file
from tests.conftest import MODEL, require

client = TestClient(app)


@pytest.fixture(scope="module")
def applicants(raw_credit) -> pd.DataFrame:
    df = raw_credit.drop(columns="credit_risk").head(60).copy()
    df.insert(0, "applicant_id", [f"A{i}" for i in range(60)])
    return df


@pytest.fixture(autouse=True)
def audit_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("NYAYA_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    return tmp_path / "audit.jsonl"


# ------------------------------------------------------------------ reading
def test_reads_csv_with_semicolons_and_any_column_case(applicants):
    text = applicants.rename(columns=str.upper).to_csv(index=False, sep=";")
    df = read_table(text.encode(), "export.csv")
    assert len(df) == 60 and "status" in df.columns and "applicant_id" in df.columns


def test_reads_excel(applicants):
    buf = io.BytesIO()
    applicants.to_excel(buf, index=False)
    assert len(read_table(buf.getvalue(), "a.xlsx")) == 60


def test_reads_pdf_table_and_pdf_forms(applicants, tmp_path):
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    import sys

    from compliance_ai.config import ROOT

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_samples

    make_samples.pdf_table(applicants.head(25), tmp_path / "t.pdf")
    make_samples.pdf_forms(applicants.head(3), tmp_path / "f.pdf")
    table = read_table((tmp_path / "t.pdf").read_bytes(), "t.pdf")
    forms = read_table((tmp_path / "f.pdf").read_bytes(), "f.pdf")
    assert len(table) == 25 and table.loc[0, "applicant_id"] == "A0"
    assert len(forms) == 3 and forms.loc[2, "purpose"] == applicants.iloc[2]["purpose"]


@pytest.mark.parametrize("data, name, message", [
    (b"", "a.csv", "empty"),
    (b"a,b\n1,2\n", "a.csv", "Missing column"),
    (b"hello", "a.docx", "CSV, Excel"),
    (b"%PDF-1.4 not really", "a.pdf", "PDF"),
])
def test_bad_files_are_explained(data, name, message):
    with pytest.raises(BatchError, match=message):
        read_table(data, name)


def test_row_limit(applicants):
    big = pd.concat([applicants] * (MAX_ROWS // 60 + 1))
    with pytest.raises(BatchError, match="limit is 3,000"):
        read_table(big.to_csv(index=False).encode(), "big.csv")


# ------------------------------------------------------------------ validation
def test_bad_rows_are_reported_not_fatal(applicants):
    df = applicants.astype(str).copy()
    df.loc[0, "amount"] = "-5"
    df.loc[1, "purpose"] = "yacht"
    df.loc[2, "age"] = ""
    df.loc[3, "duration"] = "twelve"
    df.loc[4, "housing"] = "OWN"                   # different capitalisation is fine
    valid, invalid = validate_rows(read_table(df.to_csv(index=False).encode(), "x.csv"))
    assert len(valid) == 56 and len(invalid) == 4
    assert list(invalid["row"]) == [2, 3, 4, 5]    # line numbers in the file
    assert "amount must be" in invalid.loc[0, "issues"] and "purpose: unknown" in invalid.loc[1, "issues"]
    assert valid.loc[0, "housing"] == "own" and valid["amount"].dtype.kind == "i"


# ------------------------------------------------------------------ whole batch
def test_check_file_scores_and_applies_both_rule_sets(applicants, audit_to_tmp):
    require(MODEL, "run python -m compliance_ai.credit.train")
    r = check_file(applicants.to_csv(index=False).encode(), "a.csv", ["EU", "india"])
    s = r["summary"]
    assert s["scored"] == 60 and s["passed"] + s["failed"] == 60 and s["human_review"] == s["failed"]
    assert {p["code"] for p in r["policies"]} == {"EU", "IN"}
    a = r["applicants"][0]
    assert a["reference"] == "A0" and set(a["actions"]) == {"EU", "IN"} and len(a["reasons"]) == 4
    eu = next(p for p in r["policies"] if p["code"] == "EU")
    status = {c["id"]: c["status"] for c in eu["checks"]}
    assert status["human_review"] == status["explanation"] == status["sensitive_inputs"] == "pass"
    assert status["logging"] == "pass" and r["drift"] is not None
    # The audit log has the decisions but no personal data from the file.
    record = json.loads(audit_to_tmp.read_text().splitlines()[-1])
    assert record["batch_id"] == r["batch_id"] and len(record["decisions"]) == 60
    assert "A0" not in audit_to_tmp.read_text() and "skilled" not in audit_to_tmp.read_text()


def test_audit_write_failure_is_reported(applicants, monkeypatch, tmp_path):
    require(MODEL, "run python -m compliance_ai.credit.train")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("NYAYA_AUDIT_LOG", str(blocker / "audit.jsonl"))   # parent is a file
    r = check_file(applicants.to_csv(index=False).encode(), "a.csv", ["EU"])
    eu = r["policies"][0]
    assert r["audit_log"]["written"] is False
    assert next(c for c in eu["checks"] if c["id"] == "logging")["status"] == "warn"
    assert eu["counts"]["warn"] >= 1 and eu["overall"] in {"warn", "fail"}


def test_policy_checks_catch_problems():
    batch = {"applicants": [{"result": "FAIL", "requires_human_review": False, "reasons": []}],
             "model": {"inputs": ["amount", "personal_status_sex"]}, "fairness": [],
             "summary": {"rows": 10, "invalid": 5}, "drift": {"overall": "alert", "summary": "x"},
             "drift_min_rows": 30, "batch_id": "b", "file": {"sha256": "0" * 64}}
    eu = evaluate(load_packs(["EU"])[0], batch)
    status = {c["id"]: c["status"] for c in eu["checks"]}
    assert status["human_review"] == status["explanation"] == status["sensitive_inputs"] == "fail"
    assert status["data_quality"] == "fail" and status["monitoring"] == "fail" and eu["overall"] == "fail"
    with pytest.raises(ValueError, match="Unknown jurisdiction"):
        load_packs(["US"])


# ------------------------------------------------------------------ API and page
def test_api_upload_multipart_and_raw(applicants):
    require(MODEL, "run python -m compliance_ai.credit.train")
    csv = applicants.to_csv(index=False).encode()
    r1 = client.post("/batch/check?jurisdictions=EU", files={"file": ("a.csv", csv, "text/csv")})
    r2 = client.post("/batch/check?jurisdictions=IN&filename=a.csv", content=csv,
                     headers={"Content-Type": "application/octet-stream"})
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["summary"] == r2.json()["summary"]
    assert [p["code"] for p in r2.json()["policies"]] == ["IN"]
    bad = client.post("/batch/check?filename=a.csv", content=b"x,y\n1,2\n")
    assert bad.status_code == 422 and "Missing column" in bad.json()["detail"]


def test_template_and_sample_are_valid_inputs():
    require(MODEL, "run python -m compliance_ai.credit.train")
    t = check_file(template_csv().encode(), "template.csv", ["EU"], write_audit_log=False)
    assert t["summary"]["scored"] == 1
    s = check_file(sample_file(), "sample.csv", ["EU", "IN"], write_audit_log=False)
    assert s["summary"]["rows"] == 203 and s["summary"]["invalid"] == 3


def test_batch_page_renders():
    r = client.get("/batch")
    assert r.status_code == 200 and "Batch check" in r.text
    assert 'const CHECK_URL = "/batch/check";' in r.text and "__" not in r.text.split("<script>")[0][-200:]
    assert client.get("/batch/template.csv").text.startswith("applicant_id,status")
    assert client.get("/batch/sample.csv").status_code == 200
