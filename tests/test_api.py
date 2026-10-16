import pytest
from fastapi.testclient import TestClient

from compliance_ai.api import app
from tests.conftest import MODEL, require

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["api"] == "ok"


def test_score_endpoint(applicant):
    require(MODEL, "run python -m compliance_ai.credit.train")
    r = client.post("/credit/score", json=applicant)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["p_default"] <= 1
    assert body["decision"] in {"approve", "refer_or_reject"}
    assert len(body["reasons"]) == 4


def test_score_rejects_invalid_input(applicant):
    r = client.post("/credit/score", json={**applicant, "age": 7})
    assert r.status_code == 422


def test_score_is_deterministic(applicant):
    require(MODEL, "run python -m compliance_ai.credit.train")
    a = client.post("/credit/score", json=applicant).json()["p_default"]
    b = client.post("/credit/score", json=applicant).json()["p_default"]
    assert a == pytest.approx(b)


def test_review_routes_risky_applicant_to_human(applicant, tfidf_index, monkeypatch):
    require(MODEL, "run python -m compliance_ai.credit.train")
    from compliance_ai.rag.assistant import ComplianceAssistant
    from compliance_ai.review import review

    assistant = ComplianceAssistant(llm_provider="none", index_dir=tfidf_index)
    result = review(applicant, assistant)
    assert result["model"]["decision"] == "refer_or_reject"
    assert result["requires_human_review"] is True
    assert result["memo_mode"] == "template" and result["regulation_sources"]


def test_root_redirects_to_credit_check_page():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/app"


def test_sentiment_endpoint():
    from compliance_ai.config import ROOT

    require(ROOT / "models/sentiment/tfidf_baseline/model.joblib",
            "run python -m compliance_ai.nlp.train --method tfidf_baseline")
    r = client.post("/nlp/sentiment", json={"sentences": [
        "Net sales increased by 25 % and operating profit doubled.",
        "Operating profit fell to EUR 3.2 mn from EUR 7.5 mn.",
    ]})
    assert r.status_code == 200
    labels = [x["label"] for x in r.json()]
    assert labels == ["positive", "negative"]
