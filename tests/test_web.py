"""The credit check page: form options, PASS/FAIL scoring and the rendered HTML."""
import json
import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from compliance_ai.api import app
from compliance_ai.credit.data import CATEGORIES, DataValidationError, validate
from compliance_ai.web import PRESETS, form_options, render_scorer
from tests.conftest import MODEL, require

client = TestClient(app)


def test_categories_cover_the_training_data(raw_credit):
    for col, allowed in CATEGORIES.items():
        assert set(raw_credit[col]) <= set(allowed), col


def test_unknown_category_is_rejected(applicant):
    bad = pd.DataFrame([{**applicant, "housing": "castle"}])
    with pytest.raises(DataValidationError, match="housing"):
        validate(bad, require_target=False)
    validate(bad, require_target=False, check_categories=False)     # drift monitoring still accepts it


def test_form_lists_every_field_once():
    names = [f["name"] for s in form_options()["sections"] for f in s["fields"]]
    assert sorted(names) == sorted(c for c in pd.read_csv(MODEL.parents[1] / "data/raw/german_credit.csv",
                                                          nrows=1).columns if c != "credit_risk")
    audit = [f["name"] for s in form_options()["sections"] for f in s["fields"] if not f["model_input"]]
    assert set(audit) == {"age", "personal_status_sex", "foreign_worker"}


def test_presets_are_valid_applicants():
    validate(pd.DataFrame(PRESETS.values()), require_target=False)


def test_score_has_pass_fail_consistent_with_threshold(applicant):
    require(MODEL, "run python -m compliance_ai.credit.train")
    for body in [applicant, *PRESETS.values()]:
        r = client.post("/credit/score", json=body).json()
        assert r["credit_score"] == pytest.approx(100 * (1 - r["p_default"]), abs=0.06)
        assert r["pass_mark"] == pytest.approx(100 * (1 - r["threshold"]), abs=0.06)
        assert (r["result"] == "PASS") == (r["p_default"] < r["threshold"]) == (r["decision"] == "approve")
        assert r["requires_human_review"] == (r["result"] == "FAIL")
        assert all(reason["label"] for reason in r["reasons"])


def test_presets_span_pass_and_fail():
    require(MODEL, "run python -m compliance_ai.credit.train")
    results = {name: client.post("/credit/score", json=a).json()["result"] for name, a in PRESETS.items()}
    assert results["Strong"] == "PASS" and results["High risk"] == "FAIL"


def test_audit_only_fields_do_not_change_the_score():
    require(MODEL, "run python -m compliance_ai.credit.train")
    base = PRESETS["Borderline"]
    variants = [base, {**base, "age": 20}, {**base, "age": 70, "foreign_worker": "no",
                                          "personal_status_sex": "female : divorced/separated/married"}]
    assert len({client.post("/credit/score", json=v).json()["credit_score"] for v in variants}) == 1


def test_unknown_category_returns_422(applicant):
    require(MODEL, "run python -m compliance_ai.credit.train")
    assert client.post("/credit/score", json={**applicant, "purpose": "yacht"}).status_code == 422


def test_page_renders_with_embedded_options():
    r = client.get("/app")
    assert r.status_code == 200 and "Credit check" in r.text
    assert "__OPTIONS_JSON__" not in r.text and "__SCORE_URL__" not in r.text
    options = json.loads(re.search(r"const OPTIONS = (.*);\n", r.text).group(1))
    assert options == form_options()
    assert 'const SCORE_URL = "/credit/score";' in r.text
    assert client.get("/credit/options").json() == form_options()


def test_render_escapes_script_end():
    assert "</script>x" not in render_scorer('"</script>x')
