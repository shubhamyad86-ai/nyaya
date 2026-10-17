"""Amounts in EUR, INR or DM: the model works in DM, people use their own currency."""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from compliance_ai import currency
from compliance_ai.api import app
from compliance_ai.batch import check_file, validate_rows
from compliance_ai.web import PRESETS, form_options
from tests.conftest import MODEL, require

client = TestClient(app)


def test_fixed_euro_rate_and_round_trip():
    assert currency.dm_per_unit("EUR") == pytest.approx(1.95583)
    assert currency.dm_per_unit("DM") == 1
    for code in ("EUR", "INR", "DM"):
        assert currency.to_dm(currency.from_dm(1374, code), code) == pytest.approx(1374)
    assert currency.money(1374, "EUR") == "€703" and currency.money(1374, "DM") == "1,374 DM"
    assert currency.money(18_424, "INR") == "₹10,01,068"           # Indian lakh grouping
    assert currency.money(1374, "INR") == "₹74,656" and currency.money(1, "INR") == "₹54"


def test_codebook_labels_are_rewritten_not_the_values():
    assert currency.relabel("... < 100 DM", "EUR") == "... < €51"
    assert currency.relabel("0 <= ... < 200 DM", "INR").startswith("0 <= ... < ₹10,")
    assert currency.relabel("no checking account", "EUR") == "no checking account"
    assert currency.relabel("... < 100 DM", "DM") == "... < 100 DM"
    with pytest.raises(ValueError):
        currency.check("USD")


def test_page_gets_currencies_and_labels():
    opts = form_options()
    assert opts["default_currency"] in {"EUR", "INR", "DM"}
    assert {c["code"] for c in opts["currencies"]} == {"EUR", "INR", "DM"}
    assert opts["money_labels"]["EUR"]["... < 100 DM"] == "... < €51"
    assert "inflation" in currency.info("EUR")["note"]
    page = client.get("/app").text
    assert 'id="currency"' in page and "money_labels" in page


def _file(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_euro_file_scores_like_the_dm_file():
    require(MODEL, "run python -m compliance_ai.credit.train")
    dm_rows = list(PRESETS.values())
    eur_rows = [{**r, "amount": round(r["amount"] / 1.95583, 2)} for r in dm_rows]
    inr_rows = [{**r, "amount": round(currency.from_dm(r["amount"], "INR"))} for r in dm_rows]
    dm = check_file(_file(dm_rows), "dm.csv", ["EU"], write_audit_log=False)
    eur = check_file(_file(eur_rows), "eur.csv", ["EU"], write_audit_log=False, money="EUR")
    inr = check_file(_file(inr_rows), "inr.csv", ["IN"], write_audit_log=False, money="INR")
    scores = [a["credit_score"] for a in dm["applicants"]]
    for other in (eur, inr):
        assert [a["credit_score"] for a in other["applicants"]] == scores
    assert eur["currency"]["code"] == "EUR" and dm["currency"]["code"] == "DM"
    shown = [x["value"] for a in eur["applicants"] for x in a["reasons"] if x["label"] == "Loan amount"]
    assert shown and all(str(v).startswith("€") for v in shown)


def test_amount_limits_are_reported_in_the_files_currency():
    row = {**PRESETS["Strong"], "amount": 80_000}       # 80,000 € = 156,466 DM, above the 100,000 DM limit
    _, invalid = validate_rows(pd.DataFrame([row]).astype(str), "EUR")
    assert "amount must be €1-€51,129" in invalid.loc[0, "issues"]
    valid, _ = validate_rows(pd.DataFrame([{**row, "amount": "702.50"}]).astype(str), "EUR")
    assert valid.loc[0, "amount"] == 1374                # decimals are fine in euro; whole DM for the model


def test_api_takes_the_currency_parameter():
    require(MODEL, "run python -m compliance_ai.credit.train")
    body = _file([{**PRESETS["Strong"], "amount": 703}])
    r = client.post("/batch/check?currency=EUR&jurisdictions=EU&filename=a.csv", content=body,
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200 and r.json()["currency"]["code"] == "EUR"
    bad = client.post("/batch/check?currency=USD&filename=a.csv", content=body,
                      headers={"Content-Type": "application/octet-stream"})
    assert bad.status_code == 422
