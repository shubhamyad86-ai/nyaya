import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from compliance_ai.monitoring import build_profile, drift_report, psi
from tests.conftest import MODEL, require


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25          # big shift
    assert psi([0.25] * 4, [0.27, 0.23, 0.25, 0.25]) < 0.01


def test_report_flags_shifted_feature_only():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"amount": rng.normal(3000, 800, 2000), "purpose": rng.choice(["car", "tv"], 2000)})
    profile = build_profile(ref, rng.uniform(0, 1, 2000))
    same = pd.DataFrame({"amount": rng.normal(3000, 800, 300), "purpose": rng.choice(["car", "tv"], 300)})
    shifted = same.assign(amount=same.amount * 2, purpose=rng.choice(["car", "boat"], 300))
    assert drift_report(profile, same, rng.uniform(0, 1, 300))["overall"] == "ok"
    rep = drift_report(profile, shifted)
    assert rep["overall"] == "alert"
    by_name = {f["feature"]: f for f in rep["features"]}
    assert by_name["amount"]["status"] == "significant_shift"
    assert "boat" in by_name["purpose"]["unseen_categories"]


def test_random_samples_of_training_data_rarely_alert(raw_credit):
    """PSI alone flagged 70-100% of random 30-60 row samples; the p-value guard fixes that."""
    require(MODEL, "run python -m compliance_ai.credit.train")
    from compliance_ai.monitoring import check_drift

    batch = raw_credit.drop(columns="credit_risk")
    alerts = sum(check_drift(batch.sample(40, random_state=s))["overall"] == "alert" for s in range(20))
    assert alerts <= 2


def test_small_batches_are_refused():
    profile = build_profile(pd.DataFrame({"x": np.arange(100.0)}))
    with pytest.raises(ValueError, match="at least 30"):
        drift_report(profile, pd.DataFrame({"x": np.arange(10.0)}))


def test_drift_endpoint_on_real_segments(raw_credit):
    require(MODEL, "run python -m compliance_ai.credit.train")
    from compliance_ai.api import app

    client = TestClient(app)

    def payload(df):
        rows = df.drop(columns="credit_risk").to_dict("records")
        return {"applicants": [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                               for r in rows]}

    young = client.post("/monitor/drift", json=payload(raw_credit[raw_credit.age < 25]))
    assert young.status_code == 200 and young.json()["overall"] == "alert"
    assert client.post("/monitor/drift", json=payload(raw_credit.head(5))).status_code == 422
