"""The Databricks job's shared rules must match the main package (same data rules, PSI, score)."""
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from compliance_ai.config import ROOT
from compliance_ai.credit import data as pkg_data
from compliance_ai.monitoring import psi as pkg_psi

sys.path.insert(0, str(ROOT / "databricks/notebooks"))
import common  # noqa: E402


def test_rules_match_the_package():
    assert common.CATEGORIES == pkg_data.CATEGORIES
    assert common.RANGES == pkg_data.NUMERIC_RANGES


def test_features_match_the_package(raw_credit):
    ours = common.add_features(raw_credit)
    theirs = pkg_data.add_features(raw_credit)
    for col in ["sex", "age_group", "amount_per_month", "has_checking_account", "no_savings",
                "delayed_before"]:
        pd.testing.assert_series_equal(ours[col], theirs[col], check_dtype=False)
    numeric, categorical = common.feature_columns(ours.assign(default=0))
    assert not set(numeric + categorical) & set(common.AUDIT_ONLY)
    assert "credit_risk" not in numeric + categorical and "default" not in numeric + categorical


def test_psi_matches_the_package():
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
    assert common.psi(a, b) == pytest.approx(pkg_psi(a, b))


def test_cost_threshold_and_score_fields():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.2, 0.1, 0.05])
    assert common.expected_cost(y, p, 0.5) == 0                 # both defaulters rejected
    assert common.expected_cost(y, p, 0.7) == pytest.approx(5 / 6)   # one defaulter approved costs 5
    t, cost = common.best_threshold(y, p)
    assert cost == 0 and 0.4 < t <= 0.6
    f = common.score_fields(p, 0.5)
    assert list(f["result"]) == ["FAIL", "FAIL", "PASS", "PASS", "PASS", "PASS"]
    assert f["credit_score"].tolist() == [10.0, 40.0, 60.0, 80.0, 90.0, 95.0]
    assert (f["requires_human_review"] == (f["result"] == "FAIL")).all()


def test_drift_flags_a_shifted_batch(raw_credit):
    ref = common.add_features(raw_credit)
    same = ref.sample(300, random_state=1)
    # The demo's "young_big_loans" scenario: under-25s with bigger loans.
    shifted = ref[ref.age < 25].sample(300, replace=True, random_state=1)
    shifted = shifted.assign(amount=shifted.amount * 2)
    cols = ["amount", "duration", "purpose", "housing"]
    assert (common.drift_table(ref, same, cols)["status"] != "alert").all()
    table = common.drift_table(ref, shifted, cols).set_index("feature")
    assert table.loc["amount", "status"] == "alert" and table.loc["housing", "status"] == "alert"


def test_fairness_table():
    scores = pd.DataFrame({"result": ["PASS"] * 8 + ["FAIL"] * 2 + ["PASS"] * 2 + ["FAIL"] * 8,
                           "age_group": ["25_plus"] * 10 + ["under_25"] * 10})
    t = common.fairness_table(scores, ["age_group"]).set_index("group")
    assert t.loc["25_plus", "disparate_impact"] == 1.0
    assert t.loc["under_25", "disparate_impact"] == 0.25 and t.loc["under_25", "four_fifths_rule"] == "FAIL"


def test_bundle_tasks_point_to_existing_notebooks():
    job = yaml.safe_load((ROOT / "databricks/resources/credit_risk_pipeline.job.yml").read_text())
    tasks = job["resources"]["jobs"]["credit_risk_pipeline"]["tasks"]
    assert [t["task_key"] for t in tasks] == ["ingest", "silver", "train", "score", "monitor"]
    for t in tasks:
        path = (ROOT / "databricks/resources" / t["notebook_task"]["notebook_path"]).resolve()
        assert path.exists() and path.read_text().startswith("# Databricks notebook source")
