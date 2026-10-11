import numpy as np
import pandas as pd
import pytest

from compliance_ai.credit.data import (
    DataValidationError,
    add_features,
    feature_columns,
    validate,
)
from compliance_ai.credit.fairness import fairness_summary, group_metrics
from compliance_ai.credit.metrics import best_threshold, expected_cost


def test_real_data_schema(raw_credit):
    validate(raw_credit)
    assert raw_credit.shape == (1000, 21)
    assert raw_credit["credit_risk"].mean() == pytest.approx(0.7)


def test_validation_rejects_bad_rows(raw_credit):
    bad = raw_credit.copy()
    bad.loc[0, "age"] = 7
    with pytest.raises(DataValidationError, match="age"):
        validate(bad)
    with pytest.raises(DataValidationError, match="missing columns"):
        validate(raw_credit.drop(columns="amount"))


def test_protected_attributes_are_not_model_inputs(raw_credit):
    df = add_features(raw_credit)
    numeric, categorical = feature_columns(df)
    inputs = set(numeric + categorical)
    assert not inputs & {"sex", "age_group", "foreign_worker", "personal_status_sex", "credit_risk",
                         "age", "log_amount"}
    assert {"amount_per_month", "has_checking_account"} <= inputs


def test_engineered_features(raw_credit):
    df = add_features(raw_credit.head(1))
    assert df.loc[0, "amount_per_month"] == pytest.approx(1169 / 6)
    assert df.loc[0, "sex"] == "male" and df.loc[0, "age_group"] == "25_plus"


def test_expected_cost_and_threshold():
    y = np.array([1, 1, 0, 0, 0, 0])            # 1 = default
    p = np.array([0.9, 0.3, 0.2, 0.1, 0.4, 0.05])
    # threshold 0.5: one defaulter approved (cost 5) -> 5/6
    assert expected_cost(y, p, 0.5, 5, 1) == pytest.approx(5 / 6)
    thr, cost = best_threshold(y, p, 5, 1)
    assert cost <= expected_cost(y, p, 0.5, 5, 1)
    assert 0.2 < thr <= 0.3                     # catches both defaulters, rejects one good


def test_fairness_metrics_known_values():
    y = np.zeros(8)
    p = np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9])   # group a: 3/4 approved, b: 1/4
    groups = np.array(list("aaaabbbb"))
    table = group_metrics(y, p, groups, threshold=0.5)
    assert table.loc["a", "approval_rate"] == 0.75
    assert table.loc["b", "disparate_impact_ratio"] == pytest.approx(1 / 3)
    summary = fairness_summary(table)
    assert summary["demographic_parity_difference"] == pytest.approx(0.5)
    assert summary["passes_four_fifths_rule"] is False


def test_reason_codes_sorted_by_impact():
    from compliance_ai.credit.explain import reason_codes

    shap_row = pd.Series({"amount": 0.2, "duration": -0.9, "age": 0.5})
    applicant = pd.Series({"amount": 1000, "duration": 6, "age": 30})
    codes = reason_codes(shap_row, applicant, top_n=2)
    assert [c["feature"] for c in codes] == ["duration", "age"]
    assert codes[0]["direction"] == "decreases risk"
