import numpy as np
import pytest

from compliance_ai.experiment import (
    assign_variant,
    bootstrap_diff,
    mcnemar,
    sample_size_per_arm,
    srm_check,
    two_proportion_test,
)


def test_assignment_is_stable_and_balanced():
    assert assign_variant("applicant-7", "exp") == assign_variant("applicant-7", "exp")
    arms = [assign_variant(i, "exp") for i in range(10_000)]
    assert 0.48 < arms.count("B") / len(arms) < 0.52
    # A different experiment reshuffles the same users.
    other = [assign_variant(i, "other-exp") for i in range(10_000)]
    assert sum(a != b for a, b in zip(arms, other, strict=True)) > 4_000


def test_assignment_respects_share():
    arms = [assign_variant(i, "exp", share_b=0.1) for i in range(10_000)]
    assert 0.08 < arms.count("B") / len(arms) < 0.12


def test_sample_size_matches_textbook_value():
    # 10% -> 12% conversion, alpha 0.05, power 0.8: about 3,841 per arm.
    assert 3_800 <= sample_size_per_arm(0.10, 0.02) <= 3_900
    with pytest.raises(ValueError):
        sample_size_per_arm(0.99, 0.05)


def test_two_proportion_test():
    same = two_proportion_test(50, 100, 50, 100)
    assert same["diff"] == 0 and same["p_value"] == pytest.approx(1.0)
    big = two_proportion_test(500, 1000, 600, 1000)
    assert big["p_value"] < 1e-5 and big["ci_low"] > 0


def test_bootstrap_detects_shift_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 500)
    assert bootstrap_diff(a, a + 0.5)["p_value"] < 0.01
    assert bootstrap_diff(a, rng.normal(0, 1, 500))["p_value"] > 0.01


def test_mcnemar_and_srm():
    a = np.array([True] * 30 + [False] * 10)
    assert mcnemar(a, a)["p_value"] == 1.0
    assert mcnemar(np.ones(40, bool), np.zeros(40, bool))["p_value"] < 1e-6
    assert not srm_check(5_000, 5_050)["mismatch"]
    assert srm_check(5_000, 5_600)["mismatch"]
