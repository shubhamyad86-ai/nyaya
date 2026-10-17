# A/B test: champion vs challenger credit model

- **Champion (A):** xgboost @ 0.19
- **Challenger (B):** xgboost @ 0.50
- **Test applicants:** 200 held-out applicants, unseen by both models

## 1. Plan: power analysis

Champion approval rate: 35.5%. To detect a change of 5% points with 80% power at alpha = 0.05, a live test needs **1,388 applicants per arm** (2,776 in total).

## 2. Randomised A/B split (how it runs live)

Applicants were hashed into arms with `assign_variant` (experiment `credit-model-ab-1`): A = 108, B = 92. Sample-ratio check p = 0.258 (split is fine).

| Metric | A | B | Difference (B - A) | 95% CI | p-value | Result |
|---|---|---|---|---|---|---|
| Approval rate | 38.9% | 60.9% | +22.0% | +8.4% to +35.5% | 0.002 | significant |
| Default rate among approved | 11.9% | 8.9% | -3.0% | -15.3% to +9.3% | 0.630 | not significant |
| Cost per applicant | 0.593 | 0.413 | -0.180 | -0.492 to +0.132 | 0.254 | not significant |

Each arm has about 100 applicants against the 1,388 the plan asks for, so this split is underpowered: a non-significant result here means 'not enough data', not 'no difference'.

## 3. Paired comparison (both models on every test applicant)

- Cost per applicant: A 0.645, B 0.585; difference -0.060 (95% CI -0.195 to +0.100, p = 0.422, not significant).
- Decisions that disagree: only A right on 7, only B right on 47 (McNemar p = 0.000, significant).
- Approval rate: A 35.5%, B 62.5%.

## Decision

No significant difference in cost. Keep the champion: switching models carries risk and needs evidence, and this test does not provide it.
