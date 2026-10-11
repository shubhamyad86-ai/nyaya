# Model card - credit-risk classifier

<!-- auto:card-header -->
| | |
|---|---|
| Model | XGBoost gradient-boosted trees inside a scikit-learn pipeline (scaling + one-hot encoding) |
| Task | Estimate the probability that a loan applicant defaults; refer high-risk cases to a human |
| Data | Statlog German Credit (UCI), 1,000 real applicants, 30% defaulted |
| Selection | Lowest expected business cost across 4 models, 5-fold stratified cross-validation |
| Decision rule | Refer or reject when P(default) >= 0.19 |
| Inputs | 21 features. Age, sex/marital status and foreign-worker status are excluded; `log_amount` (a duplicate of `amount`) was dropped |
| Tracking | MLflow (parameters, CV metrics, fairness metrics, artefacts, model) |
| Monitoring | PSI drift check per feature and on the scores, against a profile saved at training |
<!-- /auto:card-header -->

## Intended use

Decision support for a credit analyst, not automated refusal. GDPR Article 22 gives applicants the
right not to be subject to decisions based solely on automated processing, and the EU AI Act lists
creditworthiness assessment as high-risk (Annex III, point 5(b)). Every `refer_or_reject` outcome
therefore goes to a human reviewer with the model's reason codes and the relevant regulation passages
(`/credit/review`, `/agent/review`).

Not intended for: production lending, populations other than 1970s-90s German consumer credit,
or any decision without human review.

<!-- auto:card-performance -->
## Performance (20% held-out test set, 200 applicants)

| Metric | Value |
|---|---|
| ROC AUC | 0.791 |
| PR AUC (defaulters) | 0.671 |
| Brier score | 0.170 |
| Defaulters caught (recall) | 83% (50 of 60) |
| Precision on refer/reject | 39% |
| Approval rate | 36% |
| Expected cost per applicant | 0.645 |

Cost baselines on the same test set: approve everyone 1.50, reject everyone 0.70, this model at the default 0.5 threshold 0.585. The 5:1 cost matrix makes missed defaulters expensive, so the cost-optimal threshold is low and the approval rate is low. With 200 test applicants, expect a few points of noise on every number.

## Cross-validated comparison (800 training applicants)

| Model | ROC AUC (mean ± sd) | PR AUC | Brier | Best threshold | Expected cost |
|---|---|---|---|---|---|
| XGBoost | 0.795 ± 0.031 | 0.615 | 0.175 | 0.19 | **0.486** |
| Gradient boosting | 0.790 ± 0.029 | 0.599 | 0.167 | 0.16 | 0.505 |
| Logistic regression | 0.765 ± 0.059 | 0.595 | 0.197 | 0.27 | 0.510 |
| Random forest | 0.794 ± 0.052 | 0.643 | 0.170 | 0.27 | 0.526 |

The models are close: expected costs span only 0.040 per applicant, within cross-validation noise, so a different library version can change which one wins. XGBoost has the best ranking quality (ROC AUC). The interpretable logistic regression is a credible alternative where simplicity matters more than a few percent of cost.
<!-- /auto:card-performance -->

<!-- auto:card-fairness -->
## Fairness audit

Sex, age and foreign-worker status are never model inputs; they are kept only for auditing. Figures below use out-of-fold predictions on 800 applicants for steadier group estimates.

| Attribute | Groups (n) | Approval rate | Approval rate if repaid | Actual default rate | Disparate impact |
|---|---|---|---|---|---|
| Sex | female (249) / male (551) | 33% / 39% | 46% / 50% | 36% / 27% | 0.84 - passes 4/5 rule |
| Age | under 25 (110) / 25+ (690) | 19% / 40% | 28% / 52% | 38% / 29% | **0.48 - fails** |
| Foreign worker | yes (769) / no (31) | 36% / 58% | 48% / 64% | 31% / 10% | 0.62 - fails, tiny group |

On the 200-applicant test set the age ratio is 0.60, but only about 40 test applicants are under 25, so the out-of-fold figure is the more reliable one.

Findings
- **Age is the material issue.** Younger applicants do default more often in this data (38% vs 29%), but the model approves them at 19% versus 40%, and creditworthy under-25s are approved far less often (28% vs 52%).
- **Removing age does not fix it.** It is still far below 0.80. In a controlled experiment the age disparate impact is 0.42 with age as an input and 0.48 without it, with expected cost 0.509 vs 0.486. Age information reaches the model through proxies; the drift check on under-25 applicants points at housing and employment duration. Next step: a dedicated mitigation (group-aware thresholds or reweighing), with legal review of which approach is permissible.
- The foreign-worker result rests on 31 people and should not be read as reliable.
<!-- /auto:card-fairness -->

## Monitoring

`models/reference_profile.json` stores decile bins and category shares for every input and for the
model's scores. `POST /monitor/drift` (and `/api/drift` in Azure) compares a batch of 30 or more
applicants against it with the Population Stability Index. A feature is flagged only when PSI is
large (> 0.25 shift, 0.10-0.25 monitor) **and** a chi-square test says the change is unlikely to be
noise (p < 0.01, Bonferroni-corrected across features).

That second condition matters: PSI alone flagged 70-100% of random 30-60 applicant samples drawn
from the training data itself. With the test, false alerts fall to 0-2% while real shifts are still
caught (`reports/drift_report.md`):

<!-- auto:card-drift -->
| Batch | Result |
|---|---|
| Held-out test set | OK |
| Applicants under 25 | ALERT: housing and employment duration; mean P(default) 0.39 → 0.49 |
| Largest 25% of loans | ALERT: amount, amount per month, duration and purpose; mean P(default) 0.39 → 0.51 |
<!-- /auto:card-drift -->

## Explainability

<!-- auto:card-explain -->
SHAP values give global importance (`reports/shap_importance.png`) and per-applicant reason codes. Top drivers: checking-account status, loan amount, monthly repayment, duration, credit history. Age is not an input, so it cannot appear in reason codes; proxies for it can.
<!-- /auto:card-explain -->

## Data caveats

- The decoded CSV labels some categories inconsistently with the original codebook (Groemping, 2019,
  documented errors in the Statlog coding). Numeric values are reliable; category names are approximate,
  which is why "critical account" appears to lower risk.
- The data is old, small and from one country; it is suitable for demonstrating method, not for
  deployment.

## Unsupervised analysis

- K-means (k chosen by silhouette, best k = 2) separates a large-loan segment (234 applicants, median
  6,712 DM over 36 months, 42% default) from a small-loan segment (766, median 1,844 DM over 15 months,
  26% default).
- Isolation Forest flags 30 unusual applications (median 12,070 DM); 57% of them defaulted versus 29%
  of the rest, so the flag is a useful trigger for manual review.
