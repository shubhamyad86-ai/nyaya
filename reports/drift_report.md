# Drift monitoring report

PSI per feature against the training population. < 0.10 stable · 0.10-0.25 monitor · > 0.25 significant shift. A shift also needs a chi-square p-value below 0.01 (Bonferroni-corrected across features), so small batches are not flagged for random noise.

## Held-out test set (same population)

**OK** - No meaningful drift. (200 applicants; model-score PSI 0.09, mean P(default) 0.39 -> 0.39)

| Feature | PSI | p-value | Status |
|---|---|---|---|
| purpose | 0.073 | 0.0034 | stable |
| employment_duration | 0.048 | 0.011 | stable |
| amount_per_month | 0.045 | 0.48 | stable |
| amount | 0.041 | 0.59 | stable |
| savings | 0.037 | 0.2 | stable |

## Applicants under 25 (new segment)

**ALERT** - Significant shift in: housing, employment_duration. Review before relying on the model. (149 applicants; model-score PSI 0.26, mean P(default) 0.39 -> 0.49)

| Feature | PSI | p-value | Status |
|---|---|---|---|
| housing | 0.426 | 0 | significant_shift |
| employment_duration | 0.361 | 0 | significant_shift |
| credit_history | 0.229 | 0 | monitor |
| job | 0.212 | 0.0008 | monitor |
| purpose | 0.160 | 0.017 | stable |

## Largest 25% of loans (product change)

**ALERT** - Significant shift in: amount, amount_per_month, duration, purpose, property. Review before relying on the model. (250 applicants; model-score PSI 0.19, mean P(default) 0.39 -> 0.51)

| Feature | PSI | p-value | Status |
|---|---|---|---|
| amount | 5.756 | 0 | significant_shift |
| amount_per_month | 1.533 | 0 | significant_shift |
| duration | 0.906 | 0 | significant_shift |
| purpose | 0.291 | 0 | significant_shift |
| property | 0.276 | 0 | significant_shift |
