---
title: Nyaya Credit Check
emoji: 🏦
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Credit scoring with reasons, batch upload, EU and India rules
---

# Nyaya credit check

- **/app**: fill in an applicant and get a credit score (0-100), PASS or FAIL and the main reasons.
- **/batch**: upload a CSV, Excel or PDF with up to 3,000 applicants; each one is scored and the
  batch is checked against EU (GDPR, AI Act) and Indian (RBI, DPDP Act) rules.
- **/docs**: the API.

FAIL means "refer to a human underwriter", never an automatic rejection (GDPR Art. 22).
The model is trained on the Statlog German Credit data (1,000 applicants, amounts in Deutsche Mark).
The uploaded files are not stored. Demo only, not for real lending decisions.

Source code: see the GitHub repository.
