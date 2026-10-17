# Batch check: a whole file against EU and Indian rules

Upload a CSV, Excel or PDF file with up to 3,000 loan applicants. Every applicant gets a credit score
with PASS or FAIL and the main reasons, and the whole batch is checked against the rules a lender in
the EU or in India has to follow around an automated credit decision.

| Where | How |
|---|---|
| Web page | `start_api.bat`, then <http://127.0.0.1:8000/batch> (or **Upload a file** at the top of `/app`) |
| Command line | `python scripts/check_csv.py applicants.pdf --jurisdictions EU,IN` |
| API | `POST /batch/check?jurisdictions=EU,IN` with the file as form field `file` |
| Online | <https://nyaya-ccnp.onrender.com/batch> (free instance, see RENDER.md) |
| Azure | `https://<app>.azurewebsites.net/api/batch` after deploying (see AZURE_DEPLOY.md) |

## Files it accepts

| Format | What it needs |
|---|---|
| CSV | A header row with the 20 German Credit column names. Commas or semicolons, UTF-8 or Windows encoding. |
| Excel (.xlsx) | The same columns on the first sheet. |
| PDF, table | A table whose header row has the column names; it may continue over many pages. |
| PDF, forms | One application per page (or block) with lines like `amount: 1473`. |

Amounts can be in **EUR, INR or DM**: choose it in the **Amounts in** menu (API `?currency=EUR`,
command line `--currency EUR`). They are converted to DM, the training currency, before scoring; see
[CURRENCY.md](CURRENCY.md). Column names are matched ignoring capitals and spaces, and category values ignoring capitals. An
optional `applicant_id` (or `id`, `customer_id`, `reference`, `name`) column is carried through to the
results. An optional `credit_risk` column (1 = repaid, 0 = defaulted) adds an accuracy section.
Limits: 3,000 applicants and 15 MB per file. Scanned PDFs (photos of paper) need OCR first.

Download the template from the page, or use the samples in `data/samples/` (made by
`python scripts/make_samples.py`): 3,000 rows as CSV, 300 as Excel, 120 as a PDF table and 10 as PDF
forms, each with a few deliberate errors.

## What you get

**Per applicant:** credit score (0-100), PASS or FAIL against the pass mark, probability of default,
the four factors that moved the score most, and the action each rule set requires:

| | PASS | FAIL |
|---|---|---|
| EU | Approve | Refer to a human underwriter before any rejection; give the main reasons and a way to contest |
| India | Approve | Refer to a credit officer; if declined, tell the applicant the main reasons in writing |

**Rows with errors** (a negative amount, an unknown purpose, an empty age) are listed with what to
fix. They do not stop the rest of the file.

**For the batch:** approval rate, fairness by age group, sex and foreign-worker status, drift against
the training data, and the policy checks below. Results download as CSV; the page prints as a
compliance report (save as PDF from the print dialog).

## The checks

The rules live in `src/compliance_ai/policies/eu.yaml` and `india.yaml`, so a compliance team can
review and change the wording, the thresholds and the actions without touching Python.

### European Union

| Check | Law | How it is checked |
|---|---|---|
| No solely automated rejection | GDPR Art. 22(1) and (3); Consumer Credit Directive (EU) 2023/2225, Art. 18; EU AI Act Art. 14 | Every FAIL is routed to a person |
| Main reasons given | GDPR Art. 13(2)(f), 15(1)(h), 22(3); CCD Art. 18 | Every decision has its top factors |
| Sex, age, nationality not inputs | EU Charter Art. 21; EU AI Act Art. 10(2)(f)-(g) | Read from the model's actual input list |
| Bias examination | EU AI Act Art. 10(2)(f)-(g) | Approval-rate ratios; below 0.80 → review |
| Data quality | EU AI Act Art. 10(3); GDPR Art. 5(1)(d) | Share of invalid rows; above 5% → fail |
| Monitoring | EU AI Act Art. 15 and 72 | PSI drift against the training data |
| Logging | EU AI Act Art. 12 and 26(6) | One audit-log line per batch |
| Minimisation | GDPR Art. 5(1)(c) and (e) | The file is not stored, only its SHA-256 |

### India

| Check | Law | How it is checked |
|---|---|---|
| Reasons for rejection in writing | RBI Fair Practices Code for lenders | Every decision has its top factors |
| No discrimination by sex | RBI Fair Practices Code (sex, caste, religion) | Read from the model's input list |
| Bias examination | RBI Fair Practices Code | Approval-rate ratios |
| Human review of refusals | Good practice, not an explicit legal requirement today | Every FAIL is routed to a person |
| Data accuracy | DPDP Act 2023, Section 8(3) | Share of invalid rows |
| Monitoring | Model risk management good practice | PSI drift |
| Retention | DPDP Act 2023, Section 8(7) | The file is not stored |
| Logging for grievances | DPDP Act 2023, Section 13; RBI Fair Practices Code | One audit-log line per batch |

## The audit log

`reports/audit_log.jsonl` gets one line per batch: batch id, time, file fingerprint, model and
threshold, counts, the decision and score per row number, and the result of every check. It holds
no names, references or applicant data. On Azure the same record goes to Application Insights
(the app's file system is read-only, so the logging check shows "review" there).

## Limits, honestly

- **The model is a credit-risk model trained on 1,000 German applicants (amounts in Deutsche Mark).**
  For Indian applicants it would have to be retrained and validated on Indian data in rupees; the
  India pack says so on every report. For **insurance**, the same upload, validation, policy and
  audit pipeline would work, but it needs an insurance model and insurance rules (for example, in
  the EU, no price differences by sex); this project does not include those.
- The checks show whether the **decision process** meets each rule. They are not legal advice, and
  a pass does not replace a review by the compliance team.
- The four-fifths rule is a screening metric from US practice. EU and Indian law have no fixed
  ratio; a ratio below 0.80 means "document a review", not "discrimination proven".
- Digital-lending duties such as the Key Fact Statement and the annual percentage rate (RBI), or
  pre-contract information (EU), are product steps outside this check.
