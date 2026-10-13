# Run the credit pipeline on Databricks (Free Edition)

`notebooks/databricks_credit_pipeline.py` is a Databricks notebook in source format. It builds Delta
tables (bronze → silver + quarantine → gold), trains two models with MLflow tracking, writes scored
applicants back to a Delta table, and audits approval rates by age and sex.

## Steps (about 10 minutes)

1. Sign up for **Databricks Free Edition** at <https://www.databricks.com/learn/free-edition>
   (no card needed).
2. Upload the data: **Catalog → workspace → default → Create → Volume**, name it `raw`, open it and
   **Upload to this volume** → `data/raw/german_credit.csv` from the project folder.
3. Import the notebook: **Workspace → your user folder → ⋮ → Import → File** →
   `notebooks/databricks_credit_pipeline.py`.
4. Open it, connect to **Serverless** compute and click **Run all**.

If you used another catalog, schema or volume, change the three widgets at the top of the notebook.

## What to look at afterwards

- **Catalog → workspace → default:** the tables `credit_bronze`, `credit_silver`,
  `credit_quarantine`, `credit_risk_by_segment` and `credit_scores`. Each has a **History** tab
  (every write is a Delta version).
- **Experiments → compliance-ai-credit-risk:** one run per model with CV AUC, expected cost and
  threshold; the final run has the logged model, test metrics and the fairness table.
- Run the notebook a second time: bronze gains a new version (append), while silver and gold are
  rebuilt from the latest load.

## How it relates to the rest of the project

| Local project | Databricks notebook |
|---|---|
| `spark/credit_pipeline.py` writes Delta files to `data/delta/` | Same rules, written as Unity Catalog tables |
| `compliance_ai.credit.train`: 4 models, SHAP, MLflow in `mlflow.db` | 2 models (no extra installs needed), MLflow in the workspace |
| `reports/metrics.json` fairness audit | Fairness table logged to MLflow and displayed |

## Result of the run on Databricks Free Edition (October 2026)

| Step | Result |
|---|---|
| Silver | 1,000 rows passed the quality rules, 0 quarantined |
| MLflow | logistic regression CV AUC 0.764, expected cost 0.510; gradient boosting 0.782, 0.521 |
| Selected model | logistic regression (lowest cost) |
| Fairness | under-25 disparate impact 0.525 (fails four-fifths rule); female 0.932 (passes) |

These agree with the local pipeline: the age gap persists even though age is not a model input.
