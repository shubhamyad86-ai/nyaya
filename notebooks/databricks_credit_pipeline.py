# Databricks notebook source
# MAGIC %md
# MAGIC # Credit risk on Databricks: Delta medallion tables, MLflow training, fairness audit
# MAGIC
# MAGIC This notebook is the Databricks version of the `compliance-ai` credit pipeline. It runs on
# MAGIC **Databricks Free Edition** (serverless compute, Unity Catalog).
# MAGIC
# MAGIC **Before you run it**
# MAGIC 1. In Databricks: **Catalog → workspace → default → Create → Volume**, name it `raw`.
# MAGIC 2. Open the volume and **Upload** `data/raw/german_credit.csv` from the project.
# MAGIC 3. **Workspace → Import** this file (`notebooks/databricks_credit_pipeline.py`), attach it to
# MAGIC    serverless compute, and click **Run all**.
# MAGIC
# MAGIC **What it builds**
# MAGIC
# MAGIC | Layer | Table | Content |
# MAGIC |---|---|---|
# MAGIC | Bronze | `credit_bronze` | raw CSV rows + ingestion time, append-only |
# MAGIC | Silver | `credit_silver` | rows that pass the data-quality rules, with engineered features |
# MAGIC | Silver | `credit_quarantine` | rows that failed a rule, with the reason |
# MAGIC | Gold | `credit_risk_by_segment` | default rate by loan purpose and age group |
# MAGIC | Gold | `credit_scores` | test-set scores and decisions from the trained model |
# MAGIC
# MAGIC Then it trains two models with MLflow tracking, picks a cost-based approval threshold
# MAGIC (approving a bad loan costs 5x rejecting a good one), and audits approval rates by age and sex.
# MAGIC Age, sex, marital status and foreign-worker status are **not** model inputs; they are kept only
# MAGIC for the audit.

# COMMAND ----------

dbutils.widgets.text("source_csv", "/Volumes/workspace/default/raw/german_credit.csv", "Source CSV")
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")

SOURCE = dbutils.widgets.get("source_csv")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
PREFIX = f"{CATALOG}.{SCHEMA}" if CATALOG else SCHEMA


def table(name: str) -> str:
    return f"{PREFIX}.{name}"


print("source:", SOURCE, "| tables in:", PREFIX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze: ingest the raw file as-is

# COMMAND ----------

from pyspark.sql import functions as F

raw = (
    spark.read.option("header", True).option("inferSchema", True).csv(SOURCE)
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.lit(SOURCE.split("/")[-1]))
)
raw.write.mode("append").option("mergeSchema", "true").saveAsTable(table("credit_bronze"))
print("bronze rows added:", raw.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver: data-quality rules, quarantine, features
# MAGIC Same rules as `src/compliance_ai/credit/data.py`. Silver reads the **latest** bronze load.

# COMMAND ----------

RANGES = {
    "duration": (1, 120), "amount": (1, 100_000), "age": (18, 100),
    "installment_rate": (1, 4), "present_residence": (1, 4),
    "number_credits": (1, 10), "people_liable": (1, 5),
}

bronze = spark.table(table("credit_bronze"))
latest = bronze.agg(F.max("_ingested_at")).first()[0]
bronze = bronze.filter(F.col("_ingested_at") == latest)

checks = [F.when(F.col(c).isNull() | ~F.col(c).between(lo, hi), F.lit(f"{c} out of range"))
          for c, (lo, hi) in RANGES.items()]
checks.append(F.when(~F.col("credit_risk").isin(0, 1), F.lit("invalid credit_risk")))
checked = bronze.withColumn("_issues", F.concat_ws("; ", *checks))

quarantine = checked.filter(F.col("_issues") != "")
silver = (
    checked.filter(F.col("_issues") == "").drop("_issues")
    .withColumn("default", (1 - F.col("credit_risk")).cast("int"))
    .withColumn("sex", F.when(F.col("personal_status_sex").startswith("female"), "female").otherwise("male"))
    .withColumn("age_group", F.when(F.col("age") < 25, "under_25").otherwise("25_plus"))
    .withColumn("amount_per_month", F.col("amount") / F.col("duration"))
    .withColumn("has_checking_account", (F.col("status") != "no checking account").cast("int"))
    .withColumn("no_savings", F.lower("savings").rlike("unknown|no savings").cast("int"))
    .withColumn("delayed_before", F.lower("credit_history").contains("delay").cast("int"))
    .withColumn("applicant_id", F.monotonically_increasing_id())
)

silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("credit_silver"))
quarantine.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("credit_quarantine"))
print("silver rows:", silver.count(), "| quarantined:", quarantine.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gold: reporting table

# COMMAND ----------

silver = spark.table(table("credit_silver"))
by_segment = (
    silver.groupBy("purpose", "age_group")
    .agg(F.count("*").alias("applicants"),
         F.round(F.avg("default"), 3).alias("default_rate"),
         F.round(F.avg("amount"), 0).alias("avg_amount"),
         F.round(F.avg("duration"), 1).alias("avg_duration_months"))
    .orderBy(F.desc("applicants"))
)
(by_segment.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(table("credit_risk_by_segment")))
display(by_segment)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train with MLflow
# MAGIC 1,000 rows fit in memory, so training uses scikit-learn on a pandas copy of the silver table.
# MAGIC Spark does the data engineering; this is the usual split on Databricks for small tabular data.

# COMMAND ----------

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COST_FALSE_APPROVAL, COST_FALSE_REJECTION = 5, 1
AUDIT_ONLY = {"personal_status_sex", "foreign_worker", "age", "sex", "age_group"}
NOT_FEATURES = {"credit_risk", "default", "applicant_id", "_ingested_at", "_source_file"} | AUDIT_ONLY

pdf = silver.toPandas()
features = [c for c in pdf.columns if c not in NOT_FEATURES]
numeric = [c for c in features if pdf[c].dtype.kind in "if"]
categorical = [c for c in features if c not in numeric]

idx_train, idx_test = train_test_split(pdf.index, test_size=0.2, stratify=pdf["default"], random_state=42)
X_train, X_test = pdf.loc[idx_train, features], pdf.loc[idx_test, features]
y_train, y_test = pdf.loc[idx_train, "default"].to_numpy(), pdf.loc[idx_test, "default"].to_numpy()


def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def expected_cost(y, p_default, threshold):
    reject = p_default >= threshold
    false_approval = np.sum((~reject) & (y == 1))
    false_rejection = np.sum(reject & (y == 0))
    return (COST_FALSE_APPROVAL * false_approval + COST_FALSE_REJECTION * false_rejection) / len(y)


def best_threshold(y, p_default):
    grid = np.linspace(0.05, 0.95, 91)
    costs = [expected_cost(y, p_default, t) for t in grid]
    return float(grid[int(np.argmin(costs))]), float(min(costs))


candidates = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=300, random_state=42),
}

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/compliance-ai-credit-risk")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in candidates.items():
    with mlflow.start_run(run_name=f"cv_{name}"):
        oof = cross_val_predict(make_pipeline(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold, cost = best_threshold(y_train, oof)
        auc = roc_auc_score(y_train, oof)
        mlflow.log_params({"model": name, "cv_folds": 5, "cost_false_approval": COST_FALSE_APPROVAL})
        mlflow.log_metrics({"cv_auc": auc, "cv_expected_cost": cost, "threshold": threshold})
        results[name] = {"auc": auc, "cost": cost, "threshold": threshold}
        print(f"{name:24s} CV AUC {auc:.3f}  cost/applicant {cost:.3f}  threshold {threshold:.2f}")

winner = min(results, key=lambda n: results[n]["cost"])
threshold = results[winner]["threshold"]
print("winner:", winner)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Refit the winner, evaluate once on the held-out test set, log the model

# COMMAND ----------

final = make_pipeline(candidates[winner]).fit(X_train, y_train)
p_test = final.predict_proba(X_test)[:, 1]
reject = p_test >= threshold
test_metrics = {
    "test_auc": float(roc_auc_score(y_test, p_test)),
    "test_expected_cost": float(expected_cost(y_test, p_test, threshold)),
    "test_approval_rate": float(np.mean(~reject)),
    "test_bad_rate_among_approved": float(y_test[~reject].mean()) if (~reject).any() else 0.0,
}

with mlflow.start_run(run_name=f"final_{winner}") as run:
    mlflow.log_params({"model": winner, "threshold": threshold, "features": len(features)})
    mlflow.log_metrics(test_metrics)
    mlflow.sklearn.log_model(final, name="model", input_example=X_train.head(3),
                             signature=infer_signature(X_train, p_test))
    run_id = run.info.run_id

print({k: round(v, 3) for k, v in test_metrics.items()}, "| run:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Gold: write test-set scores and decisions

# COMMAND ----------

keep = ["applicant_id", "purpose", "amount", "duration", "age_group", "sex", "default"]
scores = pdf.loc[idx_test, keep].copy()
scores["p_default"] = p_test
scores["decision"] = np.where(reject, "refer_or_decline", "approve")
scores["model"] = winner
scores["mlflow_run_id"] = run_id
spark.createDataFrame(scores).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(table("credit_scores"))
display(spark.table(table("credit_scores")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fairness audit
# MAGIC **Disparate impact** = approval rate of a group ÷ approval rate of the most-approved group.
# MAGIC Below 0.80 fails the four-fifths rule and needs review. **Equal opportunity gap** = difference
# MAGIC in approval rates among applicants who actually repaid.

# COMMAND ----------

import pandas as pd

audit_rows = []
for attribute in ["age_group", "sex"]:
    g = scores.assign(approved=scores["decision"].eq("approve"))
    rates = g.groupby(attribute)["approved"].mean()
    good = g[g["default"] == 0].groupby(attribute)["approved"].mean()
    for group, rate in rates.items():
        audit_rows.append({
            "attribute": attribute, "group": group, "applicants": int((g[attribute] == group).sum()),
            "approval_rate": round(rate, 3),
            "disparate_impact": round(rate / rates.max(), 3),
            "approval_rate_good_payers": round(good.get(group, np.nan), 3),
            "equal_opportunity_gap": round(good.max() - good.get(group, np.nan), 3),
            "four_fifths_rule": "pass" if rate / rates.max() >= 0.8 else "FAIL",
        })

audit = pd.DataFrame(audit_rows)
with mlflow.start_run(run_id=run_id):
    for r in audit_rows:
        mlflow.log_metric(f"di_{r['attribute']}_{r['group']}", r["disparate_impact"])
    mlflow.log_table(audit, "fairness_audit.json")
display(spark.createDataFrame(audit))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Delta features worth showing
# MAGIC Every write is a versioned commit, so you can audit what changed and read an older version
# MAGIC ("time travel"). Re-run the notebook and bronze gets a new version with the new load appended.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {table('credit_bronze')}")
        .select("version", "timestamp", "operation", "operationMetrics"))

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) AS rows_in_version_0 FROM {table('credit_bronze')} VERSION AS OF 0"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC - The test set is small (200 applicants, about 40 under 25), so fairness numbers have wide
# MAGIC   uncertainty. Treat a FAIL as a signal to investigate, not a final verdict.
# MAGIC - Removing age from the inputs does not remove age effects: housing and employment length act as
# MAGIC   proxies. The local project's drift report shows the same features.
# MAGIC - To register the model in Unity Catalog: `mlflow.set_registry_uri("databricks-uc")` then
# MAGIC   `mlflow.register_model(f"runs:/{run_id}/model", f"{PREFIX}.credit_risk_model")`.
