# Databricks notebook source
# MAGIC %md
# MAGIC # 5 · Monitoring: drift and fairness → `credit_monitoring`
# MAGIC
# MAGIC **Drift.** The latest load is compared with the exact data the champion was trained on: the
# MAGIC silver table at the version stored on the model (Delta time travel). PSI per feature:
# MAGIC below 0.10 stable, 0.10-0.25 monitor, above 0.25 alert.
# MAGIC
# MAGIC **Fairness.** Approval rates of the latest scores by age group, sex and foreign-worker status.
# MAGIC A disparate impact below 0.80 fails the four-fifths rule.
# MAGIC
# MAGIC Results are appended to `credit_monitoring`, so trends can be charted in a Databricks dashboard.
# MAGIC A drift **alert** fails this task, which emails the job owner.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "nyaya")
dbutils.widgets.text("model_name", "")
dbutils.widgets.dropdown("fail_on_drift_alert", "true", ["true", "false"])
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
MODEL = dbutils.widgets.get("model_name") or f"{CATALOG}.{SCHEMA}.nyaya_credit_risk"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"  # noqa: E731

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F

from common import PSI_ALERT, drift_table, fairness_table, feature_columns

mlflow.set_registry_uri("databricks-uc")
champ = MlflowClient().get_model_version_by_alias(MODEL, "champion")
train_version = int(champ.tags["silver_version"])

reference = spark.read.option("versionAsOf", train_version).table(T("credit_silver")).toPandas()
silver = spark.table(T("credit_silver"))
latest = silver.agg(F.max("_ingested_at")).first()[0]
current = silver.filter(F.col("_ingested_at") == latest).toPandas()
numeric, categorical = feature_columns(reference)
print(f"reference: silver v{train_version} ({len(reference)} rows) | current: load at {latest} "
      f"({len(current)} rows) | champion v{champ.version}")

drift = drift_table(reference, current, numeric + categorical)
display(drift)

# COMMAND ----------

scores = (spark.table(T("credit_scores"))
          .filter((F.col("model_version") == int(champ.version)) & (F.col("loaded_at") == latest))
          .toPandas())
fairness = fairness_table(scores, ["age_group", "sex", "foreign_worker"])
display(fairness)
if len(scores) < 100:
    print(f"Note: only {len(scores)} scored applicants; small groups make fairness ratios noisy.")

# COMMAND ----------

now = pd.Timestamp.now(tz="UTC")
rows = [{"checked_at": now, "loaded_at": latest, "model_version": int(champ.version), "check": "drift",
         "subject": r.feature, "group_name": None, "value": float(r.psi), "status": r.status}
        for r in drift.itertuples()]
rows += [{"checked_at": now, "loaded_at": latest, "model_version": int(champ.version), "check": "fairness",
          "subject": r.attribute, "group_name": r.group, "value": float(r.disparate_impact),
          "status": r.four_fifths_rule} for r in fairness.itertuples()]
schema = ("checked_at timestamp, loaded_at timestamp, model_version int, check string, subject string, "
          "group_name string, value double, status string")
(spark.createDataFrame(pd.DataFrame(rows), schema=schema)
 .write.mode("append").saveAsTable(T("credit_monitoring")))

alerts = drift[drift.psi >= PSI_ALERT]
fair_fails = fairness[fairness.four_fifths_rule == "FAIL"]
print(f"drift alerts: {len(alerts)} | fairness groups failing four-fifths: {len(fair_fails)}")
if len(fair_fails):
    failing = ", ".join(f"{r.attribute}={r.group} ({r.disparate_impact})" for r in fair_fails.itertuples())
    print(f"Fairness: {failing} -> needs human review of the policy.")

if len(alerts) and dbutils.widgets.get("fail_on_drift_alert") == "true":
    shifted = ", ".join(f"{r.feature} (PSI {r.psi})" for r in alerts.itertuples())
    raise RuntimeError(f"Drift alert on: {shifted}"
                       ". The model may not fit the new applicants; review before relying on its scores.")
