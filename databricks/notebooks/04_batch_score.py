# Databricks notebook source
# MAGIC %md
# MAGIC # 4 · Batch scoring with `@champion` → `credit_scores`
# MAGIC
# MAGIC Scores the applicants from the latest load with whichever model version holds the `champion`
# MAGIC alias. Each row gets the same fields as the API and the web page: P(default), a 0-100 credit
# MAGIC score, PASS or FAIL against the pass mark, and `requires_human_review` (GDPR Article 22).
# MAGIC
# MAGIC Writes use `MERGE` on (applicant, model version), so re-running the job never duplicates scores.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "nyaya")
dbutils.widgets.text("model_name", "")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
MODEL = dbutils.widgets.get("model_name") or f"{CATALOG}.{SCHEMA}.nyaya_credit_risk"
SILVER, SCORES = f"{CATALOG}.{SCHEMA}.credit_silver", f"{CATALOG}.{SCHEMA}.credit_scores"

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F

from common import score_fields

mlflow.set_registry_uri("databricks-uc")
champ = MlflowClient().get_model_version_by_alias(MODEL, "champion")
model = mlflow.sklearn.load_model(f"models:/{MODEL}@champion")
threshold = float(champ.tags["threshold"])
print(f"champion: {MODEL} v{champ.version} ({champ.tags.get('algorithm')}), threshold {threshold}")

silver = spark.table(SILVER)
latest = silver.agg(F.max("_ingested_at")).first()[0]
batch = silver.filter(F.col("_ingested_at") == latest).toPandas()
print(f"scoring {len(batch)} applicants from the load at {latest}")

# COMMAND ----------

p_default = model.predict_proba(batch[list(model.feature_names_in_)])[:, 1]
keep = ["applicant_id", "purpose", "amount", "duration", "age_group", "sex", "foreign_worker", "default"]
scores = pd.concat([batch[keep].reset_index(drop=True), score_fields(p_default, threshold)], axis=1)
scores["model_version"] = int(champ.version)
scores["scored_at"] = pd.Timestamp.now(tz="UTC")
scores["loaded_at"] = latest

updates = spark.createDataFrame(scores)
if not spark.catalog.tableExists(SCORES):
    updates.limit(0).write.saveAsTable(SCORES)
    spark.sql(f"COMMENT ON TABLE {SCORES} IS 'Credit scores and PASS/FAIL per applicant and model version.'")
updates.createOrReplaceTempView("new_scores")
spark.sql(f"""
MERGE INTO {SCORES} AS t
USING new_scores AS s
ON t.applicant_id = s.applicant_id AND t.model_version = s.model_version
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

# COMMAND ----------

summary = scores.groupby("result").agg(applicants=("applicant_id", "size"),
                                      avg_score=("credit_score", "mean"),
                                      actual_default_rate=("default", "mean")).round(3)
display(summary.reset_index())
display(spark.table(SCORES).filter(F.col("model_version") == int(champ.version))
        .orderBy("credit_score").limit(20))
