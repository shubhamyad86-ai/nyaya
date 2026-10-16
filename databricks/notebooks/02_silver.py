# Databricks notebook source
# MAGIC %md
# MAGIC # 2 · Silver: data-quality rules, quarantine, features
# MAGIC
# MAGIC Every bronze row is checked against the same rules the API uses (`compliance_ai.credit.data`):
# MAGIC no missing values, numbers in range, categories from the codebook, label 0 or 1.
# MAGIC
# MAGIC | Table | Content |
# MAGIC |---|---|
# MAGIC | `credit_silver` | valid rows, de-duplicated, with engineered features and a stable `applicant_id` |
# MAGIC | `credit_quarantine` | rows that broke a rule, with the reasons |
# MAGIC
# MAGIC **Quality gate:** if more than 5% of the latest load fails the rules, the task fails, the job
# MAGIC stops before training, and the owner gets an email.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "nyaya")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"  # noqa: E731

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

from common import CATEGORIES, MAX_QUARANTINE_SHARE, RANGES, TARGET

bronze = spark.table(T("credit_bronze"))
business_cols = [c for c in bronze.columns if not c.startswith("_")]

checks = [F.when(F.col(c).isNull(), F.lit(f"{c} missing")) for c in business_cols]
checks += [F.when(~F.col(c).between(lo, hi), F.lit(f"{c} out of range")) for c, (lo, hi) in RANGES.items()]
checks += [F.when(~F.col(c).isin(allowed), F.lit(f"{c} unknown value")) for c, allowed in CATEGORIES.items()]
checks.append(F.when(~F.col(TARGET).isin(0, 1), F.lit(f"{TARGET} not 0/1")))

checked = (
    bronze
    .withColumn("_issues", F.concat_ws("; ", *checks))
    # The same applicant sent twice (e.g. a file re-exported) gets the same id.
    .withColumn("applicant_id",
                F.sha2(F.concat_ws("||", *[F.col(c).cast("string") for c in business_cols]), 256))
)

quarantine = checked.filter(F.col("_issues") != "")
first_seen = Window.partitionBy("applicant_id").orderBy("_ingested_at")
valid = (checked.filter(F.col("_issues") == "").drop("_issues")
         .withColumn("_n", F.row_number().over(first_seen)).filter("_n = 1").drop("_n"))

silver = (
    valid
    .withColumn("default", (1 - F.col(TARGET)).cast("int"))
    .withColumn("sex", F.when(F.col("personal_status_sex").startswith("female"), "female").otherwise("male"))
    .withColumn("age_group", F.when(F.col("age") < 25, "under_25").otherwise("25_plus"))
    .withColumn("amount_per_month", F.col("amount") / F.col("duration"))
    .withColumn("has_checking_account", (F.col("status") != "no checking account").cast("int"))
    .withColumn("no_savings", F.lower("savings").rlike("unknown|no savings").cast("int"))
    .withColumn("delayed_before", F.lower("credit_history").contains("delay").cast("int"))
)

# COMMAND ----------

silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("credit_silver"))
quarantine.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("credit_quarantine"))
spark.sql(f"COMMENT ON TABLE {T('credit_silver')} IS "
          "'Valid, de-duplicated applications with features. Rebuilt from bronze on every run.'")
spark.sql(f"COMMENT ON TABLE {T('credit_quarantine')} IS "
          "'Rows that failed a data-quality rule, with reasons.'")

# Delta CHECK constraints: any later write that breaks a rule is rejected by the table itself.
existing = {r["key"] for r in spark.sql(f"SHOW TBLPROPERTIES {T('credit_silver')}").collect()}
for name, rule in {"valid_amount": "amount BETWEEN 1 AND 100000",
                   "valid_duration": "duration BETWEEN 1 AND 120",
                   "valid_label": f"{TARGET} IN (0, 1)"}.items():
    if f"delta.constraints.{name}" not in existing:
        spark.sql(f"ALTER TABLE {T('credit_silver')} ADD CONSTRAINT {name} CHECK ({rule})")

# COMMAND ----------

latest = bronze.agg(F.max("_ingested_at")).first()[0]
latest_total = checked.filter(F.col("_ingested_at") == latest).count()
latest_bad = quarantine.filter(F.col("_ingested_at") == latest).count()
share = latest_bad / latest_total if latest_total else 0.0
print(f"silver rows: {silver.count()} | quarantined in total: {quarantine.count()} | "
      f"latest load: {latest_bad}/{latest_total} rejected ({share:.1%})")

if latest_bad:
    display(quarantine.filter(F.col("_ingested_at") == latest)
            .groupBy("_issues").count().orderBy(F.desc("count")))

if share > MAX_QUARANTINE_SHARE:
    raise ValueError(f"Quality gate: {share:.1%} of the latest load failed the rules "
                     f"(limit {MAX_QUARANTINE_SHARE:.0%}). See {T('credit_quarantine')}.")
