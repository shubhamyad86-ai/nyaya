# Databricks notebook source
# MAGIC %md
# MAGIC # Demo: drop a "new month" of applications into the landing volume
# MAGIC
# MAGIC Run this, then run the job again to see the whole loop react: `COPY INTO` picks up only the new
# MAGIC file, silver checks it, a new model version is trained and compared with the champion, the batch
# MAGIC is scored, and monitoring compares it with the training data.
# MAGIC
# MAGIC - **scenario = same** — applicants like the training data (monitoring should stay stable)
# MAGIC - **scenario = young_big_loans** — a campaign aimed at under-25s with larger loans (drift alert)
# MAGIC - **scenario = bad_file** — 20% broken rows (the silver quality gate stops the job)
# MAGIC
# MAGIC The rows are resampled from the real data with small random changes, and they keep the historical
# MAGIC outcome column, which a real new application would not have yet. That keeps the demo simple.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "nyaya")
dbutils.widgets.text("landing_volume", "landing")
dbutils.widgets.dropdown("scenario", "young_big_loans", ["same", "young_big_loans", "bad_file"])
dbutils.widgets.text("rows", "300")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
SCENARIO, N = dbutils.widgets.get("scenario"), int(dbutils.widgets.get("rows"))

# COMMAND ----------

import numpy as np
import pandas as pd

business = [c for c in spark.table(f"{CATALOG}.{SCHEMA}.credit_bronze").columns if not c.startswith("_")]
base = spark.table(f"{CATALOG}.{SCHEMA}.credit_silver").select(*business).toPandas()
rng = np.random.default_rng()

if SCENARIO == "young_big_loans":
    weights = np.where(base["age"] < 25, 6.0, 1.0)
    new = base.sample(N, replace=True, weights=weights, random_state=int(rng.integers(1e9))).copy()
    new["amount"] = (new["amount"] * rng.uniform(1.3, 1.8, N)).round().clip(1, 100_000).astype(int)
    new["duration"] = (new["duration"] + rng.choice([6, 12], N)).clip(1, 120)
else:
    new = base.sample(N, replace=True, random_state=int(rng.integers(1e9))).copy()
    new["amount"] = (new["amount"] * rng.uniform(0.9, 1.1, N)).round().astype(int)

if SCENARIO == "bad_file":
    broken = rng.choice(N, N // 5, replace=False)
    new.iloc[broken, new.columns.get_loc("amount")] = -1

name = f"applications_{pd.Timestamp.now():%Y%m%d_%H%M%S}_{SCENARIO}.csv"
path = f"/Volumes/{CATALOG}/{SCHEMA}/{dbutils.widgets.get('landing_volume')}/{name}"
new.to_csv(path, index=False)
print(f"wrote {len(new)} rows to {path}. Now run the job again.")
