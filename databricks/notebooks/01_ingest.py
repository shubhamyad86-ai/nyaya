# Databricks notebook source
# MAGIC %md
# MAGIC # 1 · Ingest: landing volume → `credit_bronze`
# MAGIC
# MAGIC New CSV files dropped into the landing volume are loaded with `COPY INTO`, which remembers the
# MAGIC files it has already loaded. Re-running the job never loads the same file twice, so the step is
# MAGIC safe to retry. Bronze keeps every row as received, plus the load time and the file name.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "nyaya")
dbutils.widgets.text("landing_volume", "landing")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
BRONZE = f"{CATALOG}.{SCHEMA}.credit_bronze"
VOLUME = dbutils.widgets.get("landing_volume")
LANDING = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
files = [f.name for f in dbutils.fs.ls(LANDING) if f.name.endswith(".csv")]
print(f"{len(files)} CSV file(s) in {LANDING}: {files}")
if not files and not spark.catalog.tableExists(BRONZE):
    raise FileNotFoundError(f"Upload data/raw/german_credit.csv to {LANDING} first (see docs/DATABRICKS.md).")

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {BRONZE}
COMMENT 'Loan applications as received (Statlog German Credit schema). Append-only.'
""")

result = spark.sql(f"""
COPY INTO {BRONZE}
FROM (
  SELECT *, current_timestamp() AS _ingested_at, _metadata.file_name AS _source_file
  FROM '{LANDING}'
)
FILEFORMAT = CSV
PATTERN = '*.csv'
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true')
""").first()

rows_loaded = int(result["num_inserted_rows"]) if result is not None else 0
print(f"rows loaded this run: {rows_loaded}")

# Read by the job's "has_new_data" condition: training only runs when new rows arrived.
dbutils.jobs.taskValues.set(key="rows_loaded", value=rows_loaded)

# COMMAND ----------

display(spark.sql(f"""
SELECT _source_file, min(_ingested_at) AS loaded_at, count(*) AS rows
FROM {BRONZE} GROUP BY _source_file ORDER BY loaded_at
"""))
