# Databricks notebook source
# MAGIC %md
# MAGIC # 1 · Ingest: landing volume → `credit_bronze`
# MAGIC
# MAGIC New CSV files dropped into the landing volume are loaded with `COPY INTO`, which remembers the
# MAGIC files it has already loaded. Re-running the job never loads the same file twice, so the step is
# MAGIC safe to retry. Bronze keeps every row as received, plus the load time and the file name.
# MAGIC
# MAGIC Every column is cast to a fixed type instead of letting Spark guess per file. Guessing breaks
# MAGIC as soon as two files look slightly different (one file's `duration` read as INT, the next as
# MAGIC BIGINT, and Delta refuses to merge them). A value that cannot be cast becomes NULL, so the row
# MAGIC is caught by the quality rules in silver instead of failing the whole load.

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

from common import CATEGORIES, RANGES, TARGET

# The applicant columns and their types: numbers as INT, categories as STRING, the outcome as INT.
COLUMNS = {**{c: "STRING" for c in CATEGORIES}, **{c: "INT" for c in RANGES}, TARGET: "INT"}

if spark.catalog.tableExists(BRONZE) and spark.table(BRONZE).columns:
    # Keep whatever types the existing table already has, so new loads always match it.
    existing = {f.name: f.dataType.simpleString().upper() for f in spark.table(BRONZE).schema}
    COLUMNS = {c: existing.get(c, t) for c, t in COLUMNS.items()}
else:
    ddl = ", ".join(f"{c} {t}" for c, t in COLUMNS.items())
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {BRONZE} ({ddl}, _ingested_at TIMESTAMP, _source_file STRING)
    COMMENT 'Loan applications as received (Statlog German Credit schema). Append-only.'
    """)

select = ",\n    ".join(f"try_cast({c} AS {t}) AS {c}" for c, t in COLUMNS.items())
result = spark.sql(f"""
COPY INTO {BRONZE}
FROM (
  SELECT
    {select},
    current_timestamp() AS _ingested_at,
    _metadata.file_name AS _source_file
  FROM '{LANDING}'
)
FILEFORMAT = CSV
PATTERN = '*.csv'
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
""").first()

rows_loaded = int(result["num_inserted_rows"]) if result is not None else 0
print(f"rows loaded this run: {rows_loaded}")

# Read by the train task: it only retrains when new rows arrived.
dbutils.jobs.taskValues.set(key="rows_loaded", value=rows_loaded)

# COMMAND ----------

display(spark.sql(f"""
SELECT _source_file, min(_ingested_at) AS loaded_at, count(*) AS rows
FROM {BRONZE} GROUP BY _source_file ORDER BY loaded_at
"""))
