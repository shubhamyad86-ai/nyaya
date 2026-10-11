"""Medallion (bronze -> silver -> gold) pipeline for the credit data with PySpark and Delta Lake.

Runs unchanged on Databricks (as a job or notebook) and locally with delta-spark:

    python spark/credit_pipeline.py --base-path data/delta            # local
    python spark/credit_pipeline.py --base-path /Volumes/main/risk/credit   # Databricks

Layers
- bronze/credit_raw          raw CSV rows + ingestion metadata, append-only
- silver/credit_clean        validated, typed rows with engineered features
- silver/credit_quarantine   rows that failed a data-quality rule, with the reason
- gold/credit_features       model-ready feature table (same features as the pandas pipeline)
- gold/risk_by_segment       default rate by loan purpose and age group, for reporting
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]

# Same rules as src/compliance_ai/credit/data.py
RANGES = {
    "duration": (1, 120), "amount": (1, 100_000), "age": (18, 100),
    "installment_rate": (1, 4), "present_residence": (1, 4),
    "number_credits": (1, 10), "people_liable": (1, 5),
}


def get_spark(fmt: str = "delta") -> SparkSession:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return SparkSession.builder.getOrCreate()      # Delta is built in on Databricks
    if fmt != "delta":
        return (SparkSession.builder.appName("credit-medallion").master("local[*]")
                .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false")
                .getOrCreate())
    # Downloads the Delta Lake jars from Maven Central on first run.
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder.appName("credit-medallion")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def bronze(spark: SparkSession, source: str) -> DataFrame:
    return (
        spark.read.option("header", True).option("inferSchema", True).csv(source)
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.lit(Path(source).name))
    )


def quality_issues(df: DataFrame) -> F.Column:
    """One string describing every failed rule, or null when the row is clean."""
    checks = [F.when(F.col(c).isNull() | ~F.col(c).between(lo, hi), F.lit(f"{c} out of range"))
              for c, (lo, hi) in RANGES.items()]
    checks.append(F.when(~F.col("credit_risk").isin(0, 1), F.lit("invalid credit_risk")))
    return F.concat_ws("; ", *checks)


def silver(raw: DataFrame) -> tuple[DataFrame, DataFrame]:
    checked = raw.withColumn("_issues", quality_issues(raw))
    quarantine = checked.filter(F.col("_issues") != "")
    clean = (
        checked.filter(F.col("_issues") == "").drop("_issues")
        .withColumn("default", (1 - F.col("credit_risk")).cast("int"))
        .withColumn("sex", F.when(F.col("personal_status_sex").startswith("female"), "female")
                    .otherwise("male"))
        .withColumn("age_group", F.when(F.col("age") < 25, "under_25").otherwise("25_plus"))
        .withColumn("amount_per_month", F.col("amount") / F.col("duration"))
        .withColumn("log_amount", F.log1p("amount"))
        .withColumn("has_checking_account", (F.col("status") != "no checking account").cast("int"))
        .withColumn("no_savings", F.lower("savings").rlike("unknown|no savings").cast("int"))
        .withColumn("delayed_before", F.lower("credit_history").contains("delay").cast("int"))
        .withColumn("applicant_id", F.monotonically_increasing_id())
    )
    return clean, quarantine


def gold(clean: DataFrame) -> tuple[DataFrame, DataFrame]:
    features = clean.drop("_ingested_at", "_source_file")
    by_segment = (
        clean.groupBy("purpose", "age_group")
        .agg(F.count("*").alias("applicants"),
             F.round(F.avg("default"), 3).alias("default_rate"),
             F.round(F.avg("amount"), 0).alias("avg_amount"),
             F.round(F.avg("duration"), 1).alias("avg_duration_months"))
        .orderBy(F.desc("applicants"))
    )
    return features, by_segment


FORMAT = "delta"


def write(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    df.write.format(FORMAT).mode(mode).option("overwriteSchema", "true").save(path)


def read(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.format(FORMAT).load(path)


def run(source: str, base: str, fmt: str = "delta") -> dict:
    """fmt="parquet" runs the same logic without Delta (e.g. when Maven Central is unreachable)."""
    global FORMAT
    FORMAT = fmt
    spark = get_spark(fmt)
    raw = bronze(spark, source)
    write(raw, f"{base}/bronze/credit_raw", mode="append")

    # Silver reads back from bronze (latest ingestion only) - the table is the contract.
    bronze_tbl = read(spark, f"{base}/bronze/credit_raw")
    latest = bronze_tbl.agg(F.max("_ingested_at")).first()[0]
    clean, quarantine = silver(bronze_tbl.filter(F.col("_ingested_at") == latest))
    write(clean, f"{base}/silver/credit_clean")
    write(quarantine, f"{base}/silver/credit_quarantine")

    features, by_segment = gold(read(spark, f"{base}/silver/credit_clean"))
    write(features, f"{base}/gold/credit_features")
    write(by_segment, f"{base}/gold/risk_by_segment")

    stats = {
        "bronze_rows_this_run": raw.count(),
        "silver_clean_rows": clean.count(),
        "quarantined_rows": quarantine.count(),
        "gold_feature_rows": features.count(),
        "format": fmt,
    }
    if fmt == "delta":   # Delta keeps a versioned transaction log (time travel / audit trail)
        stats["delta_versions_bronze"] = spark.sql(
            f"DESCRIBE HISTORY delta.`{base}/bronze/credit_raw`").count()
    print(stats)
    by_segment.show(8, truncate=False)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(ROOT / "data/raw/german_credit.csv"))
    parser.add_argument("--base-path", default=str(ROOT / "data/delta"))
    parser.add_argument("--format", choices=["delta", "parquet"], default="delta")
    args = parser.parse_args()
    run(args.source, args.base_path, args.format)
