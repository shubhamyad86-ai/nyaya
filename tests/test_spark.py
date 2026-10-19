"""Spark pipeline tests (Parquet mode, so no Maven download is needed). Skipped without Java or PySpark."""
import importlib.util
import shutil

import pytest

from compliance_ai.credit.data import add_features
from tests.conftest import CREDIT_CSV, require

pytestmark = pytest.mark.skipif(
    shutil.which("java") is None or importlib.util.find_spec("pyspark") is None,
    reason="Java or PySpark not installed")


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    require(CREDIT_CSV, "run scripts/download_data.py")
    import credit_pipeline

    base = str(tmp_path_factory.mktemp("lake"))
    stats = credit_pipeline.run(str(CREDIT_CSV), base, fmt="parquet")
    spark = credit_pipeline.get_spark("parquet")
    yield stats, base, spark


def test_row_counts(lake):
    stats, _, _ = lake
    assert stats["bronze_rows_this_run"] == 1000
    assert stats["silver_clean_rows"] + stats["quarantined_rows"] == 1000


def test_spark_features_match_pandas(lake, raw_credit):
    _, base, spark = lake
    import pandas as pd

    # collect() instead of toPandas(): PySpark 3.5's pandas bridge needs distutils (gone in Python 3.12+).
    sdf = pd.DataFrame([r.asDict() for r in spark.read.parquet(f"{base}/gold/credit_features").collect()])
    pdf = add_features(raw_credit)
    for col in ["amount_per_month", "log_amount", "has_checking_account", "no_savings", "delayed_before"]:
        assert sorted(sdf[col].round(6)) == sorted(pdf[col].round(6)), col
    assert (sdf["sex"] == "female").sum() == (pdf["sex"] == "female").sum()


def test_quarantine_catches_bad_rows(tmp_path, raw_credit):
    import credit_pipeline

    bad = raw_credit.copy()
    bad.loc[0, "age"] = 150
    src = tmp_path / "bad.csv"
    bad.to_csv(src, index=False)
    stats = credit_pipeline.run(str(src), str(tmp_path / "lake"), fmt="parquet")
    assert stats["quarantined_rows"] == 1
