"""
Tests of the data quality framework (phase 15).

The most important test in this file is
test_a_null_condition_counts_as_invalid: it is the SQL trap that lets broken
data pass for valid in most pipelines.
"""

from __future__ import annotations

import json

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.spark.data_quality import (
    ERROR_COL,
    QualityReport,
    Rule,
    apply_rules,
    assert_quality,
    count_failures,
    log_report,
    save_report,
    split_valid_invalid,
)
from src.spark.utils import get_logger

SCHEMA = StructType([
    StructField("id", IntegerType()),
    StructField("quantity", IntegerType()),
    StructField("status", StringType()),
])


@pytest.fixture
def rules(spark):
    """
    The rules have to be built AFTER Spark starts: F.col() requires an active
    SparkContext. Defining them at module level would break test COLLECTION,
    before a single test even ran.
    """
    return [
        Rule("id_not_null", F.col("id").isNotNull(), "primary key"),
        Rule("quantity_positive", F.col("quantity") > 0, "quantity > 0"),
        Rule("status_valid", F.col("status").isin(["ok", "ko"]), "known status"),
    ]


@pytest.fixture
def df(spark):
    return spark.createDataFrame([
        (1, 5, "ok"),        # valid
        (2, 0, "ok"),        # zero quantity -> invalid
        (3, -1, "ok"),       # negative quantity -> invalid
        (None, 2, "ok"),     # missing id -> invalid
        (5, None, "ok"),     # NULL quantity -> invalid (the trap)
        (6, 3, "unknown"),   # status outside the reference list -> invalid
        (7, 1, "ko"),        # valid
    ], SCHEMA)


def test_apply_rules_adds_the_error_column(spark, df, rules):
    annotated = apply_rules(df, rules)
    assert ERROR_COL in annotated.columns
    errors = {r.id: r[ERROR_COL] for r in annotated.collect()}
    assert errors[1] == [], "a valid row has no error"
    assert errors[2] == ["quantity_positive"]
    assert errors[6] == ["status_valid"]


def test_a_null_condition_counts_as_invalid(spark, df, rules):
    """
    In SQL, NULL > 0 is not False but NULL. Without the coalesce in _safe(),
    the row id=5 (quantity NULL) would trigger NO rule at all and would be
    considered valid. This is THE classic pipeline flaw.
    """
    annotated = apply_rules(df, rules)
    row = [r for r in annotated.collect() if r.id == 5][0]
    assert "quantity_positive" in row[ERROR_COL]


def test_one_row_can_violate_several_rules(spark, rules):
    bad = spark.createDataFrame([(None, -5, "xxx")], SCHEMA)
    row = apply_rules(bad, rules).collect()[0]
    assert set(row[ERROR_COL]) == {"id_not_null", "quantity_positive", "status_valid"}


def test_split_valid_invalid(spark, df, rules):
    valid, invalid = split_valid_invalid(apply_rules(df, rules))
    assert valid.count() == 2
    assert invalid.count() == 5
    assert ERROR_COL not in valid.columns, "the technical column does not reach Silver"
    assert ERROR_COL in invalid.columns, "the quarantine keeps the rejection reason"
    assert sorted(r.id for r in valid.collect()) == [1, 7]


def test_count_failures_counts_per_rule(spark, df, rules):
    failures = count_failures(apply_rules(df, rules), rules)
    assert failures == {
        "id_not_null": 1,
        "quantity_positive": 3,   # 0, -1 and NULL
        "status_valid": 1,
    }


def test_count_failures_on_an_empty_dataframe(spark, rules):
    empty = spark.createDataFrame([], SCHEMA)
    annotated = apply_rules(empty, rules)
    assert annotated.count() == 0
    assert count_failures(annotated, rules) == {
        "id_not_null": 0, "quantity_positive": 0, "status_valid": 0}


def test_quality_report_computes_the_ratio(spark):
    report = QualityReport("orders", rows_read=100, rows_valid=90, rows_invalid=10)
    assert report.valid_ratio == 0.9
    assert report.to_dict()["valid_ratio"] == 0.9


def test_quality_report_on_zero_rows_does_not_divide_by_zero():
    report = QualityReport("orders", rows_read=0, rows_valid=0)
    assert report.valid_ratio == 0.0


def test_save_report_writes_readable_json(tmp_path):
    report = QualityReport("orders", rows_read=10, rows_valid=8, rows_invalid=2,
                           failures={"quantity_positive": 2})
    path = save_report(report, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["dataset"] == "orders"
    assert data["failures_by_rule"]["quantity_positive"] == 2
    assert data["valid_ratio"] == 0.8


def test_assert_quality_blocks_below_the_threshold():
    log = get_logger("test")
    good = QualityReport("orders", rows_read=100, rows_valid=95)
    assert_quality(good, 0.80, log)  # must not raise

    bad = QualityReport("orders", rows_read=100, rows_valid=40)
    with pytest.raises(ValueError, match="pipeline stopped"):
        assert_quality(bad, 0.80, log)


def test_log_report_does_not_crash_without_failures():
    """A perfect report must print without error (a common edge case)."""
    log_report(get_logger("test"),
               QualityReport("products", rows_read=5, rows_valid=5, failures={}))


def test_duplicates_are_counted_separately_from_invalid_rows(spark, rules):
    """
    A duplicated row is not 'invalid': it is surplus. Mixing the two counters
    makes the quality report unreadable.
    """
    df = spark.createDataFrame([(1, 5, "ok"), (1, 5, "ok"), (2, 5, "ok")], SCHEMA)
    valid, invalid = split_valid_invalid(apply_rules(df, rules))
    assert invalid.count() == 0, "no row is invalid"
    before = valid.count()
    after = valid.dropDuplicates(["id"]).count()
    report = QualityReport("t", rows_read=3, rows_valid=after,
                           rows_invalid=0, rows_duplicated=before - after)
    assert report.rows_duplicated == 1
    assert report.rows_invalid == 0
