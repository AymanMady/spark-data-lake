"""
Tests of the Bronze -> Silver and Silver -> Gold transformations (phase 15).

Every test uses hand-written DataFrames of a few rows: a test has to be fast,
readable, and its result computable in your head.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.spark.bronze_to_silver import check_schema, clean_email, clean_text, parse_flexible_date
from src.spark.silver_to_gold import (
    build_fact,
    country_sales,
    customer_sales,
    daily_sales,
    product_sales,
)

# ---------------------------------------------------------------------------
# Cleaning building blocks
# ---------------------------------------------------------------------------

def test_clean_text_trims_spaces_and_normalises_case(spark):
    df = spark.createDataFrame(
        [("  france ",), ("FRANCE",), ("france",), ("  jean-pierre  ",)], ["v"])
    result = [r.c for r in df.select(clean_text("v").alias("c")).collect()]
    assert result[:3] == ["France", "France", "France"], \
        "three different spellings must collapse into ONE value"
    # initcap only capitalises AFTER a space: the hyphen is not a separator.
    # This test locks that behaviour in so it surprises nobody.
    assert result[3] == "Jean-pierre"


def test_clean_text_preserves_nulls(spark):
    df = spark.createDataFrame([(None,)], StructType([StructField("v", StringType())]))
    assert df.select(clean_text("v").alias("c")).collect()[0].c is None


def test_clean_email_normalises_valid_emails(spark):
    df = spark.createDataFrame([("  JEAN.DUPONT@Gmail.COM  ",)], ["email"])
    assert df.select(clean_email("email").alias("e")).collect()[0].e == "jean.dupont@gmail.com"


@pytest.mark.parametrize("invalid", [
    "jean.dupont.gmail.com",   # no at-sign
    "jean@",                   # no domain
    "@gmail.com",              # no local part
    "jean@gmail",              # no extension
    "",                        # empty
])
def test_clean_email_nulls_out_invalid_emails(spark, invalid):
    df = spark.createDataFrame([(invalid,)], ["email"])
    assert df.select(clean_email("email").alias("e")).collect()[0].e is None


def test_parse_flexible_date_handles_several_formats(spark):
    df = spark.createDataFrame(
        [("2024-03-15",), ("15/03/2024",), ("not a date",), (None,)],
        StructType([StructField("d", StringType())]))
    parsed = df.select(
        parse_flexible_date("d", ["yyyy-MM-dd", "dd/MM/yyyy"]).cast("date").alias("p")
    ).collect()
    assert parsed[0].p == dt.date(2024, 3, 15)
    assert parsed[1].p == dt.date(2024, 3, 15), "both formats must converge"
    assert parsed[2].p is None, "an unreadable date becomes NULL, without crashing"
    assert parsed[3].p is None


def test_check_schema_detects_a_missing_column(spark):
    df = spark.createDataFrame([(1, "a")], ["order_id", "status"])
    check_schema(df, ["order_id", "status"], "orders")  # must not raise
    with pytest.raises(ValueError, match="columns missing"):
        check_schema(df, ["order_id", "status", "quantity"], "orders")


def test_deduplication_on_the_business_key(spark):
    df = spark.createDataFrame([(1, "a"), (1, "a"), (2, "b")], ["order_id", "v"])
    assert df.count() == 3
    assert df.dropDuplicates(["order_id"]).count() == 2


# ---------------------------------------------------------------------------
# Gold layer
# ---------------------------------------------------------------------------

ORDERS_SCHEMA = StructType([
    StructField("order_id", LongType()), StructField("customer_id", LongType()),
    StructField("product_id", LongType()), StructField("quantity", IntegerType()),
    StructField("order_date", DateType()), StructField("status", StringType()),
])
PRODUCTS_SCHEMA = StructType([
    StructField("product_id", LongType()), StructField("product_name", StringType()),
    StructField("category", StringType()), StructField("price", DoubleType()),
])
CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id", LongType()), StructField("country", StringType()),
])

D1, D2 = dt.date(2024, 1, 1), dt.date(2024, 1, 2)


@pytest.fixture
def silver_dir(spark, tmp_path):
    """Write a tiny Silver data lake to disk, then return its path."""
    orders = spark.createDataFrame([
        (1, 10, 100, 2, D1, "delivered"),   # 2 x 10.0 = 20.0
        (2, 10, 101, 1, D1, "shipped"),     # 1 x 5.0  =  5.0
        (3, 11, 100, 3, D2, "cancelled"),   # cancelled ->  0.0
        (4, 11, 100, 1, D2, "delivered"),   # 1 x 10.0 = 10.0
        (5, 99, 100, 1, D2, "delivered"),   # unknown customer -> lost in the join
    ], ORDERS_SCHEMA)
    products = spark.createDataFrame([
        (100, "Lamp", "Home", 10.0),
        (101, "Mug", "Home", 5.0),
    ], PRODUCTS_SCHEMA)
    customers = spark.createDataFrame([
        (10, "France"), (11, "Spain"),
    ], CUSTOMERS_SCHEMA)

    orders.write.parquet(str(tmp_path / "orders"))
    products.write.parquet(str(tmp_path / "products"))
    customers.write.parquet(str(tmp_path / "customers"))
    return tmp_path


def test_build_fact_joins_and_computes_revenue(spark, silver_dir):
    fact = build_fact(spark, silver_dir).collect()
    assert len(fact) == 4, "the unknown customer's order is dropped by the inner join"
    revenues = {r.order_id: r.revenue for r in fact}
    assert revenues == {1: 20.0, 2: 5.0, 3: 0.0, 4: 10.0}


def test_revenue_excludes_cancelled_orders(spark, silver_dir):
    fact = build_fact(spark, silver_dir)
    cancelled = fact.filter(F.col("status") == "cancelled").collect()[0]
    assert cancelled.revenue == 0.0
    assert cancelled.quantity == 3, "the quantity is still counted, only revenue is zeroed"


def test_daily_sales(spark, silver_dir):
    rows = {r["date"]: r for r in daily_sales(build_fact(spark, silver_dir)).collect()}
    assert rows[D1].orders_count == 2
    assert rows[D1].total_quantity == 3
    assert rows[D1].total_revenue == 25.0
    assert rows[D1].average_order_value == 12.5
    assert rows[D2].total_revenue == 10.0, "the cancelled order brings in nothing"
    assert rows[D2].orders_count == 2, "but it does count as an order"


def test_product_sales(spark, silver_dir):
    rows = {r.product_id: r for r in product_sales(build_fact(spark, silver_dir)).collect()}
    assert rows[100].quantity_sold == 6
    assert rows[100].revenue == 30.0
    assert rows[100].category == "Home"
    assert rows[101].revenue == 5.0


def test_customer_sales(spark, silver_dir):
    rows = {r.customer_id: r for r in customer_sales(build_fact(spark, silver_dir)).collect()}
    assert rows[10].orders_count == 2
    assert rows[10].total_spent == 25.0
    assert rows[10].average_order_value == 12.5
    assert 99 not in rows, "a customer absent from Silver must not show up"


def test_country_sales(spark, silver_dir):
    rows = {r.country: r for r in country_sales(build_fact(spark, silver_dir)).collect()}
    assert rows["France"].revenue == 25.0
    assert rows["Spain"].revenue == 10.0


def test_the_average_does_not_use_an_already_rounded_total(spark, tmp_path):
    """
    Regression: computing average_order_value from the ALREADY rounded revenue
    gave a one-cent gap (a real bug hit on 449 customers).
    3 orders at 10.005 -> total 30.02 (rounded), exact average 10.01.
    Double rounding would also give 30.02 / 3 = 10.0067 -> 10.01, so we pick a
    case where the two really diverge: 0.335 x 3.
    """
    orders = spark.createDataFrame(
        [(1, 10, 100, 1, D1, "delivered"),
         (2, 10, 100, 1, D1, "delivered"),
         (3, 10, 100, 1, D1, "delivered")], ORDERS_SCHEMA)
    products = spark.createDataFrame([(100, "X", "C", 0.335)], PRODUCTS_SCHEMA)
    customers = spark.createDataFrame([(10, "France")], CUSTOMERS_SCHEMA)
    orders.write.parquet(str(tmp_path / "orders"))
    products.write.parquet(str(tmp_path / "products"))
    customers.write.parquet(str(tmp_path / "customers"))

    row = customer_sales(build_fact(spark, tmp_path)).collect()[0]
    raw = 0.335 * 3
    assert row.average_order_value == pytest.approx(round(raw / 3, 2))
    assert row.average_order_value == pytest.approx(0.34, abs=0.001)


def test_an_empty_dataframe_does_not_break_the_aggregates(spark, tmp_path):
    """A pipeline must survive a day with no order at all."""
    spark.createDataFrame([], ORDERS_SCHEMA).write.parquet(str(tmp_path / "orders"))
    spark.createDataFrame([(100, "X", "C", 1.0)], PRODUCTS_SCHEMA) \
        .write.parquet(str(tmp_path / "products"))
    spark.createDataFrame([(10, "France")], CUSTOMERS_SCHEMA) \
        .write.parquet(str(tmp_path / "customers"))

    fact = build_fact(spark, tmp_path)
    assert fact.count() == 0
    assert daily_sales(fact).count() == 0
    assert country_sales(fact).count() == 0
    assert customer_sales(fact).count() == 0
