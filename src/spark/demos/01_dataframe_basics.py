"""
DEMO 1 - The basic Spark DataFrame operations (phase 4).

Covers: select, filter/where, withColumn, drop, distinct,
        groupBy, agg, join, orderBy.

Run with: make demo-basics
"""

from __future__ import annotations

from pyspark.sql import functions as F

from src.spark.spark_session import describe_session, get_spark
from src.spark.utils import SILVER, get_logger

log = get_logger("demo")


def section(title: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info(f"  {title}")
    log.info("=" * 68)


def main() -> int:
    spark = get_spark("demo-dataframe-basics")
    for key, value in describe_session(spark).items():
        log.info(f"  {key:<22}: {value}")

    orders = spark.read.parquet(str(SILVER / "orders"))
    customers = spark.read.parquet(str(SILVER / "customers"))
    products = spark.read.parquet(str(SILVER / "products"))

    section("1. select() - pick columns")
    log.info("With Parquet (a columnar format), select() physically reads ONLY")
    log.info("the requested columns. This is 'column pruning'.")
    orders.select("order_id", "customer_id", "quantity").show(5, truncate=False)

    section("2. filter() / where() - filter rows")
    log.info("filter() and where() are STRICTLY identical: where is an alias")
    log.info("provided for people coming from SQL.")
    big = orders.filter(F.col("quantity") >= 4)
    same = orders.where("quantity >= 4")
    log.info(f"filter: {big.count():,} rows | where: {same.count():,} rows")

    section("3. withColumn() - add or replace a column")
    log.info("withColumn does not MODIFY the DataFrame: it returns a new one.")
    log.info("Spark DataFrames are IMMUTABLE.")
    enriched = (
        orders
        .withColumn("order_year", F.year("order_date"))
        .withColumn("is_weekend", F.dayofweek("order_date").isin([1, 7]))
    )
    enriched.select("order_id", "order_date", "order_year", "is_weekend").show(5)

    section("4. drop() - remove columns")
    log.info(f"Before: {orders.columns}")
    log.info(f"After : {orders.drop('order_ts', 'order_year', 'order_month').columns}")

    section("5. distinct() - unique values")
    log.info("distinct() triggers a SHUFFLE: Spark has to compare rows that sit")
    log.info("on different partitions.")
    statuses = orders.select("status").distinct()
    log.info(f"Distinct statuses: {[r.status for r in statuses.collect()]}")

    section("6. groupBy() + agg() - aggregate")
    log.info("groupBy on its own computes nothing: it needs an agg() after it.")
    (orders.groupBy("status")
           .agg(F.count("*").alias("orders_count"),
                F.sum("quantity").alias("total_quantity"),
                F.round(F.avg("quantity"), 2).alias("average_quantity"))
           .orderBy(F.col("orders_count").desc())
           .show(truncate=False))

    section("7. join() - combine two DataFrames")
    log.info("Joining orders to products to get the price.")
    joined = orders.join(products, "product_id", "inner")
    log.info(f"orders: {orders.count():,} | after the join: {joined.count():,}")
    log.info("The difference comes from the orders whose product was rejected")
    log.info("in Silver (invalid price): the INNER JOIN drops them.")
    joined.select("order_id", "product_name", "category", "quantity", "price") \
        .show(5, truncate=False)

    section("8. orderBy() - sort")
    log.info("CAREFUL: orderBy is a GLOBAL operation. To sort, Spark has to")
    log.info("gather the data -> SHUFFLE + reduced parallelism.")
    log.info("Sorting a whole dataset only to display 10 rows is a classic")
    log.info("waste. Here it is justified: we aggregate first.")
    (joined.groupBy("category")
           .agg(F.round(F.sum(F.col("quantity") * F.col("price")), 2).alias("revenue"))
           .orderBy(F.col("revenue").desc())
           .show(truncate=False))

    section("9. A full chain")
    log.info("The 5 best French customers, in a single chain.")
    top = (
        orders
        .join(F.broadcast(customers), "customer_id")
        .join(F.broadcast(products), "product_id")
        .where((F.col("country") == "France") & (F.col("status") == "delivered"))
        .withColumn("revenue", F.col("quantity") * F.col("price"))
        .groupBy("customer_id", "first_name", "last_name")
        .agg(F.round(F.sum("revenue"), 2).alias("total_spent"),
             F.count("*").alias("orders_count"))
        .orderBy(F.col("total_spent").desc())
        .limit(5)
    )
    top.show(truncate=False)

    log.info("")
    log.info("TAKEAWAY: everything above, except show/count/collect, was only")
    log.info("PLAN BUILDING. See demo 03 on lazy evaluation.")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
