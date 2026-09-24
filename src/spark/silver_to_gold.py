"""
SILVER -> GOLD (phases 8, 9, 11, 12).

THE ROLE OF THE GOLD LAYER: one table = one business question.

Detail is deliberately lost here (that is what an aggregate is for) in
exchange for read speed. A dashboard querying Gold reads a few thousand rows
instead of a million orders.

This job demonstrates four ideas at once:

  JOIN            enrich the orders with the price (products) and the country
                  (customers).
  BROADCAST JOIN  products (66 KB) and customers (928 KB) are tiny next to the
                  orders: we broadcast them to every executor to remove the
                  shuffle on the left-hand side.
  CACHE           the enriched fact table is read FOUR times (once per Gold
                  table). Without the cache, Spark would redo both joins four
                  times.
  SPARK SQL       --engine sql produces exactly the same results through
                  temporary views and SQL.

Usage:
    spark-submit src/spark/silver_to_gold.py
    spark-submit src/spark/silver_to_gold.py --engine sql
    spark-submit src/spark/silver_to_gold.py --no-broadcast --no-cache --explain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.spark.schemas import REVENUE_STATUSES
from src.spark.spark_session import get_spark
from src.spark.utils import GOLD, SILVER, get_logger, human_bytes, path_size, timer

log = get_logger("gold")

GOLD_TABLES = ["daily_sales", "product_sales", "customer_sales", "country_sales"]


# ---------------------------------------------------------------------------
# Enriched fact table
# ---------------------------------------------------------------------------

def build_fact(spark: SparkSession, silver: Path, use_broadcast: bool = True) -> DataFrame:
    """
    Build the fact table: orders + price + country.

    Why an INNER JOIN and not a LEFT JOIN?
    An order whose product is unknown has no price: its revenue would be NULL
    and would pollute every sum. We prefer to lose those rows explicitly (they
    are counted during the Silver phase) rather than produce wrong aggregates.
    """
    orders = spark.read.parquet(str(silver / "orders"))
    products = spark.read.parquet(str(silver / "products"))
    customers = spark.read.parquet(str(silver / "customers"))

    # Read only the useful columns: with Parquet (a COLUMNAR format), Spark
    # physically pulls only those columns off the disk.
    # This is "column pruning", impossible with CSV.
    products = products.select("product_id", "product_name", "category", "price")
    customers = customers.select("customer_id", "country")

    if use_broadcast:
        # broadcast() = "send this whole table to every executor".
        # Spark then switches from a SortMergeJoin (2 shuffles) to a
        # BroadcastHashJoin (0 shuffle on the orders side).
        # DO NOT do this when the broadcast table is large: it has to fit in
        # the memory of the driver AND of every executor.
        products = F.broadcast(products)
        customers = F.broadcast(customers)

    fact = (
        orders
        .join(products, "product_id", "inner")
        .join(customers, "customer_id", "inner")
        # Revenue only counts orders that were actually fulfilled.
        # A cancelled order exists, but brings in nothing.
        .withColumn(
            "revenue",
            F.when(F.col("status").isin(REVENUE_STATUSES),
                   F.col("quantity") * F.col("price")).otherwise(F.lit(0.0)),
        )
        .withColumn("is_revenue", F.col("status").isin(REVENUE_STATUSES))
    )
    return fact


# ---------------------------------------------------------------------------
# Aggregates - DataFrame API version
# ---------------------------------------------------------------------------

def daily_sales(fact: DataFrame) -> DataFrame:
    """
    Performance note: this uses count("order_id"), not countDistinct("order_id").
    After the Silver deduplication, order_id is already unique: countDistinct
    would cost an extra shuffle for an identical result.

    THE DOUBLE-ROUNDING TRAP (a real bug hit while building this project):
    computing the average from an ALREADY rounded total_revenue gives a
    different result from the average computed on the raw sum. The gap was
    0.01 EUR for 449 customers out of 40,052. So the unrounded sum
    (_revenue_raw) is kept for every derived calculation, and rounding happens
    only at display time.
    """
    return (
        fact.groupBy(F.col("order_date").alias("date"))
        .agg(
            F.count("order_id").alias("orders_count"),
            F.sum("quantity").alias("total_quantity"),
            F.sum("revenue").alias("_revenue_raw"),
        )
        .select(
            "date",
            "orders_count",
            "total_quantity",
            F.round("_revenue_raw", 2).alias("total_revenue"),
            F.round(F.col("_revenue_raw") / F.col("orders_count"), 2)
                .alias("average_order_value"),
        )
        .orderBy("date")
    )


def product_sales(fact: DataFrame) -> DataFrame:
    return (
        fact.groupBy("product_id", "product_name", "category")
        .agg(
            F.count("order_id").alias("orders_count"),
            F.sum("quantity").alias("quantity_sold"),
            F.round(F.sum("revenue"), 2).alias("revenue"),
        )
        .orderBy(F.col("revenue").desc())
    )


def customer_sales(fact: DataFrame) -> DataFrame:
    # Same precaution as daily_sales: the average is computed on the raw sum,
    # never on the already-rounded one (see the double-rounding trap).
    return (
        fact.groupBy("customer_id")
        .agg(
            F.count("order_id").alias("orders_count"),
            F.sum("revenue").alias("_revenue_raw"),
        )
        .select(
            "customer_id",
            "orders_count",
            F.round("_revenue_raw", 2).alias("total_spent"),
            F.round(F.col("_revenue_raw") / F.col("orders_count"), 2)
                .alias("average_order_value"),
        )
        .orderBy(F.col("total_spent").desc())
    )


def country_sales(fact: DataFrame) -> DataFrame:
    return (
        fact.groupBy("country")
        .agg(
            F.count("order_id").alias("orders_count"),
            F.round(F.sum("revenue"), 2).alias("revenue"),
        )
        .orderBy(F.col("revenue").desc())
    )


BUILDERS = {
    "daily_sales": daily_sales,
    "product_sales": product_sales,
    "customer_sales": customer_sales,
    "country_sales": country_sales,
}


# ---------------------------------------------------------------------------
# Aggregates - Spark SQL version (phase 9)
# ---------------------------------------------------------------------------

SQL_QUERIES = {
    "daily_sales": """
        SELECT
            order_date                                   AS date,
            COUNT(order_id)                              AS orders_count,
            SUM(quantity)                                AS total_quantity,
            ROUND(SUM(revenue), 2)                       AS total_revenue,
            ROUND(SUM(revenue) / COUNT(order_id), 2)     AS average_order_value
        FROM fact
        GROUP BY order_date
        ORDER BY date
    """,
    "product_sales": """
        SELECT
            product_id,
            product_name,
            category,
            COUNT(order_id)         AS orders_count,
            SUM(quantity)           AS quantity_sold,
            ROUND(SUM(revenue), 2)  AS revenue
        FROM fact
        GROUP BY product_id, product_name, category
        ORDER BY revenue DESC
    """,
    "customer_sales": """
        SELECT
            customer_id,
            COUNT(order_id)                              AS orders_count,
            ROUND(SUM(revenue), 2)                       AS total_spent,
            ROUND(SUM(revenue) / COUNT(order_id), 2)     AS average_order_value
        FROM fact
        GROUP BY customer_id
        ORDER BY total_spent DESC
    """,
    "country_sales": """
        SELECT
            country,
            COUNT(order_id)         AS orders_count,
            ROUND(SUM(revenue), 2)  AS revenue
        FROM fact
        GROUP BY country
        ORDER BY revenue DESC
    """,
}


def build_with_sql(spark: SparkSession, fact: DataFrame, table: str) -> DataFrame:
    """
    Same result, written in SQL.

    createOrReplaceTempView exposes a DataFrame as a SQL table. The view is
    TEMPORARY: it lives for the SparkSession and stores nothing.

    Key point: the DataFrame API and SQL go through the SAME Catalyst optimiser
    and produce the SAME physical plan. The choice is about readability and
    team skills, not performance.
    """
    fact.createOrReplaceTempView("fact")
    return spark.sql(SQL_QUERIES[table])


# ---------------------------------------------------------------------------

def write_gold(df: DataFrame, table: str, output: Path) -> None:
    target = output / table
    with timer(log, f"Writing gold/{table}"):
        # coalesce(1): the Gold tables are small (a few thousand rows).
        # Writing them as 8 files buys nothing and complicates the read from
        # PostgreSQL. Unlike repartition(1), coalesce triggers NO shuffle: it
        # just merges existing partitions in place.
        df.coalesce(1).write.mode("overwrite").parquet(str(target))
    log.info(f"gold/{table}: {human_bytes(path_size(target))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SILVER -> GOLD aggregation")
    parser.add_argument("--engine", choices=["dataframe", "sql"], default="dataframe",
                        help="DataFrame API or Spark SQL (identical results)")
    parser.add_argument("--tables", nargs="+", default=GOLD_TABLES, choices=GOLD_TABLES)
    parser.add_argument("--silver", type=Path, default=SILVER)
    parser.add_argument("--output", type=Path, default=GOLD)
    parser.add_argument("--no-broadcast", action="store_true",
                        help="Disable the broadcast join (for comparison)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the fact table cache")
    parser.add_argument("--explain", action="store_true",
                        help="Print the physical execution plan")
    args = parser.parse_args(argv)

    spark = get_spark(f"silver-to-gold-{args.engine}")
    log.info(f"Engine: {args.engine} | broadcast: {not args.no_broadcast} | "
             f"cache: {not args.no_cache}")

    fact = build_fact(spark, args.silver, use_broadcast=not args.no_broadcast)

    if args.explain:
        log.info("Execution plan of the fact table:")
        fact.explain("formatted")

    if not args.no_cache:
        # cache() is LAZY: nothing is stored until an action has walked the
        # data. The count() below materialises the cache.
        fact.cache()
        with timer(log, "Materialising the cache (first pass)"):
            n = fact.count()
        log.info(f"Fact table: {n:,} rows cached")

    total = 0
    for table in args.tables:
        log.info("-" * 60)
        log.info(f"Creating {table}")
        df = BUILDERS[table](fact) if args.engine == "dataframe" \
            else build_with_sql(spark, fact, table)
        if args.explain and table == args.tables[0]:
            df.explain("formatted")
        write_gold(df, table, args.output)
        total += 1

    if not args.no_cache:
        # Free the memory: a forgotten cache starves the next jobs of RAM.
        fact.unpersist()

    log.info("=" * 60)
    log.info(f"{total} Gold tables written to {args.output}")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
