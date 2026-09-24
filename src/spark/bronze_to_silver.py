"""
BRONZE -> SILVER (phases 5 and 14).

THE ROLE OF THE SILVER LAYER: produce data you can trust.

For each dataset:
    1. read Bronze (Parquet, everything as text)
    2. check the expected schema
    3. clean          (whitespace, casing, emails, categories)
    4. type           (text -> long / double / date / timestamp)
    5. normalise      (multi-format dates, statuses)
    6. validate       (data quality rules, in a single pass)
    7. quarantine     (invalid rows are ISOLATED, not thrown away)
    8. deduplicate
    9. write Parquet

TWO DISTINCT POLICIES, not to be confused:

  REPAIR   a cosmetic anomaly does not invalidate the row.
           broken email -> NULL, missing country -> "Unknown".
           The order is kept: it counts towards revenue.

  REJECT   an anomaly that makes the row unusable.
           quantity <= 0, unreadable date, missing primary key.
           The row goes to quarantine and leaves the pipeline.

Usage:
    spark-submit src/spark/bronze_to_silver.py
    spark-submit src/spark/bronze_to_silver.py --datasets orders
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.spark.data_quality import (
    QualityReport,
    Rule,
    apply_rules,
    assert_quality,
    count_failures,
    log_report,
    save_report,
    split_valid_invalid,
)
from src.spark.schemas import VALID_STATUSES
from src.spark.spark_session import get_spark
from src.spark.utils import BRONZE, DATA_DIR, SILVER, get_logger, human_bytes, path_size, timer

log = get_logger("silver")

QUARANTINE = DATA_DIR / "quarantine"
QUALITY_REPORTS = DATA_DIR / "quality"

# Minimum ratio of valid rows below which the pipeline stops.
MIN_VALID_RATIO = 0.80

EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"


# ---------------------------------------------------------------------------
# Reusable cleaning building blocks
# ---------------------------------------------------------------------------

def clean_text(col: str) -> F.Column:
    """
    Strip stray whitespace and normalise casing.

    initcap("  jean  ") -> "Jean". Without it, "FRANCE", "france" and " France "
    would be three different countries in a groupBy: the counts would be wrong.

    KNOWN LIMITATION (covered by the tests): initcap only capitalises after a
    SPACE. "jean-pierre" becomes "Jean-pierre" and "o'brien" becomes "O'brien".
    That is acceptable here: the goal is NORMALISATION (one spelling per
    value), not perfect typography. Two spellings of the same name do converge
    to the same string, and that is all a groupBy needs. If typography ever
    became a requirement, the answer would be a regex on the separators, not a
    Python UDF.
    """
    return F.initcap(F.trim(F.col(col)))


def parse_flexible_date(col: str, formats: list[str]) -> F.Column:
    """
    Try several date formats and keep the first one that works.

    to_timestamp returns NULL when the format does not match (Spark 3 in strict
    CORRECTED mode). coalesce therefore picks the first successful attempt.

    Rejected alternative: a Python UDF using dateutil. It would work, but a
    Python UDF forces Spark to serialise every row to a Python interpreter ->
    10 to 100x slower than a native function, and invisible to the Catalyst
    optimiser.
    """
    return F.coalesce(*[F.to_timestamp(F.col(col), fmt) for fmt in formats])


def clean_email(col: str) -> F.Column:
    """Normalise the email, then NULL if the result is not a valid address."""
    normalized = F.lower(F.trim(F.col(col)))
    return F.when(normalized.rlike(EMAIL_REGEX), normalized).otherwise(F.lit(None))


def check_schema(df: DataFrame, expected: list[str], name: str) -> None:
    """
    Check that the expected columns are present.

    Without this guard, a column renamed at the source would produce NULLs
    everywhere and zero revenue, without raising a single error.
    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: columns missing from Bronze -> {missing}")
    log.info(f"{name}: schema checked ({len(df.columns)} columns)")


def finalize(df_clean: DataFrame, rules: list[Rule], dedup_keys: list[str],
             name: str, rows_read: int) -> tuple[DataFrame, DataFrame, QualityReport, DataFrame]:
    """
    Apply the rules, split valid/invalid, deduplicate, produce the report.

    Performance note: the annotated DataFrame is cached because it is read FIVE
    times (counting failures, counting invalid rows, counting before and after
    deduplication, then the two writes). Without the cache, Spark would replay
    the whole cleaning on every read.

    The cached DataFrame is RETURNED instead of being freed here: the caller
    has to write Silver and the quarantine before releasing the memory.
    Freeing it too early would cancel the whole benefit of the cache at write
    time, which is precisely the most expensive step.
    """
    annotated = apply_rules(df_clean, rules).cache()

    failures = count_failures(annotated, rules)
    valid, invalid = split_valid_invalid(annotated)

    rows_invalid = invalid.count()
    before_dedup = valid.count()

    # dropDuplicates on the business key triggers a SHUFFLE: Spark has to
    # gather every row sharing a key onto the same partition.
    deduped = valid.dropDuplicates(dedup_keys)
    rows_valid = deduped.count()

    report = QualityReport(
        dataset=name,
        rows_read=rows_read,
        rows_valid=rows_valid,
        rows_invalid=rows_invalid,
        rows_duplicated=before_dedup - rows_valid,
        failures=failures,
    )
    return deduped, invalid, report, annotated


def write_silver(df: DataFrame, name: str, output: Path,
                 partition_by: list[str] | None = None) -> None:
    target = output / name
    if partition_by:
        # repartition BEFORE partitionBy: without it, every Spark partition
        # writes one file into EVERY partition directory.
        # 4 partitions x 24 months = 96 small files instead of 24.
        # This is the "small files problem", a classic data lake plague.
        writer = (df.repartition(*[F.col(col) for col in partition_by])
                    .write.mode("overwrite").partitionBy(*partition_by))
    else:
        writer = df.write.mode("overwrite")
    with timer(log, f"Writing silver/{name}"):
        writer.parquet(str(target))
    log.info(f"silver/{name}: {human_bytes(path_size(target))}")


def write_quarantine(df: DataFrame, name: str) -> None:
    target = QUARANTINE / name
    df.write.mode("overwrite").parquet(str(target))
    log.info(f"Quarantine: {target}")


# ---------------------------------------------------------------------------
# CUSTOMERS
# ---------------------------------------------------------------------------

def process_customers(spark: SparkSession, bronze: Path, output: Path) -> QualityReport:
    log.info("=" * 60)
    log.info("Cleaning customers")
    df = spark.read.parquet(str(bronze / "customers"))
    check_schema(df, ["customer_id", "first_name", "last_name", "email",
                      "country", "created_at"], "customers")
    rows_read = df.count()
    log.info(f"Rows before: {rows_read:,}")

    cleaned = df.select(
        # cast("long") returns NULL when the value is not numeric: that is
        # intended, the customer_id_not_null rule will catch it.
        F.col("customer_id").cast("long").alias("customer_id"),
        clean_text("first_name").alias("first_name"),
        clean_text("last_name").alias("last_name"),
        # REPAIR: a broken email is no reason to lose the customer.
        clean_email("email").alias("email"),
        # REPAIR: missing country -> "Unknown" rather than a NULL that would
        # silently vanish from a groupBy by country.
        F.coalesce(clean_text("country"), F.lit("Unknown")).alias("country"),
        parse_flexible_date("created_at", ["yyyy-MM-dd", "dd/MM/yyyy"])
            .cast("date").alias("created_at"),
        F.col("_ingested_at"),
    )

    rules = [
        Rule("customer_id_not_null", F.col("customer_id").isNotNull(),
             "The primary key is mandatory"),
        Rule("customer_id_positive", F.col("customer_id") > 0,
             "An identifier must be strictly positive"),
    ]

    valid, invalid, report, cached = finalize(
        cleaned, rules, ["customer_id"], "customers", rows_read)
    log.info(f"Rows after : {report.rows_valid:,}")
    log_report(log, report)

    write_silver(valid.drop("_ingested_at"), "customers", output)
    write_quarantine(invalid, "customers")
    cached.unpersist()   # the memory goes back to the next jobs
    return report


# ---------------------------------------------------------------------------
# PRODUCTS
# ---------------------------------------------------------------------------

def process_products(spark: SparkSession, bronze: Path, output: Path) -> QualityReport:
    log.info("=" * 60)
    log.info("Cleaning products")
    df = spark.read.parquet(str(bronze / "products"))
    check_schema(df, ["product_id", "product_name", "category", "price"], "products")
    rows_read = df.count()
    log.info(f"Rows before: {rows_read:,}")

    cleaned = df.select(
        F.col("product_id").cast("long").alias("product_id"),
        clean_text("product_name").alias("product_name"),
        clean_text("category").alias("category"),
        F.col("price").cast("double").alias("price"),
        F.col("_ingested_at"),
    )

    rules = [
        Rule("product_id_not_null", F.col("product_id").isNotNull(),
             "The primary key is mandatory"),
        Rule("price_not_null", F.col("price").isNotNull(),
             "A product without a price makes any revenue calculation impossible"),
        # REJECT: a price <= 0 would directly distort revenue.
        Rule("price_positive", F.col("price") > 0,
             "The price must be strictly positive"),
    ]

    valid, invalid, report, cached = finalize(cleaned, rules, ["product_id"], "products", rows_read)
    log.info(f"Rows after : {report.rows_valid:,}")
    log_report(log, report)

    write_silver(valid.drop("_ingested_at"), "products", output)
    write_quarantine(invalid, "products")
    cached.unpersist()
    return report


# ---------------------------------------------------------------------------
# ORDERS
# ---------------------------------------------------------------------------

def process_orders(spark: SparkSession, bronze: Path, output: Path) -> QualityReport:
    log.info("=" * 60)
    log.info("Cleaning orders")
    df = spark.read.parquet(str(bronze / "orders"))
    check_schema(df, ["order_id", "customer_id", "product_id", "quantity",
                      "order_date", "status"], "orders")
    rows_read = df.count()
    log.info(f"Rows before: {rows_read:,}")

    order_ts = parse_flexible_date("order_date", ["yyyy-MM-dd HH:mm:ss", "dd/MM/yyyy"])

    cleaned = (
        df.select(
            F.col("order_id").cast("long").alias("order_id"),
            F.col("customer_id").cast("long").alias("customer_id"),
            F.col("product_id").cast("long").alias("product_id"),
            F.col("quantity").cast("int").alias("quantity"),
            order_ts.alias("order_ts"),
            # The status is lowercased rather than initcap'd: it is a
            # technical enumeration value, not a displayable label.
            F.lower(F.trim(F.col("status"))).alias("status"),
            F.col("_ingested_at"),
        )
        .withColumn("order_date", F.to_date("order_ts"))
        # Physical partitioning columns (see write_silver).
        .withColumn("order_year", F.year("order_date"))
        .withColumn("order_month", F.month("order_date"))
    )

    rules = [
        Rule("order_id_not_null", F.col("order_id").isNotNull(), "Primary key"),
        Rule("customer_id_not_null", F.col("customer_id").isNotNull(),
             "Without a customer the order cannot be attributed"),
        Rule("product_id_not_null", F.col("product_id").isNotNull(),
             "Without a product the price cannot be known"),
        Rule("quantity_not_null", F.col("quantity").isNotNull(), "Quantity is mandatory"),
        Rule("quantity_positive", F.col("quantity") > 0,
             "A quantity <= 0 makes no business sense"),
        Rule("order_date_not_null", F.col("order_date").isNotNull(),
             "Unreadable or missing date: the order is unusable"),
        Rule("status_valid", F.col("status").isin(VALID_STATUSES),
             f"Status outside the reference list {VALID_STATUSES}"),
    ]

    valid, invalid, report, cached = finalize(cleaned, rules, ["order_id"], "orders", rows_read)
    log.info(f"Rows after : {report.rows_valid:,}")
    log_report(log, report)

    write_silver(valid.drop("_ingested_at"), "orders", output,
                 partition_by=["order_year", "order_month"])
    write_quarantine(invalid, "orders")
    cached.unpersist()
    return report


# ---------------------------------------------------------------------------
# Referential integrity (done afterwards: it needs all 3 datasets)
# ---------------------------------------------------------------------------

def check_referential_integrity(spark: SparkSession, output: Path) -> dict:
    """
    Find the orders that reference a non-existent customer or product.

    Technique: LEFT ANTI JOIN. A join that keeps only the left rows WITHOUT a
    match on the right. Far more efficient than a left join followed by an
    isNull filter: Spark stops at the first match found and never materialises
    the right-hand columns.
    """
    log.info("=" * 60)
    log.info("Checking referential integrity")
    orders = spark.read.parquet(str(output / "orders"))
    customers = spark.read.parquet(str(output / "customers")).select("customer_id")
    products = spark.read.parquet(str(output / "products")).select("product_id")

    orphan_customers = orders.join(customers, "customer_id", "left_anti").count()
    orphan_products = orders.join(products, "product_id", "left_anti").count()
    total = orders.count()

    log.info(f"Orders with a non-existent customer: {orphan_customers:,} "
             f"({100 * orphan_customers / total:.2f}%)")
    log.info(f"Orders with a non-existent product : {orphan_products:,} "
             f"({100 * orphan_products / total:.2f}%)")
    log.info("These orders stay in Silver but will be dropped by the Gold layer's "
             "INNER JOIN: an explicit choice, not an accident.")
    return {"orphan_customers": orphan_customers, "orphan_products": orphan_products}


# ---------------------------------------------------------------------------

PROCESSORS = {
    "customers": process_customers,
    "products": process_products,
    "orders": process_orders,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BRONZE -> SILVER cleaning")
    parser.add_argument("--datasets", nargs="+", default=["customers", "products", "orders"],
                        choices=["customers", "products", "orders"])
    parser.add_argument("--bronze", type=Path, default=BRONZE)
    parser.add_argument("--output", type=Path, default=SILVER)
    parser.add_argument("--min-valid-ratio", type=float, default=MIN_VALID_RATIO,
                        help="Threshold below which the pipeline fails")
    parser.add_argument("--skip-integrity", action="store_true")
    args = parser.parse_args(argv)

    spark = get_spark("bronze-to-silver")
    reports = []
    for name in args.datasets:
        reports.append(PROCESSORS[name](spark, args.bronze, args.output))

    for report in reports:
        save_report(report, QUALITY_REPORTS)
        assert_quality(report, args.min_valid_ratio, log)

    if not args.skip_integrity and len(args.datasets) == 3:
        check_referential_integrity(spark, args.output)

    log.info("=" * 60)
    log.info("SUMMARY")
    for r in reports:
        log.info(f"  {r.dataset:<10} {r.rows_read:>10,} read -> {r.rows_valid:>10,} valid "
                 f"({100 * r.valid_ratio:.1f}%)")
    log.info(f"JSON reports: {QUALITY_REPORTS}")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
