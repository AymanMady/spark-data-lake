"""
DEMO 5 - JOINS and BROADCAST JOIN (phase 11).

Run with: make demo-joins
"""

from __future__ import annotations

from pyspark.sql import functions as F

from src.spark.spark_session import get_spark
from src.spark.utils import SILVER, bench, get_logger, human_bytes, path_size

log = get_logger("demo")


def section(title: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info(f"  {title}")
    log.info("=" * 68)


def join_strategy(df) -> str:
    """Extract the join strategy Catalyst picked from the physical plan."""
    plan = df._jdf.queryExecution().toString()
    for strategy in ("BroadcastHashJoin", "SortMergeJoin", "ShuffledHashJoin",
                     "BroadcastNestedLoopJoin"):
        if strategy in plan:
            return strategy
    return "unknown"


def main() -> int:
    spark = get_spark("demo-joins",
                      extra_conf={"spark.ui.showConsoleProgress": "false"})

    orders = spark.read.parquet(str(SILVER / "orders"))
    customers = spark.read.parquet(str(SILVER / "customers"))
    products = spark.read.parquet(str(SILVER / "products"))

    section("1. The sizes involved: everything starts here")
    for name, df, path in [("orders", orders, SILVER / "orders"),
                           ("customers", customers, SILVER / "customers"),
                           ("products", products, SILVER / "products")]:
        log.info(f"    {name:<10} {df.count():>10,} rows   {human_bytes(path_size(path)):>9}")
    log.info("")
    log.info("products and customers are DIMENSION tables: small and stable.")
    log.info("orders is the FACT table: large and growing.")
    log.info("That imbalance is what makes the broadcast join relevant.")

    section("2. INNER JOIN: keep only the matches")
    inner = orders.join(products, "product_id", "inner")
    n_inner = inner.count()
    log.info(f"    orders {orders.count():,} INNER JOIN products -> {n_inner:,}")
    log.info(f"    {orders.count() - n_inner:,} orders LOST: their product does")
    log.info("    not exist in Silver (rejected for an invalid price, or orphaned).")
    log.info("    An inner join is a FILTER in disguise. It is the number one")
    log.info("    cause of numbers that do not add up in data engineering.")

    section("3. LEFT JOIN: keep everything on the left")
    left = orders.join(products, "product_id", "left")
    n_left = left.count()
    n_null = left.filter(F.col("price").isNull()).count()
    log.info(f"    orders LEFT JOIN products -> {n_left:,} rows (nothing lost)")
    log.info(f"    of which {n_null:,} have price = NULL")
    log.info("")
    log.info("The left join keeps a trace of the problem instead of hiding it.")
    log.info("But careful: SUM(quantity * price) will silently ignore those NULLs.")
    log.info("Choosing left or inner is a BUSINESS decision, not a technical one.")

    section("4. LEFT ANTI JOIN: what does NOT match")
    anti = orders.join(products, "product_id", "left_anti")
    log.info(f"    Orders with no known product: {anti.count():,}")
    log.info("    More efficient than a left join followed by filter(isNull):")
    log.info("    Spark never has to materialise the right-hand columns.")
    log.info("    This is the go-to tool for auditing referential integrity.")

    section("5. SORT MERGE JOIN: the default strategy")
    log.info("Without a broadcast, to join two tables Spark has to:")
    log.info("    1. SHUFFLE the left table by join key;")
    log.info("    2. SHUFFLE the right table by the same key;")
    log.info("    3. SORT each partition;")
    log.info("    4. walk both sides in parallel.")
    log.info("-> TWO full shuffles, one of them on the large table.")

    # autoBroadcastJoinThreshold = -1 disables automatic broadcasting entirely.
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    smj = orders.join(products, "product_id").groupBy("category").agg(F.sum("quantity"))
    log.info(f"    Strategy chosen: {join_strategy(smj)}")
    t_smj = bench(lambda: orders.join(products, "product_id")
                  .groupBy("category").agg(F.sum("quantity")).collect())
    log.info(f"    Time: {t_smj['best']:.2f}s  (runs: {t_smj['runs']})")

    section("6. BROADCAST HASH JOIN: removing the shuffle")
    log.info("If one table is small enough to fit in memory:")
    log.info("    1. the driver collects it in full;")
    log.info("    2. it sends it to EVERY executor;")
    log.info("    3. each executor joins its own partition locally.")
    log.info("-> ZERO shuffle on the large table. That is the whole gain.")

    bhj = (orders.join(F.broadcast(products), "product_id")
           .groupBy("category").agg(F.sum("quantity")))
    log.info(f"    Strategy chosen: {join_strategy(bhj)}")
    t_bhj = bench(lambda: orders.join(F.broadcast(products), "product_id")
                  .groupBy("category").agg(F.sum("quantity")).collect())
    log.info(f"    Time: {t_bhj['best']:.2f}s  (runs: {t_bhj['runs']})")
    log.info("")
    gain = 100 * (1 - t_bhj["best"] / t_smj["best"])
    log.info(f"    MEASURED GAIN: {gain:+.0f}% "
             f"({t_smj['best']:.2f}s -> {t_bhj['best']:.2f}s)")

    section("7. Spark often does it on its own")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")
    auto = orders.join(products, "product_id").groupBy("category").agg(F.sum("quantity"))
    log.info(f"    Automatic threshold: "
             f"{human_bytes(int(spark.conf.get('spark.sql.autoBroadcastJoinThreshold')))}")
    log.info(f"    Strategy with no explicit hint: {join_strategy(auto)}")
    log.info("")
    log.info("Spark compares the table's ESTIMATED size to the threshold (10 MB")
    log.info("by default) and broadcasts by itself. The F.broadcast() hint is")
    log.info("useful when:")
    log.info("  - the estimate is wrong (compressed files, complex filters);")
    log.info("  - the table exceeds the threshold but still fits in memory;")
    log.info("  - you want the intent to be explicit for whoever reads the code.")

    section("8. WHEN NOT TO BROADCAST")
    log.info("  - Table > a few hundred MB: the driver collects it ENTIRELY in")
    log.info("    memory -> driver OutOfMemoryError.")
    log.info("  - Many executors: the table is copied onto EACH of them.")
    log.info("    200 MB x 50 executors = 10 GB of RAM consumed in total.")
    log.info("  - Two large tables: the sort merge join is the right choice.")
    log.info("")
    log.info("Typical symptom of an abusive broadcast: the job hangs on a stage")
    log.info("with a single task, then the driver dies with an OOM.")

    section("9. The project's real use case")
    log.info("In silver_to_gold.py, orders is joined to customers AND products.")
    log.info("Both dimensions are under a megabyte: an obvious broadcast.")
    t_both = bench(lambda: orders
                   .join(F.broadcast(products), "product_id")
                   .join(F.broadcast(customers), "customer_id")
                   .groupBy("country").agg(F.sum("quantity")).collect(), runs=2)
    log.info(f"    Double broadcast join: {t_both['best']:.2f}s")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
