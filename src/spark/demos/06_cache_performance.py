"""
DEMO 6 - CACHE, PERSIST and optimisation (phase 12).

Guiding rule of this demo: no premature optimisation. For each technique it
shows (1) the problem, (2) the simple version, (3) the optimised version,
(4) the measurement, (5) the trade-off.

Run with: make demo-cache
"""

from __future__ import annotations

import time

from pyspark import StorageLevel
from pyspark.sql import functions as F

from src.spark.spark_session import get_spark
from src.spark.utils import SILVER, get_logger

log = get_logger("demo")


def section(title: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info(f"  {title}")
    log.info("=" * 68)


def main() -> int:
    spark = get_spark("demo-cache",
                      extra_conf={"spark.ui.showConsoleProgress": "false"})

    orders = spark.read.parquet(str(SILVER / "orders"))
    products = spark.read.parquet(str(SILVER / "products"))
    customers = spark.read.parquet(str(SILVER / "customers"))

    def build_fact():
        """An expensive chain: two joins plus a computation. This is what gets reused."""
        return (orders
                .join(F.broadcast(products), "product_id")
                .join(F.broadcast(customers), "customer_id")
                .withColumn("revenue", F.col("quantity") * F.col("price")))

    section("1. THE PROBLEM: Spark remembers NOTHING by default")
    log.info("A DataFrame is not a result: it is a RECIPE.")
    log.info("On every action, Spark RE-RUNS the whole recipe starting from")
    log.info("reading the files. Four aggregations = the joins four times over.")

    section("2. SIMPLE VERSION (no cache): 4 aggregations")
    fact = build_fact()
    start = time.perf_counter()
    r1 = fact.groupBy("order_date").agg(F.sum("revenue")).count()
    r2 = fact.groupBy("category").agg(F.sum("revenue")).count()
    r3 = fact.groupBy("customer_id").agg(F.sum("revenue")).count()
    r4 = fact.groupBy("country").agg(F.sum("revenue")).count()
    no_cache = time.perf_counter() - start
    log.info(f"    Results: {r1} dates, {r2} categories, {r3:,} customers, {r4} countries")
    log.info(f"    TIME WITHOUT CACHE: {no_cache:.2f}s")

    section("3. OPTIMISED VERSION (with cache)")
    fact_cached = build_fact().cache()

    start = time.perf_counter()
    n = fact_cached.count()
    materialize = time.perf_counter() - start
    log.info(f"    Materialising the cache: {materialize:.2f}s for {n:,} rows")
    log.info("    cache() is LAZY: it is this first count() that actually fills")
    log.info("    the memory. Before it, nothing is stored.")

    start = time.perf_counter()
    fact_cached.groupBy("order_date").agg(F.sum("revenue")).count()
    fact_cached.groupBy("category").agg(F.sum("revenue")).count()
    fact_cached.groupBy("customer_id").agg(F.sum("revenue")).count()
    fact_cached.groupBy("country").agg(F.sum("revenue")).count()
    with_cache = time.perf_counter() - start

    log.info(f"    The same 4 aggregations: {with_cache:.2f}s")
    log.info(f"    Total with cache       : {materialize + with_cache:.2f}s")
    log.info("")
    total_cached = materialize + with_cache
    log.info(f"    MEASURED: {no_cache:.2f}s without cache -> {total_cached:.2f}s with cache")
    log.info(f"              i.e. {100 * (1 - total_cached / no_cache):.0f}% LESS time "
             f"(materialisation included)")
    log.info(f"    On the aggregations alone: {no_cache:.2f}s -> {with_cache:.2f}s, "
             f"i.e. {100 * (1 - with_cache / no_cache):.0f}% less time")
    log.info("")
    log.info("So the cache is NOT free: you have to pay for its materialisation")
    log.info("first. It pays off from the second re-read onwards.")

    section("4. Checking that the cache is actually used")
    plan = (fact_cached.groupBy("country").agg(F.sum("revenue"))
            ._jdf.queryExecution().toString())
    log.info(f"    'InMemoryTableScan' present in the plan: "
             f"{'InMemoryTableScan' in plan}")
    log.info("    That is THE marker to look for. If it is missing although you")
    log.info("    called cache(), the DataFrame was rebuilt some other way (a")
    log.info("    transformation added after the cache, for instance).")

    section("5. cache() or persist()?")
    log.info("    cache() == persist(StorageLevel.MEMORY_AND_DISK)")
    log.info("")
    log.info("    MEMORY_ONLY          fast, but the partitions that do not fit in")
    log.info("                         RAM are RECOMPUTED on every access.")
    log.info("    MEMORY_AND_DISK      the default: spills to disk rather than")
    log.info("                         recomputing. The right default choice.")
    log.info("    MEMORY_AND_DISK_SER  serialised: 2 to 4x more compact, but pays")
    log.info("                         deserialisation CPU on every read.")
    log.info("    DISK_ONLY            when recomputing costs more than a disk")
    log.info("                         read (a very long chain).")
    log.info("")
    fact_cached.unpersist()
    ser = build_fact().persist(StorageLevel.MEMORY_AND_DISK_DESER)
    start = time.perf_counter()
    ser.count()
    log.info(f"    persist(MEMORY_AND_DISK_DESER): {time.perf_counter() - start:.2f}s")
    ser.unpersist()

    section("6. WHY cache() EVERYWHERE IS BAD PRACTICE")
    log.info("  a) Memory is SHARED with execution. The cache steals RAM from the")
    log.info("     shuffles and aggregations, which then start spilling to disk.")
    log.info("     The job gets SLOWER.")
    log.info("")
    log.info("  b) When memory fills up, Spark EVICTS partitions in LRU order.")
    log.info("     You pay the cost of writing the cache without ever enjoying")
    log.info("     the re-read. The worst of both worlds.")
    log.info("")
    log.info("  c) A DataFrame read ONCE has nothing to gain: you are adding a")
    log.info("     pure write cost.")
    log.info("")
    log.info("  d) The cache freezes a result: if the source files change, the")
    log.info("     cache silently becomes wrong.")
    log.info("")
    log.info("RULE: only cache a DataFrame that is (1) reused at least twice,")
    log.info("(2) expensive to recompute, (3) reasonably able to fit in memory.")
    log.info("And ALWAYS call unpersist() afterwards.")

    section("7. Demonstration: a useless cache costs time")
    single_use = build_fact().filter(F.col("country") == "France")

    start = time.perf_counter()
    single_use.count()
    plain = time.perf_counter() - start

    cached_once = build_fact().filter(F.col("country") == "France").cache()
    start = time.perf_counter()
    cached_once.count()
    cached = time.perf_counter() - start
    cached_once.unpersist()

    log.info(f"    A single action, no cache  : {plain:.2f}s")
    log.info(f"    A single action, with cache: {cached:.2f}s")
    delta = 100 * (cached / plain - 1)
    log.info(f"    -> {delta:+.0f}%: the cache cost time and returned nothing.")

    section("8. Order of optimisations, most to least rewarding")
    log.info("    1. Read less  : Parquet, column pruning, partition pruning.")
    log.info("    2. Filter early: shrink the volume BEFORE the joins.")
    log.info("    3. Avoid shuffles: broadcast the small dimensions.")
    log.info("    4. Tune the partitions: neither too many nor too few.")
    log.info("    5. Cache what is genuinely reused.")
    log.info("")
    log.info("Starting with point 5 is the most widespread mistake.")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
