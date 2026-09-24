"""
DEMO 2 - PARTITIONS, repartition() and coalesce() (phase 6).

A PARTITION is a chunk of the data processed by ONE task, on ONE core. It is
Spark's unit of parallelism. Every performance discussion starts with the
question: "how many partitions, and are they balanced?"

Run with: make demo-partitions
"""

from __future__ import annotations

import time

from pyspark.sql import functions as F

from src.spark.spark_session import describe_session, get_spark
from src.spark.utils import SILVER, get_logger, rows_per_partition

log = get_logger("demo")


def section(title: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info(f"  {title}")
    log.info("=" * 68)


def describe_partitions(df, label: str) -> list[int]:
    counts = rows_per_partition(df)
    non_empty = [c for c in counts if c > 0]
    log.info(f"{label}")
    log.info(f"    partitions   : {len(counts)} ({len(non_empty)} of them non-empty)")
    log.info(f"    rows/part.   : min={min(counts):,} max={max(counts):,} "
             f"total={sum(counts):,}")
    if non_empty and min(non_empty) > 0:
        ratio = max(counts) / max(min(non_empty), 1)
        log.info(f"    imbalance    : x{ratio:.1f} between the largest and the smallest")
    return counts


def timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    log.info(f"    {label:<42} {elapsed:6.2f}s")
    return result, elapsed


def main() -> int:
    spark = get_spark("demo-partitions")
    info = describe_session(spark)
    log.info(f"Cores available: {info['default_parallelism']} | "
             f"shuffle.partitions: {info['shuffle_partitions']}")

    orders = spark.read.parquet(str(SILVER / "orders"))

    section("1. How many partitions on read, and WHY?")
    n = orders.rdd.getNumPartitions()
    log.info(f"df.rdd.getNumPartitions() = {n}")
    log.info("")
    log.info("Spark does not pick that number at random. For a file it computes:")
    log.info("    maxSplitBytes = min(maxPartitionBytes, max(openCostInBytes, totalSize/cores))")
    log.info(f"    spark.sql.files.maxPartitionBytes = "
             f"{spark.conf.get('spark.sql.files.maxPartitionBytes')}")
    log.info(f"    spark.sql.files.openCostInBytes   = "
             f"{spark.conf.get('spark.sql.files.openCostInBytes')}")
    log.info("Here silver/orders is already split into 24 files (24 months):")
    log.info("Spark groups those files until each partition is full.")
    describe_partitions(orders, "Raw read")

    section("2. repartition(n) - redistribute WITH a shuffle")
    log.info("repartition triggers a FULL SHUFFLE: every row is redistributed")
    log.info("at random to obtain BALANCED partitions.")
    log.info("Expensive, but it is the only way to INCREASE parallelism.")
    rep, elapsed = timed("repartition(16) [transformation only]",
                         lambda: orders.repartition(16))
    log.info(f"    That duration ({elapsed:.2f}s) measures NOTHING: repartition is a")
    log.info("    transformation, so it is LAZY. No data has moved yet.")
    log.info("    The real cost shows up below, when an action forces the work.")
    describe_partitions(rep, "After repartition(16)")

    section("3. coalesce(n) - reduce WITHOUT a shuffle")
    log.info("coalesce MERGES existing partitions in place. No network transfer,")
    log.info("so it is very fast. But it can only REDUCE, and the result may be")
    log.info("unbalanced.")
    col, _ = timed("coalesce(2) [transformation only, lazy]",
                   lambda: orders.coalesce(2))
    describe_partitions(col, "After coalesce(2)")

    log.info("")
    log.info("Trying to INCREASE with coalesce:")
    up = orders.coalesce(32)
    log.info(f"    coalesce(32) on {n} partitions -> {up.rdd.getNumPartitions()} partitions")
    log.info("    coalesce CANNOT increase the partition count: it is ignored.")

    section("4. Measured comparison on a real action")
    log.info("Same computation (sum of quantities), three configurations:")
    base = orders.select("quantity")
    timed(f"{n} partitions (natural read)", lambda: base.agg(F.sum("quantity")).collect())
    timed("1 partition (coalesce 1 = single-threaded)",
          lambda: base.coalesce(1).agg(F.sum("quantity")).collect())
    timed("64 partitions (repartition 64 = too many)",
          lambda: base.repartition(64).agg(F.sum("quantity")).collect())
    log.info("")
    log.info("How to read the result:")
    log.info("  - 1 partition  : no parallelism, a single core does the work.")
    log.info("  - 64 partitions: the shuffle and the scheduling of 64 tasks cost")
    log.info("    more than the computation itself at this volume.")
    log.info("  - There is an optimum, and it is NOT 'as many as possible'.")

    section("5. repartition by COLUMN - partitioning by key")
    log.info("repartition('status') puts every row with the same status on the")
    log.info("SAME partition (hash partitioning).")
    log.info("Useful before several aggregations on that key: the shuffle happens")
    log.info("once. Dangerous when the key is skewed.")
    by_status = orders.repartition(F.col("status"))
    describe_partitions(by_status, "After repartition('status')")
    log.info("")
    log.info("This is DATA SKEW in action: only 5 statuses, and 'delivered' is")
    log.info("62% of the rows. One task handles far more than the others -> the")
    log.info("job is as slow as its largest partition.")

    section("6. Rules of thumb")
    log.info("  - Aim for 100 to 200 MB of data per partition.")
    log.info("  - Aim for a partition count that is a multiple of the core count.")
    log.info("  - repartition() to INCREASE or REBALANCE (shuffle, expensive).")
    log.info("  - coalesce() to REDUCE before writing (no shuffle, fast).")
    log.info("  - Too many small partitions = scheduling costs more than the work.")
    log.info("  - Too few = no parallelism, and a risk of OutOfMemory.")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
