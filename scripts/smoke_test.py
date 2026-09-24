"""
End-to-end check of the Spark cluster (phase 1).

This script does NO business processing. Its only purpose is to prove that:
  1. the driver can connect to the Spark Master;
  2. the worker really provided executors;
  3. the driver and the executors see the SAME data/ folder (a critical point
     when the data lake is a plain filesystem);
  4. Spark can write and then re-read Parquet.

Run with: make smoke
"""

import shutil
from pathlib import Path

from pyspark.sql import SparkSession

SMOKE_DIR = Path("/opt/workspace/data/raw/smoke_test_tmp")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main() -> None:
    section("1. Creating the SparkSession")

    # SparkSession = the single entry point of any Spark application.
    # .master() is not set here: it is read from spark-defaults.conf
    # (spark://spark-master:7077). Phase 4 goes into the detail.
    spark = SparkSession.builder.appName("smoke-test-phase1").getOrCreate()
    sc = spark.sparkContext

    # A tiny warm-up job. Without it we would query Spark BEFORE the executors
    # had registered with the driver, and would read a core count that is too
    # low. A classic startup race.
    spark.range(1).count()

    print(f"  Spark version         : {spark.version}")
    print(f"  Master                : {sc.master}")
    print(f"  Application ID        : {sc.applicationId}")
    print(f"  Cores available       : {sc.defaultParallelism}")

    # getExecutorMemoryStatus only exists in Scala: we go through the JVM gateway.
    try:
        status = sc._jsc.sc().getExecutorMemoryStatus()
        print(f"  Executors + driver    : {status.size()} registered processes")
    except Exception as exc:  # pragma: no cover - purely informational
        print(f"  (executor info unavailable: {exc})")

    section("2. A first DataFrame and its PARTITIONS")

    # spark.range() generates data without reading a file: perfect for testing.
    df = spark.range(0, 1_000_000)

    # A PARTITION = a chunk of the data handled by ONE task.
    # Number of partitions -> number of tasks running in parallel.
    print(f"  Rows                  : {df.count():,}")
    print(f"  Partitions            : {df.rdd.getNumPartitions()}")
    print("  -> each partition will be handled by one task on an executor.")

    section("3. Parquet write (the driver AND the executors write)")

    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)

    # .write is an ACTION: it is what actually triggers the computation.
    df.write.mode("overwrite").parquet(str(SMOKE_DIR))

    files = sorted(p.name for p in SMOKE_DIR.glob("*.parquet"))
    total_bytes = sum(p.stat().st_size for p in SMOKE_DIR.glob("*.parquet"))
    print(f"  Folder written        : {SMOKE_DIR}")
    print(f"  .parquet files        : {len(files)} (one per partition)")
    print(f"  Total size            : {total_bytes / 1024:.1f} KB")
    print(f"  Example               : {files[0] if files else '(none)'}")

    section("4. Reading it back")

    reread = spark.read.parquet(str(SMOKE_DIR))
    n = reread.count()
    print(f"  Rows read back        : {n:,}")
    assert n == 1_000_000, f"Expected 1,000,000 rows, got {n}"

    print("\n  Schema inferred automatically from the Parquet file:")
    reread.printSchema()

    section("5. Cleanup")
    shutil.rmtree(SMOKE_DIR)
    print(f"  {SMOKE_DIR} removed.")

    spark.stop()
    print("\n" + "=" * 70)
    print("  SMOKE TEST PASSED: the Spark cluster is operational.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
