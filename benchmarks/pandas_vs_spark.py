"""
BENCHMARK Pandas vs PySpark (phase 16).

The same processing on both sides:
    read a CSV  ->  clean  ->  aggregate by day

Rules of this benchmark:
  - no number is made up: everything is measured on the machine that runs it;
  - if a size exceeds what the machine can take, we SAY so and skip the
    measurement instead of making the system swap;
  - Spark is not presented as systematically better: on small volumes it is
    slower, and that is normal.

Usage:
    python3 -m benchmarks.pandas_vs_spark --sizes 10mb 100mb
    python3 -m benchmarks.pandas_vs_spark --sizes 10mb 100mb 1gb --force
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.generator.generate_data import PRESETS
from src.spark.spark_session import get_spark
from src.spark.utils import PROJECT_ROOT, get_logger, human_bytes, path_size

log = get_logger("bench")

BENCH_DIR = PROJECT_ROOT / "data" / "bench"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

# Pandas loads the CSV into memory with heavy overhead: every text value
# becomes a Python object (~50 bytes of header on its own).
# FACTOR MEASURED on this project: a 93 MB CSV pushed the process to 920 MB of
# RSS, i.e. x9.9. We keep 10, an observed value rather than an estimate.
PANDAS_MEMORY_FACTOR = 10


def available_memory_bytes() -> int:
    """Memory actually available (MemAvailable), not total memory."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def peak_rss_bytes() -> int:
    """Peak memory of the CURRENT PROCESS since it started."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# ---------------------------------------------------------------------------
# The two implementations of the SAME processing
# ---------------------------------------------------------------------------

def run_pandas(csv_path: Path) -> dict:
    """
    Pandas pipeline: everything fits (or does not) in the memory of ONE process.
    """
    gc.collect()
    rss_before = peak_rss_bytes()
    start = time.perf_counter()

    df = pd.read_csv(csv_path, dtype=str)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["order_date"] = pd.to_datetime(
        df["order_date"], format="mixed", errors="coerce").dt.date
    df["status"] = df["status"].str.strip().str.lower()

    clean = df.dropna(subset=["order_id", "customer_id", "product_id",
                              "quantity", "order_date"])
    clean = clean[clean["quantity"] > 0].drop_duplicates(subset=["order_id"])

    agg = (clean.groupby("order_date")
                .agg(orders_count=("order_id", "count"),
                     total_quantity=("quantity", "sum"))
                .reset_index()
                .sort_values("order_date"))

    elapsed = time.perf_counter() - start
    result = {
        "engine": "pandas",
        "seconds": round(elapsed, 2),
        "rows_in": int(len(df)),
        "rows_out": int(len(agg)),
        "peak_rss_bytes": max(peak_rss_bytes() - rss_before, 0),
    }
    del df, clean, agg
    gc.collect()
    return result


def run_spark(spark, csv_path: Path) -> dict:
    """
    PySpark pipeline: the same processing, distributed across the executors.
    """
    from pyspark.sql import functions as F

    start = time.perf_counter()

    df = spark.read.option("header", "true").csv(str(csv_path))
    cleaned = (
        df.withColumn("quantity", F.col("quantity").cast("int"))
          .withColumn("order_date", F.to_date(F.coalesce(
              F.to_timestamp("order_date", "yyyy-MM-dd HH:mm:ss"),
              F.to_timestamp("order_date", "dd/MM/yyyy"))))
          .withColumn("status", F.lower(F.trim(F.col("status"))))
          .dropna(subset=["order_id", "customer_id", "product_id",
                          "quantity", "order_date"])
          .filter(F.col("quantity") > 0)
          .dropDuplicates(["order_id"])
    )
    agg = (cleaned.groupBy("order_date")
                  .agg(F.count("order_id").alias("orders_count"),
                       F.sum("quantity").alias("total_quantity"))
                  .orderBy("order_date"))
    rows_out = agg.count()   # ACTION: this is what triggers the whole computation
    elapsed = time.perf_counter() - start

    return {
        "engine": "pyspark",
        "seconds": round(elapsed, 2),
        "rows_out": rows_out,
        "executor_memory": spark.conf.get("spark.executor.memory"),
        "executor_cores": spark.conf.get("spark.executor.cores"),
    }


# ---------------------------------------------------------------------------

def prepare_dataset(size: str, keep: bool) -> Path:
    """Generate (or reuse) the dataset for a given size."""
    target = BENCH_DIR / size
    csv = target / "orders" / "orders.csv"
    if csv.exists() and keep:
        log.info(f"  dataset {size} already present ({human_bytes(path_size(csv))})")
        return csv

    rows = PRESETS[size]
    log.info(f"  generating {rows:,} orders for the {size} step...")
    subprocess.run(
        [sys.executable, "-m", "src.generator.generate_data",
         "--rows", str(rows), "--output", str(target), "--clean", "--seed", "42"],
        check=True, cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
    )
    log.info(f"  -> {human_bytes(path_size(csv))}")
    return csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Pandas vs PySpark")
    parser.add_argument("--sizes", nargs="+", default=["10mb", "100mb"],
                        choices=sorted(PRESETS))
    parser.add_argument("--force", action="store_true",
                        help="Run Pandas even if memory looks insufficient")
    parser.add_argument("--keep", action="store_true", default=True,
                        help="Reuse the datasets that are already generated")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the benchmark datasets at the end")
    args = parser.parse_args(argv)

    avail = available_memory_bytes()
    log.info("=" * 70)
    log.info(f"Memory available on the machine: {human_bytes(avail)}")
    log.info(f"Steps under test: {', '.join(args.sizes)}")
    log.info("=" * 70)

    spark = get_spark("benchmark-pandas-vs-spark",
                      extra_conf={"spark.ui.showConsoleProgress": "false"})
    results = []

    for size in args.sizes:
        log.info("")
        log.info(f"--- STEP {size.upper()} " + "-" * 40)
        csv = prepare_dataset(size, args.keep)
        file_size = path_size(csv)

        entry = {"size": size, "csv_bytes": file_size,
                 "csv_human": human_bytes(file_size)}

        # --- Pandas ---
        needed = file_size * PANDAS_MEMORY_FACTOR
        if needed > avail and not args.force:
            log.warning(f"  PANDAS SKIPPED: it would need about "
                        f"{human_bytes(needed)} of RAM "
                        f"(file x{PANDAS_MEMORY_FACTOR}) while the machine "
                        f"only has {human_bytes(avail)} available.")
            log.warning("  This is exactly the ceiling Spark removes.")
            log.warning("  Use --force to try anyway (risk of swapping).")
            entry["pandas"] = {"skipped": True,
                               "reason": f"estimated need {human_bytes(needed)} "
                                         f"> {human_bytes(avail)} available"}
        else:
            log.info("  Pandas...")
            entry["pandas"] = run_pandas(csv)
            log.info(f"    {entry['pandas']['seconds']}s | peak RAM "
                     f"{human_bytes(entry['pandas']['peak_rss_bytes'])} | "
                     f"{entry['pandas']['rows_out']} rows out")

        # --- Spark ---
        log.info("  PySpark...")
        entry["spark"] = run_spark(spark, csv)
        log.info(f"    {entry['spark']['seconds']}s | "
                 f"{entry['spark']['rows_out']} rows out")

        if not entry["pandas"].get("skipped"):
            ratio = entry["spark"]["seconds"] / entry["pandas"]["seconds"]
            entry["spark_vs_pandas"] = round(ratio, 2)
            verdict = (f"Spark {ratio:.1f}x SLOWER" if ratio > 1
                       else f"Spark {1 / ratio:.1f}x faster")
            log.info(f"    -> {verdict}")

        results.append(entry)

    # --- Summary ---
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"{'Step':<8} {'CSV':>10} {'Pandas':>10} {'PySpark':>10} "
             f"{'Pandas RAM':>12} {'Verdict':>22}")
    for r in results:
        p = r["pandas"]
        if p.get("skipped"):
            log.info(f"{r['size']:<8} {r['csv_human']:>10} {'N/A':>10} "
                     f"{r['spark']['seconds']:>9.2f}s {'-':>12} "
                     f"{'Pandas impossible':>22}")
        else:
            ratio = r["spark_vs_pandas"]
            verdict = f"Spark x{ratio:.1f} slower" if ratio > 1 \
                else f"Spark x{1 / ratio:.1f} faster"
            log.info(f"{r['size']:<8} {r['csv_human']:>10} "
                     f"{p['seconds']:>9.2f}s {r['spark']['seconds']:>9.2f}s "
                     f"{human_bytes(p['peak_rss_bytes']):>12} {verdict:>22}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"benchmark_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {
            "cpu_count": os.cpu_count(),
            "available_memory": human_bytes(avail),
        },
        "spark": {
            "executor_memory": spark.conf.get("spark.executor.memory"),
            "executor_cores": spark.conf.get("spark.executor.cores"),
            "master": spark.sparkContext.master,
        },
        "results": results,
    }, indent=2))
    log.info("")
    log.info(f"Results saved: {out}")

    spark.stop()
    if args.cleanup and BENCH_DIR.exists():
        shutil.rmtree(BENCH_DIR)
        log.info(f"Benchmark datasets deleted ({BENCH_DIR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
