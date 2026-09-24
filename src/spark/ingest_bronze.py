"""
RAW -> BRONZE (phase 3).

THE ROLE OF THE BRONZE LAYER: keep the data exactly as it arrived.

This job cleans NOTHING. It fixes NOTHING. All it does is:
  1. read the source CSVs with an EXPLICIT schema (everything as text);
  2. add technical lineage metadata;
  3. write Parquet.

Why convert to Parquet if nothing is transformed?
Because Bronze will be re-read dozens of times by the Silver jobs, the tests
and ad-hoc analyses. On every read, CSV forces you:
  - to re-parse text line by line;
  - to read EVERY column even when you only want two;
  - to work without any statistics that would let you skip blocks.
Parquet is columnar, compressed, typed, and carries its own statistics.
The gain is measured in src/spark/demos/04_parquet_vs_csv.py.

The rule "Bronze = raw" is about the CONTENT, not the storage format.

Usage:
    spark-submit src/spark/ingest_bronze.py
    spark-submit src/spark/ingest_bronze.py --datasets orders --input data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.spark.schemas import RAW_SCHEMAS
from src.spark.spark_session import get_spark
from src.spark.utils import BRONZE, RAW, get_logger, human_bytes, path_size, timer

log = get_logger("bronze")


def read_raw_csv(spark: SparkSession, name: str, input_dir: Path) -> DataFrame:
    """
    Read a source CSV with an imposed schema.

    mode="PERMISSIVE" (the default): a malformed row does not fail the job, its
    fields become NULL. That is what we want in Bronze: ingest everything,
    judge later. The alternatives would be DROPMALFORMED (silent loss) or
    FAILFAST (the job dies on a single dirty row).
    """
    path = input_dir / name
    log.info(f"Reading {name} from {path}")
    return (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        # Product names can contain commas inside quotes.
        .option("quote", '"')
        .option("escape", '"')
        .schema(RAW_SCHEMAS[name])
        .csv(str(path))
    )


def add_ingestion_metadata(df: DataFrame) -> DataFrame:
    """
    Add the technical lineage columns.

    _ingested_at  : when this row entered the lake.
    _source_file  : which file it came from. input_file_name() is a Spark
                    function that exposes the file the task is reading.

    These columns answer the question you always end up asking during an
    incident: "this odd-looking row, where and when did it come from?".
    """
    return (
        df.withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.element_at(F.split(F.input_file_name(), "/"), -1))
    )


def ingest(spark: SparkSession, name: str, input_dir: Path, output_dir: Path) -> dict:
    df = add_ingestion_metadata(read_raw_csv(spark, name, input_dir))

    target = output_dir / name
    with timer(log, f"Writing bronze/{name}") as t:
        # .write is an ACTION: this is where all the work is triggered.
        df.write.mode("overwrite").parquet(str(target))

    # The counts are done AFTER the write, on the Parquet: far cheaper than
    # recounting from the CSV.
    written = spark.read.parquet(str(target))
    rows = written.count()

    csv_size = path_size(input_dir / name)
    parquet_size = path_size(target)
    ratio = csv_size / parquet_size if parquet_size else 0

    log.info(f"{name}: {rows:,} rows | CSV {human_bytes(csv_size)} "
             f"-> Parquet {human_bytes(parquet_size)} (x{ratio:.1f} more compact)")

    return {
        "dataset": name,
        "rows": rows,
        "csv_bytes": csv_size,
        "parquet_bytes": parquet_size,
        "compression_ratio": round(ratio, 2),
        "write_seconds": round(t["seconds"], 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAW -> BRONZE ingestion")
    parser.add_argument("--datasets", nargs="+", default=["customers", "products", "orders"],
                        choices=["customers", "products", "orders"])
    parser.add_argument("--input", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=BRONZE)
    args = parser.parse_args(argv)

    spark = get_spark("bronze-ingestion")
    log.info(f"Spark {spark.version} | master={spark.sparkContext.master}")

    results = []
    for name in args.datasets:
        source = args.input / name
        if not source.exists():
            log.error(f"{source} not found. Run `make generate` first.")
            spark.stop()
            return 1
        results.append(ingest(spark, name, args.input, args.output))

    log.info("-" * 60)
    total_csv = sum(r["csv_bytes"] for r in results)
    total_pq = sum(r["parquet_bytes"] for r in results)
    log.info(f"TOTAL: {sum(r['rows'] for r in results):,} rows | "
             f"CSV {human_bytes(total_csv)} -> Parquet {human_bytes(total_pq)} "
             f"(x{total_csv / total_pq:.1f})")
    log.info("Bronze layer ready. No data was modified.")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
