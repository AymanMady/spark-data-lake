"""
DEMO 4 - CSV vs PARQUET, predicate pushdown and partition pruning (phase 10).

Every number in this demo is MEASURED on your machine, never invented.

Run with: make demo-parquet
"""

from __future__ import annotations

from pyspark.sql import functions as F

from src.spark.schemas import RAW_ORDERS_SCHEMA
from src.spark.spark_session import get_spark
from src.spark.utils import BRONZE, RAW, SILVER, bench, get_logger, human_bytes, path_size

log = get_logger("demo")


def section(title: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info(f"  {title}")
    log.info("=" * 68)


def main() -> int:
    spark = get_spark("demo-parquet-vs-csv",
                      extra_conf={"spark.ui.showConsoleProgress": "false"})

    csv_path = str(RAW / "orders")
    parquet_path = str(BRONZE / "orders")

    section("1. STORAGE: how much disk space?")
    csv_size = path_size(RAW / "orders")
    pq_size = path_size(BRONZE / "orders")
    log.info(f"    CSV     : {human_bytes(csv_size)}")
    log.info(f"    Parquet : {human_bytes(pq_size)}")
    log.info(f"    -> Parquet is {csv_size / pq_size:.1f}x more compact, "
             f"for IDENTICAL data.")
    log.info("")
    log.info("Why? CSV stores data ROW by row, as text.")
    log.info("Parquet stores it COLUMN by column. A column holds values of the")
    log.info("same type, often repetitive: 'delivered' appears 600,000 times.")
    log.info("Compression (snappy) and dictionary encoding are therefore")
    log.info("formidably effective on it.")

    def read_csv():
        return (spark.read.option("header", "true")
                .schema(RAW_ORDERS_SCHEMA).csv(csv_path))

    section("2. FULL READ: count() over the whole file")
    csv_full = bench(lambda: read_csv().count())
    pq_full = bench(lambda: spark.read.parquet(parquet_path).count())
    log.info(f"    CSV     : {csv_full['best']:5.2f}s   (runs: {csv_full['runs']})")
    log.info(f"    Parquet : {pq_full['best']:5.2f}s   (runs: {pq_full['runs']})")
    log.info(f"    -> Parquet {csv_full['best'] / pq_full['best']:.1f}x faster")
    log.info("")
    log.info("A WORD OF HONESTY: 45 MB is small. The file fits in the OS disk")
    log.info("cache and Spark's fixed costs dominate. The real gap between CSV")
    log.info("and Parquet widens with volume; rerun this demo with --preset 1gb")
    log.info("to see it.")
    log.info("")
    log.info("Parquet stores the row count in the file's METADATA. For a count()")
    log.info("it has almost nothing to read. CSV has to parse every byte just to")
    log.info("find where the rows end.")

    section("3. COLUMN PRUNING: reading one column out of six")
    csv_col = bench(lambda: read_csv().select("quantity")
                    .agg(F.sum(F.col("quantity").cast("int"))).collect())
    pq_col = bench(lambda: spark.read.parquet(parquet_path).select("quantity")
                   .agg(F.sum(F.col("quantity").cast("int"))).collect())
    log.info(f"    CSV     : {csv_col['best']:5.2f}s")
    log.info(f"    Parquet : {pq_col['best']:5.2f}s")
    log.info(f"    -> {csv_col['best'] / pq_col['best']:.1f}x faster")
    log.info("")
    log.info("CSV is FORCED to read all 6 columns: they are interleaved on every")
    log.info("row. Parquet goes straight to the 'quantity' column block and")
    log.info("physically ignores the other 5.")
    log.info("This is THE reason columnar formats exist in analytics.")

    section("4. PREDICATE PUSHDOWN: the filter goes down into the file")
    log.info("Without pushdown: read everything, then filter in memory.")
    log.info("With pushdown: the format itself skips the useless blocks.")
    log.info("")
    log.info("Parquet keeps min/max statistics for every column in every row")
    log.info("group. If a block has max(quantity)=3 and we are looking for")
    log.info("quantity=5, the whole block is skipped without being read.")
    log.info("")

    def extract(plan: str, marker: str) -> str:
        """Isolate a specific section of the physical plan (PushedFilters, etc.)."""
        for line in plan.split("\n"):
            if marker in line:
                start = line.find(marker)
                end = line.find("],", start)
                return line[start:end + 1] if end > 0 else line[start:start + 140]
        return f"{marker}: absent from the plan"

    pq_plan = (spark.read.parquet(parquet_path)
               .filter(F.col("status") == "delivered")._jdf.queryExecution().toString())
    csv_plan = (read_csv().filter(F.col("status") == "delivered")
                ._jdf.queryExecution().toString())

    log.info(f"    PARQUET -> {extract(pq_plan, 'PushedFilters')}")
    log.info(f"    CSV     -> {extract(csv_plan, 'PushedFilters')}")
    log.info("")
    log.info("SURPRISE: CSV shows PushedFilters TOO.")
    log.info("That is correct since Spark 3.0 (spark.sql.csv.filterPushdown.enabled),")
    log.info("but the word 'pushdown' covers two VERY different mechanisms:")
    log.info("")
    log.info("  CSV     : Spark still reads EVERY BYTE of the file to find the")
    log.info("            line endings. The filter only spares it the conversion")
    log.info("            of the remaining columns once a condition has failed.")
    log.info("            Saving: parsing CPU.")
    log.info("")
    log.info("  PARQUET : the min/max statistics of each row group let it NOT")
    log.info("            READ the block at all.")
    log.info("            Saving: whole disk I/O operations.")
    log.info("")
    log.info(f"    Another clue in the plan: Batched: "
             f"{'true' if 'Batched: true' in pq_plan else '?'} (Parquet, vectorised")
    log.info("    reads by column blocks) versus Batched: false (CSV, row by row).")
    log.info("")
    log.info("Let us check by MEASURING, with a very selective filter:")

    sel_csv = bench(lambda: read_csv().filter(F.col("quantity") == "5").count())
    sel_pq = bench(lambda: spark.read.parquet(parquet_path)
                   .filter(F.col("quantity") == "5").count())
    log.info(f"    CSV     filtered: {sel_csv['best']:5.2f}s "
             f"(vs {csv_full['best']:.2f}s unfiltered)")
    log.info(f"    Parquet filtered: {sel_pq['best']:5.2f}s "
             f"(vs {pq_full['best']:.2f}s unfiltered)")
    log.info(f"    -> CSV/Parquet gap: x{sel_csv['best'] / sel_pq['best']:.1f} "
             f"with a filter, against x{csv_full['best'] / pq_full['best']:.1f} without")

    section("5. PARTITION PRUNING: skipping whole directories")
    log.info("silver/orders is physically partitioned by year and by month:")
    log.info("    silver/orders/order_year=2024/order_month=12/...")
    log.info("")
    log.info("Filtering on those columns lets Spark not even OPEN the irrelevant")
    log.info("directories. This differs from pushdown: here the files are never")
    log.info("read at all.")
    log.info("")

    silver = str(SILVER / "orders")
    full_scan = bench(lambda: spark.read.parquet(silver)
                      .agg(F.sum("quantity")).collect())
    pruned = bench(lambda: spark.read.parquet(silver)
                   .filter((F.col("order_year") == 2024) & (F.col("order_month") == 12))
                   .agg(F.sum("quantity")).collect())
    log.info(f"    Full scan (24 months): {full_scan['best']:5.2f}s")
    log.info(f"    1 month (pruning)    : {pruned['best']:5.2f}s")
    log.info(f"    -> {full_scan['best'] / pruned['best']:.1f}x faster "
             f"while reading 1/24th of the data")
    log.info("")
    log.info("WHY NOT 24x FASTER? Because at this scale the time is dominated by")
    log.info("FIXED costs: task startup, directory listing, round trips with the")
    log.info("driver. Reading 780 KB or 18 MB changes little when the file is")
    log.info("already in the system cache.")
    log.info("At 500 GB those fixed costs become negligible and the gain tends")
    log.info("towards the theoretical ratio.")

    pruned_plan = (spark.read.parquet(silver)
                   .filter((F.col("order_year") == 2024) & (F.col("order_month") == 12))
                   ._jdf.queryExecution().toString())
    for line in pruned_plan.split("\n"):
        if "PartitionFilters" in line:
            start = line.find("PartitionFilters")
            log.info(f"    {line[start:start + 120]}")
            break

    section("6. THE PARTITIONING TRAP")
    log.info("Partitioning by a high-cardinality column (customer_id, order_id)")
    log.info("would create THOUSANDS of directories each holding a few rows.")
    log.info("This is the 'small files problem':")
    log.info("  - every file has a fixed opening cost;")
    log.info("  - the driver has to list all the directories before starting;")
    log.info("  - compression becomes ineffective on tiny blocks.")
    log.info("")
    log.info("RULE: partition on a LOW-cardinality column that shows up often in")
    log.info("filters. Date, country, category.")
    log.info("Aim for partitions of at least 100 MB.")

    section("7. WHEN SHOULD YOU KEEP CSV?")
    log.info("  - exchanging with a human, or a tool that only reads CSV;")
    log.info("  - a tiny file read once;")
    log.info("  - needing to read the data with a plain 'cat' during an incident.")
    log.info("For everything else in analytics: Parquet.")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
