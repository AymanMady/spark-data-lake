"""
GOLD -> POSTGRESQL (phase 13).

DATA LAKE vs DATA WAREHOUSE: why both?

  DATA LAKE (data/, Parquet)
    - stores EVERYTHING, including what you do not yet know what to do with;
    - schema-on-read: the structure is decided at read time;
    - very cheap per GB, designed for massive processing (Spark);
    - bad at answering 500 small queries per second.

  DATA WAREHOUSE (PostgreSQL)
    - stores the RESULT, modelled and constrained;
    - schema-on-write: the structure is enforced at write time;
    - indexes, transactions, concurrency, BI tools connecting directly;
    - bad at storing 50 TB of raw logs.

So only the Gold layer is loaded: a few tens of thousands of aggregated rows,
which Metabase, Power BI or an API read in milliseconds.

WHY GO THROUGH SPARK RATHER THAN pandas.to_sql?
Because Spark writes IN PARALLEL: each partition opens its own JDBC
connection. pandas would serialise everything into a single process.

Usage:
    spark-submit src/warehouse/load_postgres.py
    spark-submit src/warehouse/load_postgres.py --tables daily_sales --mode append
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from src.spark.spark_session import get_spark
from src.spark.utils import GOLD, get_logger, timer

log = get_logger("warehouse")

TABLES = ["daily_sales", "product_sales", "customer_sales", "country_sales"]

# Indexes created after loading: they speed up the dashboard queries.
# They are created AFTER the insert, never before: maintaining an index during
# a bulk load slows the write down badly.
POST_LOAD_INDEXES = {
    "daily_sales": ["CREATE INDEX IF NOT EXISTS idx_daily_sales_date ON gold.daily_sales(date)"],
    "product_sales": [
        "CREATE INDEX IF NOT EXISTS idx_product_sales_cat ON gold.product_sales(category)",
        "CREATE INDEX IF NOT EXISTS idx_product_sales_rev ON gold.product_sales(revenue DESC)",
    ],
    "customer_sales": [
        "CREATE INDEX IF NOT EXISTS idx_customer_sales_spent"
        " ON gold.customer_sales(total_spent DESC)",
    ],
    "country_sales": [],
}


def pg_config() -> dict[str, str]:
    """
    Read the PostgreSQL configuration from the environment.

    No hardcoded credentials: the variables come from the .env file injected by
    docker compose. The code can therefore be published on GitHub safely.
    """
    missing = [k for k in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                           "POSTGRES_USER", "POSTGRES_PASSWORD") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}. "
                           f"Check your .env file.")
    return {
        "url": (f"jdbc:postgresql://{os.environ['POSTGRES_HOST']}:"
                f"{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"),
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "driver": "org.postgresql.Driver",
    }


def load_table(spark: SparkSession, table: str, gold_dir: Path, cfg: dict,
               mode: str, num_partitions: int, batch_size: int) -> int:
    source = gold_dir / table
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run `make gold` first.")

    df: DataFrame = spark.read.parquet(str(source))
    rows = df.count()

    # numPartitions controls the number of SIMULTANEOUS JDBC connections.
    # Too few: the parallelism goes unused.
    # Too many: PostgreSQL's connection pool saturates (default: 100).
    # For Gold tables of a few thousand rows, 2 to 4 is enough.
    df = df.repartition(num_partitions)

    log.info(f"Loading PostgreSQL: gold.{table} ({rows:,} rows, "
             f"{num_partitions} connections, mode={mode})")

    with timer(log, f"Writing gold.{table}"):
        (df.write
           .format("jdbc")
           .option("url", cfg["url"])
           .option("dbtable", f"gold.{table}")
           .option("user", cfg["user"])
           .option("password", cfg["password"])
           .option("driver", cfg["driver"])
           # batchsize: rows per grouped INSERT. The default (1000) is already
           # fine; going too high eats memory.
           .option("batchsize", str(batch_size))
           # truncate=true: TRUNCATE instead of DROP + CREATE. Preserves the
           # indexes, the grants and the views that depend on the table.
           .option("truncate", "true")
           .mode(mode)
           .save())
    return rows


def create_indexes(table: str, cfg: dict) -> None:
    """Create the indexes through psycopg2: Spark JDBC cannot do it."""
    statements = POST_LOAD_INDEXES.get(table, [])
    if not statements:
        return
    import psycopg2

    with psycopg2.connect(
        host=os.environ["POSTGRES_HOST"], port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"], user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    ) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
    log.info(f"  {len(statements)} index(es) created on gold.{table}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GOLD -> PostgreSQL loading")
    parser.add_argument("--tables", nargs="+", default=TABLES, choices=TABLES)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--mode", choices=["overwrite", "append"], default="overwrite",
                        help="overwrite = replace the contents, append = add to them")
    parser.add_argument("--num-partitions", type=int, default=2,
                        help="Number of parallel JDBC connections")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--skip-indexes", action="store_true")
    args = parser.parse_args(argv)

    cfg = pg_config()
    # The password is never logged.
    log.info(f"Target: {cfg['url']} (user {cfg['user']})")

    spark = get_spark("gold-to-postgres")
    total = 0
    for table in args.tables:
        total += load_table(spark, table, args.gold, cfg, args.mode,
                            args.num_partitions, args.batch_size)
        if not args.skip_indexes:
            create_indexes(table, cfg)

    log.info("=" * 60)
    log.info(f"{len(args.tables)} tables loaded, {total:,} rows in total.")
    log.info("Check it: make psql  then  SELECT * FROM gold.country_sales;")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
