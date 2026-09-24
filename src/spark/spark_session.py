"""
SparkSession factory (phase 4).

The SparkSession is the single ENTRY POINT of any Spark application.
It wraps:
  - the SparkContext : the connection to the cluster;
  - the SQLContext   : the SQL engine and the table catalogue;
  - Catalyst         : the query optimiser.

Its creation is centralised here for three reasons:
  1. not duplicating the configuration in every job;
  2. being able to switch cluster <-> local[*] without touching business code;
  3. letting the tests get a lightweight local session.
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(
    app_name: str,
    master: str | None = None,
    local: bool = False,
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """
    Create (or retrieve) a SparkSession.

    Args:
        app_name:   name shown in the Spark UI. Always explicit: it is what
                    lets you find a job again in the history.
        master:     cluster URL. Defaults to $SPARK_MASTER_URL, otherwise the
                    value from spark-defaults.conf.
        local:      True -> force local[*] (used by the tests: no cluster
                    needed, everything runs in a single JVM process).
        extra_conf: extra configuration, specific to one job.
    """
    builder = SparkSession.builder.appName(app_name)

    # --- Where to run? -----------------------------------------------------
    # local[*]            : everything in the driver JVM, * = all cores.
    # spark://host:7077   : standalone cluster (our case).
    # yarn / k8s          : production resource managers.
    if local:
        builder = builder.master("local[*]")
    elif master or os.getenv("SPARK_MASTER_URL"):
        builder = builder.master(master or os.environ["SPARK_MASTER_URL"])
    # Otherwise: let spark-defaults.conf decide.

    # --- Configuration shared by every job in the project ------------------
    defaults = {
        # Write dates/timestamps in UTC: a pipeline must never depend on the
        # time zone of the machine running it.
        "spark.sql.session.timeZone": "UTC",
        # By default Spark 3 rejects ambiguous dates rather than silently
        # converting them wrongly. We keep that strict behaviour.
        "spark.sql.legacy.timeParserPolicy": "CORRECTED",
        # Default Parquet compression. snappy = a good speed/size trade-off;
        # zstd compresses better but costs more CPU.
        "spark.sql.parquet.compression.codec": "snappy",
        # Writes min/max statistics per block: essential for PREDICATE
        # PUSHDOWN to be able to skip whole blocks on read.
        "spark.sql.parquet.filterPushdown": "true",
    }
    for key, value in {**defaults, **(extra_conf or {})}.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()

    # Cuts the noise: we only want to see our own application logs.
    spark.sparkContext.setLogLevel("WARN")
    return spark


def describe_session(spark: SparkSession) -> dict[str, str]:
    """
    Return the information needed to understand what is about to run.

    The spark.range(1).count() is not decorative: executors register
    ASYNCHRONOUSLY. Read too early, defaultParallelism would return the core
    count of the only executor already up (2 instead of 4). This tiny job
    forces the wait before the counters are read.
    """
    spark.range(1).count()
    sc = spark.sparkContext
    return {
        "spark_version": spark.version,
        "master": sc.master,
        "app_id": sc.applicationId,
        "app_name": sc.appName,
        "default_parallelism": str(sc.defaultParallelism),
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "adaptive_enabled": spark.conf.get("spark.sql.adaptive.enabled"),
    }
