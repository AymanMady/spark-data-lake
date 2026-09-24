"""
DEMO 3 - LAZY EVALUATION, TRANSFORMATIONS vs ACTIONS, DAG and SHUFFLE
(phase 7, together with reading the execution plans from the "EXPLAIN" phase).

Run with: make demo-lazy
"""

from __future__ import annotations

import time

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
    spark = get_spark("demo-lazy-shuffle")
    orders = spark.read.parquet(str(SILVER / "orders"))

    section("1. LAZY EVALUATION: building a plan costs almost nothing")

    start = time.perf_counter()
    chained = (
        orders
        .filter(F.col("quantity") > 1)
        .withColumn("big_order", F.col("quantity") >= 4)
        .select("order_id", "customer_id", "product_id", "quantity", "status", "big_order")
        .filter(F.col("status") != "cancelled")
        .withColumn("quantity_x2", F.col("quantity") * 2)
    )
    build_time = time.perf_counter() - start
    log.info(f"5 chained transformations: {build_time * 1000:.1f} ms")
    log.info("On ~950,000 rows. Impossible: nothing was computed.")
    log.info("Spark only built a LOGICAL PLAN.")

    start = time.perf_counter()
    n = chained.count()
    action_time = time.perf_counter() - start
    log.info("")
    log.info(f"Then ONE action (count): {action_time:.2f}s for {n:,} rows")
    log.info(f"Ratio: the action cost {action_time / build_time:.0f} times more")
    log.info("than building all 5 transformations put together.")

    section("2. Why laziness is an OPTIMISATION, not idleness")
    log.info("Because it waits, Spark sees the WHOLE query before acting.")
    log.info("Catalyst can then:")
    log.info("  - merge the two filter() calls into a single pass;")
    log.info("  - push the filters BEFORE the read (predicate pushdown);")
    log.info("  - read only the columns that are used (column pruning);")
    log.info("  - drop 'quantity_x2' if it is never consumed.")
    log.info("")
    log.info("An engine that executed every line immediately could do NONE of")
    log.info("these optimisations.")
    log.info("")
    log.info("Physical plan actually executed:")
    chained.explain("formatted")

    section("3. TRANSFORMATION vs ACTION: how to tell them apart")
    log.info("Simple rule: look at the RETURN TYPE.")
    log.info("")
    log.info(f"  orders.filter(...)  -> {type(orders.filter(F.col('quantity') > 1)).__name__}"
             "  = a DataFrame  -> TRANSFORMATION")
    log.info("  orders.count()      -> int         = a value      -> ACTION")
    log.info("")
    log.info("TRANSFORMATIONS: select, filter, where, withColumn, drop, join,")
    log.info("                 groupBy, orderBy, distinct, repartition...")
    log.info("ACTIONS        : count, collect, show, take, first, write,")
    log.info("                 toPandas, foreach...")

    section("4. NARROW vs WIDE: where the shuffle comes from")
    log.info("NARROW: each output partition depends on a SINGLE input partition.")
    log.info("        No network exchange.")
    log.info("        -> filter, select, withColumn, map")
    log.info("")
    log.info("WIDE  : one output partition depends on SEVERAL input partitions.")
    log.info("        The data has to cross the network.")
    log.info("        -> groupBy, join, distinct, orderBy, repartition")
    log.info("")
    log.info("A Spark job is cut into STAGES. The boundary between two stages is")
    log.info("ALWAYS a shuffle. Counting the 'Exchange' nodes in a plan = counting")
    log.info("the shuffles.")

    section("5. A shuffle, up close")
    log.info("Before the shuffle: rows sharing a status are SCATTERED across the")
    log.info("                    4 partitions.")
    log.info("       |")
    log.info("       v  each task writes intermediate files to DISK")
    log.info("          (shuffle write), one per destination partition")
    log.info("       |")
    log.info("       v  the next stage's tasks READ those files over the")
    log.info("          NETWORK (shuffle read)")
    log.info("       |")
    log.info("After : every row with the same status is on the SAME partition")
    log.info("")
    log.info("Cost: serialisation + disk write + network transfer +")
    log.info("      deserialisation. It is almost always the slowest operation")
    log.info("      in a Spark job.")

    narrow_only = orders.filter(F.col("quantity") > 2).select("order_id", "quantity")
    with_shuffle = orders.groupBy("status").agg(F.sum("quantity"))

    log.info("")
    log.info("Plan WITHOUT a shuffle (narrow only):")
    for line in narrow_only._jdf.queryExecution().simpleString().split("\n")[:6]:
        log.info(f"    {line}")
    plan = narrow_only._jdf.queryExecution().toString()
    log.info(f"    -> Exchange present: {'Exchange' in plan}")

    log.info("")
    log.info("Plan WITH a shuffle (groupBy):")
    plan = with_shuffle._jdf.queryExecution().toString()
    log.info(f"    -> Exchange present: {'Exchange' in plan}")
    log.info(f"    -> number of Exchanges: {plan.count('Exchange hashpartitioning')}")

    section("6. Measuring the cost of a shuffle")
    t0 = time.perf_counter()
    narrow_only.count()
    t_narrow = time.perf_counter() - t0

    t0 = time.perf_counter()
    with_shuffle.collect()
    t_wide = time.perf_counter() - t0

    log.info(f"    filter + select (narrow)   : {t_narrow:.2f}s")
    log.info(f"    groupBy + sum   (wide)     : {t_wide:.2f}s")
    log.info("")
    log.info("Note: at this volume the gap stays moderate. At 100 GB, a badly")
    log.info("handled shuffle turns a 5-minute job into several hours.")

    section("7. spark.sql.shuffle.partitions and AQE")

    def bench(label: str, aqe: str, parts: str, runs: int = 3) -> float:
        """Take the BEST of several runs: we are measuring the cost of the
        computation, not that of the first pass's cold disk cache."""
        spark.conf.set("spark.sql.adaptive.enabled", aqe)
        spark.conf.set("spark.sql.shuffle.partitions", parts)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            orders.groupBy("customer_id").agg(F.sum("quantity")).count()
            times.append(time.perf_counter() - t0)
        best = min(times)
        log.info(f"    {label:<42} {best:5.2f}s  (best of {runs})")
        return best

    # Unmeasured warm-up: otherwise the first configuration pays the cold cache.
    orders.groupBy("customer_id").agg(F.sum("quantity")).count()

    t_8_noaqe = bench("AQE off | shuffle.partitions = 8", "false", "8")
    t_200_noaqe = bench("AQE off | shuffle.partitions = 200", "false", "200")
    t_200_aqe = bench("AQE ON  | shuffle.partitions = 200", "true", "200")

    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", "8")

    log.info("")
    log.info("An HONEST reading of these numbers:")
    log.info(f"  - Without AQE, going from 8 to 200 partitions costs "
             f"{100 * (t_200_noaqe / t_8_noaqe - 1):+.0f}%: that is 200 tiny tasks")
    log.info("    to schedule for a volume that does not call for it.")
    log.info(f"  - With AQE, those 200 partitions come back to {t_200_aqe:.2f}s "
             f"({100 * (t_200_aqe / t_200_noaqe - 1):+.0f}%):")
    log.info("    AQE looks at the real shuffle size and MERGES the partitions")
    log.info("    that turned out too small.")
    log.info("")
    log.info("Conclusion: since Spark 3.2, AQE makes up for much of a bad")
    log.info("shuffle.partitions setting. Tuning it by hand still helps, but it")
    log.info("is no longer the magic lever it used to be.")
    log.info("At the scale of several TB, the gap becomes major again.")

    section("8. The DAG")
    log.info("    read parquet   (narrow)")
    log.info("         |")
    log.info("       filter       (narrow)   ---+")
    log.info("         |                        |  STAGE 1")
    log.info("      withColumn    (narrow)   ---+")
    log.info("         |")
    log.info("    === SHUFFLE (Exchange) ===")
    log.info("         |")
    log.info("      groupBy       (wide)     ---+  STAGE 2")
    log.info("         |                        |")
    log.info("       write        (action)   ---+")
    log.info("")
    log.info("JOB  : triggered by ONE action.")
    log.info("STAGE: a portion of the job with no shuffle in it.")
    log.info("TASK : one partition handled by one core. #tasks = #partitions.")
    log.info("")
    log.info("All of this is visible in the 'Stages' tab of the Spark UI (port 4040).")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
