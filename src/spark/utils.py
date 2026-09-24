"""
Cross-cutting utilities: logging, timing, DataFrame inspection.

The goal is for the OUTPUT of the jobs to tell the story of what happens:
    [BRONZE]  Reading orders
    [SILVER]  Rows before: 1,015,000
    [SILVER]  Rows after : 998,500
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pyspark.sql import DataFrame

# Project root, whatever directory the script is launched from.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/workspace"))
DATA_DIR = PROJECT_ROOT / "data"
RAW = DATA_DIR / "raw"
BRONZE = DATA_DIR / "bronze"
SILVER = DATA_DIR / "silver"
GOLD = DATA_DIR / "gold"

_CONFIGURED = False


def get_logger(stage: str) -> logging.Logger:
    """
    Logger prefixed by the pipeline stage: [BRONZE], [SILVER], [GOLD]...

    We use logging rather than print(): levels, timestamps, and output that can
    be redirected to a file or a log collector in production.
    """
    global _CONFIGURED
    name = f"[{stage.upper()}]"
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)-11s %(message)s", "%H:%M:%S")
        )
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
        _CONFIGURED = True
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(logger: logging.Logger, label: str) -> Iterator[dict]:
    """
    Time a block of code.

    WATCH OUT, classic trap: because of LAZY EVALUATION, timing transformations
    measures nothing at all (they do not run). You have to wrap an ACTION
    (count, write, collect) to measure a real duration. That is why every use
    of this timer in the project surrounds an action.
    """
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["seconds"] = elapsed
        logger.info(f"{label} -> {elapsed:.2f}s")


def bench(fn, runs: int = 3, warmup: int = 1) -> dict:
    """
    Reliable measurement of a function's execution time.

    Two precautions are essential if the published figures are to mean
    anything:
      - WARMUP: the first run pays for the cold disk cache, the compilation of
        the code Catalyst generates and the allocation of the executors. It is
        not representative.
      - SEVERAL RUNS: we keep the BEST time, which approximates the real cost
        of the computation without the machine's noise (other processes, GC).
    """
    import statistics
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {
        "best": min(times),
        "median": statistics.median(times),
        "worst": max(times),
        "runs": [round(t, 3) for t in times],
    }


# ---------------------------------------------------------------------------
# Sizes and paths
# ---------------------------------------------------------------------------

def human_bytes(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def path_size(path: Path) -> int:
    """Total size of a file or a directory (recursive)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def reset_dir(path: Path) -> Path:
    """Delete then recreate a directory (so the jobs stay idempotent)."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# DataFrame inspection
# ---------------------------------------------------------------------------

def partition_report(df: DataFrame, label: str = "") -> str:
    """A DataFrame's partition count: the basis of any performance reasoning."""
    n = df.rdd.getNumPartitions()
    return f"{label + ': ' if label else ''}{n} partition(s)"


def rows_per_partition(df: DataFrame) -> list[int]:
    """
    Number of rows per partition.

    Useful to diagnose DATA SKEW: if one partition holds 90% of the rows, a
    single task does the work while the others wait, and the job is as slow as
    a single-threaded one.

    Cost: triggers an ACTION (mapPartitions + collect). Keep it for debugging
    and small volumes.
    """
    return df.rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()


def log_dataframe(logger: logging.Logger, df: DataFrame, name: str,
                  count: bool = True) -> int | None:
    """Log a DataFrame summary. count=False avoids an expensive action."""
    logger.info(f"{name}: {len(df.columns)} columns, {partition_report(df)}")
    if not count:
        return None
    n = df.count()
    logger.info(f"{name}: {n:,} rows")
    return n
