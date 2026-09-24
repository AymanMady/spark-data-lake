"""
Shared test configuration (phase 15).

Principle: the tests do NOT need the cluster or large volumes. They run on a
local SparkSession, with hand-built DataFrames of a few rows. A test has to be
fast and deterministic.
"""

from __future__ import annotations

import logging

import pytest

# py4j tries to write one last message after pytest has already closed stdout,
# which pollutes the output with a "Logging error" that has nothing to do with
# the tests. So we silence that logger.
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

from src.spark.spark_session import get_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    """
    Local SparkSession, shared by the whole test session.

    scope="session" matters: starting a JVM costs several seconds. Recreating
    it for every test would make the suite unbearably slow.

    local[2] rather than local[*]: two partitions are enough to exercise the
    parallel code, and it avoids saturating the machine during the tests.
    """
    session = get_spark(
        "pytest-suite",
        master="local[2]",
        extra_conf={
            # 2 shuffle partitions: otherwise Spark creates 200 for 5 rows.
            "spark.sql.shuffle.partitions": "2",
            "spark.ui.enabled": "false",
            "spark.sql.adaptive.enabled": "false",
        },
    )
    yield session
    session.stop()
