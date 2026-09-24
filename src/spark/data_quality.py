"""
Data quality framework (phase 14).

Principle: an invalid row is NEVER dropped silently.
It is:
  1. counted (per rule, so you know what is breaking);
  2. isolated in a quarantine area you can inspect;
  3. recorded in a timestamped JSON report.

This is the difference between a script and a production pipeline: when
revenue drops 30%, you have to be able to answer "is it the business, or is it
my pipeline?".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class Rule:
    """
    A quality rule.

    name:        short identifier, used in the counters and the reports.
    condition:   Spark expression that must be TRUE for a row to be valid.
    description: readable explanation, carried into the report.
    """
    name: str
    condition: Column
    description: str = ""


@dataclass
class QualityReport:
    dataset: str
    rows_read: int = 0
    rows_valid: int = 0
    rows_invalid: int = 0
    rows_duplicated: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def valid_ratio(self) -> float:
        return self.rows_valid / self.rows_read if self.rows_read else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "generated_at": self.generated_at,
            "rows_read": self.rows_read,
            "rows_valid": self.rows_valid,
            "rows_invalid": self.rows_invalid,
            "rows_duplicated": self.rows_duplicated,
            "valid_ratio": round(self.valid_ratio, 6),
            "failures_by_rule": self.failures,
        }


def _safe(condition: Column) -> Column:
    """
    Turn a condition into a "safe" one.

    Fundamental SQL trap: NULL > 0 is not False, it is NULL. Without this
    coalesce, a missing quantity would trigger NO rule at all and pass as
    valid. So we force NULL -> False (= invalid).
    """
    return F.coalesce(condition, F.lit(False))


ERROR_COL = "_dq_errors"


def apply_rules(df: DataFrame, rules: list[Rule]) -> DataFrame:
    """
    Add a _dq_errors column holding the list of violated rules.

    A single pass over the data: we do not filter rule by rule (which would
    trigger as many reads as there are rules).
    """
    error_array = F.array_compact(F.array(*[
        F.when(~_safe(rule.condition), F.lit(rule.name)) for rule in rules
    ]))
    return df.withColumn(ERROR_COL, error_array)


def split_valid_invalid(df_with_errors: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split the valid rows from the quarantined ones."""
    valid = df_with_errors.filter(F.size(ERROR_COL) == 0).drop(ERROR_COL)
    invalid = df_with_errors.filter(F.size(ERROR_COL) > 0)
    return valid, invalid


def count_failures(df_with_errors: DataFrame, rules: list[Rule]) -> dict[str, int]:
    """
    Count violations per rule in a SINGLE pass (one action).

    Naive: one action per rule -> N reads of the dataset.
    Here : one agg with N sum(when(...)) expressions -> 1 read.
    """
    exprs = [
        F.sum(F.when(F.array_contains(ERROR_COL, rule.name), 1).otherwise(0)).alias(rule.name)
        for rule in rules
    ]
    row = df_with_errors.agg(*exprs).collect()[0].asDict()
    return {name: int(value or 0) for name, value in row.items()}


def log_report(logger: logging.Logger, report: QualityReport) -> None:
    """Print the quality report in the agreed format."""
    logger.info(f"Rows read    : {report.rows_read:,}")
    logger.info(f"Rows valid   : {report.rows_valid:,}")
    logger.info(f"Rows invalid : {report.rows_invalid:,}")
    logger.info(f"Rows removed : {report.rows_read - report.rows_valid:,} "
                f"({report.rows_duplicated:,} of them duplicates)")
    logger.info(f"Valid ratio  : {100 * report.valid_ratio:.2f}%")
    if report.failures:
        logger.info("Breakdown of the violated rules:")
        for name, count in sorted(report.failures.items(), key=lambda kv: -kv[1]):
            if count:
                pct = 100 * count / report.rows_read if report.rows_read else 0
                logger.info(f"    {name:<28} {count:>10,}  ({pct:.2f}%)")


def save_report(report: QualityReport, directory: Path) -> Path:
    """Persist the report as JSON: it becomes auditable and comparable over time."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.dataset}_quality.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def assert_quality(report: QualityReport, min_valid_ratio: float,
                   logger: logging.Logger) -> None:
    """
    Guard rail: stop the pipeline if quality collapses.

    A deliberate trade-off: we do not block on a single invalid row (real data
    is never perfect), but we refuse to propagate a massively broken dataset to
    Gold and then to the data warehouse.
    """
    if report.valid_ratio < min_valid_ratio:
        msg = (f"{report.dataset}: valid ratio {100 * report.valid_ratio:.2f}% "
               f"< threshold {100 * min_valid_ratio:.2f}% -> pipeline stopped")
        logger.error(msg)
        raise ValueError(msg)
