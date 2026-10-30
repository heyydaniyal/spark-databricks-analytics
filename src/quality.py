"""Reusable data-quality report.

Phase 1 rule: we MEASURE problems, we do not fix them. Bronze stays a faithful copy of the
source; what to do about each issue is decided in Phase 2 (silver) and logged in
docs/decisions.md.

`quality_report` runs the same checks on any DataFrame and returns them as one tidy Spark
DataFrame (one row per check), so results for several datasets can be stacked and saved
as a single table.

Most per-column checks are computed in ONE aggregation pass over the data (one Spark job),
instead of one `.count()` per column per check.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, StringType, StructField, StructType

from src.ingest import META_COLS
from src.schemas import CORRUPT_COL

REPORT_SCHEMA = StructType([
    StructField("dataset", StringType()),
    StructField("check", StringType()),
    StructField("column", StringType()),
    StructField("observed", StringType()),
    StructField("expected", StringType()),
    StructField("status", StringType()),  # PASS / WARN / FAIL / INFO
])


def _status(ok: bool, fail_level: str = "FAIL") -> str:
    return "PASS" if ok else fail_level


def quality_report(
    df: DataFrame,
    dataset: str,
    expected_rows: Optional[int] = None,
    key_col: Optional[str] = None,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    allowed_values: Optional[Dict[str, Sequence]] = None,
    rate_cols: Iterable[str] = (),
) -> DataFrame:
    """Run standard quality checks and return one row per check.

    Args:
        df: data to check (typically a bronze table).
        dataset: short name used in the report ("churn", "transactions").
        expected_rows: row count we expect; mismatch = FAIL.
        key_col: column that must be unique (an ID); duplicates = FAIL.
        ranges: {column: (min, max)} valid range, inclusive; values outside = FAIL.
        allowed_values: {column: [values]} valid categories; anything else = FAIL.
        rate_cols: 0/1 or boolean columns whose share of 1/True is reported (e.g. the target).
    """
    ranges = ranges or {}
    allowed_values = allowed_values or {}
    data_cols = [c for c in df.columns if c not in META_COLS and c != CORRUPT_COL]
    types = dict(df.dtypes)

    # ---- 1. One aggregation pass for all per-column counts ----
    aggs = [F.count(F.lit(1)).alias("__rows")]
    if CORRUPT_COL in df.columns:
        aggs.append(F.count(CORRUPT_COL).alias("__corrupt"))
    for c in data_cols:
        aggs.append(F.sum(F.col(c).isNull().cast("int")).alias(f"null__{c}"))
        if types[c] == "string":
            aggs.append(F.sum((F.trim(F.col(c)) == "").cast("int")).alias(f"blank__{c}"))
        if isinstance(df.schema[c].dataType, (DoubleType, FloatType)):
            aggs.append(F.sum(F.isnan(F.col(c)).cast("int")).alias(f"nan__{c}"))
    for c, (lo, hi) in ranges.items():
        aggs += [
            F.min(c).alias(f"min__{c}"),
            F.max(c).alias(f"max__{c}"),
            F.sum(((F.col(c) < lo) | (F.col(c) > hi)).cast("int")).alias(f"oor__{c}"),
        ]
    for c, values in allowed_values.items():
        aggs.append(
            F.sum((F.col(c).isNotNull() & ~F.col(c).isin(list(values))).cast("int"))
            .alias(f"bad__{c}")
        )
    for c in rate_cols:
        aggs.append(F.avg(F.col(c).cast("double")).alias(f"rate__{c}"))

    stats = df.agg(*aggs).first().asDict()
    n_rows = stats["__rows"]
    rows = []

    def add(check, column, observed, expected, status):
        rows.append((dataset, check, column, str(observed), str(expected), status))

    # Row count and malformed rows
    add("row_count", "*", n_rows, expected_rows if expected_rows is not None else "-",
        _status(n_rows == expected_rows) if expected_rows is not None else "INFO")
    if "__corrupt" in stats:
        add("malformed_rows", CORRUPT_COL, stats["__corrupt"], 0, _status(stats["__corrupt"] == 0))

    # Missing values: NULL, blank strings, NaN (NaN is not NULL in Spark)
    for c in data_cols:
        add("nulls", c, stats[f"null__{c}"], 0, _status(stats[f"null__{c}"] == 0, "WARN"))
        if f"blank__{c}" in stats:
            n = stats[f"blank__{c}"] or 0
            add("blank_strings", c, n, 0, _status(n == 0, "WARN"))
        if f"nan__{c}" in stats:
            n = stats[f"nan__{c}"] or 0
            add("nan_values", c, n, 0, _status(n == 0, "WARN"))

    # Ranges (observed min..max reported alongside the violation count)
    for c, (lo, hi) in ranges.items():
        n_bad = stats[f"oor__{c}"] or 0
        add("out_of_range", c, f"{n_bad} rows (observed {stats[f'min__{c}']}..{stats[f'max__{c}']})",
            f"{lo}..{hi}", _status(n_bad == 0))

    # Categories
    for c, values in allowed_values.items():
        n_bad = stats[f"bad__{c}"] or 0
        add("unexpected_category", c, n_bad, "/".join(map(str, values)), _status(n_bad == 0))

    # Rates (informational, e.g. churn rate or fraud rate)
    for c in rate_cols:
        rate = stats[f"rate__{c}"]
        add("positive_rate", c, f"{rate:.4f}" if rate is not None else None, "-", "INFO")

    # ---- 2. Duplicates (need their own passes) ----
    data_only = df.select(*data_cols)
    dup_rows = n_rows - data_only.distinct().count()
    add("duplicate_rows", "*", dup_rows, 0, _status(dup_rows == 0, "WARN"))

    if key_col:
        dup_keys = df.groupBy(key_col).count().filter("count > 1").count()
        add("duplicate_keys", key_col, dup_keys, 0, _status(dup_keys == 0))

    return df.sparkSession.createDataFrame(rows, REPORT_SCHEMA)


def category_counts(df: DataFrame, columns: Iterable[str]) -> DataFrame:
    """Value counts for several categorical columns, stacked into one table.

    Output columns: column, value, n, pct. Computed in Spark; only the small result is shown.
    """
    total = df.count()
    parts = [
        df.groupBy(F.col(c).cast("string").alias("value"))
        .count()
        .select(F.lit(c).alias("column"), "value", F.col("count").alias("n"))
        for c in columns
    ]
    stacked = parts[0]
    for p in parts[1:]:
        stacked = stacked.unionByName(p)
    return (
        stacked.withColumn("pct", F.round(F.col("n") / F.lit(total) * 100, 2))
        .orderBy("column", F.desc("n"))
    )
