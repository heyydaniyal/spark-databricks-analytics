"""Small helpers that do not belong to one phase: timing and the synthetic scale-up data."""

import time
from contextlib import contextmanager

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@contextmanager
def timer(label: str, results: dict):
    """Measure the wall-clock time of a block and store it in `results[label]` (seconds)."""
    start = time.perf_counter()
    yield
    results[label] = round(time.perf_counter() - start, 1)


def replicate_with_noise(df: DataFrame, factor: int, seed: int, key_col: str = "CustomerId") -> DataFrame:
    """Make `factor` copies of every row, with small random changes. FOR THE SCALABILITY TEST ONLY.

    - Each copy gets a new unique ID: original ID * factor + copy number.
    - Balance and EstimatedSalary are moved by up to +/-1%, so the copies are not exact
      duplicates (zero balances stay zero). All other columns are copied as they are.
    - The result is synthetic. It is used to measure how the pipeline behaves on ~1M rows,
      never to report anything about customers.
    """
    copies = df.sparkSession.range(factor).withColumnRenamed("id", "_copy")

    def jitter(col: str, col_seed: int):
        return F.round(F.col(col) * (1 + (F.rand(col_seed) - 0.5) * 0.02), 2)

    return (
        df.crossJoin(copies)
        .withColumn(key_col, F.col(key_col) * factor + F.col("_copy"))
        .withColumn("Balance", jitter("Balance", seed))
        .withColumn("EstimatedSalary", jitter("EstimatedSalary", seed + 1))
        .drop("_copy")
    )
