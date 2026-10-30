"""Reading raw CSVs and writing bronze Delta tables."""

from typing import Optional, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

# Metadata columns added to every bronze table (lineage: when and from which file)
META_COLS = ("_ingested_at", "_source_file")


def read_raw_csv(spark: SparkSession, path: str, schema: StructType, **options) -> DataFrame:
    """Read a CSV with an explicit schema, keeping malformed rows instead of dropping them.

    - header=True skips the header line; column names come from `schema`.
    - mode=PERMISSIVE + columnNameOfCorruptRecord: rows that do not fit the schema land in
      `_corrupt_record`, so the quality report can count them.
    - Extra options (encoding, quote, escape) are passed through per file.
    """
    reader = (
        spark.read.format("csv")
        .schema(schema)
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
    )
    for key, value in options.items():
        reader = reader.option(key, value)

    return (
        reader.load(path)
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


def write_bronze(df: DataFrame, table_name: str) -> None:
    """Write (or fully replace) a bronze Delta table.

    Overwrite makes the notebook re-runnable: running it twice gives the same table,
    not duplicated rows.
    """
    write_table(df, table_name)


def write_table(df: DataFrame, table_name: str, partition_by: Optional[Sequence[str]] = None) -> None:
    """Write (or fully replace) any Delta table: bronze, silver or gold.

    - overwrite + overwriteSchema: re-running a notebook rebuilds the table from scratch.
    - partition_by: optional column(s) to split the files on disk by (e.g. Geography), so
      queries that filter on that column only read the matching folders.
    """
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", True)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.saveAsTable(table_name)
