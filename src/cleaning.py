"""Bronze -> silver cleaning (Phase 2).

Bronze is an untouched copy of the files. Silver is the cleaned version: only the columns we
use, correct types, values parsed into a usable form. Every decision here comes from a finding
in GX_01 and is logged in docs/decisions.md.

Each function takes a bronze DataFrame and returns a silver DataFrame. Nothing is written here:
the notebook decides where to save, so the same functions also work on new data (e.g. the
streaming files in Phase 7) or on the synthetic copy used in the scalability test.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.schemas import CORRUPT_COL

# 0/1 columns in the customer file (HasCrCard, IsActiveMember are inputs; Exited is the target)
BINARY_COLS = ("HasCrCard", "IsActiveMember", "Exited")

# Geolocation text looks like "34.0522 N, -74.006 W": number, hemisphere letter, comma, number, letter.
GEO_PATTERN = r"^(-?[0-9.]+) ([NS]), (-?[0-9.]+) ([EW])$"

TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"


def drop_corrupt_rows(df: DataFrame) -> DataFrame:
    """Keep only rows that matched the schema when the CSV was read (GX_01 found 0 bad rows)."""
    if CORRUPT_COL not in df.columns:
        return df
    return df.filter(F.col(CORRUPT_COL).isNull())


def clean_churn(bronze: DataFrame) -> DataFrame:
    """Silver customers.

    - Dropped: RowNumber (just the line number in the file), Surname (personal data, never used),
      the corrupt-record column and the ingestion lineage columns.
    - Kept: CustomerId, only as a key to identify customers. It is never a model input.
    - Text columns are trimmed, so " France" and "France" can never become two countries.
    - 0/1 columns are cast to int explicitly (they already are; this makes the contract visible).
    """
    return drop_corrupt_rows(bronze).select(
        F.col("CustomerId").cast("long").alias("CustomerId"),
        F.col("CreditScore").cast("int").alias("CreditScore"),
        F.trim("Geography").alias("Geography"),
        F.trim("Gender").alias("Gender"),
        F.col("Age").cast("int").alias("Age"),
        F.col("Tenure").cast("int").alias("Tenure"),
        F.col("Balance").cast("double").alias("Balance"),
        F.col("NumOfProducts").cast("int").alias("NumOfProducts"),
        F.col("HasCrCard").cast("int").alias("HasCrCard"),
        F.col("IsActiveMember").cast("int").alias("IsActiveMember"),
        F.col("EstimatedSalary").cast("double").alias("EstimatedSalary"),
        F.col("Exited").cast("int").alias("Exited"),
    )


def clean_transactions(bronze: DataFrame) -> DataFrame:
    """Silver transactions.

    - timestamp_raw (text) -> event_time (real timestamp).
    - geolocation_raw (text) -> latitude, longitude (numbers). The hemisphere letters contradict
      the numbers (GX_01: every row says "W", even for positive longitudes), so we trust the
      number's sign and ignore the letter.
    - pin_code is dropped: a security code has no analytical value and should not be stored.
    """
    geo = F.col("geolocation_raw")
    event_time = F.to_timestamp("timestamp_raw", TIMESTAMP_FORMAT)
    return drop_corrupt_rows(bronze).select(
        "transaction_id",
        "sender_account_id",
        "receiver_account_id",
        "amount",
        "transaction_type",
        event_time.alias("event_time"),
        "transaction_status",
        "fraud_flag",
        F.regexp_extract(geo, GEO_PATTERN, 1).cast("double").alias("latitude"),
        F.regexp_extract(geo, GEO_PATTERN, 3).cast("double").alias("longitude"),
        "device_used",
        "network_slice_id",
        "latency_ms",
        "slice_bandwidth_mbps",
    )
