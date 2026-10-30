"""Explicit schemas for the raw CSV files.

Why not inferSchema=True:
- it scans the file an extra time to guess types (costly at scale);
- the guess can change silently if a future file has an odd value;
- an explicit schema documents what we expect and makes bad rows visible.

Bronze keeps the raw values. The only change is column names in the transactions file:
Delta tables do not accept spaces or brackets in column names, so they become snake_case.
Values that need interpretation (timestamp, geolocation, PIN) stay as strings here and are
parsed or dropped in Phase 2 (silver).

Each schema ends with `_corrupt_record`: in PERMISSIVE mode Spark puts any row that does not
match the schema there instead of failing or silently dropping it.
"""

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

CORRUPT_COL = "_corrupt_record"

CHURN_SCHEMA = StructType([
    StructField("RowNumber", IntegerType()),
    StructField("CustomerId", LongType()),
    StructField("Surname", StringType()),
    StructField("CreditScore", IntegerType()),
    StructField("Geography", StringType()),
    StructField("Gender", StringType()),
    StructField("Age", IntegerType()),
    StructField("Tenure", IntegerType()),
    StructField("Balance", DoubleType()),
    StructField("NumOfProducts", IntegerType()),
    StructField("HasCrCard", IntegerType()),
    StructField("IsActiveMember", IntegerType()),
    StructField("EstimatedSalary", DoubleType()),
    StructField("Exited", IntegerType()),
    StructField(CORRUPT_COL, StringType()),
])

# Original header -> bronze column name (order matches the file)
TRANSACTIONS_COLUMNS = {
    "Transaction ID": "transaction_id",
    "Sender Account ID": "sender_account_id",
    "Receiver Account ID": "receiver_account_id",
    "Transaction Amount": "amount",
    "Transaction Type": "transaction_type",
    "Timestamp": "timestamp_raw",
    "Transaction Status": "transaction_status",
    "Fraud Flag": "fraud_flag",
    "Geolocation (Latitude/Longitude)": "geolocation_raw",
    "Device Used": "device_used",
    "Network Slice ID": "network_slice_id",
    "Latency (ms)": "latency_ms",
    "Slice Bandwidth (Mbps)": "slice_bandwidth_mbps",
    "PIN Code": "pin_code",
}

TRANSACTIONS_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("sender_account_id", StringType()),
    StructField("receiver_account_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("transaction_type", StringType()),
    StructField("timestamp_raw", StringType()),       # parsed in silver
    StructField("transaction_status", StringType()),
    StructField("fraud_flag", BooleanType()),
    StructField("geolocation_raw", StringType()),     # e.g. "34.0522 N, -74.006 W", parsed in silver
    StructField("device_used", StringType()),
    StructField("network_slice_id", StringType()),
    StructField("latency_ms", IntegerType()),
    StructField("slice_bandwidth_mbps", IntegerType()),
    StructField("pin_code", StringType()),            # string keeps leading zeros; dropped in silver
    StructField(CORRUPT_COL, StringType()),
])
