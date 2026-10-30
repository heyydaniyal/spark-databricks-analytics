"""Single source of truth for storage locations and table names.

Every notebook imports from here, so a path or table name is changed in one place only.
Data lives in a Unity Catalog Volume (not in the Git repo, not in DBFS), as required on
Databricks Free Edition (serverless).
"""

CATALOG = "workspace"
SCHEMA = "churn"
VOLUME = "raw"

# Raw files, uploaded once to the Volume (see GX_00_setup)
RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
CHURN_CSV = f"{RAW_PATH}/churn.csv"
TRANSACTIONS_CSV = f"{RAW_PATH}/bank_transactions.csv"


def table(name: str) -> str:
    """Fully qualified Unity Catalog table name, e.g. workspace.churn.bronze_churn."""
    return f"{CATALOG}.{SCHEMA}.{name}"


# Bronze layer: raw data, values unchanged, plus ingestion metadata
BRONZE_CHURN = table("bronze_churn")
BRONZE_TRANSACTIONS = table("bronze_transactions")

# Output of the Phase 1 quality checks
QUALITY_REPORT = table("quality_report")

# Silver layer: cleaned data (Phase 2)
SILVER_CHURN = table("silver_churn")
SILVER_TRANSACTIONS = table("silver_transactions")

# Gold layer: customer features + fixed train/val/test assignment (Phase 2).
# Every later notebook (EDA, model, segments, graph) reads this one table.
GOLD_CHURN = table("gold_churn_features")

# Synthetic copy of the customers, used ONLY for the scalability test (never for insights)
SCALE_TEST_CHURN = table("scaletest_churn_x100")
SCALE_FACTOR = 100

# Train / validation / test shares, stratified by Exited (see src/ml.py)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Expected sizes, taken from the dataset audit. Ingestion fails loudly if they change.
EXPECTED_ROWS = {"churn": 10_000, "transactions": 1_000}

# One seed for every random operation in the project (splits, sampling, models)
SEED = 42
