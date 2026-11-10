"""Single source of truth for storage locations and table names.

Every notebook imports from here, so a path or table name is changed in one place only.
Data lives in a Unity Catalog Volume (not in the Git repo, not in DBFS), as required on
Databricks Free Edition (serverless).
"""

import os

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

# Phase 5: segment of every customer (CustomerId, split, segment_id, segment_name)
CUSTOMER_SEGMENTS = table("customer_segments")

# Fitted models live in their own Volume (saved by GX_04 and GX_06, loaded by later notebooks)
MODELS_VOLUME = f"{CATALOG}.{SCHEMA}.models"
MODELS_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/models"
CHURN_MODEL_PATH = f"{MODELS_PATH}/churn_final_model"          # written by GX_04 Step 12
CHURN_MODEL_SETTINGS = f"{MODELS_PATH}/churn_final_model_settings.json"
SEGMENT_MODEL_PATH = f"{MODELS_PATH}/customer_segments_model"  # written by GX_06
SEGMENT_MODEL_SETTINGS = f"{MODELS_PATH}/customer_segments_model_settings.json"

# Phase 6 (GX_07) runs on a local machine, not on Databricks: GraphFrames' algorithms need a
# Spark server plug-in that serverless does not allow (see GX_00). It reads the same raw CSVs
# from the repo's data/raw folder and rebuilds every table it needs with the same src/ code.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_RAW_PATH = os.path.join(REPO_ROOT, "data", "raw")
LOCAL_CHURN_CSV = os.path.join(LOCAL_RAW_PATH, "churn.csv")
LOCAL_TRANSACTIONS_CSV = os.path.join(LOCAL_RAW_PATH, "bank_transactions.csv")
# GraphFrames library for Spark 4 / Scala 2.13. Built against Spark 4.1, so GX_07 runs on
# pyspark 4.1.x. The Python side: pip install graphframes-py==0.12.3 (same version).
GRAPHFRAMES_PACKAGE = "io.graphframes:graphframes-spark4_2.13:0.12.3"

# Synthetic copy of the customers, used ONLY for the scalability test (never for insights)
SCALE_TEST_CHURN = table("scaletest_churn_x100")
SCALE_FACTOR = 100

# Train / validation / test shares, stratified by Exited (see src/ml.py)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Expected sizes, taken from the dataset audit. Ingestion fails loudly if they change.
EXPECTED_ROWS = {"churn": 10_000, "transactions": 1_000}

# One seed for every random operation in the project (splits, sampling, models)
SEED = 42
