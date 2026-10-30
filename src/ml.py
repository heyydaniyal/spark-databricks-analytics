"""Model preparation shared by every modelling notebook (Phase 2 defines it, Phase 4 uses it).

Two things live here:

1. `add_split_column`: a fixed, stratified train / val / test label for every customer.
   Stored in the gold table, so the model (Phase 4), the segments (Phase 5) and the graph
   (Phase 6) all use exactly the same split, and the test customers are never seen in training.

2. `build_preprocessing_stages`: the steps that turn customer columns into the single numeric
   vector Spark MLlib models need. They are returned as a list of *unfitted* stages, so Phase 4
   can do `Pipeline(stages=build_preprocessing_stages() + [classifier])` and fit everything
   on the training rows only.
"""

from typing import Dict, List, Sequence

from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

LABEL_COL = "Exited"
KEY_COL = "CustomerId"
SPLIT_COL = "split"

# Default model inputs. Text columns are one-hot encoded; number columns go in as they are.
# CustomerId, the group labels (AgeGroup, ...) and Exited are deliberately NOT here.
# The ratio goes in as its log (LogBalanceToSalary): the raw ratio has extreme outliers.
CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "ZeroBalance", "LogBalanceToSalary", "Engagement",
]


def add_split_column(
    df: DataFrame,
    ratios: Dict[str, float],
    seed: int,
    label_col: str = LABEL_COL,
    key_col: str = KEY_COL,
) -> DataFrame:
    """Label every row 'train', 'val' or 'test', stratified by the label.

    Stratified: churners and stayers are split separately, so all three parts keep the same
    churn rate (20.4%). Plain randomSplit does not guarantee this.

    Deterministic: rows are ordered by a hash of (CustomerId, seed), not by rand(). rand()
    depends on how the data happens to be partitioned, so it can give a different split on
    another run; the hash gives the same split every time, on any cluster.
    """
    order = F.xxhash64(F.col(key_col).cast("string"), F.lit(seed))
    position = F.percent_rank().over(Window.partitionBy(label_col).orderBy(order))
    train_cut = ratios["train"]
    val_cut = ratios["train"] + ratios["val"]
    return df.withColumn(
        SPLIT_COL,
        F.when(position < train_cut, "train").when(position < val_cut, "val").otherwise("test"),
    )


def build_preprocessing_stages(
    categorical: Sequence[str] = tuple(CATEGORICAL_FEATURES),
    numeric: Sequence[str] = tuple(NUMERIC_FEATURES),
    output_col: str = "features",
) -> List:
    """StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler (all unfitted).

    - StringIndexer: text -> number (France=0, Spain=1, ...). handleInvalid="keep" means a
      country never seen in training (e.g. in a streaming file) gets its own slot instead of
      crashing the job.
    - OneHotEncoder: number -> 0/1 columns, so the model does not read "Spain=2" as "twice France".
    - VectorAssembler: all inputs -> one vector column, the format MLlib models expect.
    - StandardScaler: every input to mean 0, standard deviation 1, so a balance of 100,000 does
      not dominate an age of 40. Mean and std are learned from the training rows only.
    """
    idx_cols = [f"{c}_idx" for c in categorical]
    ohe_cols = [f"{c}_ohe" for c in categorical]
    return [
        StringIndexer(inputCols=list(categorical), outputCols=idx_cols, handleInvalid="keep"),
        OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols),
        VectorAssembler(inputCols=ohe_cols + list(numeric), outputCol=f"{output_col}_raw"),
        StandardScaler(inputCol=f"{output_col}_raw", outputCol=output_col, withMean=True, withStd=True),
    ]


def vector_slot_names(df: DataFrame, vector_col: str = "features_raw") -> List[str]:
    """Name of every position in an assembled vector (read from the column's ML metadata).

    Needed later to turn model coefficients / importances back into readable names.
    """
    attrs = df.schema[vector_col].metadata["ml_attr"]["attrs"]
    slots = [a for group in attrs.values() for a in group]
    return [a["name"] for a in sorted(slots, key=lambda a: a["idx"])]
