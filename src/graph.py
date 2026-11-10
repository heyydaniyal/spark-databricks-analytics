"""Graph helpers (Phase 6, GX_07).

Two networks are built in GX_07:

1. A customer look-alike network. Two customers are linked when they look alike on the same
   four inputs as the segments (age, balance, products, activity). The links are found with
   MLlib's LSH (locality-sensitive hashing) instead of comparing every pair of customers, so
   the same code works for millions of customers.

2. The transaction network from bank_transactions.csv: accounts are the points, transfers
   are the links.

Everything here returns plain Spark DataFrames (vertices with an `id` column, edges with
`src` and `dst` columns). Only GX_07 turns them into a GraphFrame, so the inputs can be built
anywhere and only the graph algorithms depend on where GraphFrames runs.

Determinism: every "pick the top ones" step breaks ties by customer number, so the network
is the same on every run and every machine.
"""

from typing import Sequence

from pyspark.ml.feature import BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from src.ml import KEY_COL

# Look-alike rule: the 10 closest customers within a distance of 0.5 (on scaled inputs,
# 0.5 = half a standard deviation). Chosen in GX_07 Step 4.
LOOKALIKE_K = 10
LOOKALIKE_MAX_DISTANCE = 0.5

# LSH settings: bucket width and number of independent hash tables (more tables = fewer
# missed neighbours, more work). Checked against an exact search in GX_07 Step 5.
LSH_BUCKET_LENGTH = 1.0
LSH_HASH_TABLES = 3


def fit_lookalike_lsh(prepared: DataFrame, features_col: str, seed: int) -> BucketedRandomProjectionLSHModel:
    """Fit the LSH model that puts customers into buckets of nearby customers.

    BucketedRandomProjectionLSH projects every customer onto a few random directions and
    cuts each direction into buckets of width LSH_BUCKET_LENGTH. Customers close to each
    other usually land in the same bucket, so only customers sharing a bucket are compared.
    The random directions come from `seed`, so the buckets are the same on every run.
    """
    return BucketedRandomProjectionLSH(
        inputCol=features_col,
        outputCol="lsh_hashes",
        bucketLength=LSH_BUCKET_LENGTH,
        numHashTables=LSH_HASH_TABLES,
        seed=seed,
    ).fit(prepared)


def _top_k(pairs: DataFrame, k: int) -> DataFrame:
    """Keep each customer's k closest partners; ties broken by the partner's number."""
    closest_first = Window.partitionBy("src").orderBy("distance", "dst")
    return (
        pairs.filter(F.col("src") != F.col("dst"))
        .withColumn("rank", F.row_number().over(closest_first))
        .filter(F.col("rank") <= k)
    )


def nearest_lookalikes(
    lsh_model: BucketedRandomProjectionLSHModel,
    customers: DataFrame,
    candidates: DataFrame,
    k: int = LOOKALIKE_K,
    max_distance: float = LOOKALIKE_MAX_DISTANCE,
    key_col: str = KEY_COL,
) -> DataFrame:
    """For every customer in `customers`: its k closest customers in `candidates` (LSH search).

    Returns one row per (customer, look-alike): src, dst, distance, rank (1 = closest).
    A customer with fewer than k candidates within `max_distance` gets fewer rows.
    `customers` and `candidates` can be the same DataFrame (look-alikes inside one group) or
    different ones (e.g. validation customers looking for look-alikes among training ones).
    """
    pairs = lsh_model.approxSimilarityJoin(customers, candidates, max_distance, distCol="distance").select(
        F.col(f"datasetA.{key_col}").alias("src"),
        F.col(f"datasetB.{key_col}").alias("dst"),
        "distance",
    )
    return _top_k(pairs, k)


def euclidean_distance(a: Column, b: Column) -> Column:
    """Distance between two array<double> columns of the same length (no Python UDF)."""
    squared = F.zip_with(a, b, lambda x, y: (x - y) * (x - y))
    return F.sqrt(F.aggregate(squared, F.lit(0.0), lambda acc, x: acc + x))


def exact_lookalikes(
    customers: DataFrame,
    candidates: DataFrame,
    features_col: str,
    k: int = LOOKALIKE_K,
    max_distance: float = LOOKALIKE_MAX_DISTANCE,
    key_col: str = KEY_COL,
) -> DataFrame:
    """Same result shape as `nearest_lookalikes`, but by comparing EVERY pair (cross join).

    Only used to check LSH on a sample: at millions of customers this is the work LSH avoids.
    """
    left = customers.select(F.col(key_col).alias("src"), vector_to_array(features_col).alias("a"))
    right = candidates.select(F.col(key_col).alias("dst"), vector_to_array(features_col).alias("b"))
    pairs = (
        left.crossJoin(right)
        .withColumn("distance", euclidean_distance(F.col("a"), F.col("b")))
        .filter(F.col("distance") <= max_distance)
        .select("src", "dst", "distance")
    )
    return _top_k(pairs, k)


def undirected_links(pairs: DataFrame) -> DataFrame:
    """Look-alike pairs -> network links, stored in BOTH directions.

    "A is among B's 10 closest" is one-sided; a link exists if either side picked the other.
    GraphFrames stores directed edges, so every link is kept as A->B and B->A. Then
    "neighbours" means the same thing for every algorithm, whatever the direction.
    """
    both = pairs.select("src", "dst", "distance").unionByName(
        pairs.select(F.col("dst").alias("src"), F.col("src").alias("dst"), "distance")
    )
    return both.groupBy("src", "dst").agg(F.min("distance").alias("distance"))


def most_common_label(pairs: DataFrame, labels: DataFrame, label_col: str) -> DataFrame:
    """Give each `src` customer the label most common among its look-alikes (`dst`).

    `labels`: one row per look-alike, columns `id` and `label_col` (e.g. a community).
    Ties go to the smallest label, the same rule label propagation uses.
    Returns src, label_col.
    """
    counted = (
        pairs.join(labels.withColumnRenamed("id", "dst"), "dst")
        .groupBy("src", label_col)
        .count()
    )
    most_first = Window.partitionBy("src").orderBy(F.col("count").desc(), F.col(label_col))
    return (
        counted.withColumn("rank", F.row_number().over(most_first))
        .filter(F.col("rank") == 1)
        .select("src", label_col)
    )


def degree_band(col: str = "degree") -> Column:
    """Readable bands for the number of look-alike links (labels sort in natural order)."""
    c = F.col(col)
    return (
        F.when(c.isNull() | (c == 0), "1. none")
        .when(c < 5, "2. 1-4")
        .when(c < 10, "3. 5-9")
        .when(c <= 15, "4. 10-15")
        .otherwise("5. 16+")
    )


def churn_band(col: str) -> Column:
    """Bands of a churn rate (0-1), used to group customers by the churn of their community."""
    c = F.col(col)
    return (
        F.when(c < 0.10, "1. under 10%")
        .when(c < 0.20, "2. 10-20%")
        .when(c < 0.35, "3. 20-35%")
        .when(c < 0.50, "4. 35-50%")
        .otherwise("5. 50% or more")
    )


def transaction_graph_inputs(silver_transactions: DataFrame, edge_cols: Sequence[str]):
    """Transactions -> (vertices, edges) for a GraphFrame.

    Vertices: every account that sends or receives at least once, with how many
    transactions it appears in. Edges: one per transaction, sender -> receiver.
    """
    edges = silver_transactions.select(
        F.col("sender_account_id").alias("src"),
        F.col("receiver_account_id").alias("dst"),
        *edge_cols,
    )
    appearances = edges.select(F.col("src").alias("id")).unionAll(edges.select(F.col("dst").alias("id")))
    vertices = appearances.groupBy("id").agg(F.count("*").alias("transactions"))
    return vertices, edges
