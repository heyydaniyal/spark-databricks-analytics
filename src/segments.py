"""Customer segmentation helpers (Phase 5, GX_06).

The churn model (GX_04) answers *who* is likely to leave. Segments answer *what kind of
customer* they are, so the bank can choose *which offer* to make. Everything a later notebook
needs to put a customer into a segment lives here: the input columns, the preparation steps,
the clustering search and the naming rule.

Three design points:

1. Few, meaningful inputs. Only columns that GX_03 found linked to leaving AND that describe
   the customer's relationship with the bank (age, balance, products, activity). Columns with
   no link (tenure, credit score, salary, credit card) only add noise to the distances.
   Country and gender are left out on purpose: segments should be about how people use the
   bank, not who they are. Both are reported per segment afterwards.

2. Reproducible clustering. Spark's KMeans picks its starting points from a random sample
   whose result depends on how the rows are split into partitions. Two runs with the same
   seed can therefore land on different segments (tested in GX_06). We fix the row order and
   use one partition (`fixed_layout`), then try several seeds and keep the solution with the
   lowest within-segment distance (`fit_best_kmeans`), like scikit-learn's n_init.

3. Names from the data, not from cluster numbers. KMeans numbers its clusters arbitrarily,
   so the names are given by a rule on each segment's profile (`segment_name`).
"""

from typing import Dict, List, Sequence, Tuple

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.ml import KEY_COL, LABEL_COL

# Inputs of the segmentation (Exited is never an input: segments must exist before anyone leaves).
SEGMENT_FEATURES = ["Age", "Balance", "NumOfProducts", "IsActiveMember"]

SEGMENT_VECTOR_COL = "segment_features"
CLUSTER_COL = "cluster"          # raw KMeans number (0..k-1), arbitrary order
SEGMENT_ID_COL = "segment_id"    # 1..k, ordered by churn rate on the training customers
SEGMENT_NAME_COL = "segment_name"
OLDER_SEGMENT_NAME = "Older customers"


def build_segment_prep_stages(features: Sequence[str] = tuple(SEGMENT_FEATURES)) -> List:
    """VectorAssembler -> StandardScaler (both unfitted).

    Scaling matters even more for clustering than for the model: KMeans measures plain
    distances, so without it a balance difference of EUR 10,000 would outweigh 40 years of age.
    """
    return [
        VectorAssembler(inputCols=list(features), outputCol=f"{SEGMENT_VECTOR_COL}_raw"),
        StandardScaler(
            inputCol=f"{SEGMENT_VECTOR_COL}_raw", outputCol=SEGMENT_VECTOR_COL,
            withMean=True, withStd=True,
        ),
    ]


def fixed_layout(df: DataFrame, key_col: str = KEY_COL) -> DataFrame:
    """Same rows, same order, one partition -> KMeans gives the same result on any cluster.

    Fine for a few thousand customers. At millions of rows one would keep the partitions
    and instead save the fitted model, so every later step uses the same segments.
    """
    return df.orderBy(key_col).coalesce(1)


def fit_best_kmeans(
    prepared: DataFrame,
    k: int,
    seeds: Sequence[int],
    max_iter: int = 100,
) -> Tuple[KMeansModel, List[Tuple[int, float]]]:
    """Fit KMeans once per seed; return the model with the lowest cost and every (seed, cost).

    Cost = sum of squared distances from each customer to the centre of its segment
    (lower = tighter segments). Different seeds can end in different local optima; keeping
    the lowest-cost one makes the result less dependent on luck.
    """
    best_model, best_cost, runs = None, None, []
    for seed in seeds:
        model = KMeans(
            k=k, seed=seed, maxIter=max_iter,
            featuresCol=SEGMENT_VECTOR_COL, predictionCol=CLUSTER_COL,
        ).fit(prepared)
        cost = float(model.summary.trainingCost)
        runs.append((seed, cost))
        if best_cost is None or cost < best_cost:
            best_model, best_cost = model, cost
    return best_model, runs


def segment_profile(df: DataFrame, *segment_cols: str, label_col: str = LABEL_COL) -> DataFrame:
    """One row per segment: size, churn rate and the averages that describe the segment.

    `segment_cols`: the column(s) that identify a segment, e.g. CLUSTER_COL, or
    SEGMENT_ID_COL + SEGMENT_NAME_COL. share_pct is the segment's share of `df`.
    """
    total = df.count()

    def pct(condition):
        """Share of the segment's customers for whom `condition` is true, in %."""
        return F.round(F.avg(condition.cast("int")) * 100, 1)

    return (
        df.groupBy(*segment_cols)
        .agg(
            F.count("*").alias("customers"),
            F.round(F.count("*") * 100 / total, 1).alias("share_pct"),
            F.sum(label_col).alias("leavers"),
            F.round(F.avg(label_col) * 100, 1).alias("churn_pct"),
            F.round(F.avg("Age"), 1).alias("avg_age"),
            pct(F.col("Age") >= 50).alias("age_50plus_pct"),
            F.round(F.avg("Balance")).cast("long").alias("avg_balance"),
            pct(F.col("Balance") == 0).alias("zero_balance_pct"),
            F.round(F.avg("NumOfProducts"), 2).alias("avg_products"),
            pct(F.col("NumOfProducts") == 1).alias("one_product_pct"),
            pct(F.col("NumOfProducts") == 2).alias("two_products_pct"),
            pct(F.col("NumOfProducts") >= 3).alias("three_plus_pct"),
            pct(F.col("IsActiveMember") == 1).alias("active_pct"),
            pct(F.col("Geography") == "Germany").alias("germany_pct"),
            pct(F.col("Gender") == "Female").alias("female_pct"),
        )
        .orderBy(*segment_cols)
    )


def segment_name(avg_age: float, active_pct: float, zero_balance_pct: float) -> str:
    """Plain-language name of a segment, from its profile (percentages 0-100).

    Older customers form their own group whatever their balance; everyone else is described
    by the two levers the bank can act on: activity and money held with the bank.
    """
    if avg_age >= 50:
        return OLDER_SEGMENT_NAME
    activity = "Active" if active_pct >= 50 else "Inactive"
    if zero_balance_pct < 50:
        return f"{activity} savers"
    return f"{activity}, little or no savings"


def segment_lookup(train_profile_rows, cluster_col: str = CLUSTER_COL) -> Dict[int, Tuple[int, str]]:
    """KMeans cluster number -> (segment_id, segment_name).

    segment_id orders the segments by churn rate on the training customers (1 = lowest), so
    the numbering means something and does not depend on KMeans' arbitrary cluster numbers.
    """
    ordered = sorted(train_profile_rows, key=lambda r: r["churn_pct"])
    lookup = {
        r[cluster_col]: (i + 1, segment_name(r["avg_age"], r["active_pct"], r["zero_balance_pct"]))
        for i, r in enumerate(ordered)
    }
    names = [name for _, name in lookup.values()]
    assert len(set(names)) == len(names), f"Two segments got the same name: {names}"
    return lookup


def add_segment_labels(df: DataFrame, lookup: Dict[int, Tuple[int, str]], cluster_col: str = CLUSTER_COL) -> DataFrame:
    """Add segment_id and segment_name columns from the raw cluster number."""
    seg_id, seg_name = F.lit(None).cast("int"), F.lit(None).cast("string")
    for cluster, (sid, name) in lookup.items():
        seg_id = F.when(F.col(cluster_col) == cluster, F.lit(sid)).otherwise(seg_id)
        seg_name = F.when(F.col(cluster_col) == cluster, F.lit(name)).otherwise(seg_name)
    return df.withColumn(SEGMENT_ID_COL, seg_id).withColumn(SEGMENT_NAME_COL, seg_name)
