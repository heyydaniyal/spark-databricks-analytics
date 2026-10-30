"""Customer group definitions shared by EDA (Phase 3) and the gold feature table (Phase 2).

Defined once here so the bands shown on the slides are exactly the bands the model sees.
Cut points are analytical choices (readable, roughly equal-width), not business thresholds;
the reasoning is logged in docs/decisions.md.

Labels start with a number so they sort in natural order in tables and charts.
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


def age_group(col: str = "Age") -> Column:
    c = F.col(col)
    return (
        F.when(c <= 30, "1. 18-30")
        .when(c <= 40, "2. 31-40")
        .when(c <= 50, "3. 41-50")
        .when(c <= 60, "4. 51-60")
        .otherwise("5. 60+")
    )


def tenure_group(col: str = "Tenure") -> Column:
    c = F.col(col)
    return F.when(c <= 2, "1. 0-2").when(c <= 6, "2. 3-6").otherwise("3. 7-10")


def balance_group(col: str = "Balance") -> Column:
    c = F.col(col)
    return (
        F.when(c == 0, "1. Zero")
        .when(c < 50_000, "2. <50k")
        .when(c < 100_000, "3. 50-100k")
        .when(c < 150_000, "4. 100-150k")
        .otherwise("5. 150k+")
    )


def credit_score_group(col: str = "CreditScore") -> Column:
    c = F.col(col)
    return (
        F.when(c < 500, "1. <500")
        .when(c < 600, "2. 500-599")
        .when(c < 700, "3. 600-699")
        .when(c < 800, "4. 700-799")
        .otherwise("5. 800+")
    )


def products_flag(col: str = "NumOfProducts") -> Column:
    """1, 2, or 3+ products (3 and 4 are rare and behave alike)."""
    c = F.col(col)
    return F.when(c == 1, "1").when(c == 2, "2").otherwise("3+")


def add_customer_groups(df: DataFrame) -> DataFrame:
    """Add all group columns at once. The input DataFrame is not modified."""
    return df.select(
        "*",
        age_group().alias("AgeGroup"),
        tenure_group().alias("TenureGroup"),
        balance_group().alias("BalanceGroup"),
        credit_score_group().alias("CreditScoreGroup"),
        products_flag().alias("ProductsFlag"),
    )


# ---------------------------------------------------------------------------
# Phase 2: engineered features for the gold table
# ---------------------------------------------------------------------------

def add_engineered_features(df: DataFrame) -> DataFrame:
    """New customer descriptors, each motivated by a GX_01 / EDA finding.

    - ZeroBalance (0/1): 36% of customers have exactly 0 balance; they churn much less.
    - BalanceToSalary: balance relative to income. Null if salary is 0 or missing (never
      happens here, but avoids a division by zero).
    - LogBalanceToSalary: log(1 + BalanceToSalary). The plain ratio is extremely skewed
      (median 0.75, max 10,615) because some salaries are unrealistically small (min 11.58);
      the log keeps the order but stops a few extreme customers from dominating the model.
    - Engagement (0-3): IsActiveMember + HasCrCard + (holds 2 or more products).
    """
    salary = F.col("EstimatedSalary")
    ratio = F.when(salary > 0, F.round(F.col("Balance") / salary, 4))
    return df.select(
        "*",
        (F.col("Balance") == 0).cast("int").alias("ZeroBalance"),
        ratio.alias("BalanceToSalary"),
        F.round(F.log1p(ratio), 4).alias("LogBalanceToSalary"),
        (
            F.col("IsActiveMember")
            + F.col("HasCrCard")
            + (F.col("NumOfProducts") >= 2).cast("int")
        ).alias("Engagement"),
    )


def build_gold_features(silver_churn: DataFrame) -> DataFrame:
    """Silver customers -> gold customers: customer groups + engineered features."""
    return silver_churn.transform(add_customer_groups).transform(add_engineered_features)
