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
