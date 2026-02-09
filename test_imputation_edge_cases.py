#!/usr/bin/env python3
"""
Test imputation with edge cases (all NULL columns, etc.)
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, BooleanType
from src.features.imputation import impute_missing_values


def test_edge_cases():
    """Test imputation with edge cases."""

    spark = SparkSession.builder \
        .appName("TestImputationEdgeCases") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print("=" * 80)
    print("TESTING IMPUTATION WITH EDGE CASES")
    print("=" * 80)

    # Create sample data with edge cases
    schema = StructType([
        StructField("sid", IntegerType(), False),
        StructField("year", IntegerType(), False),
        StructField("all_null_numeric", DoubleType(), True),
        StructField("all_null_boolean", BooleanType(), True),
        StructField("normal_numeric", DoubleType(), True),
        StructField("normal_boolean", BooleanType(), True),
    ])

    data = [
        (1, 2020, None, None, 100.0, True),
        (1, 2021, None, None, None, True),
        (1, 2022, None, None, 120.0, None),
        (2, 2020, None, None, 200.0, False),
        (2, 2021, None, None, None, False),
        (2, 2022, None, None, 220.0, True),
    ]

    df = spark.createDataFrame(data, schema)

    print("\n📊 BEFORE IMPUTATION:")
    print("Note: all_null_* columns have 100% NULLs (edge case)")
    df.show()

    # Apply imputation
    df_imputed = impute_missing_values(
        df,
        strategy="smart",
        exclude_cols=['sid', 'year'],
        verbose=True
    )

    print("\n📊 AFTER IMPUTATION:")
    df_imputed.show()

    # Verify no NULLs remain
    null_count = df_imputed.select(
        [F.sum(F.col(c).isNull().cast('int')).alias(c)
         for c in df_imputed.columns if c not in ['sid', 'year']]
    ).first()

    total_nulls = sum([v for v in null_count.asDict().values() if v is not None])

    if total_nulls == 0:
        print("\n✅ SUCCESS: All NULLs imputed, including edge cases!")
    else:
        print(f"\n❌ FAILED: {total_nulls} NULLs remain")

    spark.stop()
    print("\n✓ Test complete")
    print("=" * 80)


if __name__ == "__main__":
    test_edge_cases()
