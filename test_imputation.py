#!/usr/bin/env python3
"""
Test imputation functionality to verify NULLs are properly handled.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, BooleanType
from src.features.imputation import impute_missing_values, get_null_summary


def test_imputation():
    """Test the imputation functionality with sample data."""

    # Create Spark session
    spark = SparkSession.builder \
        .appName("TestImputation") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print("=" * 80)
    print("TESTING IMPUTATION FUNCTIONALITY")
    print("=" * 80)

    # Create sample data with NULLs
    schema = StructType([
        StructField("sid", IntegerType(), False),
        StructField("year", IntegerType(), False),
        StructField("numeric_feature", DoubleType(), True),
        StructField("boolean_feature", BooleanType(), True),
        StructField("another_numeric", DoubleType(), True),
    ])

    data = [
        (1, 2020, 100.0, True, 50.0),
        (1, 2021, None, True, 60.0),   # NULL numeric
        (1, 2022, 120.0, None, 70.0),  # NULL boolean
        (2, 2020, 200.0, False, None), # NULL numeric
        (2, 2021, None, False, 80.0),  # NULL numeric
        (2, 2022, 220.0, True, 90.0),
        (3, 2020, 150.0, True, 55.0),
        (3, 2021, 160.0, None, 65.0),  # NULL boolean
        (3, 2022, None, True, None),   # Multiple NULLs
    ]

    df = spark.createDataFrame(data, schema)

    print("\n📊 BEFORE IMPUTATION:")
    print(f"Total rows: {df.count()}")
    df.show()

    # Get NULL summary
    null_summary = get_null_summary(df, exclude_cols=['sid', 'year'])
    print("\nNULL Summary:")
    for col, stats in null_summary.items():
        print(f"  {col}: {stats['count']} NULLs ({stats['percent']:.1f}%)")

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
    null_summary_after = get_null_summary(df_imputed, exclude_cols=['sid', 'year'])

    if not null_summary_after:
        print("\n✅ SUCCESS: All NULLs have been imputed!")
    else:
        print("\n❌ FAILED: Some NULLs remain:")
        for col, stats in null_summary_after.items():
            print(f"  {col}: {stats['count']} NULLs ({stats['percent']:.1f}%)")

    spark.stop()
    print("\n✓ Test complete")
    print("=" * 80)


if __name__ == "__main__":
    test_imputation()
