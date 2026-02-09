"""
Imputation utilities for handling missing values in demographic forecasting.

This module handles NULLs that arise from:
1. Lagged features (first N years have NULLs)
2. Rolling windows and trend features
3. External data joins with incomplete matches
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, NumericType, IntegralType, DecimalType, DoubleType, FloatType
from typing import Dict, List, Optional, Tuple


def impute_missing_values(
    df: DataFrame,
    strategy: str = "smart",
    exclude_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> DataFrame:
    """
    Comprehensive missing value imputation for all feature columns.

    Args:
        df: Spark DataFrame with potential NULL values
        strategy: Imputation strategy
            - "smart": Use median for numeric, mode for boolean, forward-fill for others
            - "median": Median for all numeric features
            - "mean": Mean for all numeric features
            - "zero": Fill with 0
        exclude_cols: Columns to exclude from imputation (e.g., ID, label columns)
        verbose: Print imputation details

    Returns:
        DataFrame with imputed values
    """
    if exclude_cols is None:
        exclude_cols = ['sid', 'year', 'refnis', 'y_moved', 'label']

    if verbose:
        print("=" * 80)
        print("IMPUTING MISSING VALUES")
        print("=" * 80)
        print(f"Strategy: {strategy}")

    # Count NULLs before imputation
    null_counts_before = {}
    total_nulls_before = 0

    for col in df.columns:
        if col not in exclude_cols:
            null_count = df.filter(F.col(col).isNull()).count()
            if null_count > 0:
                null_counts_before[col] = null_count
                total_nulls_before += null_count

    if verbose and null_counts_before:
        print(f"\n📊 Found {len(null_counts_before)} columns with NULLs")
        print(f"   Total NULL values: {total_nulls_before:,}")

        # Show top 10 columns with most NULLs
        sorted_nulls = sorted(null_counts_before.items(), key=lambda x: x[1], reverse=True)
        print(f"\n   Top columns with NULLs:")
        for col, count in sorted_nulls[:10]:
            pct = count / df.count() * 100
            print(f"      {col:<50} {count:>10,} ({pct:>5.1f}%)")

    if not null_counts_before:
        if verbose:
            print("\n✓ No NULL values found - no imputation needed!")
        return df

    # Impute based on strategy
    imputed_cols = []
    imputation_log = {}

    for col in df.columns:
        if col in exclude_cols or col not in null_counts_before:
            continue

        # Get column data type
        col_type = df.schema[col].dataType

        # Determine imputation method based on type and strategy
        if isinstance(col_type, BooleanType):
            # Boolean: use mode (most common value)
            try:
                mode_row = df.groupBy(col).count() \
                            .orderBy(F.desc("count")) \
                            .first()
                if mode_row is not None:
                    mode_val = mode_row[0]
                    if mode_val is not None:
                        df = df.fillna({col: mode_val})
                        imputed_cols.append(col)
                        imputation_log[col] = f"mode={mode_val}"
                    else:
                        # All values are NULL, fill with False
                        df = df.fillna({col: False})
                        imputed_cols.append(col)
                        imputation_log[col] = "False (all NULL)"
                else:
                    # All values are NULL, fill with False
                    df = df.fillna({col: False})
                    imputed_cols.append(col)
                    imputation_log[col] = "False (all NULL)"
            except Exception as e:
                # Fallback: fill with False
                df = df.fillna({col: False})
                imputed_cols.append(col)
                imputation_log[col] = f"False (error: {e})"

        elif isinstance(col_type, (NumericType, IntegralType, DecimalType, DoubleType, FloatType)):
            # Numeric: use median, mean, or zero based on strategy
            if strategy == "zero":
                df = df.fillna({col: 0.0})
                imputed_cols.append(col)
                imputation_log[col] = "0.0"
            elif strategy == "mean":
                mean_val = df.agg(F.mean(col)).first()[0]
                if mean_val is not None:
                    df = df.fillna({col: float(mean_val)})
                    imputed_cols.append(col)
                    imputation_log[col] = f"mean={mean_val:.3f}"
                else:
                    # All values are NULL, fill with 0
                    df = df.fillna({col: 0.0})
                    imputed_cols.append(col)
                    imputation_log[col] = "0.0 (all NULL)"
            else:  # median or smart
                quantiles = df.approxQuantile(col, [0.5], 0.01)
                if quantiles and quantiles[0] is not None:
                    median_val = quantiles[0]
                    df = df.fillna({col: float(median_val)})
                    imputed_cols.append(col)
                    imputation_log[col] = f"median={median_val:.3f}"
                else:
                    # Try mean as fallback
                    mean_val = df.agg(F.mean(col)).first()[0]
                    if mean_val is not None:
                        df = df.fillna({col: float(mean_val)})
                        imputed_cols.append(col)
                        imputation_log[col] = f"mean={mean_val:.3f} (median failed)"
                    else:
                        # All values are NULL, fill with 0
                        df = df.fillna({col: 0.0})
                        imputed_cols.append(col)
                        imputation_log[col] = "0.0 (all NULL)"
        else:
            # String or other types: forward fill or leave as-is
            # For demographic forecasting, we usually don't have string features
            pass

    # Verify imputation worked
    total_nulls_after = 0
    remaining_nulls = {}

    for col in null_counts_before.keys():
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            remaining_nulls[col] = null_count
            total_nulls_after += null_count

    if verbose:
        print(f"\n✅ Imputation complete")
        print(f"   Columns imputed: {len(imputed_cols)}")
        print(f"   NULLs before: {total_nulls_before:,}")
        print(f"   NULLs after:  {total_nulls_after:,}")
        print(f"   Reduction: {total_nulls_before - total_nulls_after:,} ({(1 - total_nulls_after/total_nulls_before)*100:.1f}%)")

        if remaining_nulls:
            print(f"\n⚠️  WARNING: {len(remaining_nulls)} columns still have NULLs:")
            sorted_remaining = sorted(remaining_nulls.items(), key=lambda x: x[1], reverse=True)
            for col, count in sorted_remaining[:5]:
                pct = count / df.count() * 100
                print(f"      {col:<50} {count:>10,} ({pct:>5.1f}%)")

    print("=" * 80)

    return df


def impute_with_group_medians(
    df: DataFrame,
    group_col: str,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> DataFrame:
    """
    Impute missing values using group-specific medians (e.g., per municipality).

    This is more sophisticated than global imputation: it uses the median
    within each group (e.g., municipality) rather than the overall median.

    Args:
        df: Spark DataFrame
        group_col: Column to group by (e.g., 'refnis' for municipality)
        feature_cols: Specific columns to impute (None = all numeric columns)
        exclude_cols: Columns to exclude from imputation
        verbose: Print progress

    Returns:
        DataFrame with imputed values
    """
    if exclude_cols is None:
        exclude_cols = ['sid', 'year', 'refnis', 'y_moved', 'label']

    if verbose:
        print("=" * 80)
        print(f"IMPUTING WITH GROUP-SPECIFIC MEDIANS (by {group_col})")
        print("=" * 80)

    # Auto-detect numeric columns if not specified
    if feature_cols is None:
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                col_type = df.schema[col].dataType
                if isinstance(col_type, (NumericType, IntegralType, DecimalType, DoubleType, FloatType)):
                    # Check if it has NULLs
                    null_count = df.filter(F.col(col).isNull()).count()
                    if null_count > 0:
                        feature_cols.append(col)

    if not feature_cols:
        if verbose:
            print("✓ No columns need imputation")
        return df

    if verbose:
        print(f"Imputing {len(feature_cols)} columns using {group_col}-specific medians...")

    # Compute group-specific medians for each feature
    from pyspark.sql.window import Window

    for col in feature_cols:
        # Window partitioned by group
        w = Window.partitionBy(group_col)

        # Compute median within each group (using approx percentile)
        # Note: We use a DataFrame join approach since window functions don't support percentile_approx

        # Approach: Compute group medians, then join back and fill
        group_medians = df.groupBy(group_col).agg(
            F.expr(f"percentile_approx({col}, 0.5, 100) as {col}_median")
        )

        # Join back
        df = df.join(group_medians, on=group_col, how='left')

        # Fill NULLs with group median
        df = df.withColumn(
            col,
            F.when(F.col(col).isNull(), F.col(f"{col}_median")).otherwise(F.col(col))
        )

        # Drop temporary median column
        df = df.drop(f"{col}_median")

    # Fill any remaining NULLs with global median (for groups with all-NULL values)
    if verbose:
        print("Filling remaining NULLs with global medians...")

    df = impute_missing_values(df, strategy="smart", exclude_cols=exclude_cols, verbose=False)

    if verbose:
        # Count remaining NULLs
        total_nulls = 0
        for col in feature_cols:
            null_count = df.filter(F.col(col).isNull()).count()
            total_nulls += null_count

        print(f"✅ Group-based imputation complete")
        print(f"   Remaining NULLs: {total_nulls:,}")
        print("=" * 80)

    return df


def get_null_summary(df: DataFrame, exclude_cols: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Get a comprehensive summary of NULL values in the DataFrame.

    Args:
        df: Spark DataFrame
        exclude_cols: Columns to exclude from analysis

    Returns:
        Dictionary with NULL statistics per column
    """
    if exclude_cols is None:
        exclude_cols = ['sid', 'year', 'refnis']

    total_rows = df.count()
    null_summary = {}

    for col in df.columns:
        if col not in exclude_cols:
            null_count = df.filter(F.col(col).isNull()).count()
            if null_count > 0:
                null_summary[col] = {
                    'count': null_count,
                    'percent': null_count / total_rows * 100,
                    'dtype': str(df.schema[col].dataType)
                }

    return null_summary
