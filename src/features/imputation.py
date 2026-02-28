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

    Optimized to use minimal Spark jobs (1 aggregation pass instead of per-column scans).

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

    # Classify columns by type
    exclude_set = set(exclude_cols)
    numeric_cols = []
    boolean_cols = []

    for col in df.columns:
        if col in exclude_set:
            continue
        col_type = df.schema[col].dataType
        if isinstance(col_type, BooleanType):
            boolean_cols.append(col)
        elif isinstance(col_type, (NumericType, IntegralType, DecimalType, DoubleType, FloatType)):
            numeric_cols.append(col)

    # --- SINGLE PASS: count NULLs for all columns at once ---
    null_count_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in numeric_cols + boolean_cols
    ]

    if not null_count_exprs:
        if verbose:
            print("\n✓ No imputable columns found")
            print("=" * 80)
        return df

    null_counts_row = df.agg(*null_count_exprs).first()
    null_counts_before = {}
    for c in numeric_cols + boolean_cols:
        cnt = null_counts_row[c]
        if cnt and cnt > 0:
            null_counts_before[c] = cnt

    total_nulls_before = sum(null_counts_before.values())

    if verbose and null_counts_before:
        total_rows = df.count()
        print(f"\n📊 Found {len(null_counts_before)} columns with NULLs")
        print(f"   Total NULL values: {total_nulls_before:,}")
        sorted_nulls = sorted(null_counts_before.items(), key=lambda x: x[1], reverse=True)
        print(f"\n   Top columns with NULLs:")
        for col, count in sorted_nulls[:10]:
            pct = count / total_rows * 100
            print(f"      {col:<50} {count:>10,} ({pct:>5.1f}%)")

    if not null_counts_before:
        if verbose:
            print("\n✓ No NULL values found - no imputation needed!")
            print("=" * 80)
        return df

    imputation_log = {}

    # --- BOOLEAN COLUMNS: single pass to get modes ---
    bool_cols_with_nulls = [c for c in boolean_cols if c in null_counts_before]
    bool_fill = {}
    if bool_cols_with_nulls:
        # Count True values in one pass; mode = True if count_true > count_false
        bool_aggs = []
        for c in bool_cols_with_nulls:
            bool_aggs.append(F.sum(F.when(F.col(c) == True, 1).otherwise(0)).alias(f"{c}__true"))
            bool_aggs.append(F.sum(F.when(F.col(c).isNotNull(), 1).otherwise(0)).alias(f"{c}__nonnull"))
        bool_row = df.agg(*bool_aggs).first()

        for c in bool_cols_with_nulls:
            true_count = bool_row[f"{c}__true"] or 0
            nonnull_count = bool_row[f"{c}__nonnull"] or 0
            if nonnull_count == 0:
                bool_fill[c] = False
                imputation_log[c] = "False (all NULL)"
            else:
                mode_val = true_count > (nonnull_count - true_count)
                bool_fill[c] = mode_val
                imputation_log[c] = f"mode={mode_val}"

        if bool_fill:
            df = df.fillna(bool_fill)

    # --- NUMERIC COLUMNS ---
    num_cols_with_nulls = [c for c in numeric_cols if c in null_counts_before]
    num_fill = {}

    if num_cols_with_nulls:
        if strategy == "zero":
            for c in num_cols_with_nulls:
                num_fill[c] = 0.0
                imputation_log[c] = "0.0"
        elif strategy == "mean":
            # Single pass for all means
            mean_exprs = [F.mean(c).alias(c) for c in num_cols_with_nulls]
            mean_row = df.agg(*mean_exprs).first()
            for c in num_cols_with_nulls:
                val = mean_row[c]
                if val is not None:
                    num_fill[c] = float(val)
                    imputation_log[c] = f"mean={val:.3f}"
                else:
                    num_fill[c] = 0.0
                    imputation_log[c] = "0.0 (all NULL)"
        else:
            # median or smart: single approxQuantile call for ALL columns at once
            medians = df.approxQuantile(num_cols_with_nulls, [0.5], 0.01)
            for i, c in enumerate(num_cols_with_nulls):
                if medians[i] and medians[i][0] is not None:
                    num_fill[c] = float(medians[i][0])
                    imputation_log[c] = f"median={medians[i][0]:.3f}"
                else:
                    num_fill[c] = 0.0
                    imputation_log[c] = "0.0 (all NULL)"

        if num_fill:
            df = df.fillna(num_fill)

    imputed_cols = list(bool_fill.keys()) + list(num_fill.keys())

    if verbose:
        print(f"\n✅ Imputation complete")
        print(f"   Columns imputed: {len(imputed_cols)}")
        print(f"   NULLs before: {total_nulls_before:,}")
        print(f"   Strategy: {strategy}")
        for c in imputed_cols:
            print(f"      {c:<50} → {imputation_log.get(c, '?')}")

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

    Optimized: computes all group medians in a single aggregation, then joins once.

    Args:
        df: Spark DataFrame
        group_col: Column to group by (e.g., 'refnis' for municipality)
        feature_cols: Specific columns to impute (None = all numeric columns with NULLs)
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

    exclude_set = set(exclude_cols)

    # Auto-detect numeric columns with NULLs in a single pass
    if feature_cols is None:
        candidate_cols = [
            c for c in df.columns
            if c not in exclude_set
            and isinstance(df.schema[c].dataType, (NumericType, IntegralType, DecimalType, DoubleType, FloatType))
        ]
        if candidate_cols:
            null_exprs = [
                F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
                for c in candidate_cols
            ]
            null_row = df.agg(*null_exprs).first()
            feature_cols = [c for c in candidate_cols if (null_row[c] or 0) > 0]
        else:
            feature_cols = []

    if not feature_cols:
        if verbose:
            print("✓ No columns need imputation")
        return df

    if verbose:
        print(f"Imputing {len(feature_cols)} columns using {group_col}-specific medians...")

    # Compute ALL group medians in a single aggregation
    median_aggs = [
        F.expr(f"percentile_approx(`{c}`, 0.5, 100) as `{c}_median`")
        for c in feature_cols
    ]
    group_medians = df.groupBy(group_col).agg(*median_aggs)

    # Single join
    df = df.join(F.broadcast(group_medians), on=group_col, how='left')

    # Fill NULLs with group median, then drop temp columns
    for c in feature_cols:
        median_col = f"{c}_median"
        df = df.withColumn(
            c,
            F.when(F.col(c).isNull(), F.col(median_col)).otherwise(F.col(c))
        )
    # Drop all median columns at once
    df = df.drop(*[f"{c}_median" for c in feature_cols])

    # Fill any remaining NULLs with global median
    if verbose:
        print("Filling remaining NULLs with global medians...")

    df = impute_missing_values(df, strategy="smart", exclude_cols=list(exclude_set), verbose=False)

    if verbose:
        print(f"✅ Group-based imputation complete")
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

    exclude_set = set(exclude_cols)
    target_cols = [c for c in df.columns if c not in exclude_set]

    if not target_cols:
        return {}

    # Single pass for all null counts + total rows
    agg_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in target_cols
    ]
    agg_exprs.append(F.count("*").alias("__total_rows__"))

    result = df.agg(*agg_exprs).first()
    total_rows = result["__total_rows__"]

    null_summary = {}
    for col in target_cols:
        null_count = result[col] or 0
        if null_count > 0:
            null_summary[col] = {
                'count': null_count,
                'percent': null_count / total_rows * 100,
                'dtype': str(df.schema[col].dataType)
            }

    return null_summary
