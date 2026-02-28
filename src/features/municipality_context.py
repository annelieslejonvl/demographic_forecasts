"""
Municipality-level contextual features for demographic forecasting.

These features capture the socio-economic environment that influences
individual transition decisions (moving, fertility, divorce).

CRITICAL: All municipality features must be LAGGED by at least 1 year
to avoid temporal leakage!
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import List, Optional


def create_municipality_aggregates(
    df: DataFrame,
    id_col: str = "id",
    time_col: str = "year",
    muni_col: str = "refnis_lag1",  # CHANGED: Use ORIGIN municipality!
    lag_years: int = 1
) -> DataFrame:
    """
    Create municipality-level aggregate features.

    These capture the local context (economic conditions, demographics, migration patterns)
    that influence individual decisions.

    CRITICAL: This function expects muni_col='refnis_lag1' (ORIGIN municipality),
    not 'refnis' (which might be DESTINATION if person moved).

    Also expects LAGGED individual features: MS_ADI_PP_lag1, age_lag1, etc.

    Args:
        df: Spark DataFrame with refnis_lag1, MS_ADI_PP_lag1, age_lag1 already created
        id_col: Individual identifier
        time_col: Time column
        muni_col: Municipality identifier (should be 'refnis_lag1' to avoid leakage!)
        lag_years: Number of years to lag (default: 1)

    Returns:
        DataFrame with municipality-level features added
    """
    print(f"Creating municipality-level aggregates using {muni_col} (origin municipality)...")
    print(f"  Aggregating LAGGED individual features to avoid temporal leakage")

    # Window for municipality × year aggregation
    # CRITICAL: Uses refnis_lag1 (where they WERE) not refnis (where they might have moved TO)
    w_muni_year = Window.partitionBy(muni_col, time_col)

    # 1. MIGRATION PATTERNS (most important for moving prediction!)
    # Out-migration rate: % of people who moved OUT of this municipality
    df = df.withColumn('muni_out_migration_rate',
                       F.avg(F.col('y_moved_lag1')).over(w_muni_year))

    # 2. ECONOMIC CONDITIONS
    # Median income in municipality (using LAGGED income)
    df = df.withColumn('muni_median_income',
                       F.expr(f'percentile_approx(MS_ADI_PP_lag1, 0.5)').over(w_muni_year))

    # Income inequality (std dev / mean) (using LAGGED income)
    df = df.withColumn('muni_income_std',
                       F.stddev('MS_ADI_PP_lag1').over(w_muni_year))
    df = df.withColumn('muni_income_mean',
                       F.avg('MS_ADI_PP_lag1').over(w_muni_year))
    df = df.withColumn('muni_income_inequality',
                       F.col('muni_income_std') / (F.col('muni_income_mean') + 1))

    # 3. DEMOGRAPHIC COMPOSITION
    # Age structure (using LAGGED age)
    df = df.withColumn('muni_median_age',
                       F.expr('percentile_approx(age_lag1, 0.5)').over(w_muni_year))

    # Share of young adults (18-35) (using LAGGED age)
    df = df.withColumn('muni_share_young_adults',
                       F.avg(F.when((F.col('age_lag1') >= 17) & (F.col('age_lag1') <= 34), 1).otherwise(0)).over(w_muni_year))

    # Share with children
    df = df.withColumn('has_children_indicator',
                       F.when(F.col('hh_pos') > 2, 1).otherwise(0))  # Rough proxy
    df = df.withColumn('muni_share_families',
                       F.avg('has_children_indicator').over(w_muni_year))

    # 4. LIFE EVENTS (aggregate rates)
    # Birth rate
    df = df.withColumn('muni_birth_rate',
                       F.avg(F.col('birth1_event_lag1')).over(w_muni_year))

    # Divorce rate
    df = df.withColumn('muni_divorce_rate',
                       F.avg(F.col('divorce_event_lag1')).over(w_muni_year))

    # Partnership formation rate
    df = df.withColumn('muni_partnership_rate',
                       F.avg(F.col('getalifeother_event_lag1')).over(w_muni_year))

    # 5. DIVERSITY
    # Nationality diversity (share non-Belgian)
    # Note: eerste_nationaliteit might be numeric code or string
    # Cast to string for safe comparison
    df = df.withColumn('muni_share_foreign',
                       F.avg(F.when(F.col('eerste_nationaliteit').cast('string') != '1.0', 1).otherwise(0)).over(w_muni_year))

    # 6. SIZE/DENSITY (static, but useful)
    # Population size of municipality (log scale for better distribution)
    df = df.withColumn('muni_population',
                       F.count('*').over(w_muni_year))
    df = df.withColumn('muni_log_population',
                       F.log1p(F.col('muni_population')))

    print(f"✓ Created {15} municipality-level features")

    # CRITICAL: LAG ALL FEATURES!
    # These represent conditions in year T-1 that influence decisions in year T
    print(f"⏱️  Lagging features by {lag_years} year(s) to prevent temporal leakage...")

    # Checkpoint before lagging (optional, for large datasets)
    # df = df.checkpoint()

    # Create window for lagging: partition by (muni, sid), order by year
    w_lag = Window.partitionBy(muni_col, id_col).orderBy(time_col)

    # List of all municipality features to lag
    muni_features = [
        'muni_out_migration_rate',
        'muni_median_income',
        'muni_income_inequality',
        'muni_median_age',
        'muni_share_young_adults',
        'muni_share_families',
        'muni_birth_rate',
        'muni_divorce_rate',
        'muni_partnership_rate',
        'muni_share_foreign',
        'muni_log_population'
    ]

    # Lag each feature
    for feature in muni_features:
        df = df.withColumn(f'{feature}_lag{lag_years}',
                          F.lag(feature, lag_years).over(w_lag))
        # Drop original (unlagged) version to prevent accidental use
        df = df.drop(feature)

    # Also drop temporary columns
    df = df.drop('muni_income_std', 'muni_income_mean', 'has_children_indicator', 'muni_population')

    print(f"✓ Lagged features created: {', '.join([f + '_lag1' for f in muni_features[:3]])}...")

    return df


def create_municipality_trends(
    df: DataFrame,
    muni_col: str = "refnis_lag1",  # CHANGED: Use ORIGIN municipality
    time_col: str = "year",
    window_years: int = 3
) -> DataFrame:
    """
    Create municipality-level TREND features (change over time).

    These capture dynamics: is the municipality growing/shrinking,
    getting richer/poorer, etc.

    Args:
        df: DataFrame with lagged municipality features already created
        muni_col: Municipality identifier (should be 'refnis_lag1')
        time_col: Time column
        window_years: Number of years to compute trends over (default: 3)

    Returns:
        DataFrame with trend features added
    """
    print(f"Creating municipality-level trends using {muni_col} (over {window_years} years)...")

    # For trend calculation, we need a window over the past N years
    # IMPORTANT: Use lagged features to compute trends!
    w_muni_id = Window.partitionBy(muni_col, 'id').orderBy(time_col)

    # Income trend: is median income rising or falling?
    df = df.withColumn('muni_income_trend_3yr',
                       F.col('muni_median_income_lag1') -
                       F.lag('muni_median_income_lag1', window_years - 1).over(w_muni_id))

    # Migration trend: is out-migration increasing or decreasing?
    df = df.withColumn('muni_migration_trend_3yr',
                       F.col('muni_out_migration_rate_lag1') -
                       F.lag('muni_out_migration_rate_lag1', window_years - 1).over(w_muni_id))

    # Age structure trend: is municipality aging?
    df = df.withColumn('muni_aging_trend_3yr',
                       F.col('muni_median_age_lag1') -
                       F.lag('muni_median_age_lag1', window_years - 1).over(w_muni_id))

    print("✓ Created municipality trend features")

    return df


def create_individual_vs_municipality_features(
    df: DataFrame,
    id_col: str = "id",
    time_col: str = "year",
    muni_col: str = "refnis_lag1",
    lag_years: int = 1
) -> DataFrame:
    """
    Create features comparing individual characteristics to municipality averages.

    These capture relative position: "Am I richer/poorer than my neighbors?"

    CRITICAL: Uses LAGGED individual features and LAGS the output to avoid temporal leakage!

    Args:
        df: DataFrame with both individual and municipality features (already lagged)
        id_col: Individual identifier
        time_col: Time column
        muni_col: Municipality identifier
        lag_years: Number of years to lag output features (default: 1)

    Returns:
        DataFrame with comparison features added (and lagged)
    """
    print("Creating individual vs municipality comparison features...")

    # Use LAGGED individual features to compare with LAGGED municipality features
    # Income relative to municipality median (both from T-1)
    df = df.withColumn('income_vs_muni_median',
                       (F.col('MS_ADI_PP_lag1') - F.col('muni_median_income_lag1')) /
                       (F.col('muni_median_income_lag1') + 1))

    # Age relative to municipality median (both from T-1)
    df = df.withColumn('age_vs_muni_median',
                       F.col('age_lag1') - F.col('muni_median_age_lag1'))

    # Is individual income in top/bottom quartile of municipality? (using T-1 data)
    df = df.withColumn('high_income_in_muni',
                       (F.col('MS_ADI_PP_lag1') > F.col('muni_median_income_lag1') * 1.5).cast('int'))

    df = df.withColumn('low_income_in_muni',
                       (F.col('MS_ADI_PP_lag1') < F.col('muni_median_income_lag1') * 0.7).cast('int'))

    print("  ✓ Created 4 comparison features using LAGGED individual data")

    # Now LAG these comparison features too!
    # These represent comparisons from T-1 that influence decisions at T
    print(f"  ⏱️  Lagging comparison features by {lag_years} year(s)...")

    w_lag = Window.partitionBy(muni_col, id_col).orderBy(time_col)

    comparison_features = [
        'income_vs_muni_median',
        'age_vs_muni_median',
        'high_income_in_muni',
        'low_income_in_muni'
    ]

    for feature in comparison_features:
        df = df.withColumn(f'{feature}_lag{lag_years}',
                          F.lag(feature, lag_years).over(w_lag))
        # Drop original (unlagged) version to prevent accidental use
        df = df.drop(feature)

    print(f"  ✓ Lagged comparison features: {', '.join([f + '_lag1' for f in comparison_features])}")

    return df


def create_all_municipality_features(
    df: DataFrame,
    id_col: str = "id",
    time_col: str = "year",
    muni_col: str = "refnis_lag1",  # CHANGED: Use ORIGIN municipality!
    lag_years: int = 1,
    include_trends: bool = True
) -> DataFrame:
    """
    Create all municipality-level contextual features.

    This is the main entry point. It creates:
    1. Municipality aggregates (lagged)
    2. Municipality trends (optional)
    3. Individual vs municipality comparisons

    CRITICAL: Expects DataFrame with refnis_lag1, MS_ADI_PP_lag1, age_lag1 already created!

    Args:
        df: Spark DataFrame (with lagged municipality and individual features)
        id_col: Individual ID
        time_col: Time column
        muni_col: Municipality ID (should be 'refnis_lag1' to avoid leakage!)
        lag_years: Years to lag (default: 1)
        include_trends: Whether to compute trends (default: True)

    Returns:
        DataFrame with all municipality features
    """
    print("="*80)
    print("CREATING MUNICIPALITY-LEVEL FEATURES")
    print("="*80)

    # Step 1: Aggregates
    df = create_municipality_aggregates(df, id_col, time_col, muni_col, lag_years)

    # Step 2: Trends (optional)
    if include_trends:
        df = create_municipality_trends(df, muni_col, time_col, window_years=3)

    # Step 3: Individual comparisons (with lagging)
    df = create_individual_vs_municipality_features(df, id_col, time_col, muni_col, lag_years)

    n_features_added = 11 + (3 if include_trends else 0) + 4  # aggregates + trends + comparisons
    print(f"\n✅ Added {n_features_added} municipality-level features")
    print("="*80)

    return df


# Utility: Check for temporal leakage in municipality features
def check_municipality_features_for_leakage(columns: List[str]) -> List[str]:
    """
    Identify municipality features that are NOT properly lagged.

    Args:
        columns: List of column names

    Returns:
        List of potentially leaky municipality features
    """
    leaky = []

    # Prefixes for municipality-related features
    municipality_prefixes = ['muni_', 'income_vs_muni', 'age_vs_muni', 'high_income_in_muni', 'low_income_in_muni']

    for col in columns:
        # Check if it's a municipality feature
        if any(col.startswith(prefix) for prefix in municipality_prefixes):
            # Check if it's lagged (should have '_lag' or be a trend feature)
            if not ('_lag' in col or '_trend' in col):
                leaky.append(col)

    return leaky
