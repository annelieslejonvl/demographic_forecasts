"""
Socioeconomic feature engineering for demographic forecasting.

This module contains functions to create derived features from socioeconomic data
while avoiding temporal leakage.

IMPORTANT: All features must use lagged versions of variables that can change as
a result of the outcome (moving). This includes:
- hh_pos (household position can change when moving)
- Any event that occurs in year t
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import List


def cast_events_to_bool(df: DataFrame) -> DataFrame:
    """
    Cast all event columns to boolean type.

    Args:
        df: Spark DataFrame with event columns

    Returns:
        DataFrame with event columns cast to boolean
    """
    event_cols = [
        'birth1_event', 'birth2_event', 'divorce_event', 'getalifeother_event',
        'birth1_event_lag1', 'birth2_event_lag1', 'divorce_event_lag1', 'getalifeother_event_lag1',
        'birth1_event_lag2', 'birth2_event_lag2', 'divorce_event_lag2', 'getalifeother_event_lag2',
        'y_moved_lag1', 'y_moved_lag2'
    ]

    for col in event_cols:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast('boolean'))

    return df


def create_hh_pos_features(
    df: DataFrame,
    id_col: str = "sid",
    time_col: str = "year"
) -> DataFrame:
    """
    Create household position features with proper temporal handling.

    IMPORTANT: hh_pos can change as a result of moving, so we must use
    lagged versions to avoid temporal leakage.

    Features created:
    - hh_pos_lag1: Household position in previous year (SAFE)
    - hh_pos_changed: Whether household position changed (1 if changed, 0 otherwise)

    Args:
        df: Spark DataFrame with hh_pos column
        id_col: Individual identifier column
        time_col: Time column

    Returns:
        DataFrame with new hh_pos features
    """
    window_spec = Window.partitionBy(id_col).orderBy(time_col)

    # Create lagged household position
    df = df.withColumn('hh_pos_lag1', F.lag('hh_pos', 1).over(window_spec))

    # Detect changes in household position
    # This is useful for understanding household dynamics
    df = df.withColumn(
        'hh_pos_changed',
        F.when(
            (F.col('hh_pos') != F.col('hh_pos_lag1')) & F.col('hh_pos_lag1').isNotNull(),
            1
        ).otherwise(0).cast('int')
    )

    return df


def create_income_features_spark(
    df: DataFrame,
    id_col: str = "sid",
    time_col: str = "year"
) -> DataFrame:
    """
    Create income-related features.

    Features created:
    - income_lag1: Lagged income
    - income_change: Relative income change
    - income_norm: Income normalized by median
    - income_drop: Boolean indicator for significant income drop
    - income_rise: Boolean indicator for significant income rise
    - low_income_birth: Low income × recent birth interaction
    - high_income_divorce: High income × divorce interaction
    - low_income_divorce: Low income × divorce interaction
    - partnership_income: Income at time of partnership formation
    - income_quintile: Income quintile (1-5)

    Args:
        df: Spark DataFrame with income columns
        id_col: Individual identifier column
        time_col: Time column

    Returns:
        DataFrame with income features
    """
    window_spec = Window.partitionBy(id_col).orderBy(time_col)

    # 1. Lagged income
    df = df.withColumn('income_lag1', F.lag('MS_ADI_PP', 1).over(window_spec))

    # 2. Income change
    df = df.withColumn(
        'income_change',
        (F.col('MS_ADI_PP') - F.col('income_lag1')) / (F.col('income_lag1') + 1)
    )

    # 3. Income normalized
    median_income = df.approxQuantile('MS_ADI_PP', [0.5], 0.01)[0]
    df = df.withColumn('income_norm', F.col('MS_ADI_PP') / median_income)

    # 4. Income shocks (cast to boolean!)
    df = df.withColumn(
        'income_drop',
        (F.col('income_change') < -0.15).cast('boolean')
    )

    df = df.withColumn(
        'income_rise',
        (F.col('income_change') > 0.20).cast('boolean')
    )

    # 5. Income interactions (cast to boolean!)
    df = df.withColumn(
        'low_income_birth',
        ((F.col('income_norm') < 0.7) &
         (F.col('birth1_event_lag1') | F.col('birth2_event_lag1'))).cast('boolean')
    )

    df = df.withColumn(
        'high_income_divorce',
        ((F.col('income_norm') > 1.5) & F.col('divorce_event_lag1')).cast('boolean')
    )

    df = df.withColumn(
        'low_income_divorce',
        ((F.col('income_norm') < 0.8) & F.col('divorce_event_lag1')).cast('boolean')
    )

    # 6. Income * partnership
    df = df.withColumn(
        'partnership_income',
        F.when(F.col('getalifeother_event_lag1'), F.col('income_norm')).otherwise(0)
    )

    # 7. Income quintiles (int)
    quantiles = df.approxQuantile('MS_ADI_PP', [0.2, 0.4, 0.6, 0.8], 0.01)
    df = df.withColumn(
        'income_quintile',
        F.when(F.col('MS_ADI_PP') < quantiles[0], 1)
         .when(F.col('MS_ADI_PP') < quantiles[1], 2)
         .when(F.col('MS_ADI_PP') < quantiles[2], 3)
         .when(F.col('MS_ADI_PP') < quantiles[3], 4)
         .otherwise(5)
    )

    return df


def create_event_interactions_spark(df: DataFrame) -> DataFrame:
    """
    Create event interaction features.

    CRITICAL: Uses hh_pos_lag1 instead of hh_pos to avoid temporal leakage,
    since household position can change as a result of moving.

    Features created:
    - recent_birth: Any birth in past year
    - new_family: Birth + partnership formation
    - family_expansion: Second birth after first
    - family_break_safe: Divorce + lagged household size > 1 (SAFE - uses lag!)
    - constrained_young_family_safe: Young + low income + birth + large household (SAFE - uses lag!)
    - recent_mover: Moved in past 1-2 years
    - frequent_mover: Moved in both past years
    - age_x_life_event: Age × life event interaction
    - mobility_history: Sum of recent moves
    - total_recent_events: Count of recent life events
    - multiple_events: Boolean for multiple simultaneous events
    - coupled_x_birth: Relationship status × birth interaction
    - any_life_event_lag1: Any life event in past year
    - recent_move_x_life_event: Recent move × life event interaction

    Args:
        df: Spark DataFrame with event and hh_pos_lag1 columns

    Returns:
        DataFrame with event interaction features
    """
    df = df.withColumn(
        'recent_birth',
        (F.col('birth1_event_lag1') | F.col('birth2_event_lag1')).cast('boolean')
    )

    df = df.withColumn(
        'new_family',
        (F.col('recent_birth') & F.col('getalifeother_event_lag1')).cast('boolean')
    )

    # FIXED: Only use lagged events to avoid temporal leakage
    df = df.withColumn(
        'family_expansion',
        (F.col('birth2_event_lag1') &
         (F.col('birth1_event_lag1') | F.col('birth1_event_lag2'))).cast('boolean')
    )

    # FIXED: Use hh_pos_lag1 instead of hh_pos to avoid temporal leakage
    # Renamed to family_break_safe to indicate this is the safe version
    df = df.withColumn(
        'family_break_safe',
        (F.col('divorce_event_lag1') & (F.col('hh_pos_lag1') > 1)).cast('boolean')
    )

    df = df.withColumn(
        'recent_mover',
        (F.col('y_moved_lag1') | F.col('y_moved_lag2')).cast('boolean')
    )

    df = df.withColumn(
        'frequent_mover',
        (F.col('y_moved_lag1') & F.col('y_moved_lag2')).cast('boolean')
    )

    # FIX: Cast booleans to int before multiplication
    df = df.withColumn(
        'age_x_life_event',
        (F.col('age') * F.greatest(
            F.col('birth1_event_lag1').cast('int'),
            F.col('divorce_event_lag1').cast('int')
        )).cast('int')
    )

    # FIX: Use correct column names and cast
    df = df.withColumn(
        'mobility_history',
        (F.col('y_moved_lag1').cast('int') + F.col('y_moved_lag2').cast('int'))
    )

    df = df.withColumn(
        'total_recent_events',
        F.col('birth1_event_lag1').cast('int') +
        F.col('birth2_event_lag1').cast('int') +
        F.col('divorce_event_lag1').cast('int') +
        F.col('getalifeother_event_lag1').cast('int')
    )

    df = df.withColumn(
        "multiple_events",
        (F.col("total_recent_events") > 1).cast("boolean")
    )

    # Coupled × birth event
    df = df.withColumn(
        "coupled_x_birth",
        (F.col("coupled").cast("int") * F.col("birth1_event_lag1").cast("int"))
    )

    # Any recent life event (binary indicator)
    df = df.withColumn(
        "any_life_event_lag1",
        (
            F.col("birth1_event_lag1") |
            F.col("birth2_event_lag1") |
            F.col("divorce_event_lag1")
        ).cast("int")
    )

    # Recently moved × life event
    df = df.withColumn(
        "recent_move_x_life_event",
        (F.col("y_moved_lag1").cast("int") * F.col("any_life_event_lag1").cast('int'))
    )

    return df


def create_age_interactions_spark(df: DataFrame) -> DataFrame:
    """
    Create age-related interaction features.

    CRITICAL: Uses hh_pos_lag1 instead of hh_pos in constrained_young_family_safe
    to avoid temporal leakage.

    Features created:
    - age_norm: Age normalized to [0, 1] range
    - divorce_x_age: Divorce × age interaction
    - birth_x_age: Birth × age interaction
    - young_parent: Young age + recent birth
    - constrained_young_family_safe: Young + low income + birth + large household (SAFE - uses lag!)
    - older_parent: Older age + first birth

    Args:
        df: Spark DataFrame with age and event columns

    Returns:
        DataFrame with age interaction features
    """
    df = df.withColumn('age_norm', F.col('age') / 100)

    # Age * events (floats where applicable)
    df = df.withColumn(
        'divorce_x_age',
        F.when(F.col('divorce_event_lag1'), F.col('age_norm')).otherwise(0)
    )

    df = df.withColumn(
        'birth_x_age',
        F.when(F.col('recent_birth'), F.col('age_norm')).otherwise(0)
    )

    # Life stage indicators (cast to boolean!)
    df = df.withColumn(
        'young_parent',
        ((F.col('age') < 40) & F.col('recent_birth')).cast('boolean')
    )

    # FIXED: Use hh_pos_lag1 instead of hh_pos to avoid temporal leakage
    # Renamed to constrained_young_family_safe to indicate this is the safe version
    df = df.withColumn(
        'constrained_young_family_safe',
        ((F.col('age') < 35) &
         (F.col('income_norm') < 0.7) &
         F.col('recent_birth') &
         (F.col('hh_pos_lag1') > 2)).cast('boolean')
    )

    df = df.withColumn(
        'older_parent',
        ((F.col('age') > 35) & F.col('birth1_event_lag1')).cast('boolean')
    )

    return df


def create_all_socioeconomic_features(
    df: DataFrame,
    id_col: str = "sid",
    time_col: str = "year",
    include_hh_pos_features: bool = True
) -> DataFrame:
    """
    Apply all socioeconomic feature engineering transformations.

    This is the main entry point for feature engineering. It applies
    all transformations in the correct order.

    IMPORTANT PREREQUISITES:
    Before calling this function, you MUST have already created event history features
    using the event history transformation. Specifically, these columns must exist:
    - birth1_event_lag1, birth1_event_lag2
    - birth2_event_lag1, birth2_event_lag2
    - divorce_event_lag1, divorce_event_lag2
    - getalifeother_event_lag1, getalifeother_event_lag2
    - y_moved_lag1, y_moved_lag2

    See the example notebook or examples/socioeconomic_features_example.py for how to
    create these columns first.

    Args:
        df: Spark DataFrame with raw features AND event history features (lag1, lag2)
        id_col: Individual identifier column
        time_col: Time column
        include_hh_pos_features: Whether to create hh_pos features (default True)

    Returns:
        DataFrame with all engineered features

    Raises:
        ValueError: If required event history columns are missing
    """
    # Validate required columns exist
    required_columns = [
        'birth1_event_lag1', 'birth2_event_lag1',
        'divorce_event_lag1', 'getalifeother_event_lag1',
        'y_moved_lag1', 'y_moved_lag2'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required event history columns: {missing_columns}\n\n"
            "Before calling create_all_socioeconomic_features(), you must create "
            "event history features (first, at_risk, censored, lag1, lag2) using "
            "create_event_history_features() or similar.\n\n"
            "See examples/socioeconomic_features_example.py for the correct order."
        )

    # Apply transformations in order
    df = cast_events_to_bool(df)

    if include_hh_pos_features:
        df = create_hh_pos_features(df, id_col, time_col)

    df = create_income_features_spark(df, id_col, time_col)
    df = create_event_interactions_spark(df)
    df = create_age_interactions_spark(df)

    return df


def get_socioeconomic_feature_list() -> List[str]:
    """
    Get list of all features created by this module.

    Returns:
        List of feature names
    """
    return [
        # Household position features
        'hh_pos_lag1',
        'hh_pos_changed',

        # Income features
        'income_lag1',
        'income_change',
        'income_norm',
        'income_drop',
        'income_rise',
        'low_income_birth',
        'high_income_divorce',
        'low_income_divorce',
        'partnership_income',
        'income_quintile',

        # Event interactions
        'recent_birth',
        'new_family',
        'family_expansion',
        'family_break_safe',  # Uses hh_pos_lag1
        'recent_mover',
        'frequent_mover',
        'age_x_life_event',
        'mobility_history',
        'total_recent_events',
        'multiple_events',
        'coupled_x_birth',
        'any_life_event_lag1',
        'recent_move_x_life_event',

        # Age interactions
        'age_norm',
        'divorce_x_age',
        'birth_x_age',
        'young_parent',
        'constrained_young_family_safe',  # Uses hh_pos_lag1
        'older_parent',
    ]
