"""
Moving History Feature Engineering for Internal Migration Prediction.

Creates features that capture past mobility patterns, which are the strongest
predictors of future moves in migration research.
"""
from typing import List, Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_moving_history_features(
    df: DataFrame,
    max_lag: int = 5,
    group_cols: Optional[List[str]] = None
) -> DataFrame:
    """
    Create moving history features from longitudinal person-year data.

    Args:
        df: DataFrame with columns [sid, year, y_moved, y_moved_lag1, y_moved_lag2, ...]
        max_lag: Maximum number of years to look back (default 5)
        group_cols: Optional grouping columns for aggregations (e.g., ['eerste_nationaliteit', 'age_group'])

    Returns:
        DataFrame with additional moving history features
    """

    # =============================================================================
    # 1. MOVING FREQUENCY FEATURES (most important)
    # =============================================================================

    # Count moves in different time windows
    # Cast boolean columns to int for arithmetic operations
    df = df.withColumn(
        'moves_2yr',
        F.coalesce(F.col('y_moved_lag1').cast('int'), F.lit(0)) +
        F.coalesce(F.col('y_moved_lag2').cast('int'), F.lit(0))
    )

    # If you have more lags, add them
    # Construct dynamically based on available columns
    lag_cols = [c for c in df.columns if c.startswith('y_moved_lag')]
    lag_cols_available = sorted(lag_cols, key=lambda x: int(x.split('lag')[1]))[:max_lag]

    if len(lag_cols_available) >= 3:
        df = df.withColumn(
            'moves_3yr',
            sum(F.coalesce(F.col(c).cast('int'), F.lit(0)) for c in lag_cols_available[:3])
        )

    if len(lag_cols_available) >= 5:
        df = df.withColumn(
            'moves_5yr',
            sum(F.coalesce(F.col(c).cast('int'), F.lit(0)) for c in lag_cols_available[:5])
        )

    # =============================================================================
    # 2. SERIAL MOVER INDICATORS
    # =============================================================================

    # Binary: moved 2+ times in recent years
    df = df.withColumn(
        'is_serial_mover',
        F.when(F.col('moves_2yr') >= 2, 1).otherwise(0)
    )

    if 'moves_5yr' in df.columns:
        df = df.withColumn(
            'is_frequent_mover',
            F.when(F.col('moves_5yr') >= 3, 1).otherwise(0)
        )

    # =============================================================================
    # 3. TIME SINCE LAST MOVE (critical stability indicator)
    # =============================================================================

    # Estimate years since last move from lags
    # If y_moved_lag1 = 1 (or True), then moved 1 year ago
    # If y_moved_lag1 = 0 (or False) but y_moved_lag2 = 1, then moved 2 years ago, etc.

    years_since_move_expr = F.lit(None).cast('int')
    for i, col in enumerate(lag_cols_available, start=1):
        years_since_move_expr = F.when(
            F.col(col).cast('boolean') == True,
            F.lit(i)
        ).otherwise(years_since_move_expr)

    df = df.withColumn('years_since_last_move', years_since_move_expr)

    # If no move detected in lag window, person is "settled"
    df = df.withColumn(
        'years_since_last_move',
        F.coalesce(F.col('years_since_last_move'), F.lit(max_lag + 1))
    )

    # Binary indicators for recent vs settled
    df = df.withColumn(
        'moved_recently',
        F.when(F.col('years_since_last_move') <= 2, 1).otherwise(0)
    )

    df = df.withColumn(
        'is_settled',
        F.when(F.col('years_since_last_move') > 5, 1).otherwise(0)
    )

    # =============================================================================
    # 4. MOVING RATE (average propensity to move)
    # =============================================================================

    # Average mobility over observed period
    if 'moves_5yr' in df.columns:
        df = df.withColumn(
            'avg_moves_per_year',
            F.col('moves_5yr') / F.lit(5.0)
        )
    elif 'moves_3yr' in df.columns:
        df = df.withColumn(
            'avg_moves_per_year',
            F.col('moves_3yr') / F.lit(3.0)
        )

    # =============================================================================
    # 5. LIFE EVENT CLUSTERING (moves often cluster with life events)
    # =============================================================================

    # Count life events in recent years
    life_event_cols = [
        'birth1_event_lag1', 'birth1_event_lag2',
        'birth2_event_lag1', 'birth2_event_lag2',
        'divorce_event_lag1', 'divorce_event_lag2',
        'getalifeother_event_lag1', 'getalifeother_event_lag2',
    ]

    available_life_events = [c for c in life_event_cols if c in df.columns]

    if available_life_events:
        df = df.withColumn(
            'life_events_2yr',
            sum(F.coalesce(F.col(c).cast('int'), F.lit(0)) for c in available_life_events)
        )

        # Life event density (events per year)
        df = df.withColumn(
            'life_event_density',
            F.col('life_events_2yr') / F.lit(2.0)
        )

    # =============================================================================
    # 6. INTERACTION: MOVES + LIFE EVENTS (key predictor)
    # =============================================================================

    if 'life_events_2yr' in df.columns:
        # Did person move AND have life event recently?
        df = df.withColumn(
            'move_with_life_event',
            F.when(
                (F.col('moves_2yr') >= 1) & (F.col('life_events_2yr') >= 1),
                1
            ).otherwise(0)
        )

        # Ratio of moves to life events (some people move for every event, others don't)
        df = df.withColumn(
            'moves_per_life_event',
            F.when(
                F.col('life_events_2yr') > 0,
                F.col('moves_2yr') / F.col('life_events_2yr')
            ).otherwise(F.lit(0.0))
        )

    # =============================================================================
    # 7. SPECIFIC LIFE EVENT TRIGGERS
    # =============================================================================

    # Birth + move (common: need more space)
    if 'birth2_event_lag1' in df.columns:
        df = df.withColumn(
            'birth_and_move_lag1',
            F.when(
                (F.col('birth2_event_lag1').cast('boolean') == True) &
                (F.col('y_moved_lag1').cast('boolean') == True),
                1
            ).otherwise(0)
        )

    # Divorce + move (common: separation)
    if 'divorce_event_lag1' in df.columns:
        df = df.withColumn(
            'divorce_and_move_lag1',
            F.when(
                (F.col('divorce_event_lag1').cast('boolean') == True) &
                (F.col('y_moved_lag1').cast('boolean') == True),
                1
            ).otherwise(0)
        )

    # =============================================================================
    # 8. MOVING PATTERNS (timing and spacing)
    # =============================================================================

    # Consecutive moves (mover momentum)
    if 'y_moved_lag1' in df.columns and 'y_moved_lag2' in df.columns:
        df = df.withColumn(
            'consecutive_moves',
            F.when(
                (F.col('y_moved_lag1').cast('boolean') == True) &
                (F.col('y_moved_lag2').cast('boolean') == True),
                1
            ).otherwise(0)
        )

    # Regular mover (moves every 2-3 years)
    if len(lag_cols_available) >= 4:
        # Check for pattern like: 1, 0, 1, 0 (regular moves)
        df = df.withColumn(
            'is_regular_mover',
            F.when(
                (F.col('moves_5yr') >= 2) & (F.col('moves_5yr') <= 3),
                1
            ).otherwise(0)
        )

    # =============================================================================
    # 9. STABILITY COMPOSITE SCORE
    # =============================================================================

    # Combined stability indicator (high = stable, low = mobile)
    stability_expr = (
        F.when(F.col('years_since_last_move') > 5, 1.0).otherwise(0.0) * 0.4 +
        F.when(F.col('moves_2yr') == 0, 1.0).otherwise(0.0) * 0.3
    )

    # Add coupled component if column exists
    if 'coupled' in df.columns:
        stability_expr = stability_expr + F.when(
            F.col('coupled').cast('boolean') == True, 1.0
        ).otherwise(0.0) * 0.15

    # Add age component if column exists
    if 'age' in df.columns:
        stability_expr = stability_expr + F.when(
            F.col('age') > 40, 1.0
        ).otherwise(0.0) * 0.15

    df = df.withColumn('stability_score', stability_expr)

    # =============================================================================
    # 10. GROUP-LEVEL AGGREGATIONS (mobility rates by subgroups)
    # =============================================================================

    if group_cols:
        # Compute historical mobility rates for demographic groups
        # Note: Use historical data only (years < current year) to avoid leakage

        for group_col in group_cols:
            if group_col in df.columns:
                # Average mobility rate by group (e.g., nationality, age group)
                window_group = Window.partitionBy(group_col)

                # Cast to int for aggregation
                df = df.withColumn(
                    f'{group_col}_avg_mobility',
                    F.avg(F.col('y_moved').cast('int')).over(window_group)
                )

                # Group mobility rate in recent years (more relevant)
                if 'y_moved_lag1' in df.columns:
                    df = df.withColumn(
                        f'{group_col}_recent_mobility',
                        F.avg(F.col('y_moved_lag1').cast('int')).over(window_group)
                    )

    return df


def create_cross_sectional_features(df: DataFrame) -> DataFrame:
    """
    Create additional features for cross-sectional prediction.
    These combine current state with moving history.

    Args:
        df: DataFrame with moving history features already added

    Returns:
        DataFrame with interaction features
    """

    # =============================================================================
    # INTERACTIONS: Current state × Moving history
    # =============================================================================

    # Age × Moving frequency
    if 'age' in df.columns and 'moves_2yr' in df.columns:
        df = df.withColumn(
            'age_x_moves',
            F.col('age') * F.col('moves_2yr')
        )

        # Young frequent mover (high risk)
        df = df.withColumn(
            'young_frequent_mover',
            F.when(
                (F.col('age') < 35) & (F.col('moves_2yr') >= 2),
                1
            ).otherwise(0)
        )

    # Coupled × Stability
    if 'coupled' in df.columns and 'years_since_last_move' in df.columns:
        df = df.withColumn(
            'coupled_settled',
            F.when(
                (F.col('coupled').cast('boolean') == True) &
                (F.col('years_since_last_move') > 3),
                1
            ).otherwise(0)
        )

    # Recent life event × Past mover
    if 'life_events_2yr' in df.columns and 'moves_2yr' in df.columns:
        df = df.withColumn(
            'life_event_x_mover',
            F.col('life_events_2yr') * F.col('moves_2yr')
        )

    # Income × Moving history (higher income = more stable, unless recent move)
    if 'MS_ADI_PP' in df.columns and 'moves_2yr' in df.columns:
        df = df.withColumn(
            'income_x_moves',
            F.col('MS_ADI_PP') * F.col('moves_2yr')
        )

    return df


def create_all_moving_features(
    df: DataFrame,
    max_lag: int = 5,
    group_cols: Optional[List[str]] = None
) -> DataFrame:
    """
    Convenience function to create all moving history features.

    Args:
        df: Input DataFrame
        max_lag: Number of years to look back
        group_cols: Grouping columns for aggregations

    Returns:
        DataFrame with all moving history features
    """
    df = create_moving_history_features(df, max_lag=max_lag, group_cols=group_cols)
    df = create_cross_sectional_features(df)
    return df


# =============================================================================
# FEATURE LISTS FOR MODELING
# =============================================================================

MOVING_HISTORY_FEATURES = [
    # Frequency
    'moves_2yr',
    'moves_3yr',
    'moves_5yr',
    'is_serial_mover',
    'is_frequent_mover',

    # Timing
    'years_since_last_move',
    'moved_recently',
    'is_settled',

    # Rates
    'avg_moves_per_year',

    # Life events
    'life_events_2yr',
    'life_event_density',
    'move_with_life_event',
    'moves_per_life_event',

    # Specific triggers
    'birth_and_move_lag1',
    'divorce_and_move_lag1',

    # Patterns
    'consecutive_moves',
    'is_regular_mover',

    # Composite
    'stability_score',
]

INTERACTION_FEATURES = [
    'age_x_moves',
    'young_frequent_mover',
    'coupled_settled',
    'life_event_x_mover',
    'income_x_moves',
]


def get_feature_list(df: DataFrame) -> List[str]:
    """
    Get list of available moving history features from DataFrame.

    Args:
        df: DataFrame with moving features

    Returns:
        List of feature column names that exist in df
    """
    all_features = MOVING_HISTORY_FEATURES + INTERACTION_FEATURES
    return [f for f in all_features if f in df.columns]
