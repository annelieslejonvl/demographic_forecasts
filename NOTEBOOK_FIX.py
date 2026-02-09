"""
Copy this code into your notebook to replace the old feature engineering.

IMPORTANT: Run this AFTER you've created the event history features
(the cell that creates _first, _at_risk, _censored, _lag1, _lag2 columns)
"""

# ============================================================================
# Option 1: Use the new module (RECOMMENDED)
# ============================================================================
from src.features.socioeconomic import create_all_socioeconomic_features

# Make sure df2 already has event history features (_lag1, _lag2 columns)!
df_with_features = create_all_socioeconomic_features(
    df2,  # This is your df AFTER creating event history features
    id_col='sid',
    time_col='year',
    include_hh_pos_features=True
)

print(f"✅ Features created! Total columns: {len(df_with_features.columns)}")

# ============================================================================
# Option 2: Keep your existing code but fix it (if you prefer)
# ============================================================================
# If you want to keep your inline function definitions, you need to:
# 1. Add the hh_pos lag creation FIRST
# 2. Update family_break and constrained_young_family to use hh_pos_lag1

from pyspark.sql import Window
from pyspark.sql import functions as F

window_spec = Window.partitionBy('sid').orderBy('year')

def cast_events_to_bool(df):
    """Cast alle event columns naar boolean"""
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


def create_hh_pos_features(df):
    """NEW: Create lagged hh_pos to avoid temporal leakage"""
    df = df.withColumn('hh_pos_lag1', F.lag('hh_pos', 1).over(window_spec))

    df = df.withColumn(
        'hh_pos_changed',
        F.when(
            (F.col('hh_pos') != F.col('hh_pos_lag1')) & F.col('hh_pos_lag1').isNotNull(),
            1
        ).otherwise(0).cast('int')
    )

    return df


def create_income_features_spark(df):
    """Income features"""

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


def create_event_interactions_spark(df):
    """Event interactions - all cast to boolean"""

    df = df.withColumn(
        'recent_birth',
        (F.col('birth1_event_lag1') | F.col('birth2_event_lag1')).cast('boolean')
    )

    df = df.withColumn(
        'new_family',
        (F.col('recent_birth') & F.col('getalifeother_event_lag1')).cast('boolean')
    )

    df = df.withColumn(
        'family_expansion',
        (F.col('birth2_event_lag1') &
         (F.col('birth1_event') | F.col('birth1_event_lag1'))).cast('boolean')
    )

    # FIXED: Use hh_pos_lag1 instead of hh_pos
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


def create_age_interactions_spark(df):
    """Age interactions"""

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

    # FIXED: Use hh_pos_lag1 instead of hh_pos
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


# Apply the pipeline - NOTE: create_hh_pos_features is NEW and MUST come first!
df2 = (df2
       .transform(cast_events_to_bool)
       .transform(create_hh_pos_features)  # NEW - creates hh_pos_lag1
       .transform(create_income_features_spark)
       .transform(create_event_interactions_spark)
       .transform(create_age_interactions_spark))

print(f"✅ Features created! Total columns: {len(df2.columns)}")
