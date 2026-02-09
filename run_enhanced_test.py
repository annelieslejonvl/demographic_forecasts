"""
Test script with enhanced features to recover performance after removing leakage.

This script uses the enhanced_features module to add:
1. Extended lags (lag3)
2. Cumulative history features
3. Rolling average features
4. Life stage indicators

Expected performance:
- Before: 0.60 AUC-ROC, 0.08 AUC-PR (basic features, no leakage)
- After: 0.68-0.74 AUC-ROC, 0.25-0.35 AUC-PR (enhanced features, no leakage)

Usage:
    python run_enhanced_test.py --max-rows=1000000
"""
import sys
import argparse

# Import everything from run_test
from run_test import (
    main, load_model_config, cast_events_to_bool, create_hh_pos_features,
    create_income_features_spark, create_event_interactions_spark,
    create_age_interactions_spark, drop_cols_leakage, get_leaky_columns,
    run_hyperparameter_tuning, fix_dtypes, train_in_memory, train_incremental
)

# Import enhanced features
from enhanced_features import create_all_enhanced_features, create_life_stage_features

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
import pathlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from src.features.socioeconomic import create_all_socioeconomic_features
import mlflow

global window_spec


def main_enhanced(reuse_processed=False, max_rows=None, config_path=None, tune=False, n_trials=50):
    """
    Main training pipeline with ENHANCED features.
    """
    global window_spec

    # Load model configuration if provided
    config = None
    if config_path:
        config = load_model_config(config_path)

    # Output path for processed features
    output_path = "data/processed_features_enhanced"

    # Check if we can reuse
    import os
    success_file = os.path.join(output_path, "_SUCCESS")
    can_reuse = reuse_processed and os.path.isdir(output_path) and os.path.exists(success_file)

    if can_reuse:
        print("="*60)
        print("♻️  REUSING EXISTING ENHANCED FEATURES")
        print("="*60)
        print(f"  Found: {output_path}/")

        import pandas as pd
        import pyarrow.parquet as pq

        parquet_dataset = pq.ParquetDataset(output_path)
        total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
        print(f"  Total rows: {total_rows:,}")

        if tune:
            return run_hyperparameter_tuning(output_path, total_rows, config, n_trials)

        use_incremental = total_rows > 5_000_000

        if use_incremental:
            print(f"  ✓ Using INCREMENTAL training")
            model, metrics = train_incremental(output_path, total_rows, config)
        else:
            print(f"  ✓ Using IN-MEMORY training")
            df_pandas = pd.read_parquet(output_path)
            model, metrics = train_in_memory(df_pandas, config)

        return model, metrics

    # Full pipeline with enhanced features
    print("="*60)
    print("🚀 RUNNING ENHANCED FEATURE ENGINEERING PIPELINE")
    print("="*60)
    print("This adds:")
    print("  • Extended lags (lag1, lag2, lag3)")
    print("  • Cumulative history counts")
    print("  • Rolling 3-year averages")
    print("  • Life stage indicators")
    print("="*60)

    # Spark session
    spark = SparkSession.builder \
        .appName("DemographicForecasts_Enhanced") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.shuffle.partitions", "400") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.python.worker.memory", "4g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.2") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.driver.memoryOverhead", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import tempfile
    checkpoint_dir = tempfile.mkdtemp(prefix="spark_checkpoint_")
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    print("✓ Spark session created")

    window_spec = Window.partitionBy('sid').orderBy('year')

    # Load data
    DATA_PATH = "data/synthetic_with_demographics_fixed"
    files = list(pathlib.Path(DATA_PATH).rglob("*.parquet"))
    df = spark.read.parquet(*[str(file) for file in files])
    print(f"✓ Data loaded: {len(df.columns)} columns")

    # Sample if needed
    total_rows = df.count()
    print(f"Total rows in dataset: {total_rows:,}")

    if max_rows is not None and total_rows > max_rows:
        sample_fraction = max_rows / total_rows
        df = df.sample(fraction=sample_fraction, seed=42)
        print(f"✓ Sampled to {max_rows:,} rows ({sample_fraction:.1%} of data)")
    else:
        print(f"✓ Using full dataset: {total_rows:,} rows")

    # Rename columns
    df = df.withColumnRenamed('moved', 'y_moved')
    df = df.withColumnRenamed('gol', 'getalifeother_event')
    df = df.withColumnRenamed('income_pp', 'MS_ADI_PP')
    df = df.withColumnRenamed('income_hh', 'MS_ADI_HH')
    print("✓ Columns renamed")

    event_cols = [c for c in df.columns if c.endswith("_event")]
    event_cols.extend(['y_moved'])

    # Repartition
    num_partitions = max(200, df.select("sid").distinct().count() // 1000)
    print(f"Repartitioning by 'sid' into {num_partitions} partitions...")
    df = df.repartition(num_partitions, "sid")

    # *** USE ENHANCED FEATURES HERE ***
    print("\n🚀 Creating ENHANCED features...")
    df2 = create_all_enhanced_features(df, event_cols, id_col='sid', t_col='year')
    print("✓ Enhanced event history features created")

    # Checkpoint
    df2 = df2.checkpoint(eager=True)
    row_count = df2.count()
    print(f"✓ Checkpointed after enhanced lags: {row_count:,} rows")

    # Apply rest of pipeline
    df2 = (df2
        .transform(cast_events_to_bool)
        .transform(lambda df: create_hh_pos_features(df, window_spec))
        .transform(lambda df: create_income_features_spark(df, window_spec))
        .transform(create_event_interactions_spark)
        .transform(create_age_interactions_spark))

    print(f"✅ Base features created! Total columns: {len(df2.columns)}")

    # Add life stage features (requires total_recent_events from interactions)
    df2 = create_life_stage_features(df2)
    print(f"✓ Life stage features added")

    # Socioeconomic features
    df2 = create_all_socioeconomic_features(
        df2,
        id_col='sid',
        time_col='year',
        include_hh_pos_features=True
    )
    print(f"✓ Socioeconomic features created: {len(df2.columns)} columns")

    # Final checkpoint
    df2 = df2.checkpoint(eager=True)
    final_count = df2.count()
    print(f"✓ Final df2 checkpointed: {final_count:,} rows, {len(df2.columns)} columns")

    # *** CRITICAL: IMPUTE MISSING VALUES ***
    # Lagged features and window operations create NULLs that must be filled
    print("\n🔧 IMPUTING MISSING VALUES...")
    from src.features.imputation import impute_missing_values
    df2 = impute_missing_values(
        df2,
        strategy="smart",  # median for numeric, mode for boolean
        exclude_cols=['sid', 'year', 'refnis', 'y_moved'],  # Don't impute these
        verbose=True
    )
    print("✓ Imputation complete")

    # Write to disk
    print("\n📁 Writing enhanced processed data to disk...")
    df2.write.mode("overwrite").parquet(output_path)
    print(f"✓ Data written to {output_path}")

    # Stop Spark
    spark.stop()
    print("✓ Spark session stopped")

    # Training
    print("\n📊 Preparing data for training...")
    import pandas as pd
    import pyarrow.parquet as pq

    parquet_dataset = pq.ParquetDataset(output_path)
    total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
    print(f"Total rows in processed data: {total_rows:,}")

    if tune:
        print("\n🔍 HYPERPARAMETER TUNING MODE")
        return run_hyperparameter_tuning(output_path, total_rows, config, n_trials)

    incremental_threshold = 10_000_000
    safe_in_memory_limit = 5_000_000

    if total_rows > incremental_threshold:
        print(f"✓ Dataset is large ({total_rows:,} rows) - using INCREMENTAL training")
        model, metrics = train_incremental(output_path, total_rows, config)
        return model, metrics
    elif total_rows > safe_in_memory_limit:
        print(f"⚠️  Dataset is large ({total_rows:,} rows) - sampling to {safe_in_memory_limit:,} rows")
        df_pandas = pd.read_parquet(output_path)
        if len(df_pandas) > safe_in_memory_limit:
            df_pandas = df_pandas.sample(n=safe_in_memory_limit, random_state=42)
        print(f"✓ Sampled to {len(df_pandas):,} rows")
    else:
        print(f"✓ Dataset size OK ({total_rows:,} rows) - loading into memory")
        df_pandas = pd.read_parquet(output_path)
        print(f"✓ Loaded {len(df_pandas):,} rows")

    model, metrics = train_in_memory(df_pandas, config)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ENHANCED feature engineering pipeline")
    parser.add_argument('--reuse', '-r', action='store_true', help='Reuse existing processed features')
    parser.add_argument('--max-rows', type=int, help='Limit dataset to N rows')
    parser.add_argument('--config', type=str, help='Path to model config YAML')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of tuning trials')

    args = parser.parse_args()

    print("="*60)
    print("🚀 ENHANCED FEATURE PIPELINE")
    print("="*60)
    if args.reuse:
        print("🔄 Reuse flag: Will skip feature engineering if possible")
    if args.max_rows:
        print(f"📊 Max rows: {args.max_rows:,}")
    if args.config:
        print(f"📝 Config: {args.config}")
    if args.tune:
        print(f"🔍 Tuning: {args.n_trials} trials")
    print("="*60)
    print()

    main_enhanced(
        reuse_processed=args.reuse,
        max_rows=args.max_rows,
        config_path=args.config,
        tune=args.tune,
        n_trials=args.n_trials
    )
