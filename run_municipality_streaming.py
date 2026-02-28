"""
STREAMING year-by-year pipeline with municipality features.

This version processes ONE YEAR AT A TIME from start to finish, avoiding
the massive shuffle operations that occur when processing all years together.

Key differences from run_with_municipality_features.py:
- Each year is loaded, processed, and written independently
- Municipality aggregates computed only for data up to that year
- Minimal memory and disk usage (processes ~5M rows at a time instead of 80M)
- No intermediate checkpoints or full-dataset materializations

Expected disk usage: ~5-10GB peak (vs 50GB+ for full pipeline)
Expected memory usage: ~10-15GB peak (vs 24GB+ for full pipeline)

Usage:
    python run_municipality_streaming.py
    python run_municipality_streaming.py --start-year 2015 --end-year 2020
"""
import sys
import argparse
import os
from pathlib import Path

# Always set JAVA_HOME to JDK 17
_jdk_path = Path.home() / "jdk-17.0.18+8"
if _jdk_path.exists():
    os.environ["JAVA_HOME"] = str(_jdk_path)

# Hadoop winutils.exe required on Windows
if not os.environ.get("HADOOP_HOME"):
    _hadoop_path = Path.home() / "hadoop"
    if _hadoop_path.exists():
        os.environ["HADOOP_HOME"] = str(_hadoop_path)

# Setup logging
from src.utils.logging_setup import setup_logging, configure_spark_logging, log_stage_start, log_stage_complete, log_memory_usage

# Import base functions
from run_enhanced_test import (
    load_model_config, cast_events_to_bool, create_hh_pos_features,
    create_income_features_spark, create_event_interactions_spark,
    create_age_interactions_spark, fix_dtypes, train_in_memory, train_incremental
)
from enhanced_features import create_all_enhanced_features, create_life_stage_features

# Import municipality features
from src.features.municipality_context import (
    create_all_municipality_features,
    check_municipality_features_for_leakage
)

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
import pathlib
from typing import Dict, Any, Optional, List
from src.features.socioeconomic import create_all_socioeconomic_features
from src.features.imputation import impute_missing_values
import pandas as pd
import tempfile
import shutil
import atexit


def process_single_year(
    spark: SparkSession,
    df_all_years: "DataFrame",
    target_year: int,
    event_cols: List[str],
    output_path: str,
    use_municipality_features: bool = False  # DISABLED to save disk space
) -> None:
    """
    Process a single year through the entire feature pipeline.

    This function:
    1. Filters to years <= target_year (to include history for lags)
    2. Creates all features (enhanced, municipality, socioeconomic)
    3. Filters to only target_year for output
    4. Writes to disk as a single parquet file

    Args:
        spark: Active Spark session
        df_all_years: DataFrame with ALL years loaded
        target_year: Year to process and write
        event_cols: List of event column names
        output_path: Directory to write output
        use_municipality_features: Whether to create municipality features
    """
    log_stage_start(f"Year {target_year} - Data Filtering")
    print(f"\n{'='*60}")
    print(f"PROCESSING YEAR {target_year}")
    print(f"{'='*60}")

    # Filter to years <= target_year (need history for lags)
    # But limit how far back we go to control memory
    # lag3 needs 3 years, trends need 3 years → minimum 3 years lookback
    lookback_years = 3  # Only keep 3 years of history for lags
    min_year = max(2010, target_year - lookback_years)

    df = df_all_years.filter(
        (F.col('year') >= min_year) & (F.col('year') <= target_year)
    )

    # Repartition for this subset - use fewer partitions to reduce shuffle overhead
    df = df.repartition(2, "id")

    row_count = df.count()
    print(f"  Data range: {min_year} - {target_year}")
    print(f"  Total rows: {row_count:,}")
    log_stage_complete(f"Year {target_year} - Data Filtering")
    log_memory_usage()

    # Enhanced features
    log_stage_start(f"Year {target_year} - Enhanced Features")
    print(f"  Creating enhanced features...")
    df2 = create_all_enhanced_features(df, event_cols, id_col='id', t_col='year')
    print(f"  ✓ Enhanced features created")
    log_stage_complete(f"Year {target_year} - Enhanced Features")
    log_memory_usage()

    # Municipality features (if enabled and column exists)
    if use_municipality_features:
        log_stage_start(f"Year {target_year} - Municipality Features")
        print(f"  Creating municipality context...")

        # Create lags
        w_individual = Window.partitionBy('id').orderBy('year')
        df2 = df2.withColumn('refnis_lag1', F.lag('refnis', 1).over(w_individual))
        df2 = df2.withColumn('refnis_lag2', F.lag('refnis', 2).over(w_individual))
        df2 = df2.withColumn('MS_ADI_PP_lag1', F.lag('MS_ADI_PP', 1).over(w_individual))
        df2 = df2.withColumn('MS_ADI_HH_lag1', F.lag('MS_ADI_HH', 1).over(w_individual))
        df2 = df2.withColumn('age_lag1', F.lag('age', 1).over(w_individual))

        # Create municipality features
        df2 = create_all_municipality_features(
            df2,
            id_col='id',
            time_col='year',
            muni_col='refnis_lag1',
            lag_years=1,
            include_trends=True
        )
        print(f"  ✓ Municipality features created")
        log_stage_complete(f"Year {target_year} - Municipality Features")
        log_memory_usage()

        # External socioeconomic data
        log_stage_start(f"Year {target_year} - External Socioeconomic")
        try:
            from load_socioeconomic_data import add_socioec_features_to_spark_df
            df2 = add_socioec_features_to_spark_df(
                df2,
                spark,
                socioec_csv_path='df_socioec.csv',
                strategy='static'
            )
            print(f"  ✓ External socioec data added")
        except (FileNotFoundError, Exception) as e:
            print(f"  ⚠️  Skipping external data: {e}")
        log_stage_complete(f"Year {target_year} - External Socioeconomic")
        log_memory_usage()

    # Base features pipeline
    log_stage_start(f"Year {target_year} - Base Features")
    print(f"  Creating base features...")
    df2 = (df2
        .transform(cast_events_to_bool)
        .transform(create_hh_pos_features)
        .transform(create_income_features_spark)
        .transform(create_event_interactions_spark)
        .transform(create_age_interactions_spark))
    print(f"  ✓ Base features created")
    log_stage_complete(f"Year {target_year} - Base Features")
    log_memory_usage()

    # Life stage features
    log_stage_start(f"Year {target_year} - Life Stage Features")
    df2 = create_life_stage_features(df2)
    print(f"  ✓ Life stage features added")
    log_stage_complete(f"Year {target_year} - Life Stage Features")
    log_memory_usage()

    # Socioeconomic features
    log_stage_start(f"Year {target_year} - Socioeconomic Features")
    df2 = create_all_socioeconomic_features(
        df2,
        id_col='id',
        time_col='year',
        include_hh_pos_features=True
    )
    print(f"  ✓ Socioeconomic features created: {len(df2.columns)} columns")
    log_stage_complete(f"Year {target_year} - Socioeconomic Features")
    log_memory_usage()

    # Imputation
    log_stage_start(f"Year {target_year} - Imputation")
    print(f"  Imputing missing values...")
    df2 = impute_missing_values(
        df2,
        strategy="smart",
        exclude_cols=['id', 'year', 'refnis', 'y_moved'],
        verbose=False  # Reduce logging noise
    )
    print(f"  ✓ Imputation complete")
    log_stage_complete(f"Year {target_year} - Imputation")
    log_memory_usage()

    # Filter to ONLY target year for output
    log_stage_start(f"Year {target_year} - Write to Disk")
    print(f"  Filtering to year {target_year} only...")
    df_target = df2.filter(F.col('year') == target_year)

    # Convert to pandas and write
    print(f"  Converting to pandas...", end=" ", flush=True)
    pdf = df_target.toPandas()
    print(f"({len(pdf):,} rows)", end=" ", flush=True)

    year_file = os.path.join(output_path, f"year_{target_year}.parquet")
    pdf.to_parquet(year_file, index=False, engine="pyarrow")

    file_size_mb = os.path.getsize(year_file) / 1024**2
    print(f"✓ Written ({file_size_mb:.1f} MB)")
    log_stage_complete(f"Year {target_year} - Write to Disk")
    log_memory_usage()

    # Cleanup
    del df, df2, df_target, pdf

    print(f"{'='*60}")
    print(f"✓ YEAR {target_year} COMPLETE")
    print(f"{'='*60}\n")


def main_streaming(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    reuse_processed: bool = False,
    config_path: Optional[str] = None,
    tune: bool = False,
    n_trials: int = 50,
    feature_config_path: Optional[str] = None,
    rolling_importance: bool = False,
    target_batch_rows: Optional[int] = None,
    rolling_importance_config: Optional[Dict[str, Any]] = None,
):
    """
    Main streaming pipeline - processes each year independently.
    """
    from datetime import datetime
    log_file = setup_logging(
        log_file=f"municipality_streaming_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        log_dir='.'
    )
    print(f"📝 Logging to: {log_file}")
    log_memory_usage()

    # Load model configuration if provided
    config = None
    if config_path:
        config = load_model_config(config_path)

    output_path = "data/processed_features_with_municipality"

    # Check if we can reuse
    success_file = os.path.join(output_path, "_SUCCESS")
    can_reuse = reuse_processed and os.path.isdir(output_path) and os.path.exists(success_file)

    if can_reuse:
        log_stage_start("Reusing Existing Features")
        print("="*60)
        print("♻️  REUSING EXISTING FEATURES")
        print("="*60)
        print(f"  Found: {output_path}/")

        import pyarrow.parquet as pq
        parquet_dataset = pq.ParquetDataset(output_path)
        total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
        print(f"  Total rows: {total_rows:,}")
        log_stage_complete("Reusing Existing Features")
        log_memory_usage()

        # Continue to training...
        if tune:
            from run_enhanced_test import run_hyperparameter_tuning
            result = run_hyperparameter_tuning(output_path, total_rows, config, n_trials)
            return result

        use_incremental = total_rows > 5_000_000
        if use_incremental:
            model, metrics = train_incremental(
                output_path, total_rows, config,
                feature_config_path=feature_config_path,
                target_batch_rows=target_batch_rows,
                rolling_importance=rolling_importance,
                rolling_importance_config=rolling_importance_config,
            )
        else:
            df_pandas = pd.read_parquet(output_path)
            model, metrics = train_in_memory(
                df_pandas, config,
                feature_config_path=feature_config_path,
                rolling_importance=rolling_importance,
            )

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        return model, metrics

    # Full streaming pipeline
    print("="*60)
    print("STREAMING YEAR-BY-YEAR PIPELINE")
    print("="*60)
    print("This processes each year independently to minimize disk usage")
    print("="*60)

    log_stage_start("Spark Session Initialization")

    # Create temp dir for Spark
    spark_temp_dir = tempfile.mkdtemp(prefix="spark_stream_")

    # Lighter Spark configuration for streaming (reduced to minimize disk spill)
    spark = SparkSession.builder \
        .appName("DemographicForecasts_Streaming") \
        .master("local[4]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.memory.fraction", "0.75") \
        .config("spark.memory.storageFraction", "0.1") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.default.parallelism", "2") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "64MB") \
        .config("spark.sql.autoBroadcastJoinThreshold", "128MB") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000") \
        .config("spark.python.worker.memory", "2g") \
        .config("spark.local.dir", spark_temp_dir) \
        .config("spark.shuffle.spill.compress", "true") \
        .config("spark.shuffle.compress", "true") \
        .config("spark.io.compression.codec", "snappy") \
        .config("spark.driver.extraJavaOptions",
                "-XX:+UseG1GC "
                "-XX:InitiatingHeapOccupancyPercent=50 "
                "-XX:G1HeapRegionSize=16M "
                "-XX:MaxGCPauseMillis=500 "
                "-XX:+ParallelRefProcEnabled "
                "-XX:ParallelGCThreads=8 "
                "-XX:ConcGCThreads=2") \
        .config("spark.ui.enabled", "false") \
        .config("spark.cleaner.periodicGC.interval", "3min") \
        .getOrCreate()

    configure_spark_logging(spark, log_level="WARN")

    # Register cleanup
    def cleanup_spark_temp():
        try:
            if os.path.exists(spark_temp_dir):
                shutil.rmtree(spark_temp_dir)
                print(f"✓ Cleaned up Spark temp: {spark_temp_dir}")
        except Exception as e:
            print(f"⚠️  Could not clean up: {e}")

    atexit.register(cleanup_spark_temp)

    print("✓ Spark session created (streaming config)")
    print(f"  Temp dir: {spark_temp_dir}")
    log_stage_complete("Spark Session Initialization")
    log_memory_usage()

    # Load ALL years (lazy - won't materialize yet)
    log_stage_start("Data Loading")
    DATA_PATH = r"..\AI_datagen\data\real_panel_long"
    files = list(pathlib.Path(DATA_PATH).rglob("*.parquet"))
    df_all = spark.read.parquet(*[str(file) for file in files])

    # Rename columns
    df_all = df_all.withColumnRenamed('moved', 'y_moved')
    df_all = df_all.withColumnRenamed('gol', 'getalifeother_event')
    df_all = df_all.withColumnRenamed('income_pp', 'MS_ADI_PP')
    df_all = df_all.withColumnRenamed('income_hh', 'MS_ADI_HH')

    event_cols = [c for c in df_all.columns if c.endswith("_event")]
    event_cols.extend(['y_moved'])

    # Municipality features DISABLED to save disk space (8.6GB per year!)
    # The municipality aggregates cause massive shuffle operations
    use_municipality_features = False
    print("⚠️  Municipality features DISABLED (to save disk space)")
    print("   This avoids 8.6GB shuffle files per year")
    print("   Enable by setting use_municipality_features=True if you have external disk")

    # Get year range
    years_in_data = [row.year for row in df_all.select('year').distinct().orderBy('year').collect()]
    min_year_data = min(years_in_data)
    max_year_data = max(years_in_data)

    # Determine which years to process
    start = start_year if start_year is not None else min_year_data
    end = end_year if end_year is not None else max_year_data

    years_to_process = [y for y in range(start, end + 1) if y in years_in_data]

    print(f"✓ Data loaded")
    print(f"  Available years: {min_year_data} - {max_year_data}")
    print(f"  Will process: {len(years_to_process)} years ({years_to_process[0]} - {years_to_process[-1]})")
    log_stage_complete("Data Loading")
    log_memory_usage()

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Process each year
    for i, year in enumerate(years_to_process):
        print(f"\n[{i+1}/{len(years_to_process)}] ", end="")
        process_single_year(
            spark=spark,
            df_all_years=df_all,
            target_year=year,
            event_cols=event_cols,
            output_path=output_path,
            use_municipality_features=use_municipality_features
        )

        # Force GC between years
        import gc
        gc.collect()

    # Write SUCCESS marker
    with open(success_file, 'w') as f:
        f.write(f"Completed {len(years_to_process)} years")
    print(f"✓ Success marker created")

    # Stop Spark
    spark.stop()
    print("✓ Spark session stopped")

    # Training
    print("\n📊 Preparing data for training...")
    import pyarrow.parquet as pq
    parquet_dataset = pq.ParquetDataset(output_path)
    total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
    print(f"Total rows: {total_rows:,}")

    if tune:
        from run_enhanced_test import run_hyperparameter_tuning
        result = run_hyperparameter_tuning(output_path, total_rows, config, n_trials)
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        return result

    use_incremental = total_rows > 5_000_000
    if use_incremental:
        print(f"✓ Using INCREMENTAL training ({total_rows:,} rows)")
        model, metrics = train_incremental(
            output_path, total_rows, config,
            feature_config_path=feature_config_path,
            target_batch_rows=target_batch_rows,
            rolling_importance=rolling_importance,
            rolling_importance_config=rolling_importance_config,
        )
    else:
        print(f"✓ Using IN-MEMORY training ({total_rows:,} rows)")
        df_pandas = pd.read_parquet(output_path)
        model, metrics = train_in_memory(
            df_pandas, config,
            feature_config_path=feature_config_path,
            rolling_importance=rolling_importance,
        )

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming year-by-year pipeline")
    parser.add_argument('--start-year', type=int, help='First year to process')
    parser.add_argument('--end-year', type=int, help='Last year to process')
    parser.add_argument('--reuse', '-r', action='store_true', help='Reuse existing processed features')
    parser.add_argument('--config', type=str, help='Path to model config YAML')
    parser.add_argument('--features', type=str, help='Path to feature selection config YAML')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of tuning trials')
    parser.add_argument('--rolling-importance', action='store_true', help='Enable rolling window feature importance')
    parser.add_argument('--target-batch-rows', type=int, help='Target rows per incremental batch')

    # Rolling window config
    parser.add_argument('--rolling-train-years', type=int, help='Rolling train window size')
    parser.add_argument('--rolling-test-years', type=int, help='Rolling test window size')
    parser.add_argument('--rolling-step-years', type=int, help='Rolling step size')
    parser.add_argument('--rolling-sample-fraction', type=float, help='Rolling sample fraction')
    parser.add_argument('--rolling-max-windows', type=int, help='Max rolling windows')
    parser.add_argument('--rolling-min-train-rows', type=int, help='Minimum rows in rolling train')
    parser.add_argument('--rolling-min-test-rows', type=int, help='Minimum rows in rolling test')
    parser.add_argument('--rolling-external-memory', action='store_true', help='Use external memory for rolling')
    parser.add_argument('--rolling-external-memory-dir', type=str, help='Directory for rolling cache')
    parser.add_argument('--rolling-device', type=str, help='Device for rolling models')
    parser.add_argument('--rolling-tree-method', type=str, help='Tree method for rolling models')

    args = parser.parse_args()

    print("="*60)
    print("STREAMING YEAR-BY-YEAR PIPELINE")
    print("="*60)
    if args.start_year or args.end_year:
        print(f"Year range: {args.start_year or 'first'} - {args.end_year or 'last'}")
    if args.reuse:
        print("Reuse: Will skip processing if data exists")
    if args.config:
        print(f"Model config: {args.config}")
    if args.features:
        print(f"Feature selection: {args.features}")
    if args.tune:
        print(f"Tuning: {args.n_trials} trials")
    if args.rolling_importance:
        print("Rolling window feature importance enabled")
    print("="*60)
    print()

    rolling_importance_config = {}
    if args.rolling_train_years is not None:
        rolling_importance_config["train_years"] = args.rolling_train_years
    if args.rolling_test_years is not None:
        rolling_importance_config["test_years"] = args.rolling_test_years
    if args.rolling_step_years is not None:
        rolling_importance_config["step_years"] = args.rolling_step_years
    if args.rolling_sample_fraction is not None:
        rolling_importance_config["sample_fraction"] = args.rolling_sample_fraction
    if args.rolling_max_windows is not None:
        rolling_importance_config["max_windows"] = args.rolling_max_windows
    if args.rolling_min_train_rows is not None:
        rolling_importance_config["min_train_rows"] = args.rolling_min_train_rows
    if args.rolling_min_test_rows is not None:
        rolling_importance_config["min_test_rows"] = args.rolling_min_test_rows
    if args.rolling_external_memory:
        rolling_importance_config["external_memory"] = True
    if args.rolling_external_memory_dir:
        rolling_importance_config["external_memory_dir"] = args.rolling_external_memory_dir
    if args.rolling_device:
        rolling_importance_config["device"] = args.rolling_device
    if args.rolling_tree_method:
        rolling_importance_config["tree_method"] = args.rolling_tree_method

    main_streaming(
        start_year=args.start_year,
        end_year=args.end_year,
        reuse_processed=args.reuse,
        config_path=args.config,
        feature_config_path=args.features,
        tune=args.tune,
        n_trials=args.n_trials,
        rolling_importance=args.rolling_importance,
        target_batch_rows=args.target_batch_rows,
        rolling_importance_config=rolling_importance_config or None,
    )
