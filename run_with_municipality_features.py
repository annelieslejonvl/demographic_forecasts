"""
Enhanced training pipeline with municipality-level contextual features.

This adds socio-economic context that captures local conditions
(housing market, economy, demographics) that drive transitions.

Expected improvement:
- AUPRC: 0.086 → 0.12-0.18 (+40-100%)
- Group-level R²: 0.10 → 0.40-0.60 (+300-500%)

Usage:
    python run_with_municipality_features.py --max-rows=1000000
    python run_with_municipality_features.py --reuse --config configs/models/xgboost_classifier_tuned.yaml
"""
import sys
import argparse
import os
from pathlib import Path
# Always set JAVA_HOME to JDK 17 (required by Spark 4.x; JDK 25+ is incompatible)
_jdk_path = Path.home() / "jdk-17.0.18+8"
if _jdk_path.exists():
    os.environ["JAVA_HOME"] = str(_jdk_path)

# Hadoop winutils.exe required on Windows
if not os.environ.get("HADOOP_HOME"):
    _hadoop_path = Path.home() / "hadoop"
    if _hadoop_path.exists():
        os.environ["HADOOP_HOME"] = str(_hadoop_path)

# Setup logging FIRST (captures all output)
from src.utils.logging_setup import setup_logging, configure_spark_logging, log_stage_start, log_stage_complete, log_memory_usage

# Import base functions from run_enhanced_test
from run_enhanced_test import (
    load_model_config, cast_events_to_bool, create_hh_pos_features,
    create_income_features_spark, create_event_interactions_spark,
    create_age_interactions_spark, get_leaky_columns,
    run_hyperparameter_tuning, fix_dtypes, train_in_memory, train_incremental
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
from typing import Dict, Any, Optional
from src.features.socioeconomic import create_all_socioeconomic_features
from src.features.imputation import impute_missing_values
import mlflow

global window_spec


def main_with_municipality(
    reuse_processed=False,
    max_rows=None,
    config_path=None,
    tune=False,
    n_trials=50,
    feature_config_path=None,
    rolling_importance=False,
    target_batch_rows=None,
    rolling_importance_config: Optional[Dict[str, Any]] = None,
):
    """
    Main training pipeline with MUNICIPALITY features.

    Args:
        reuse_processed: Reuse existing processed features if available
        max_rows: Limit dataset to N rows
        config_path: Path to model config YAML
        tune: Run hyperparameter tuning
        n_trials: Number of tuning trials
        feature_config_path: Path to feature selection config YAML (e.g., 'configs/data/features_mixed.yaml')
    """
    global window_spec

    # Setup logging FIRST (captures all output to file)
    from datetime import datetime
    log_file = setup_logging(
        log_file=f"municipality_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        log_dir='.'
    )
    print(f"📝 Logging to: {log_file}")
    log_memory_usage()

    # Load model configuration if provided
    config = None
    if config_path:
        config = load_model_config(config_path)

    # Output path
    output_path = "data/processed_features_with_municipality"

    # Check if we can reuse
    import os
    success_file = os.path.join(output_path, "_SUCCESS")
    can_reuse = reuse_processed and os.path.isdir(output_path) and os.path.exists(success_file)

    if can_reuse:
        log_stage_start("Reusing Existing Features")
        print("="*60)
        print("♻️  REUSING EXISTING FEATURES (with municipality context)")
        print("="*60)
        print(f"  Found: {output_path}/")

        import pandas as pd
        import pyarrow.parquet as pq

        parquet_dataset = pq.ParquetDataset(output_path)
        total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
        print(f"  Total rows: {total_rows:,}")
        log_stage_complete("Reusing Existing Features")
        log_memory_usage()

        if tune:
            log_stage_start("Hyperparameter Tuning (Reuse)")
            result = run_hyperparameter_tuning(output_path, total_rows, config, n_trials)
            log_stage_complete("Hyperparameter Tuning (Reuse)")
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            return result

        use_incremental = total_rows > 5_000_000

        if use_incremental:
            log_stage_start("Incremental Training (Reuse)")
            print(f"  ✓ Using INCREMENTAL training")
            model, metrics = train_incremental(
                output_path,
                total_rows,
                config,
                feature_config_path=feature_config_path,
                target_batch_rows=target_batch_rows,
                rolling_importance=rolling_importance,
                rolling_importance_config=rolling_importance_config,
            )
            log_stage_complete("Incremental Training (Reuse)")
        else:
            log_stage_start("In-Memory Training (Reuse)")
            print(f"  ✓ Using IN-MEMORY training")
            df_pandas = pd.read_parquet(output_path)
            model, metrics = train_in_memory(
                df_pandas,
                config,
                feature_config_path=feature_config_path,
                rolling_importance=rolling_importance,
            )
            log_stage_complete("In-Memory Training (Reuse)")

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        return model, metrics

    # Full pipeline
    print("="*60)
    print("🏘️  RUNNING PIPELINE WITH MUNICIPALITY FEATURES")
    print("="*60)
    print("This adds:")
    print("  • Extended lag features (lag3)")
    print("  • Cumulative history")
    print("  • Rolling averages")
    print("  • Life stage indicators")
    print("  • 🆕 MUNICIPALITY-LEVEL CONTEXT:")
    print("      - Migration rates per municipality")
    print("      - Economic conditions (median income, inequality)")
    print("      - Demographic composition (age, families)")
    print("      - Life event rates (births, divorces, partnerships)")
    print("      - Trends over time")
    print("="*60)

    log_stage_start("Spark Session Initialization")

    import tempfile
    import shutil
    import atexit

    # Create temp dir for Spark local storage and checkpoints
    spark_temp_dir = tempfile.mkdtemp(prefix="spark_local_")

    # Spark session optimized for 32GB system with year-by-year processing
    # In local mode, driver and executor share same JVM - so use master="local[*]"
    spark = SparkSession.builder \
        .appName("DemographicForecasts_Municipality") \
        .master("local[*]") \
        .config("spark.driver.memory", "22g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.2") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "128MB") \
        .config("spark.sql.autoBroadcastJoinThreshold", "256MB") \
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
                "-XX:InitiatingHeapOccupancyPercent=45 "
                "-XX:G1HeapRegionSize=32M "
                "-XX:MaxGCPauseMillis=500 "
                "-XX:+ParallelRefProcEnabled "
                "-XX:ParallelGCThreads=8 "
                "-XX:ConcGCThreads=2 "
                "-XX:ReservedCodeCacheSize=256m "
                "-XX:NonProfiledCodeHeapSize=128m") \
        .config("spark.ui.enabled", "false") \
        .config("spark.cleaner.periodicGC.interval", "5min") \
        .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
        .getOrCreate()

    # Configure Spark logging
    configure_spark_logging(spark, log_level="WARN")

    log_stage_complete("Spark Session Initialization")
    log_memory_usage()

    # Set checkpoint directory
    checkpoint_dir = os.path.join(spark_temp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    # Register cleanup function to remove temp directory on exit
    def cleanup_spark_temp():
        try:
            if os.path.exists(spark_temp_dir):
                shutil.rmtree(spark_temp_dir)
                print(f"✓ Cleaned up Spark temp directory: {spark_temp_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up Spark temp directory: {e}")

    atexit.register(cleanup_spark_temp)

    print("✓ Spark session created")
    print(f"  Temp dir: {spark_temp_dir} (will be cleaned up on exit)")
    print(f"  Disk space available: 57 GB")

    window_spec = Window.partitionBy('id').orderBy('year')

    # Load data
    log_stage_start("Data Loading")
    DATA_PATH = r"..\AI_datagen\data\real_panel_long"
    files = list(pathlib.Path(DATA_PATH).rglob("*.parquet"))
    df = spark.read.parquet(*[str(file) for file in files])
    print(f"✓ Data loaded: {len(df.columns)} columns")
    log_stage_complete("Data Loading")
    log_memory_usage()

    # Check if refnis column exists
    if 'refnis' not in df.columns:
        print("⚠️  WARNING: 'refnis' column not found!")
        print("   Municipality features require a municipality identifier.")
        print("   Continuing without municipality features...")
        use_municipality_features = False
    else:
        use_municipality_features = True
        print(f"✓ Found municipality column: refnis")
        n_municipalities = df.select('refnis').distinct().count()
        print(f"  {n_municipalities} unique municipalities in dataset")

    # Sample if needed
    log_stage_start("Row Count")
    total_rows = df.count()
    print(f"Total rows in dataset: {total_rows:,}")
    log_stage_complete("Row Count")
    log_memory_usage()

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

    # Repartition - fewer partitions to reduce memory overhead
    # Each partition should be 100-200MB for optimal memory usage
    num_partitions = 8  # Match parallelism for minimal overhead
    print(f"Repartitioning by 'id' into {num_partitions} partitions...")
    df = df.repartition(num_partitions, "id")

    # Enhanced features
    log_stage_start("Enhanced Features Creation")
    print("\n🚀 Creating ENHANCED features...")
    df2 = create_all_enhanced_features(df, event_cols, id_col='id', t_col='year')
    print("✓ Enhanced event history features created")

    # SKIP checkpoints entirely - all computation deferred to incremental write
    # Modern Spark handles long DAGs well, and incremental write breaks computation naturally
    print(f"  (No checkpoint - computation will happen during incremental write)")
    log_stage_complete("Enhanced Features Creation")
    log_memory_usage()

    # *** NEW: MUNICIPALITY FEATURES ***
    if use_municipality_features:
        # CRITICAL: Create municipality and income lags FIRST to avoid temporal leakage
        log_stage_start("Municipality History Creation")
        print("\n📍 Creating municipality history (refnis_lag1, refnis_lag2)...")
        print("   This prevents temporal leakage - we use ORIGIN municipality, not DESTINATION")

        w_individual = Window.partitionBy('id').orderBy('year')
        df2 = df2.withColumn('refnis_lag1', F.lag('refnis', 1).over(w_individual))
        df2 = df2.withColumn('refnis_lag2', F.lag('refnis', 2).over(w_individual))

        # Also lag individual income for use in municipality aggregates
        df2 = df2.withColumn('MS_ADI_PP_lag1', F.lag('MS_ADI_PP', 1).over(w_individual))
        df2 = df2.withColumn('MS_ADI_HH_lag1', F.lag('MS_ADI_HH', 1).over(w_individual))
        df2 = df2.withColumn('age_lag1', F.lag('age', 1).over(w_individual))

        print("   ✓ Created: refnis_lag1, refnis_lag2, MS_ADI_PP_lag1, MS_ADI_HH_lag1, age_lag1")
        log_stage_complete("Municipality History Creation")
        log_memory_usage()

        log_stage_start("Municipality Features Creation")
        print("\n🏘️  Creating MUNICIPALITY contextual features...")
        df2 = create_all_municipality_features(
            df2,
            id_col='id',
            time_col='year',
            muni_col='refnis_lag1',  # Use ORIGIN municipality (where they were)
            lag_years=1,
            include_trends=True
        )
        print(f"✓ Municipality features created: {len(df2.columns)} total columns")
        print("   ⚠️  NOTE: Features use ORIGIN municipality (refnis_lag1), not DESTINATION (refnis)")

        # SKIP checkpoint here - even lazy checkpoint triggers shuffle OOM
        # Computation will be deferred to incremental write (year by year)
        print(f"  (Skipping checkpoint to avoid OOM - computation deferred to write phase)")
        log_stage_complete("Municipality Features Creation")
        log_memory_usage()

        # *** NEW: EXTERNAL SOCIO-ECONOMIC DATA ***
        log_stage_start("External Socioeconomic Data")
        print("\n📊 Adding EXTERNAL socio-economic data from df_socioec.csv...")
        try:
            from load_socioeconomic_data import add_socioec_features_to_spark_df
            df2 = add_socioec_features_to_spark_df(
                df2,
                spark,
                socioec_csv_path='df_socioec.csv',
                strategy='static'  # Use 2020 values for all years
            )
            print(f"✓ External socioec data added: {len(df2.columns)} total columns")

            # SKIP checkpoint - defer computation to incremental write
            print(f"  (Skipping checkpoint to avoid OOM)")
            log_stage_complete("External Socioeconomic Data")
            log_memory_usage()

        except FileNotFoundError:
            print("⚠️  df_socioec.csv not found, skipping external data")
            print("   Internal municipality aggregates still available")
            log_stage_complete("External Socioeconomic Data (Skipped)")
        except Exception as e:
            print(f"⚠️  Error loading external data: {e}")
            print("   Continuing with internal municipality aggregates only")
            log_stage_complete("External Socioeconomic Data (Error)")

    # *** SKIP CHECKPOINT - Process lazily to avoid OOM ***
    # Don't materialize the whole dataset - let Spark handle it lazily
    # The DAG will be evaluated incrementally when we process year-by-year
    print("\n⚡ Skipping intermediate checkpoint to avoid OOM")
    print("   DAG will be evaluated incrementally during year-by-year processing")

    # Rest of pipeline — now runs on a flat DataFrame
    log_stage_start("Base Features Pipeline")
    df2 = (df2
        .transform(cast_events_to_bool)
        .transform(create_hh_pos_features)  # Uses default id_col='sid', time_col='year'
        .transform(create_income_features_spark)  # Uses default parameters
        .transform(create_event_interactions_spark)
        .transform(create_age_interactions_spark))

    print(f"✅ Base features created! Total columns: {len(df2.columns)}")
    log_stage_complete("Base Features Pipeline")
    log_memory_usage()

    # Life stage features
    log_stage_start("Life Stage Features")
    df2 = create_life_stage_features(df2)
    print(f"✓ Life stage features added")
    log_stage_complete("Life Stage Features")
    log_memory_usage()

    # Socioeconomic features
    log_stage_start("Socioeconomic Features")
    df2 = create_all_socioeconomic_features(
        df2,
        id_col='id',
        time_col='year',
        include_hh_pos_features=True
    )
    print(f"✓ Socioeconomic features created: {len(df2.columns)} columns")
    print(f"  Total columns: {len(df2.columns)}")
    log_stage_complete("Socioeconomic Features")
    log_memory_usage()

    # Check for leakage in municipality features
    leaky_muni = check_municipality_features_for_leakage(df2.columns)
    if leaky_muni:
        print(f"\n⚠️  WARNING: Found {len(leaky_muni)} potentially leaky municipality features:")
        for col in leaky_muni[:5]:
            print(f"  - {col}")
        print("  These should be lagged! Check municipality_context.py")

    # *** CRITICAL: IMPUTE MISSING VALUES ***
    # Lagged features and window operations create NULLs that must be filled.

    log_stage_start("Missing Value Imputation")
    print("\n🔧 IMPUTING MISSING VALUES...")
    df2 = impute_missing_values(
        df2,
        strategy="smart",  # median for numeric, mode for boolean
        exclude_cols=['id', 'year', 'refnis', 'y_moved'],  # Don't impute these
        verbose=True
    )
    print("✓ Imputation complete")
    log_stage_complete("Missing Value Imputation")
    log_memory_usage()

    # Write to disk INCREMENTALLY (year by year to avoid OOM)
    # Using pandas/pyarrow for parquet writes to bypass Hadoop NativeIO on Windows
    log_stage_start("Incremental Write to Disk")
    print("\n📁 Writing processed data to disk (incremental by year)...")
    print("   Processing each year separately to minimize memory usage")
    print("   Using pandas+pyarrow to bypass Hadoop NativeIO on Windows")

    # Get list of years FIRST (small operation)
    log_stage_start("Getting Year List")
    years = [row.year for row in df2.select('year').distinct().orderBy('year').collect()]
    print(f"  Found {len(years)} years: {min(years)} - {max(years)}")
    log_stage_complete("Getting Year List")
    log_memory_usage()

    # Write year by year - each year is processed independently
    # This avoids loading entire dataset into memory
    os.makedirs(output_path, exist_ok=True)
    for i, year in enumerate(years):
        log_stage_start(f"Writing Year {year}")
        print(f"  [{i+1}/{len(years)}] Processing year {year}...", end=" ", flush=True)

        # Filter to single year and persist to break DAG between years
        df_year = df2.filter(F.col('year') == year).persist()

        # Force evaluation and get count
        count = df_year.count()
        print(f"({count:,} rows)...", end=" ", flush=True)

        # Convert to pandas and write
        pdf = df_year.toPandas()
        year_file = os.path.join(output_path, f"year_{year}.parquet")
        pdf.to_parquet(year_file, index=False, engine="pyarrow")

        # Unpersist to free memory for next year
        df_year.unpersist()
        del pdf

        print(f"✓")
        log_stage_complete(f"Writing Year {year}")
        log_memory_usage()

    print(f"✓ Data written to {output_path}")
    log_stage_complete("Incremental Write to Disk")
    log_memory_usage()

    # Write SUCCESS marker
    import os
    success_file = os.path.join(output_path, "_SUCCESS")
    with open(success_file, 'w') as f:
        f.write("Success")
    print(f"✓ Success marker created")

    # Stop Spark
    log_stage_start("Spark Session Cleanup")
    spark.stop()
    print("✓ Spark session stopped")
    log_stage_complete("Spark Session Cleanup")
    log_memory_usage()

    # Training
    log_stage_start("Training Preparation")
    print("\n📊 Preparing data for training...")
    import pandas as pd
    import pyarrow.parquet as pq

    parquet_dataset = pq.ParquetDataset(output_path)
    total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
    print(f"Total rows in processed data: {total_rows:,}")
    log_stage_complete("Training Preparation")
    log_memory_usage()

    if tune:
        log_stage_start("Hyperparameter Tuning")
        print("\n🔍 HYPERPARAMETER TUNING MODE")
        result = run_hyperparameter_tuning(output_path, total_rows, config, n_trials)
        log_stage_complete("Hyperparameter Tuning")
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        return result

    incremental_threshold = 10_000_000
    safe_in_memory_limit = 5_000_000

    if total_rows > incremental_threshold:
        log_stage_start("Incremental Training")
        print(f"✓ Dataset is large ({total_rows:,} rows) - using INCREMENTAL training")
        model, metrics = train_incremental(output_path, total_rows, config, feature_config_path=feature_config_path)
        log_stage_complete("Incremental Training")
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        return model, metrics
    elif total_rows > safe_in_memory_limit:
        log_stage_start("Data Loading with Sampling (Chunked)")
        print(f"⚠️  Dataset is large ({total_rows:,} rows) - sampling to {safe_in_memory_limit:,} rows")
        print("  Loading in chunks to avoid OOM...")

        # Calculate sampling fraction
        sample_fraction = safe_in_memory_limit / total_rows

        # Read in chunks and sample
        chunk_size = 500_000  # Read 500k rows at a time
        chunks_to_sample = []
        rows_read = 0

        parquet_file = pq.ParquetFile(output_path)
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            df_chunk = batch.to_pandas()
            rows_read += len(df_chunk)

            # Sample from this chunk
            n_sample_chunk = int(len(df_chunk) * sample_fraction)
            if n_sample_chunk > 0:
                df_chunk_sampled = df_chunk.sample(n=n_sample_chunk, random_state=42 + batch_idx)
                chunks_to_sample.append(df_chunk_sampled)
                print(f"    Chunk {batch_idx + 1}: sampled {n_sample_chunk:,} rows", flush=True)

            del df_chunk

            if rows_read >= total_rows:
                break

        # Combine sampled chunks
        df_pandas = pd.concat(chunks_to_sample, ignore_index=True)
        del chunks_to_sample

        # Final sample to exact size if needed
        if len(df_pandas) > safe_in_memory_limit:
            df_pandas = df_pandas.sample(n=safe_in_memory_limit, random_state=42)

        print(f"✓ Sampled to {len(df_pandas):,} rows")
        log_stage_complete("Data Loading with Sampling (Chunked)")
        log_memory_usage()
    else:
        log_stage_start("Data Loading (In-Memory)")
        print(f"✓ Dataset size OK ({total_rows:,} rows) - loading into memory")

        # Even for "small" datasets, use chunked loading to be safe
        if total_rows > 2_000_000:
            print("  Using chunked loading for safety...")
            chunks = []
            parquet_file = pq.ParquetFile(output_path)
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=500_000)):
                df_chunk = batch.to_pandas()
                chunks.append(df_chunk)
                print(f"    Loaded chunk {batch_idx + 1} ({len(df_chunk):,} rows)", flush=True)
            df_pandas = pd.concat(chunks, ignore_index=True)
            del chunks
        else:
            df_pandas = pd.read_parquet(output_path)

        print(f"✓ Loaded {len(df_pandas):,} rows")
        log_stage_complete("Data Loading (In-Memory)")
        log_memory_usage()

    log_stage_start("Model Training (In-Memory)")
    model, metrics = train_in_memory(
        df_pandas,
        config,
        feature_config_path=feature_config_path,
        rolling_importance=rolling_importance,
        rolling_importance_config=rolling_importance_config,
    )
    log_stage_complete("Model Training (In-Memory)")
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline with MUNICIPALITY features")
    parser.add_argument('--reuse', '-r', action='store_true', help='Reuse existing processed features')
    parser.add_argument('--max-rows', type=int, help='Limit dataset to N rows')
    parser.add_argument('--config', type=str, help='Path to model config YAML')
    parser.add_argument('--features', type=str, help='Path to feature selection config YAML (e.g., configs/data/features_mixed.yaml)')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of tuning trials')
    parser.add_argument('--rolling-importance', action='store_true', help='Enable rolling window feature importance (stable importance)')
    parser.add_argument('--target-batch-rows', type=int, help='Target rows per incremental batch (default: 10000000)')
    parser.add_argument('--rolling-train-years', type=int, help='Rolling train window size (default: 8)')
    parser.add_argument('--rolling-test-years', type=int, help='Rolling test window size (default: 1)')
    parser.add_argument('--rolling-step-years', type=int, help='Rolling step size (default: 1)')
    parser.add_argument('--rolling-sample-fraction', type=float, help='Rolling sample fraction (default: 0.5)')
    parser.add_argument('--rolling-max-windows', type=int, help='Max rolling windows (default: no limit)')
    parser.add_argument('--rolling-min-train-rows', type=int, help='Minimum rows in rolling train window (default: 20000)')
    parser.add_argument('--rolling-min-test-rows', type=int, help='Minimum rows in rolling test window (default: 5000)')
    parser.add_argument('--rolling-external-memory', action='store_true', help='Use external-memory (disk) for rolling windows')
    parser.add_argument('--rolling-external-memory-dir', type=str, help='Directory for rolling window cache files')
    parser.add_argument('--rolling-device', type=str, help='Device for rolling models (cpu or cuda)')
    parser.add_argument('--rolling-tree-method', type=str, help='Tree method for rolling models (e.g., hist)')

    args = parser.parse_args()

    print("="*60)
    print("🏘️  MUNICIPALITY FEATURES PIPELINE")
    print("="*60)
    if args.reuse:
        print("🔄 Reuse flag: Will skip feature engineering if possible")
    if args.max_rows:
        print(f"📊 Max rows: {args.max_rows:,}")
    if args.config:
        print(f"📝 Model config: {args.config}")
    if args.features:
        print(f"🎯 Feature selection: {args.features}")
    if args.tune:
        print(f"🔍 Tuning: {args.n_trials} trials")
    if args.rolling_importance:
        print("🧭 Rolling window feature importance enabled")
    if args.target_batch_rows:
        print(f"📦 Target batch rows: {args.target_batch_rows:,}")
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
    if args.rolling_importance and rolling_importance_config:
        print(f"🧭 Rolling config: {rolling_importance_config}")
    print("="*60)
    print()

    main_with_municipality(
        reuse_processed=args.reuse,
        max_rows=args.max_rows,
        config_path=args.config,
        feature_config_path=args.features,
        tune=args.tune,
        n_trials=args.n_trials,
        rolling_importance=args.rolling_importance,
        target_batch_rows=args.target_batch_rows,
        rolling_importance_config=rolling_importance_config or None,
    )
