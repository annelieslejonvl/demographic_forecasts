
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
import pathlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from src.data import DataSource
from run import run_experiments
from src.features.socioeconomic import create_all_socioeconomic_features
from src.utils.feature_selection import filter_features_by_config, load_feature_config
global window_spec
import mlflow
from sklearn.metrics import precision_score , recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import argparse
label_col = 'birth1_event'
def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.

    Args:
        config_path: Path to YAML config file (e.g., 'configs/models/xgboost_linear.yaml')

    Returns:
        Dictionary with model configuration
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ Loaded config from {config_path}")
    print(f"  Backend: {config.get('backend')}")
    print(f"  Model type: {config['model'].get('type')}")

    # Extract booster type if present
    booster = config['model']['params'].get('booster', 'gbtree')
    print(f"  Booster: {booster}")

    return config


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
"""
Copy this code into your notebook to replace the old feature engineering.

IMPORTANT: Run this AFTER you've created the event history features
(the cell that creates _first, _at_risk, _censored, _lag1, _lag2 columns)
"""





def create_hh_pos_features(df, window_spec=None):
    """NEW: Create lagged hh_pos to avoid temporal leakage"""
    if window_spec is None:
        # Try to use global window_spec if available
        window_spec = globals().get('window_spec')
        if window_spec is None:
            # Create default window spec
            window_spec = Window.partitionBy('id').orderBy('year')

    df = df.withColumn('hh_pos_lag1', F.lag('hh_pos', 1).over(window_spec))

    df = df.withColumn(
        'hh_pos_changed',
        F.when(
            (F.col('hh_pos') != F.col('hh_pos_lag1')) & F.col('hh_pos_lag1').isNotNull(),
            1
        ).otherwise(0).cast('int')
    )

    return df


def create_income_features_spark(df, window_spec=None):
    """Income features"""
    if window_spec is None:
        window_spec = globals().get('window_spec')
        if window_spec is None:
            window_spec = Window.partitionBy('id').orderBy('year')

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

    # FIXED: Only use lagged events to avoid temporal leakage
    df = df.withColumn(
        'family_expansion',
        (F.col('birth2_event_lag1') &
         (F.col('birth1_event_lag1') | F.col('birth1_event_lag2'))).cast('boolean')
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


def create_lags(df, event_cols):

    id_col = "id"
    t_col = "year"



    lags = [1, 2]  # change to [1,2,3,...] if needed

    w_time = Window.partitionBy(id_col).orderBy(t_col)
    w_id = Window.partitionBy(id_col)

    # last observed time per sid (needed for censoring)
    df = df.withColumn("_t_last", F.max(F.col(t_col)).over(w_id))

    for e in event_cols:
        # first year with event==1 (NULL if never)
        t_event = F.min(F.when(F.col(e) == 1, F.col(t_col))).over(w_id)

        # first-occurrence indicator (only 1 at first event time)
        e_first = F.when(F.col(t_col) == t_event, F.lit(1)).otherwise(F.lit(0)).cast("int")

        # at-risk: 1 strictly before event time; if never occurs -> 1 for all rows
        e_at_risk = (
            F.when(t_event.isNull(), F.lit(1))
            .when(F.col(t_col) < t_event, F.lit(1))
            .otherwise(F.lit(0))
            .cast("int")
        )

        # right-censored: 1 only on last observed year if event never occurs
        e_cens = (
            F.when(t_event.isNull() & (F.col(t_col) == F.col("_t_last")), F.lit(1))
            .otherwise(F.lit(0))
            .cast("int")
        )

        df = (
            df
            .withColumn(f"{e}_first", e_first)
            .withColumn(f"{e}_at_risk", e_at_risk)
            .withColumn(f"{e}_censored", e_cens)
        )

        for k in lags:
            df = df.withColumn(f"{e}_lag{k}", F.lag(F.col(f"{e}_first"), k).over(w_time))
        
    return df 

def drop_cols_leakage(df):
    cols_leaks = [col for col in df.columns if col.endswith('_first') or col.endswith('_censored') or col.endswith('_at_risk')]
    return df.drop(*cols_leaks)

def get_leaky_columns(columns):
    """
    Identify columns that cause temporal leakage.

    Returns list of columns that should be excluded from features:
    - Columns ending with _first, _censored, _at_risk
    - Raw event columns without lag (e.g., birth1_event but NOT birth1_event_lag1)

    Safe columns:
    - Lagged event columns (birth1_event_lag1, birth1_event_lag2, etc.)
    - State variables measured at start of year (age, income, coupled, etc.)
    """
    leaky = []

    # Event column names (without the _lag suffix)
    event_base_names = [
        'birth1_event',
        'birth2_event',
        'divorce_event',
        'getalifeother_event',
    ]

    for col in columns:
        # Filter out survival analysis columns
        if col.endswith('_first') or col.endswith('_censored') or col.endswith('_at_risk'):
            leaky.append(col)
            continue

        # Filter out raw event columns (without lag)
        # birth1_event is leaky, but birth1_event_lag1 is safe
        for event_name in event_base_names:
            if col == event_name:  # Exact match - no lag suffix
                leaky.append(col)
                break

    return leaky
def run_hyperparameter_tuning(parquet_path, total_rows, base_config=None, n_trials=50):
    """
    Run hyperparameter tuning using Optuna.

    Args:
        parquet_path: Path to processed features
        total_rows: Total number of rows
        base_config: Optional base configuration (can be None to use defaults)
        n_trials: Number of tuning trials

    Returns:
        Tuple of (best_model, best_metrics, tuning_results)
    """
    print("="*60)
    print("🔍 HYPERPARAMETER TUNING")
    print("="*60)
    print(f"  Trials: {n_trials}")
    print(f"  Metric: AUC-PR (optimized for imbalanced data)")
    print(f"  Strategy: Stratified sampling + sequential trials")
    print(f"  Memory: Aggressive cleanup between trials")
    print("="*60)

    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    import gc

    # Load data with stratified sampling by year for memory efficiency
    tuning_sample_size = min(2_000_000, total_rows)  # Max 1M for tuning (reduced from 2M)
    print(f"\n📊 Loading data with stratified sampling (target: {tuning_sample_size:,} rows)...")

    # Strategy: Use only recent years for tuning (2020-2023 for train, 2024+ for val)
    # This is more memory efficient and still representative
    print("  Strategy: Using recent years only (2020-2024) for faster tuning")
    print("  Loading in chunks to avoid OOM...")

    # Chunked loading with year filtering
    import pyarrow.parquet as pq

    chunks = []
    rows_loaded = 0

    try:
        # Try filtering by year during load
        parquet_dataset = pq.ParquetDataset(parquet_path, filters=[('year', '>=', 2020)])

        # Load in chunks
        chunk_size = 500_000
        for fragment in parquet_dataset.fragments:
            for batch in fragment.to_batches(batch_size=chunk_size):
                df_chunk = batch.to_pandas()
                chunks.append(df_chunk)
                rows_loaded += len(df_chunk)
                print(f"    Loaded chunk: {len(df_chunk):,} rows (total: {rows_loaded:,})", flush=True)

                # Stop if we have enough for tuning
                if rows_loaded >= tuning_sample_size * 2:  # Load 2x target for sampling flexibility
                    break
            if rows_loaded >= tuning_sample_size * 2:
                break

        df_pandas = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        del chunks
        print(f"  ✓ Loaded {len(df_pandas):,} rows from years 2020+ (chunked)")

    except Exception as e:
        print(f"  ⚠️  Chunked loading with filter failed: {e}")
        print("  Trying chunked loading without filter...")

        # Fallback: load in chunks without filter, then filter
        chunks = []
        parquet_file = pq.ParquetFile(parquet_path)

        chunk_size = 500_000
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_chunk = batch.to_pandas()

            # Filter for recent years if year column exists
            if 'year' in df_chunk.columns:
                df_chunk = df_chunk[df_chunk['year'] >= 2020]

            if len(df_chunk) > 0:
                chunks.append(df_chunk)
                rows_loaded += len(df_chunk)
                print(f"    Loaded chunk: {len(df_chunk):,} rows (total: {rows_loaded:,})", flush=True)

            # Stop if we have enough
            if rows_loaded >= tuning_sample_size * 2:
                break

        df_pandas = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        del chunks
        print(f"  ✓ Loaded {len(df_pandas):,} rows (chunked, filtered to 2020+)")

    # Stratified sampling by year to ensure temporal representation
    if len(df_pandas) > tuning_sample_size and 'year' in df_pandas.columns:
        print(f"  Applying stratified sampling by year...")
        # Calculate samples per year
        years = df_pandas['year'].unique()
        samples_per_year = tuning_sample_size // len(years)

        sampled_dfs = []
        for year in sorted(years):
            year_df = df_pandas[df_pandas['year'] == year]
            n_samples = min(samples_per_year, len(year_df))
            sampled_dfs.append(year_df.sample(n=n_samples, random_state=42))

        df_pandas = pd.concat(sampled_dfs, ignore_index=True)
        print(f"  ✓ Stratified sample: {len(df_pandas):,} rows ({len(years)} years, ~{samples_per_year:,} rows/year)")
    elif len(df_pandas) > tuning_sample_size:
        # Simple random sample if no year column
        df_pandas = df_pandas.sample(n=tuning_sample_size, random_state=42)
        print(f"  ✓ Random sample: {len(df_pandas):,} rows")

    # Prepare data
    drop_cols = ['id', 'year', 'refnis', 'y_moved']
    leak_cols = get_leaky_columns(df_pandas.columns)

    feature_cols = [c for c in df_pandas.columns if c not in drop_cols and c not in leak_cols]

    print(f"  ⚠️  Filtered out {len(leak_cols)} leaky columns to prevent temporal leakage")
    print(f"  ✓ Using {len(feature_cols)} safe features")
    df_pandas = fix_dtypes(df_pandas, feature_cols)

    # Time-based split
    if 'year' in df_pandas.columns:
        train_df = df_pandas[df_pandas['year'] < 2023]
        val_df = df_pandas[df_pandas['year'] >= 2023]
        print(f"  Train years: {train_df['year'].min()}-{train_df['year'].max()}")
        print(f"  Val years: {val_df['year'].min()}-{val_df['year'].max()}")
    else:
        split_idx = int(len(df_pandas) * 0.8)
        train_df = df_pandas.iloc[:split_idx]
        val_df = df_pandas.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df['y_moved']
    X_val, y_val = val_df[feature_cols], val_df['y_moved']

    print(f"✓ Train: {len(X_train):,} rows, Val: {len(X_val):,} rows")
    print(f"✓ Features: {len(feature_cols)}")
    if len(y_train) > 0 and len(y_val) > 0:
        train_pos_rate = float(y_train.mean())
        val_pos_rate = float(y_val.mean())
        print(f"✓ Positive rate: train={train_pos_rate:.4%}, val={val_pos_rate:.4%}")
    else:
        print("⚠️  Positive rate: skipped (empty train or val set)")
    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ Hyperparameter tuning aborted: empty train or val split.")
        return None, None, None
    if y_train.nunique(dropna=False) < 2 or y_val.nunique(dropna=False) < 2:
        print("❌ Hyperparameter tuning aborted: train or val split has a single class.")
        print(f"  Train class counts: {y_train.value_counts(dropna=False).to_dict()}")
        print(f"  Val class counts: {y_val.value_counts(dropna=False).to_dict()}")
        return None, None, None

    # Free memory
    del df_pandas, train_df, val_df
    gc.collect()

    # Define objective function
    def objective(trial):
        """Optuna objective function with memory management."""
        try:
            # Determine booster type from base_config or default to tree
            if base_config and 'model' in base_config:
                booster = base_config['model']['params'].get('booster', 'gbtree')
            else:
                booster = 'gbtree'

            if booster == 'gblinear':
                # Linear booster hyperparameters
                params = {
                    'booster': 'gblinear',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'device': 'cuda',
                    'random_state': 42,
                }
            else:
                # Tree booster hyperparameters
                params = {
                    'booster': 'gbtree',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'tree_method': 'hist',
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'device': 'cuda',
                    'random_state': 42,
                }

            # Train model
            model = XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_pr = average_precision_score(y_val, y_pred_proba)

            # Store additional metrics
            auc_roc = roc_auc_score(y_val, y_pred_proba)
            thresholds = [i / 100 for i in range(5, 96, 5)]
            best_f1 = -1.0
            best_thresh = 0.5
            for thresh in thresholds:
                f1 = f1_score(y_val, (y_pred_proba >= thresh).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

            trial.set_user_attr('auc_roc', auc_roc)
            trial.set_user_attr('auc_pr', auc_pr)
            trial.set_user_attr('f1', best_f1)
            trial.set_user_attr('f1_threshold', best_thresh)

            # CRITICAL: Delete model and predictions to free memory
            del model, y_pred_proba
            gc.collect()

            return best_f1  # Maximize F1

        except Exception as e:
            # Log error and return worst value
            print(f"  ⚠️  Trial {trial.number} failed: {e}")
            gc.collect()
            return 0.0  # Worst value for maximization

    # Setup MLflow tracking for hyperparameter tuning
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("demographic_forecasts_tuning")

    # Create MLflow callback for automatic trial logging
    from optuna.integration.mlflow import MLflowCallback
    mlflc = MLflowCallback(
        tracking_uri="http://127.0.0.1:5000",
        metric_name="auc_pr",
    )

    # Create study
    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization (IMPORTANT: n_jobs=1 for sequential execution to avoid memory issues)
    print(f"\n🎯 Starting optimization with {n_trials} trials...")
    print("  Running trials sequentially (n_jobs=1) to prevent memory issues")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False, callbacks=[mlflc])

    print(f"\n✓ Optimization complete: {len(study.trials)} trials run")

    # Extract results
    best_params = study.best_params
    best_trial = study.best_trial

    print("\n" + "="*60)
    print("✅ TUNING COMPLETE")
    print("="*60)
    print(f"Best AUC-PR: {best_trial.value:.4f}")
    print(f"\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nAll metrics:")
    print(f"  AUC-ROC:  {best_trial.user_attrs['auc_roc']:.4f}")
    print(f"  AUC-PR:   {best_trial.user_attrs['auc_pr']:.4f}")
    print(f"  F1 Score: {best_trial.user_attrs['f1']:.4f}")
    print("="*60)

    # Save results
    import json
    results_path = "tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_metrics': {
                'auc_roc': best_trial.user_attrs['auc_roc'],
                'auc_pr': best_trial.user_attrs['auc_pr'],
                'f1': best_trial.user_attrs['f1'],
            },
            'n_trials': n_trials,
        }, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")

    # Train final model with best parameters and log to MLflow
    print(f"\n🎯 Training final model with best parameters...")

    with mlflow.start_run(run_name="best_model_from_tuning"):
        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))

        # Train model
        best_model = XGBClassifier(**best_params, early_stopping_rounds=30)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)

        # Final evaluation
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        auc_pr = average_precision_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, (y_pred_proba >= 0.5).astype(int))

        final_metrics = (auc_roc, auc_pr, f1)

        # Log metrics
        mlflow.log_metric("final_auc_roc", auc_roc)
        mlflow.log_metric("final_auc_pr", auc_pr)
        mlflow.log_metric("final_f1_score", f1)

        # Log model
        mlflow.xgboost.log_model(best_model, "best_model")

        # Save local checkpoint with feature importance
        import os
        from datetime import datetime
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        booster_name = best_params.get('booster', 'gbtree')
        checkpoint_path = os.path.join(checkpoint_dir, f"model_tuned_{booster_name}_{timestamp}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model
        best_model.save_model(os.path.join(checkpoint_path, "model.json"))

        # Save feature importance
        import pandas as pd
        importance_types = ['weight', 'gain', 'cover']
        for imp_type in importance_types:
            try:
                importance = best_model.get_booster().get_score(importance_type=imp_type)
                if importance:
                    imp_df = pd.DataFrame([
                        {'feature': k, 'importance': v}
                        for k, v in importance.items()
                    ]).sort_values('importance', ascending=False)
                    imp_df.to_csv(
                        os.path.join(checkpoint_path, f"feature_importance_{imp_type}.csv"),
                        index=False
                    )
            except:
                pass

        # Save metadata
        import json
        metadata = {
            'timestamp': timestamp,
            'model_params': best_params,
            'training_mode': 'tuning',
            'n_trials': n_trials,
            'metrics': {
                'auc_roc': float(auc_roc),
                'auc_pr': float(auc_pr),
                'f1_score': float(f1)
            },
            'data': {
                'train_rows': len(X_train),
                'val_rows': len(X_val),
                'n_features': len(feature_cols)
            }
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n📊 MLflow: Best model logged to experiment 'demographic_forecasts_tuning'")
        print(f"💾 Checkpoint saved to: {checkpoint_path}")

    return best_model, final_metrics, study


def main(
    reuse_processed=False,
    max_rows=None,
    config_path=None,
    tune=False,
    n_trials=50,
    feature_config_path=None,
    rolling_importance=False,
    rolling_importance_config: Optional[Dict[str, Any]] = None,
    target_batch_rows=None,
):
    """
    Main training pipeline.

    Args:
        reuse_processed: If True, skip feature engineering and reuse existing processed_features.parquet
        max_rows: Number of rows to sample (None = use all data)
        config_path: Path to YAML config file (e.g., 'configs/models/xgboost_linear.yaml')
        tune: If True, run hyperparameter tuning
        n_trials: Number of tuning trials (default: 50)
        feature_config_path: Path to feature selection config YAML (e.g., 'configs/data/features_mixed.yaml')
    """
    global window_spec

    # Load model configuration if provided
    config = None
    if config_path:
        config = load_model_config(config_path)

    # Output path for processed features (Spark creates a directory with part files)
    output_path = "data/processed_features"

    # Check if we can reuse existing processed data
    import os
    # Spark writes a directory with part-*.parquet files and a _SUCCESS marker
    success_file = os.path.join(output_path, "_SUCCESS")
    can_reuse = reuse_processed and os.path.isdir(output_path) and os.path.exists(success_file)

    if can_reuse:
        print("="*60)
        print("♻️  REUSING EXISTING PROCESSED FEATURES")
        print("="*60)
        print(f"  Found: {output_path}/")
        print(f"  Part files: {len([f for f in os.listdir(output_path) if f.endswith('.parquet')])}")
        print("  Skipping feature engineering...")

        # Skip directly to training
        import pandas as pd
        import pyarrow.parquet as pq

        # Read the parquet dataset (handles multiple part files automatically)
        parquet_dataset = pq.ParquetDataset(output_path)
        total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
        print(f"  Total rows: {total_rows:,}")

        # Check if we should run hyperparameter tuning
        if tune:
            print("\n🔍 HYPERPARAMETER TUNING MODE")
            return run_hyperparameter_tuning(output_path, total_rows, config, n_trials)

        use_incremental = total_rows > 5_000_000

        if use_incremental:
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
        else:
            print(f"  ✓ Using IN-MEMORY training")
            # Pandas can read Spark parquet directories directly
            df_pandas = pd.read_parquet(output_path)
            model, metrics = train_in_memory(
                df_pandas,
                config,
                feature_config_path=feature_config_path,
                rolling_importance=rolling_importance,
                rolling_importance_config=rolling_importance_config,
            )

        return model, metrics

    # If not reusing, do full feature engineering
    if reuse_processed:
        print(f"⚠️  Reuse requested but {output_path} not found. Running full pipeline...")

    print("="*60)
    print("🔧 RUNNING FULL FEATURE ENGINEERING PIPELINE")
    print("="*60)

    # Optimized Spark configuration for large datasets with memory constraints
    spark = SparkSession.builder \
        .appName("DemographicForecasts") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.driver.memoryOverhead", "2g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.5") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.default.parallelism", "16") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "64MB") \
        .config("spark.sql.autoBroadcastJoinThreshold", "256MB") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000") \
        .config("spark.python.worker.memory", "2g") \
        .config("spark.driver.extraJavaOptions",
                "-XX:+UseG1GC "
                "-XX:InitiatingHeapOccupancyPercent=35 "
                "-XX:G1HeapRegionSize=16M "
                "-XX:MaxGCPauseMillis=200 "
                "-XX:+ParallelRefProcEnabled "
                "-XX:ParallelGCThreads=8 "
                "-XX:ConcGCThreads=2 "
                "-XX:ReservedCodeCacheSize=512m "
                "-XX:NonProfiledCodeHeapSize=256m") \
        .config("spark.executor.extraJavaOptions",
                "-XX:+UseG1GC "
                "-XX:InitiatingHeapOccupancyPercent=35 "
                "-XX:G1HeapRegionSize=16M "
                "-XX:MaxGCPauseMillis=200 "
                "-XX:+ParallelRefProcEnabled "
                "-XX:ParallelGCThreads=8 "
                "-XX:ConcGCThreads=2 "
                "-XX:ReservedCodeCacheSize=512m "
                "-XX:NonProfiledCodeHeapSize=256m") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Set checkpoint directory for breaking lineage
    import tempfile
    checkpoint_dir = tempfile.mkdtemp(prefix="spark_checkpoint_")
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    print("✓ Spark session created with optimized settings")
    print(f"  Checkpoint dir: {checkpoint_dir}")

    window_spec = Window.partitionBy('id').orderBy('year')

    # ============================================================
    # STEP 2: READ DATA (NOW spark.read works!)
    # ============================================================
    DATA_PATH = "data/synthetic_with_demographics_fixed"
    files = list(pathlib.Path(DATA_PATH).rglob("*.parquet"))
    df = spark.read.parquet(*[str(file) for file in files])
    print(f"✓ Data loaded: {len(df.columns)} columns")

    # ============================================================
    # CRITICAL: Sample data EARLY to prevent OOM
    # ============================================================
    total_rows = df.count()
    print(f"Total rows in dataset: {total_rows:,}")

    # Sample configuration - can be overridden by function parameter
    if max_rows is None:
        max_rows = None  # Default: use all data

    if max_rows is not None and total_rows > max_rows:
        sample_fraction = max_rows / total_rows
        df = df.sample(fraction=sample_fraction, seed=42)
        print(f"✓ Sampled to {max_rows:,} rows ({sample_fraction:.1%} of data)")
    else:
        print(f"✓ Using full dataset: {total_rows:,} rows")

    # ============================================================
    # STEP 3: RENAME COLUMNS
    # ============================================================
    df = df.withColumnRenamed('moved', 'y_moved')
    df = df.withColumnRenamed('gol', 'getalifeother_event')
    df = df.withColumnRenamed('income_pp', 'MS_ADI_PP')
    df = df.withColumnRenamed('income_hh', 'MS_ADI_HH')
    print("✓ Columns renamed")
    event_cols = [c for c in df.columns if c.endswith("_event")]
    event_cols.extend(['y_moved'])  # birth1_event, birth2_event, divorce_event, ...

    # Repartition by sid for better window operation performance
    num_partitions = max(200, df.select("id").distinct().count() // 1000)
    print(f"Repartitioning by 'id' into {num_partitions} partitions...")
    df = df.repartition(num_partitions, "id")

    df2 = create_lags(df, event_cols)
    print("✓ Event history features created")

    # Use lazy checkpoint to truncate lineage (avoids OOM on large datasets)
    df2 = df2.checkpoint(eager=False)
    print(f"✓ Checkpointed (lazy) after lags")


        # Apply the pipeline - NOTE: create_hh_pos_features is NEW and MUST come first!
    df2 = (df2
        .transform(cast_events_to_bool)
        .transform(create_hh_pos_features)  # NEW - creates hh_pos_lag1
        .transform(create_income_features_spark)
        .transform(create_event_interactions_spark)
        .transform(create_age_interactions_spark))

    print(f"✅ Features created! Total columns: {len(df2.columns)}")

    df2 = create_all_socioeconomic_features(
        df2,  # This is your df AFTER creating event history features
        id_col='id',
        time_col='year',
        include_hh_pos_features=True
    )
    print(f"✓ Socioeconomic features created: {len(df2.columns)} columns")

    # Final lazy checkpoint (truncates lineage, avoids OOM)
    df2 = df2.checkpoint(eager=False)
    print(f"✓ Final df2 checkpointed (lazy): {len(df2.columns)} columns")

    # *** CRITICAL: IMPUTE MISSING VALUES ***
    # Lagged features and window operations create NULLs that must be filled
    print("\n🔧 IMPUTING MISSING VALUES...")
    from src.features.imputation import impute_missing_values
    df2 = impute_missing_values(
        df2,
        strategy="smart",  # median for numeric, mode for boolean
        exclude_cols=['id', 'year', 'refnis', 'y_moved'],  # Don't impute these
        verbose=True
    )
    print("✓ Imputation complete")

    # CRITICAL: Write to disk INCREMENTALLY (year by year to avoid OOM)
    print("\n📁 Writing processed data to disk (incremental by year)...")
    print("   This avoids Out-Of-Memory errors on large datasets")

    # Get list of years in the dataset
    years = [row.year for row in df2.select('year').distinct().orderBy('year').collect()]
    print(f"  Found {len(years)} years: {min(years)} - {max(years)}")

    # Write year by year to avoid OOM
    for i, year in enumerate(years):
        print(f"  [{i+1}/{len(years)}] Writing year {year}...", end=" ", flush=True)

        df_year = df2.filter(F.col('year') == year)

        # Coalesce to reduce number of files (improves read performance later)
        # Use 10 partitions per year for balance between parallelism and file count
        df_year = df_year.coalesce(10)

        if i == 0:
            # First year: overwrite
            df_year.write.mode("overwrite").parquet(output_path)
        else:
            # Subsequent years: append
            df_year.write.mode("append").parquet(output_path)

        # Get row count for this year
        year_count = df_year.count()
        print(f"{year_count:,} rows")

    print(f"✓ Data written to {output_path}")

    # Write SUCCESS marker
    import os
    success_file = os.path.join(output_path, "_SUCCESS")
    with open(success_file, 'w') as f:
        f.write("Success")
    print(f"✓ Success marker created")

    # Stop Spark to free memory
    spark.stop()
    print("✓ Spark session stopped")

    # Check dataset size to determine training strategy
    print("\n📊 Preparing data for training...")
    import pandas as pd
    import pyarrow.parquet as pq

    # Get dataset info without loading all data
    # Spark creates a directory with multiple part files
    parquet_dataset = pq.ParquetDataset(output_path)
    total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
    print(f"Total rows in processed data: {total_rows:,}")

    # Check if we should run hyperparameter tuning
    if tune:
        print("\n🔍 HYPERPARAMETER TUNING MODE")
        return run_hyperparameter_tuning(output_path, total_rows, config, n_trials)

    # Strategy: Use incremental training for very large datasets
    # With label encoding (fixed categorical issues), incremental is now stable
    incremental_threshold = 10_000_000   # Use incremental for >10M rows
    safe_in_memory_limit = 5_000_000     # Sample to 5M if using in-memory

    if total_rows > incremental_threshold:
        # LARGE dataset: use incremental training (now stable with label encoding)
        print(f"✓ Dataset is large ({total_rows:,} rows)")
        print(f"  Using INCREMENTAL training to process all data")
        print(f"  (This will take ~30-60 minutes)")

        use_incremental = True
        model, metrics = train_incremental(
            output_path,
            total_rows,
            config,
            feature_config_path=feature_config_path,
            target_batch_rows=target_batch_rows,
            rolling_importance=rolling_importance,
            rolling_importance_config=rolling_importance_config,
        )
        return model, metrics

    elif total_rows > safe_in_memory_limit:
        # Large dataset: sample to safe size
        print(f"⚠️  Dataset is large ({total_rows:,} rows)")
        print(f"   Sampling to {safe_in_memory_limit:,} rows for stable in-memory training")
        print(f"   💡 Tip: Use --max-rows=N to control sample size during feature engineering")

        # Load and sample
        print("   Loading data...")
        df_pandas = pd.read_parquet(output_path)
        if len(df_pandas) > safe_in_memory_limit:
            df_pandas = df_pandas.sample(n=safe_in_memory_limit, random_state=42)
        print(f"   ✓ Sampled to {len(df_pandas):,} rows")
    else:
        print(f"✓ Dataset size OK ({total_rows:,} rows) - loading into memory")
        df_pandas = pd.read_parquet(output_path)
        print(f"✓ Loaded {len(df_pandas):,} rows as Pandas DataFrame")

    # Always use in-memory training (proven stable)
    model, metrics = train_in_memory(
        df_pandas,
        config,
        feature_config_path=feature_config_path,
        rolling_importance=rolling_importance,
        rolling_importance_config=rolling_importance_config,
    )

    return model, metrics


def find_optimal_threshold(y_true, y_pred_proba, metric='f1', thresholds=None):
    """
    Find optimal classification threshold by maximizing a given metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        thresholds: List of thresholds to try (default: 0.1 to 0.9 in steps of 0.01)

    Returns:
        optimal_threshold: Threshold that maximizes the metric
        best_score: Best score achieved
        all_scores: Dict with scores for all thresholds
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.01)

    scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    # Find best threshold
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    all_scores = {
        'thresholds': thresholds,
        'scores': scores
    }

    return optimal_threshold, best_score, all_scores


def fix_dtypes(df_pandas, feature_cols, categorical_mappings=None):
    """Fix data types for XGBoost compatibility.

    Args:
        df_pandas: DataFrame to fix
        feature_cols: List of feature columns
        categorical_mappings: Optional dict of {col: list_of_all_categories}
                            For consistent categorical encoding across batches
    """
    print("\n🔧 Fixing data types...")

    # 1. Convert object columns to proper types
    boolean_cols = [col for col in feature_cols if df_pandas[col].dtype == 'object']
    for col in boolean_cols:
        if col in df_pandas.columns:
            try:
                df_pandas[col] = df_pandas[col].astype('bool').astype('int')
                print(f"  ✓ Converted {col}: object → int (was boolean)")
            except:
                print(f"  ⚠️  Skipping {col} (not convertible)")

    # 2. Convert categorical columns
    # CRITICAL: Use label encoding instead of categorical dtype to avoid
    # category mismatch issues between batches (different unique values per batch)
    categorical_cols = ['eerste_nationaliteit', 'hh_pos', 'gender', 'age_group', 'income_quintile']

    use_label_encoding = True  # More stable than categorical dtype

    if use_label_encoding:
        # Use simple numeric encoding (stable, no category alignment issues)
        from sklearn.preprocessing import LabelEncoder

        for col in categorical_cols:
            if col in feature_cols and col in df_pandas.columns:
                # Convert to numeric codes
                df_pandas[col] = df_pandas[col].astype('category').cat.codes
                print(f"  ✓ Converted {col} to int codes ({df_pandas[col].nunique()} unique values)")
    else:
        # Use pandas categorical (ONLY safe for single-batch/in-memory training)
        for col in categorical_cols:
            if col in feature_cols:
                if categorical_mappings and col in categorical_mappings:
                    # Use pre-fitted categories (for batch consistency)
                    df_pandas[col] = pd.Categorical(df_pandas[col],
                                                   categories=categorical_mappings[col])
                else:
                    df_pandas[col] = df_pandas[col].astype('category')
                print(f"  ✓ Converted {col} to category ({df_pandas[col].nunique()} unique values)")

    # 3. Ensure numeric columns are float32
    numeric_cols = df_pandas[feature_cols].select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df_pandas[col] = df_pandas[col].astype('float32')

    return df_pandas


def rolling_window_feature_importance(
    df_pandas,
    feature_cols,
    model_params,
    time_col="year",
    label_col="y_moved",
    train_years=8,
    test_years=1,
    step_years=1,
    min_train_rows=20000,
    min_test_rows=5000,
    sample_fraction=None,
    max_windows=None,
    importance_types=None,
    random_state=42,
    output_dir=None,
    stream_write=False,
):
    """
    Train models on rolling time windows and aggregate feature importances.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    import pandas as pd
    import numpy as np
    import os

    if time_col not in df_pandas.columns:
        print(f"⚠️  Rolling window importance skipped: no '{time_col}' column")
        return None

    years = sorted(df_pandas[time_col].dropna().unique())
    window_span = train_years + test_years
    if len(years) < window_span:
        print("⚠️  Rolling window importance skipped: not enough years for window span")
        return None

    if importance_types is None:
        importance_types = ["gain", "weight", "cover"]

    importance_rows = []
    metrics_rows = []
    windows_run = 0
    detail_path = None
    metrics_path = None
    detail_header_written = False
    metrics_header_written = False
    summary_stats = {}
    trend_stats = {}

    if stream_write and output_dir:
        detail_path = os.path.join(output_dir, "feature_importance_rolling_detail.csv")
        metrics_path = os.path.join(output_dir, "feature_importance_rolling_metrics.csv")

    for window_index, start_idx in enumerate(range(0, len(years) - window_span + 1, step_years), start=1):
        train_years_list = years[start_idx:start_idx + train_years]
        test_years_list = years[start_idx + train_years:start_idx + window_span]

        train_df = df_pandas[df_pandas[time_col].isin(train_years_list)]
        test_df = df_pandas[df_pandas[time_col].isin(test_years_list)]

        if sample_fraction and sample_fraction < 1.0:
            train_df = train_df.sample(frac=sample_fraction, random_state=random_state)
            test_df = test_df.sample(frac=sample_fraction, random_state=random_state)

        if len(train_df) < min_train_rows or len(test_df) < min_test_rows:
            continue

        if train_df[label_col].nunique() < 2 or test_df[label_col].nunique() < 2:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        X_test = test_df[feature_cols]
        y_test = test_df[label_col]

        model = XGBClassifier(**model_params)
        fit_kwargs = {"verbose": False}
        if model_params.get("early_stopping_rounds"):
            fit_kwargs["eval_set"] = [(X_test, y_test)]

        model.fit(X_train, y_train, **fit_kwargs)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_pred)

        window_id = (
            f"{train_years_list[0]}-{train_years_list[-1]}__"
            f"{test_years_list[0]}-{test_years_list[-1]}"
        )
        metrics_row = {
            "window_id": window_id,
            "window_index": window_index,
            "train_years": f"{train_years_list[0]}-{train_years_list[-1]}",
            "test_years": f"{test_years_list[0]}-{test_years_list[-1]}",
            "train_start_year": int(train_years_list[0]),
            "train_end_year": int(train_years_list[-1]),
            "test_start_year": int(test_years_list[0]),
            "test_end_year": int(test_years_list[-1]),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "auc_roc": float(auc),
            "auc_pr": float(auc_pr),
        }
        if stream_write and metrics_path:
            pd.DataFrame([metrics_row]).to_csv(
                metrics_path, mode="a", header=not metrics_header_written, index=False
            )
            metrics_header_written = True
        else:
            metrics_rows.append(metrics_row)

        for imp_type in importance_types:
            importance = model.get_booster().get_score(importance_type=imp_type)
            for feature in feature_cols:
                importance_value = float(importance.get(feature, 0.0))
                detail_row = {
                    "window_id": window_id,
                    "window_index": window_index,
                    "importance_type": imp_type,
                    "feature": feature,
                    "importance": importance_value,
                }
                if stream_write and detail_path:
                    pd.DataFrame([detail_row]).to_csv(
                        detail_path, mode="a", header=not detail_header_written, index=False
                    )
                    detail_header_written = True
                else:
                    importance_rows.append(detail_row)

                key = (imp_type, feature)
                stats = summary_stats.get(key, {"count": 0, "sum": 0.0, "sumsq": 0.0})
                stats["count"] += 1
                stats["sum"] += importance_value
                stats["sumsq"] += importance_value * importance_value
                summary_stats[key] = stats

                trend = trend_stats.get(key, {
                    "count": 0, "sumx": 0.0, "sumy": 0.0, "sumxy": 0.0, "sumx2": 0.0,
                    "last_importance": 0.0,
                })
                trend["count"] += 1
                trend["sumx"] += window_index
                trend["sumy"] += importance_value
                trend["sumxy"] += window_index * importance_value
                trend["sumx2"] += window_index * window_index
                trend["last_importance"] = importance_value
                trend_stats[key] = trend

        windows_run += 1
        if max_windows and windows_run >= max_windows:
            break

    if windows_run == 0:
        print("⚠️  Rolling window importance skipped: no valid windows")
        return None

    importance_df = pd.DataFrame(importance_rows) if importance_rows else None
    metrics_df = pd.DataFrame(metrics_rows)
    summary_rows = []
    for (imp_type, feature), stats in summary_stats.items():
        count = stats["count"]
        mean = stats["sum"] / count if count else 0.0
        variance = (stats["sumsq"] / count) - (mean * mean) if count else 0.0
        std = float(np.sqrt(max(variance, 0.0)))
        summary_rows.append({
            "importance_type": imp_type,
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": std,
            "windows": int(count),
        })
    summary_df = pd.DataFrame(summary_rows)

    trend_rows = []
    for (imp_type, feature), trend in trend_stats.items():
        count = trend["count"]
        if count < 2:
            continue
        denom = (count * trend["sumx2"]) - (trend["sumx"] * trend["sumx"])
        slope = 0.0
        if denom != 0:
            slope = ((count * trend["sumxy"]) - (trend["sumx"] * trend["sumy"])) / denom
        trend_rows.append({
            "importance_type": imp_type,
            "feature": feature,
            "slope": slope,
            "last_importance": float(trend["last_importance"]),
            "mean_importance": float(trend["sumy"] / count),
            "windows": int(count),
        })
    trend_df = pd.DataFrame(trend_rows)

    return {
        "importance_detail": importance_df,
        "importance_summary": summary_df,
        "window_metrics": metrics_df,
        "importance_trend": trend_df,
        "windows": windows_run,
        "stream_write": bool(stream_write and output_dir),
    }


def rolling_window_feature_importance_from_parquet(
    parquet_path,
    feature_cols,
    model_params,
    time_col="year",
    label_col="y_moved",
    train_years=8,
    test_years=1,
    step_years=1,
    min_train_rows=20000,
    min_test_rows=5000,
    sample_fraction=None,
    max_windows=None,
    importance_types=None,
    random_state=42,
    output_dir=None,
    stream_write=False,
    external_memory=False,
    external_memory_dir=None,
):
    """
    Rolling window feature importance with refit per window using parquet filters.
    """
    from xgboost import XGBClassifier
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    import pandas as pd
    import numpy as np
    import pyarrow.dataset as ds
    import os

    dataset = ds.dataset(parquet_path, format="parquet")
    years_set = set()
    for batch in dataset.to_batches(columns=[time_col], batch_size=1_000_000):
        years_set.update(batch.column(0).to_pylist())
    years = sorted(y for y in years_set if y is not None)

    window_span = train_years + test_years
    if len(years) < window_span:
        print("⚠️  Rolling window importance skipped: not enough years for window span")
        return None

    if external_memory and sample_fraction is not None and sample_fraction < 1.0:
        sample_fraction = 1.0

    if importance_types is None:
        importance_types = ["gain", "weight", "cover"]

    importance_rows = []
    metrics_rows = []
    windows_run = 0
    detail_path = None
    metrics_path = None
    detail_header_written = False
    metrics_header_written = False
    summary_stats = {}
    trend_stats = {}

    if stream_write and output_dir:
        detail_path = os.path.join(output_dir, "feature_importance_rolling_detail.csv")
        metrics_path = os.path.join(output_dir, "feature_importance_rolling_metrics.csv")

    for window_index, start_idx in enumerate(range(0, len(years) - window_span + 1, step_years), start=1):
        train_years_list = years[start_idx:start_idx + train_years]
        test_years_list = years[start_idx + train_years:start_idx + window_span]

        if external_memory:
            def _scan_stats(years_list):
                row_count = 0
                label_values = set()
                scanner = dataset.scanner(
                    filter=ds.field(time_col).isin(years_list),
                    columns=feature_cols + [label_col],
                    batch_size=100_000,
                )
                for batch in scanner.to_batches():
                    batch_df = batch.to_pandas()
                    if sample_fraction and sample_fraction < 1.0:
                        batch_df = batch_df.sample(frac=sample_fraction, random_state=random_state)
                    if len(batch_df) == 0:
                        continue
                    labels = batch_df[label_col].to_numpy()
                    label_values.update(set(labels.tolist()))
                    row_count += len(batch_df)
                return row_count, len(label_values)

            train_rows, train_label_count = _scan_stats(train_years_list)
            test_rows, test_label_count = _scan_stats(test_years_list)

            if train_rows < min_train_rows or test_rows < min_test_rows:
                continue
            if train_label_count < 2 or test_label_count < 2:
                continue

            class ParquetDataIter(xgb.DataIter):
                def __init__(self, years_list):
                    super().__init__()
                    self.years_list = years_list
                    self._scanner = None
                    self._batch_iter = None
                    self._had_data = False

                def reset(self):
                    self._scanner = dataset.scanner(
                        filter=ds.field(time_col).isin(self.years_list),
                        columns=feature_cols + [label_col],
                        batch_size=100_000,
                    )
                    self._batch_iter = iter(self._scanner.to_batches())
                    self._had_data = False

                def next(self, input_data):
                    while True:
                        try:
                            batch = next(self._batch_iter)
                        except StopIteration:
                            if not self._had_data:
                                raise ValueError("No data yielded for DataIter batch.")
                            return 0
                        batch_df = batch.to_pandas()
                        if sample_fraction and sample_fraction < 1.0:
                            batch_df = batch_df.sample(frac=sample_fraction, random_state=random_state)
                        if len(batch_df) == 0:
                            continue
                        batch_df = fix_dtypes(batch_df, feature_cols)
                        for col in feature_cols:
                            if batch_df[col].isnull().any():
                                batch_df[col] = batch_df[col].fillna(0)
                        labels = batch_df[label_col].to_numpy().astype(np.float32)
                        input_data(
                            data=batch_df[feature_cols].to_numpy(),
                            label=labels,
                        )
                        self._had_data = True
                        return 1

            dtrain_iter = ParquetDataIter(train_years_list)
            dtest_iter = ParquetDataIter(test_years_list)

            max_bin = model_params.get("max_bin", 256)
            dtrain = xgb.QuantileDMatrix(dtrain_iter, max_bin=max_bin)
            dtest = xgb.QuantileDMatrix(dtest_iter, max_bin=max_bin, ref=dtrain)

            xgb_params = model_params.copy()
            num_boost_round = int(xgb_params.pop("n_estimators", 200))
            xgb_params.pop("early_stopping_rounds", None)
            xgb_params.pop("verbose_eval", None)

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtest, "test")],
                verbose_eval=False,
            )
            y_pred = model.predict(dtest)
        else:
            train_df = pd.read_parquet(parquet_path, filters=[(time_col, "in", train_years_list)])
            test_df = pd.read_parquet(parquet_path, filters=[(time_col, "in", test_years_list)])

            if sample_fraction and sample_fraction < 1.0:
                train_df = train_df.sample(frac=sample_fraction, random_state=random_state)
                test_df = test_df.sample(frac=sample_fraction, random_state=random_state)

            if len(train_df) < min_train_rows or len(test_df) < min_test_rows:
                continue

            if train_df[label_col].nunique() < 2 or test_df[label_col].nunique() < 2:
                continue

            train_df = fix_dtypes(train_df, feature_cols)
            test_df = fix_dtypes(test_df, feature_cols)

            for col in feature_cols:
                if train_df[col].isnull().any():
                    if train_df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                        median_val = train_df[col].median()
                        train_df[col] = train_df[col].fillna(median_val)
                    elif train_df[col].dtype == 'bool':
                        train_df[col] = train_df[col].fillna(False)
                if test_df[col].isnull().any():
                    if test_df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                        median_val = test_df[col].median()
                        test_df[col] = test_df[col].fillna(median_val)
                    elif test_df[col].dtype == 'bool':
                        test_df[col] = test_df[col].fillna(False)

            X_train = train_df[feature_cols]
            y_train = train_df[label_col]
            X_test = test_df[feature_cols]
            y_test = test_df[label_col]

            model = XGBClassifier(**model_params)
            fit_kwargs = {"verbose": False}
            if model_params.get("early_stopping_rounds"):
                fit_kwargs["eval_set"] = [(X_test, y_test)]

            model.fit(X_train, y_train, **fit_kwargs)

            y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_pred)

        window_id = (
            f"{train_years_list[0]}-{train_years_list[-1]}__"
            f"{test_years_list[0]}-{test_years_list[-1]}"
        )
        metrics_row = {
            "window_id": window_id,
            "window_index": window_index,
            "train_years": f"{train_years_list[0]}-{train_years_list[-1]}",
            "test_years": f"{test_years_list[0]}-{test_years_list[-1]}",
            "train_start_year": int(train_years_list[0]),
            "train_end_year": int(train_years_list[-1]),
            "test_start_year": int(test_years_list[0]),
            "test_end_year": int(test_years_list[-1]),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "auc_roc": float(auc),
            "auc_pr": float(auc_pr),
        }
        if stream_write and metrics_path:
            pd.DataFrame([metrics_row]).to_csv(
                metrics_path, mode="a", header=not metrics_header_written, index=False
            )
            metrics_header_written = True
        else:
            metrics_rows.append(metrics_row)

        for imp_type in importance_types:
            if external_memory:
                importance = model.get_score(importance_type=imp_type)
                for idx, feature in enumerate(feature_cols):
                    importance_value = float(importance.get(f"f{idx}", 0.0))
                    detail_row = {
                        "window_id": window_id,
                        "window_index": window_index,
                        "importance_type": imp_type,
                        "feature": feature,
                        "importance": importance_value,
                    }
                    if stream_write and detail_path:
                        pd.DataFrame([detail_row]).to_csv(
                            detail_path, mode="a", header=not detail_header_written, index=False
                        )
                        detail_header_written = True
                    else:
                        importance_rows.append(detail_row)

                    key = (imp_type, feature)
                    stats = summary_stats.get(key, {"count": 0, "sum": 0.0, "sumsq": 0.0})
                    stats["count"] += 1
                    stats["sum"] += importance_value
                    stats["sumsq"] += importance_value * importance_value
                    summary_stats[key] = stats

                    trend = trend_stats.get(key, {
                        "count": 0, "sumx": 0.0, "sumy": 0.0, "sumxy": 0.0, "sumx2": 0.0,
                        "last_importance": 0.0,
                    })
                    trend["count"] += 1
                    trend["sumx"] += window_index
                    trend["sumy"] += importance_value
                    trend["sumxy"] += window_index * importance_value
                    trend["sumx2"] += window_index * window_index
                    trend["last_importance"] = importance_value
                    trend_stats[key] = trend
            else:
                importance = model.get_booster().get_score(importance_type=imp_type)
                for feature in feature_cols:
                    importance_value = float(importance.get(feature, 0.0))
                    detail_row = {
                        "window_id": window_id,
                        "window_index": window_index,
                        "importance_type": imp_type,
                        "feature": feature,
                        "importance": importance_value,
                    }
                    if stream_write and detail_path:
                        pd.DataFrame([detail_row]).to_csv(
                            detail_path, mode="a", header=not detail_header_written, index=False
                        )
                        detail_header_written = True
                    else:
                        importance_rows.append(detail_row)

                    key = (imp_type, feature)
                    stats = summary_stats.get(key, {"count": 0, "sum": 0.0, "sumsq": 0.0})
                    stats["count"] += 1
                    stats["sum"] += importance_value
                    stats["sumsq"] += importance_value * importance_value
                    summary_stats[key] = stats

                    trend = trend_stats.get(key, {
                        "count": 0, "sumx": 0.0, "sumy": 0.0, "sumxy": 0.0, "sumx2": 0.0,
                        "last_importance": 0.0,
                    })
                    trend["count"] += 1
                    trend["sumx"] += window_index
                    trend["sumy"] += importance_value
                    trend["sumxy"] += window_index * importance_value
                    trend["sumx2"] += window_index * window_index
                    trend["last_importance"] = importance_value
                    trend_stats[key] = trend

        windows_run += 1
        if max_windows and windows_run >= max_windows:
            break

    if windows_run == 0:
        print("⚠️  Rolling window importance skipped: no valid windows")
        return None

    importance_df = pd.DataFrame(importance_rows) if importance_rows else None
    metrics_df = pd.DataFrame(metrics_rows)
    summary_rows = []
    for (imp_type, feature), stats in summary_stats.items():
        count = stats["count"]
        mean = stats["sum"] / count if count else 0.0
        variance = (stats["sumsq"] / count) - (mean * mean) if count else 0.0
        std = float(np.sqrt(max(variance, 0.0)))
        summary_rows.append({
            "importance_type": imp_type,
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": std,
            "windows": int(count),
        })
    summary_df = pd.DataFrame(summary_rows)

    trend_rows = []
    for (imp_type, feature), trend in trend_stats.items():
        count = trend["count"]
        if count < 2:
            continue
        denom = (count * trend["sumx2"]) - (trend["sumx"] * trend["sumx"])
        slope = 0.0
        if denom != 0:
            slope = ((count * trend["sumxy"]) - (trend["sumx"] * trend["sumy"])) / denom
        trend_rows.append({
            "importance_type": imp_type,
            "feature": feature,
            "slope": slope,
            "last_importance": float(trend["last_importance"]),
            "mean_importance": float(trend["sumy"] / count),
            "windows": int(count),
        })
    trend_df = pd.DataFrame(trend_rows)

    return {
        "importance_detail": importance_df,
        "importance_summary": summary_df,
        "window_metrics": metrics_df,
        "importance_trend": trend_df,
        "windows": windows_run,
        "stream_write": bool(stream_write and output_dir),
    }


def evaluate_transition_rates_by_group(
    df_pandas,
    y_true,
    y_pred_proba,
    group_cols,
    min_group_size=100,
    y_pred_binary=None,
):
    """
    Evaluate transition rates by semi-aggregated groups.
    """
    import pandas as pd
    import numpy as np

    if not group_cols:
        return None, None

    eval_df = df_pandas[group_cols].copy()
    eval_df["y_true"] = np.asarray(y_true)
    eval_df["y_pred"] = np.asarray(y_pred_proba)
    if y_pred_binary is not None:
        eval_df["y_pred_binary"] = np.asarray(y_pred_binary)
    eval_df = eval_df.dropna(subset=group_cols)

    grouped = (
        eval_df
        .groupby(group_cols, dropna=True)
        .agg(
            count=("y_true", "size"),
            actual_rate=("y_true", "mean"),
            pred_rate=("y_pred", "mean"),
        )
        .reset_index()
    )

    if min_group_size:
        grouped = grouped[grouped["count"] >= min_group_size]

    if grouped.empty:
        return grouped, None

    grouped["abs_error"] = (grouped["pred_rate"] - grouped["actual_rate"]).abs()
    grouped["sq_error"] = (grouped["pred_rate"] - grouped["actual_rate"]) ** 2
    if "y_pred_binary" in eval_df.columns:
        grouped_binary = (
            eval_df
            .groupby(group_cols, dropna=True)
            .agg(pred_rate_thresholded=("y_pred_binary", "mean"))
            .reset_index()
        )
        grouped = grouped.merge(grouped_binary, on=group_cols, how="left")
        grouped["abs_error_thresholded"] = (
            grouped["pred_rate_thresholded"] - grouped["actual_rate"]
        ).abs()
        grouped["sq_error_thresholded"] = (
            grouped["pred_rate_thresholded"] - grouped["actual_rate"]
        ) ** 2

    weights = grouped["count"].to_numpy()
    summary = {
        "groups": int(len(grouped)),
        "min_group_size": int(min_group_size),
        "mae_weighted": float(np.average(grouped["abs_error"], weights=weights)),
        "mse_weighted": float(np.average(grouped["sq_error"], weights=weights)),
        "rmse_weighted": float(np.sqrt(np.average(grouped["sq_error"], weights=weights))),
        "mae_unweighted": float(grouped["abs_error"].mean()),
        "mse_unweighted": float(grouped["sq_error"].mean()),
        "rmse_unweighted": float(np.sqrt(grouped["sq_error"].mean())),
    }
    if "sq_error_thresholded" in grouped.columns:
        summary.update({
            "mae_weighted_thresholded": float(np.average(grouped["abs_error_thresholded"], weights=weights)),
            "mse_weighted_thresholded": float(np.average(grouped["sq_error_thresholded"], weights=weights)),
            "rmse_weighted_thresholded": float(np.sqrt(np.average(grouped["sq_error_thresholded"], weights=weights))),
            "mae_unweighted_thresholded": float(grouped["abs_error_thresholded"].mean()),
            "mse_unweighted_thresholded": float(grouped["sq_error_thresholded"].mean()),
            "rmse_unweighted_thresholded": float(np.sqrt(grouped["sq_error_thresholded"].mean())),
        })

    return grouped, summary


def tune_thresholds_by_refnis(
    df_pandas,
    y_true,
    y_pred_proba,
    refnis_col="refnis",
    subgroup_cols=None,
    thresholds=None,
    min_group_size=100,
    default_threshold=0.5,
):
    """
    Tune per-refnis thresholds to minimize mean squared error of group rates.
    """
    import numpy as np
    import pandas as pd

    if refnis_col not in df_pandas.columns:
        return None

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    eval_df = df_pandas[[refnis_col]].copy()
    eval_df["y_true"] = np.asarray(y_true)
    eval_df["y_pred"] = np.asarray(y_pred_proba)
    eval_df = eval_df.dropna(subset=[refnis_col])

    rows = []
    for refnis_value, group in eval_df.groupby(refnis_col, dropna=True):
        if len(group) < min_group_size:
            rows.append({
                refnis_col: refnis_value,
                "count": int(len(group)),
                "best_threshold": float(default_threshold),
                "actual_rate": float(group["y_true"].mean()),
                "pred_rate": float((group["y_pred"] >= default_threshold).mean()),
                "sq_error": float((group["y_true"].mean() - (group["y_pred"] >= default_threshold).mean()) ** 2),
                "used_default": True,
            })
            continue

        actual_rate = group["y_true"].mean()
        best_threshold = default_threshold
        best_pred_rate = (group["y_pred"] >= default_threshold).mean()
        best_sq_error = (actual_rate - best_pred_rate) ** 2
        best_score = None
        best_msqe = None

        for thr in thresholds:
            pred_binary = (group["y_pred"] >= thr).astype(int)
            pred_rate = pred_binary.mean()
            sq_error = (actual_rate - pred_rate) ** 2
            msqe = None

            if subgroup_cols:
                subgroup_df = group[[refnis_col]].copy()
                for col in subgroup_cols:
                    subgroup_df[col] = df_pandas.loc[group.index, col].values
                subgroup_df["y_true"] = group["y_true"].values
                subgroup_df["y_pred_binary"] = pred_binary
                subgroup_df = subgroup_df.dropna(subset=subgroup_cols)
                subgroup_stats = (
                    subgroup_df
                    .groupby(subgroup_cols, dropna=True)
                    .agg(
                        count=("y_true", "size"),
                        actual_rate=("y_true", "mean"),
                        pred_rate=("y_pred_binary", "mean"),
                    )
                    .reset_index()
                )
                if not subgroup_stats.empty:
                    subgroup_stats["sq_error"] = (subgroup_stats["pred_rate"] - subgroup_stats["actual_rate"]) ** 2
                    weights = subgroup_stats["count"].to_numpy()
                    msqe = float(np.average(subgroup_stats["sq_error"], weights=weights))

            score = msqe if msqe is not None else sq_error
            if best_score is None or score < best_score:
                best_score = score
                best_sq_error = sq_error
                best_threshold = thr
                best_pred_rate = pred_rate
                best_msqe = msqe

        rows.append({
            refnis_col: refnis_value,
            "count": int(len(group)),
            "best_threshold": float(best_threshold),
            "actual_rate": float(actual_rate),
            "pred_rate": float(best_pred_rate),
            "sq_error": float(best_sq_error),
            "msqe_weighted": float(best_msqe) if best_msqe is not None else None,
            "used_default": False,
        })

    return pd.DataFrame(rows)


def train_in_memory(
    df_pandas,
    config: Optional[Dict[str, Any]] = None,
    feature_config_path: Optional[str] = None,
    rolling_importance: bool = False,
    rolling_importance_config: Optional[Dict[str, Any]] = None,
):
    """
    Train XGBoost on small dataset that fits in memory.

    Args:
        df_pandas: Pandas DataFrame with features and labels
        config: Optional model configuration dict from YAML file
        feature_config_path: Optional path to feature selection config (e.g., 'configs/data/features_mixed.yaml')
    """
    print("\n🚀 Training XGBoost (in-memory)...")

    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

    # Setup MLflow tracking URI (connect to tracking server)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Start MLflow run
    mlflow.set_experiment("demographic_forecasts_training")
    with mlflow.start_run(run_name="in_memory_training"):
        try:
            # Drop non-feature columns
            drop_cols = ['id', 'year', 'refnis', 'y_moved']
            leak_cols = get_leaky_columns(df_pandas.columns)
            feature_cols = [c for c in df_pandas.columns if c not in drop_cols and c not in leak_cols]

            print(f"⚠️  Filtered out {len(leak_cols)} leaky columns to prevent temporal leakage")
            if len(leak_cols) > 0:
                print(f"   Removed: {', '.join(sorted(leak_cols)[:5])}{'...' if len(leak_cols) > 5 else ''}")

            # Apply feature selection from config if provided
            if feature_config_path is not None:
                feature_cols = filter_features_by_config(
                    feature_cols,
                    feature_config_path=feature_config_path,
                    verbose=True
                )
            else:
                print(f"\nℹ️  No feature config provided, using all {len(feature_cols)} available features")

            # CRITICAL: Validate feature_cols is a proper list of strings
            if not isinstance(feature_cols, list):
                raise TypeError(f"feature_cols must be a list, got {type(feature_cols)}")

            if len(feature_cols) == 0:
                raise ValueError("feature_cols is empty after filtering!")

            # Ensure all items are strings
            non_string_features = [f for f in feature_cols if not isinstance(f, str)]
            if non_string_features:
                raise ValueError(f"feature_cols contains non-string items: {non_string_features[:5]}")

            print(f"  ✓ Using {len(feature_cols)} features for training")

            # Fix dtypes
            df_pandas = fix_dtypes(df_pandas, feature_cols)

            # CRITICAL: Fill NULL values to prevent row dropping by XGBoost
            print("\n🔧 Imputing NULL values...")
            null_counts_before = df_pandas[feature_cols].isnull().sum().sum()
            if null_counts_before > 0:
                print(f"  Found {null_counts_before:,} NULL values across features")

                # Impute numeric features with median
                numeric_cols = df_pandas[feature_cols].select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
                imputed_count = 0
                for col in numeric_cols:
                    if df_pandas[col].isnull().any():
                        median_val = df_pandas[col].median()
                        df_pandas[col] = df_pandas[col].fillna(median_val)
                        imputed_count += 1

                # Boolean features with False
                bool_cols = df_pandas[feature_cols].select_dtypes(include=['bool']).columns
                for col in bool_cols:
                    if df_pandas[col].isnull().any():
                        df_pandas[col] = df_pandas[col].fillna(False)
                        imputed_count += 1

                null_counts_after = df_pandas[feature_cols].isnull().sum().sum()
                print(f"  ✓ Imputed {imputed_count} columns")
                print(f"  ✓ Remaining NULLs: {null_counts_after:,} (from {null_counts_before:,})")
            else:
                print("  ✓ No NULL values found")

            # CRITICAL: Time-based split to avoid temporal leakage
            # Train on earlier years, test on recent years (like in run.py)
            print("📅 Time-based split (avoiding temporal leakage):")

            if 'year' in df_pandas.columns:
                # Use year-based split
                train_df = df_pandas[df_pandas['year'] < 2023]  # Train: all years before 2023
                test_df = df_pandas[df_pandas['year'] >= 2023]  # Test: 2023-2025

                X_train = train_df[feature_cols]
                y_train = train_df['y_moved']
                X_test = test_df[feature_cols]
                y_test = test_df['y_moved']

                print(f"  Train years: {train_df['year'].min()}-{train_df['year'].max()}")
                print(f"  Test years: {test_df['year'].min()}-{test_df['year'].max()}")
            else:
                # Fallback: temporal split by row index (assuming sorted by time)
                print("  ⚠️  No 'year' column, using 80/20 temporal split by index")
                split_idx = int(len(df_pandas) * 0.8)
                train_df = df_pandas.iloc[:split_idx]
                test_df = df_pandas.iloc[split_idx:]

                X_train = train_df[feature_cols]
                y_train = train_df['y_moved']
                X_test = test_df[feature_cols]
                y_test = test_df['y_moved']

            print(f"✓ Train: {len(X_train):,} rows, Test: {len(X_test):,} rows")
            print(f"✓ Features: {len(feature_cols)}")

            # Check class balance
            print(f"  Train class balance: {y_train.mean():.2%} positive")
            print(f"  Test class balance: {y_test.mean():.2%} positive")

            # Get model parameters from config or use defaults
            if config is not None:
                model_params = config['model']['params'].copy()
                # Remove non-XGBClassifier params
                model_params.pop('verbose_eval', None)
                booster_type = model_params.get('booster', 'gbtree')
                print(f"\n🔧 Using config-based parameters (booster={booster_type})")
            else:
                # Default tree-based parameters
                print(f"\n🔧 Using default tree-based parameters")
                model_params = {
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'n_estimators': 500,
                    'tree_method': 'hist',
                    'device': 'cuda',
                    'objective': 'binary:logistic',
                    'eval_metric': ['aucpr', 'logloss'],
                    'early_stopping_rounds': 30,
                    'random_state': 42
                }

            # Log parameters to MLflow
            mlflow.log_params(model_params)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("training_mode", "in_memory")

            # Train XGBoost
            # NOTE: Using label encoding (int codes) instead of categorical dtype
            # This avoids category mismatch issues and is more stable
            model = XGBClassifier(**model_params)

            print("\n🎯 Training model...")
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)

            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            group_eval_df = None
            group_eval_summary = None
            thresholds_by_refnis = None

            # Compute AUC metrics (threshold-independent)
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_pr = average_precision_score(y_test, y_pred_proba)

            # Find optimal threshold for F1 score
            print("\n🎯 Finding optimal classification threshold...")
            optimal_threshold, best_f1, threshold_scores = find_optimal_threshold(
                y_test, y_pred_proba, metric='f1'
            )
            print(f"  ✓ Optimal threshold: {optimal_threshold:.3f}")
            print(f"  ✓ F1 score at optimal threshold: {best_f1:.4f}")

            # Semi-aggregated evaluation: transition rates by (sex, age_group, municipality)
            sex_col = "sex" if "sex" in test_df.columns else ("gender" if "gender" in test_df.columns else None)
            age_col = "age_group" if "age_group" in test_df.columns else None
            muni_col = "refnis" if "refnis" in test_df.columns else ("municipality" if "municipality" in test_df.columns else None)
            group_cols = [col for col in [sex_col, age_col, muni_col] if col is not None]
            if len(group_cols) == 3:
                thresholds_by_refnis = tune_thresholds_by_refnis(
                    df_pandas=test_df,
                    y_true=y_test,
                    y_pred_proba=y_pred_proba,
                    refnis_col=muni_col,
                    subgroup_cols=[sex_col, age_col],
                    min_group_size=100,
                    default_threshold=optimal_threshold,
                )
                if thresholds_by_refnis is not None and not thresholds_by_refnis.empty:
                    threshold_map = thresholds_by_refnis.set_index(muni_col)["best_threshold"]
                    thresholds_applied = test_df[muni_col].map(threshold_map).fillna(optimal_threshold)
                    y_pred_binary_refnis = (y_pred_proba >= thresholds_applied).astype(int)
                else:
                    y_pred_binary_refnis = None

                group_eval_df, group_eval_summary = evaluate_transition_rates_by_group(
                    df_pandas=test_df,
                    y_true=y_test,
                    y_pred_proba=y_pred_proba,
                    group_cols=group_cols,
                    min_group_size=100,
                    y_pred_binary=y_pred_binary_refnis,
                )
                if group_eval_summary:
                    mlflow.log_metric("group_rate_msqe_weighted", group_eval_summary["mse_weighted"])
                    if "mse_weighted_thresholded" in group_eval_summary:
                        mlflow.log_metric("group_rate_msqe_weighted_thresholded", group_eval_summary["mse_weighted_thresholded"])
                    if "rmse_weighted_thresholded" in group_eval_summary:
                        mlflow.log_metric("group_rate_rmsqe_weighted_thresholded", group_eval_summary["rmse_weighted_thresholded"])
                    if "mae_weighted_thresholded" in group_eval_summary:
                        mlflow.log_metric("group_rate_mae_weighted_thresholded", group_eval_summary["mae_weighted_thresholded"])
                    mlflow.log_metric("group_rate_mae_weighted", group_eval_summary["mae_weighted"])
            else:
                print("⚠️  Skipping semi-aggregated evaluation: required columns missing")

            # Compute metrics at optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            recall_optimal = recall_score(y_test, y_pred_optimal)
            prec_optimal = precision_score(y_test, y_pred_optimal)
            f1_optimal = f1_score(y_test, y_pred_optimal)

            # Also compute metrics at default threshold (what model.predict() uses)
            y_pred_default = model.predict(X_test)
            recall_default = recall_score(y_test, y_pred_default)
            prec_default = precision_score(y_test, y_pred_default)
            f1_default = f1_score(y_test, y_pred_default)

            print(f"\n📈 Metrics comparison:")
            print(f"  Default threshold (0.5):")
            print(f"    Precision: {prec_default:.4f} | Recall: {recall_default:.4f} | F1: {f1_default:.4f}")
            print(f"  Optimal threshold ({optimal_threshold:.3f}):")
            print(f"    Precision: {prec_optimal:.4f} | Recall: {recall_optimal:.4f} | F1: {f1_optimal:.4f}")
            if f1_default > 0:
                print(f"  F1 improvement: {(f1_optimal - f1_default) / f1_default * 100:+.1f}%")
            else:
                print(f"  F1 improvement: N/A (default F1 = 0, optimal F1 = {f1_optimal:.4f})")

            # Log metrics to MLflow (using optimal threshold)
            mlflow.log_metric("auc_roc", auc)
            mlflow.log_metric("auc_pr", auc_pr)
            mlflow.log_metric("optimal_threshold", optimal_threshold)
            mlflow.log_metric("f1_score", f1_optimal)
            mlflow.log_metric("precision", prec_optimal)
            mlflow.log_metric("recall", recall_optimal)
            mlflow.log_metric("f1_score_default", f1_default)
            mlflow.log_metric("precision_default", prec_default)
            mlflow.log_metric("recall_default", recall_default)
            mlflow.log_metric("train_class_balance", y_train.mean())
            mlflow.log_metric("test_class_balance", y_test.mean())

            # Log model
            mlflow.xgboost.log_model(model, "model")

            # Save local checkpoint with feature importance
            import os
            from datetime import datetime
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            booster_name = model_params.get('booster', 'gbtree')
            checkpoint_path = os.path.join(checkpoint_dir, f"model_{booster_name}_{timestamp}")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save model
            model.save_model(os.path.join(checkpoint_path, "model.json"))

            # Save feature importance
            import pandas as pd
            importance_types = ['weight', 'gain', 'cover']
            for imp_type in importance_types:
                try:
                    importance = model.get_booster().get_score(importance_type=imp_type)
                    if importance:
                        imp_df = pd.DataFrame([
                            {'feature': k, 'importance': v}
                            for k, v in importance.items()
                        ]).sort_values('importance', ascending=False)
                        imp_df.to_csv(
                            os.path.join(checkpoint_path, f"feature_importance_{imp_type}.csv"),
                            index=False
                        )
                except:
                    pass

            # Save semi-aggregated transition rates
            if group_eval_df is not None and not group_eval_df.empty:
                group_eval_df.to_csv(
                    os.path.join(checkpoint_path, "transition_rates_by_group.csv"),
                    index=False
                )
            if thresholds_by_refnis is not None and not thresholds_by_refnis.empty:
                thresholds_by_refnis.to_csv(
                    os.path.join(checkpoint_path, "thresholds_by_refnis.csv"),
                    index=False
                )

            # Rolling window feature importance for stability
            rolling_summary = None
            rolling_detail = None
            rolling_metrics = None
            rolling_windows = 0
            if rolling_importance:
                print("\n🧭 Rolling window feature importance...")
                rolling_cfg = {
                    "time_col": "year",
                    "label_col": "y_moved",
                    "train_years": 8,
                    "test_years": 1,
                    "step_years": 1,
                    "min_train_rows": 20000,
                    "min_test_rows": 5000,
                    "sample_fraction": 0.5,
                    "max_windows": None,
                    "importance_types": ["gain", "weight", "cover"],
                    "random_state": 42,
                    "external_memory": False,
                    "external_memory_dir": None,
                    "device": None,
                    "tree_method": None,
                }
                if rolling_importance_config:
                    rolling_cfg.update(rolling_importance_config)

                rolling_results = rolling_window_feature_importance(
                    df_pandas=df_pandas,
                    feature_cols=feature_cols,
                    model_params=model_params,
                    time_col=rolling_cfg["time_col"],
                    label_col=rolling_cfg["label_col"],
                    train_years=rolling_cfg["train_years"],
                    test_years=rolling_cfg["test_years"],
                    step_years=rolling_cfg["step_years"],
                    min_train_rows=rolling_cfg["min_train_rows"],
                    min_test_rows=rolling_cfg["min_test_rows"],
                    sample_fraction=rolling_cfg["sample_fraction"],
                    max_windows=rolling_cfg["max_windows"],
                    importance_types=rolling_cfg["importance_types"],
                    random_state=rolling_cfg["random_state"],
                    output_dir=checkpoint_path,
                    stream_write=True,
                )

                if rolling_results:
                    rolling_summary = rolling_results["importance_summary"]
                    rolling_detail = rolling_results["importance_detail"]
                    rolling_metrics = rolling_results["window_metrics"]
                    rolling_trend = rolling_results["importance_trend"]
                    rolling_windows = rolling_results["windows"]

                    rolling_summary.to_csv(
                        os.path.join(checkpoint_path, "feature_importance_rolling_summary.csv"),
                        index=False
                    )
                    if rolling_detail is not None and not rolling_results.get("stream_write"):
                        rolling_detail.to_csv(
                            os.path.join(checkpoint_path, "feature_importance_rolling_detail.csv"),
                            index=False
                        )
                    if rolling_metrics is not None and not rolling_results.get("stream_write"):
                        rolling_metrics.to_csv(
                            os.path.join(checkpoint_path, "feature_importance_rolling_metrics.csv"),
                            index=False
                        )
                    if rolling_trend is not None and not rolling_trend.empty:
                        rolling_trend.to_csv(
                            os.path.join(checkpoint_path, "feature_importance_rolling_trend.csv"),
                            index=False
                        )

                    print(f"  ✓ Rolling windows: {rolling_windows}")
                else:
                    print("  ⚠️  Rolling window importance not produced")

            # Save metadata
            import json
            metadata = {
                'timestamp': timestamp,
                'model_params': model_params,
                'optimal_threshold': float(optimal_threshold),
                'metrics': {
                    'auc_roc': float(auc),
                    'auc_pr': float(auc_pr),
                    'f1_score': float(f1_optimal),
                    'precision': float(prec_optimal),
                    'recall': float(recall_optimal),
                    'f1_score_default': float(f1_default),
                    'precision_default': float(prec_default),
                    'recall_default': float(recall_default)
                },
                'data': {
                    'train_rows': len(X_train),
                    'test_rows': len(X_test),
                    'n_features': len(feature_cols),
                    'train_years': f"{train_df['year'].min()}-{train_df['year'].max()}" if 'year' in df_pandas.columns else 'N/A',
                    'test_years': f"{test_df['year'].min()}-{test_df['year'].max()}" if 'year' in df_pandas.columns else 'N/A'
                },
                'rolling_importance': {
                    'enabled': bool(rolling_importance),
                    'windows': int(rolling_windows),
                },
                'semi_aggregated_evaluation': {
                    'enabled': bool(group_eval_summary),
                    'group_cols': group_cols if len(group_cols) == 3 else [],
                    'summary': group_eval_summary or {},
                },
            }
            with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            print("\n" + "="*60)
            print("📊 FINAL RESULTS (Optimal Threshold)")
            print("="*60)
            print(f"  AUC-ROC:         {auc:.4f}")
            print(f"  AUC-PR:          {auc_pr:.4f}")
            print(f"  Optimal Thresh:  {optimal_threshold:.3f}")
            print(f"  F1 Score:        {f1_optimal:.4f}")
            print(f"  Precision:       {prec_optimal:.4f}")
            print(f"  Recall:          {recall_optimal:.4f}")
            print("="*60)
            print(f"\n📊 MLflow: Run logged to experiment 'demographic_forecasts_training'")
            print(f"💾 Checkpoint saved to: {checkpoint_path}")

            return model, (auc, auc_pr, f1_optimal)

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise


def train_incremental(
    parquet_path,
    total_rows,
    config: Optional[Dict[str, Any]] = None,
    feature_config_path: Optional[str] = None,
    target_batch_rows: Optional[int] = None,
    rolling_importance: bool = False,
    rolling_importance_config: Optional[Dict[str, Any]] = None,
):
    """
    Train XGBoost incrementally on large dataset using batched loading.

    Args:
        parquet_path: Path to Spark-written parquet directory (contains part-*.parquet files)
        total_rows: Total number of rows in the dataset
        config: Optional model configuration dict from YAML file
        feature_config_path: Optional path to feature selection config (e.g., 'configs/data/features_mixed.yaml')
    """
    print("\n🚀 Training XGBoost (incremental batches)...")

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
    import pandas as pd

    # Setup MLflow tracking URI (connect to tracking server)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Start MLflow run for incremental training
    mlflow.set_experiment("demographic_forecasts_training")
    with mlflow.start_run(run_name="incremental_training"):
        # Configuration - VERY conservative for memory
        test_size = 100_000      # Smaller test set: 100K rows

        # Get schema by reading a small sample
        print("📋 Reading schema...")
        try:
            sample_df = pd.read_parquet(parquet_path, filters=[('year', '==', 2011)])
        except:
            # Fallback: read all and take first 1000 rows
            sample_df = pd.read_parquet(parquet_path).head(1000)

        if len(sample_df) == 0:
            raise ValueError("No data found in parquet files!")

        drop_cols = ['id', 'year', 'refnis', 'y_moved']
        leak_cols = get_leaky_columns(sample_df.columns)
        feature_cols = [c for c in sample_df.columns if c not in drop_cols and c not in leak_cols]

        print(f"✓ Features: {len(feature_cols)}")
        print(f"⚠️  Filtered out {len(leak_cols)} leaky columns to prevent temporal leakage")

        # Apply feature selection from config if provided
        if feature_config_path is not None:
            feature_cols = filter_features_by_config(
                feature_cols,
                feature_config_path=feature_config_path,
                verbose=True
            )
        else:
            print(f"\nℹ️  No feature config provided, using all {len(feature_cols)} available features")

        # CRITICAL: Validate feature_cols is a proper list of strings
        if not isinstance(feature_cols, list):
            raise TypeError(f"feature_cols must be a list, got {type(feature_cols)}")

        if len(feature_cols) == 0:
            raise ValueError("feature_cols is empty after filtering!")

        # Ensure all items are strings
        non_string_features = [f for f in feature_cols if not isinstance(f, str)]
        if non_string_features:
            raise ValueError(f"feature_cols contains non-string items: {non_string_features[:5]}")

        print(f"  ✓ Using {len(feature_cols)} features for training")

        # Step 1: Load test set - TIME-BASED SPLIT (avoiding temporal leakage)
        print(f"\n📅 Time-based train/test split:")
        print(f"  Train years: 2011-2023")
        print(f"  Test years: 2024-2025")

        print(f"\n📊 Loading test set (chunked to avoid OOM)...")
        import pyarrow.parquet as pq

        # Load test set in chunks to avoid OOM
        test_chunks = []
        total_test_rows = 0

        # Open parquet dataset and filter for test years
        parquet_dataset = pq.ParquetDataset(parquet_path, filters=[('year', '>=', 2024)])

        # Count total test rows first
        for fragment in parquet_dataset.fragments:
            total_test_rows += fragment.metadata.num_rows

        print(f"  Total test rows available: {total_test_rows:,}")

        # Calculate sampling fraction
        if total_test_rows > test_size:
            sample_fraction = test_size / total_test_rows
            print(f"  Sampling fraction: {sample_fraction:.2%}")
        else:
            sample_fraction = 1.0
            print(f"  Loading all test data (< {test_size:,} rows)")

        # Load and sample in chunks
        chunk_size = 100_000  # Read 100k rows at a time
        rows_loaded = 0

        for fragment in parquet_dataset.fragments:
            # Read this fragment (file) in batches
            for batch in fragment.to_batches(batch_size=chunk_size):
                df_chunk = batch.to_pandas()
                rows_loaded += len(df_chunk)

                # Sample from this chunk
                if sample_fraction < 1.0:
                    n_sample = int(len(df_chunk) * sample_fraction)
                    if n_sample > 0:
                        df_chunk = df_chunk.sample(n=n_sample, random_state=42)
                    else:
                        continue  # Skip this chunk if sample size is 0

                test_chunks.append(df_chunk)

                # Stop if we have enough data
                if sum(len(c) for c in test_chunks) >= test_size:
                    break

            if sum(len(c) for c in test_chunks) >= test_size:
                break

        # Combine chunks
        if len(test_chunks) == 0:
            raise ValueError("No test data found for years >= 2024!")

        test_df = pd.concat(test_chunks, ignore_index=True)
        del test_chunks

        # Final sampling if we have too many rows
        if len(test_df) > test_size:
            print(f"  Sampling final test set to {test_size:,} rows")
            test_df = test_df.sample(n=test_size, random_state=42)

        print(f"✓ Loaded test set: {len(test_df):,} rows")

        test_df = fix_dtypes(test_df, feature_cols)

        # Impute NULLs in test set
        print(f"  🔧 Imputing NULL values in test set...")
        for col in feature_cols:
            if test_df[col].isnull().any():
                if test_df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                    median_val = test_df[col].median()
                    test_df[col] = test_df[col].fillna(median_val)
                elif test_df[col].dtype == 'bool':
                    test_df[col] = test_df[col].fillna(False)

        X_test = test_df[feature_cols]
        y_test = test_df['y_moved']
        print(f"✓ Test set: {len(X_test):,} rows (years 2024-2025)")
        print(f"  Class balance: {y_test.mean():.2%} positive")

        # Step 2: Incremental training on remaining data
        print(f"\n🎯 Starting incremental training...")
        print("  ✓ Using label encoding (categorical issues resolved)")
        print("  ✓ GPU acceleration enabled")

        model = None

        # Get model parameters from config or use defaults
        if config is not None:
            params = config['model']['params'].copy()
            # Remove XGBClassifier-only params that don't work with xgb.train()
            params.pop('n_estimators', None)  # We use num_boost_round instead
            params.pop('verbose_eval', None)
            params.pop('early_stopping_rounds', None)
            booster_type = params.get('booster', 'gbtree')
            print(f"🔧 Using config-based parameters (booster={booster_type})")
        else:
            # Default tree-based parameters
            print(f"🔧 Using default tree-based parameters")
            params = {
                'max_depth': 5,
                'learning_rate': 0.05,
                'objective': 'binary:logistic',
                'eval_metric': ['aucpr', 'logloss'],
                'tree_method': 'hist',
                'device': 'cuda',  # GPU now safe with label encoding
                'random_state': 42,
                'max_bin': 256,
            }

        # Log parameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("total_rows", total_rows)
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("training_mode", "incremental")

        batch_num = 0
        total_trained = 0

        # Read in batches using year-based filtering (more efficient than chunksize)
        # Train years: 2011-2023 (split into batches)
        train_years = list(range(2011, 2024))

        # Batch size strategy based on dataset size
        rows_per_year = total_rows // 15  # Approximate rows per year
        if target_batch_rows is None:
            target_batch_rows = 10_000_000  # Target rows per batch (safe for GPU)
        years_per_batch = max(1, int(target_batch_rows / rows_per_year))

        print(f"  Strategy: ~{rows_per_year:,} rows/year → {years_per_batch} year(s) per batch")

        for i in range(0, len(train_years), years_per_batch):
            year_batch = train_years[i:i+years_per_batch]
            year_str = str(year_batch[0]) if len(year_batch) == 1 else f"{year_batch[0]}-{year_batch[-1]}"

            print(f"\n  Batch {batch_num + 1}: Loading year {year_str}...")

            # Load data for these years
            batch = pd.read_parquet(parquet_path,
                                    filters=[('year', 'in', year_batch)])

            if len(batch) == 0:
                print(f"    ⚠️  No data for years {year_batch}, skipping")
                continue

            batch_num += 1
            original_size = len(batch)
            print(f"    ✓ Loaded {original_size:,} rows")

            # With label encoding, we can process all data (no sampling)
            batch = fix_dtypes(batch, feature_cols)

            # Impute NULLs in batch
            for col in feature_cols:
                if batch[col].isnull().any():
                    if batch[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                        median_val = batch[col].median()
                        batch[col] = batch[col].fillna(median_val)
                    elif batch[col].dtype == 'bool':
                        batch[col] = batch[col].fillna(False)

            # CRITICAL: Check for duplicate columns before selection
            # Duplicates cause X_batch[col] to return DataFrame instead of Series
            if len(feature_cols) != len(set(feature_cols)):
                duplicates = [col for col in set(feature_cols) if feature_cols.count(col) > 1]
                raise ValueError(f"Duplicate features in feature_cols: {duplicates}")

            X_batch = batch[feature_cols]
            y_batch = batch['y_moved']

            # CRITICAL: Validate X_batch structure before passing to XGBoost
            # Ensure all columns are Series, not nested DataFrames
            if not isinstance(X_batch, pd.DataFrame):
                raise TypeError(f"X_batch should be DataFrame, got {type(X_batch)}")

            # Check for duplicate column names in result (pandas allows this)
            if len(X_batch.columns) != len(set(X_batch.columns)):
                dup_cols = [col for col in set(X_batch.columns) if list(X_batch.columns).count(col) > 1]
                raise ValueError(f"X_batch has duplicate column names: {dup_cols}")

            # Check for any column that might be a DataFrame instead of Series
            problem_cols = []
            for col in X_batch.columns:
                if isinstance(X_batch[col], pd.DataFrame):
                    problem_cols.append(col)

            if problem_cols:
                raise ValueError(f"Found DataFrame columns (should be Series): {problem_cols}")

            print(f"    📊 Training on {len(X_batch):,} rows...")

            # Create DMatrix
            # Using numeric codes, not categorical dtype
            dtrain = xgb.DMatrix(X_batch, label=y_batch)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Train incrementally
            # More trees per batch with GPU (faster training)
            rounds = 50 if model is None else 20  # First batch: 50 trees, then: 20/batch
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=rounds,
                xgb_model=model,  # Continue from previous model
                evals=[(dtest, 'test')],
                verbose_eval=10  # Show progress every 10 rounds
            )

            total_trained += len(X_batch)
            print(f"    ✓ Trained: {len(X_batch):,} rows | Total: {total_trained:,} | Trees: {model.num_boosted_rounds()}")

            # Evaluate on test set and log metrics per batch/year
            y_pred_batch = model.predict(dtest)
            auc_batch = roc_auc_score(y_test, y_pred_batch)
            auc_pr_batch = average_precision_score(y_test, y_pred_batch)

            # Log metrics with year/batch identifier
            mlflow.log_metric(f"auc_roc_year_{year_str}", auc_batch, step=batch_num)
            mlflow.log_metric(f"auc_pr_year_{year_str}", auc_pr_batch, step=batch_num)
            mlflow.log_metric("cumulative_rows_trained", total_trained, step=batch_num)
            mlflow.log_metric("total_trees", model.num_boosted_rounds(), step=batch_num)

            # Aggressive memory cleanup
            del batch, X_batch, y_batch, dtrain, dtest, y_pred_batch
            import gc
            gc.collect()

            # Show memory usage
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"    💾 Memory: {mem_mb:.0f} MB")

        # Final evaluation
        print("\n📊 Final evaluation...")
        dtest = xgb.DMatrix(X_test)  # Using numeric codes, not categorical dtype
        y_pred_proba = model.predict(dtest)

        group_eval_df = None
        group_eval_summary = None
        thresholds_by_refnis = None

        # Compute AUC metrics (threshold-independent)
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)

        # Find optimal threshold for F1 score
        print("\n🎯 Finding optimal classification threshold...")
        optimal_threshold, best_f1, threshold_scores = find_optimal_threshold(
            y_test, y_pred_proba, metric='f1'
        )
        print(f"  ✓ Optimal threshold: {optimal_threshold:.3f}")
        print(f"  ✓ F1 score at optimal threshold: {best_f1:.4f}")

        # Semi-aggregated evaluation: transition rates by (sex, age_group, municipality)
        sex_col = "sex" if "sex" in test_df.columns else ("gender" if "gender" in test_df.columns else None)
        age_col = "age_group" if "age_group" in test_df.columns else None
        muni_col = "refnis" if "refnis" in test_df.columns else ("municipality" if "municipality" in test_df.columns else None)
        group_cols = [col for col in [sex_col, age_col, muni_col] if col is not None]
        if len(group_cols) == 3:
            thresholds_by_refnis = tune_thresholds_by_refnis(
                df_pandas=test_df,
                y_true=y_test,
                y_pred_proba=y_pred_proba,
                refnis_col=muni_col,
                subgroup_cols=[sex_col, age_col],
                min_group_size=100,
                default_threshold=optimal_threshold,
            )
            if thresholds_by_refnis is not None and not thresholds_by_refnis.empty:
                threshold_map = thresholds_by_refnis.set_index(muni_col)["best_threshold"]
                thresholds_applied = test_df[muni_col].map(threshold_map).fillna(optimal_threshold)
                y_pred_binary_refnis = (y_pred_proba >= thresholds_applied).astype(int)
            else:
                y_pred_binary_refnis = None

            group_eval_df, group_eval_summary = evaluate_transition_rates_by_group(
                df_pandas=test_df,
                y_true=y_test,
                y_pred_proba=y_pred_proba,
                group_cols=group_cols,
                min_group_size=100,
                y_pred_binary=y_pred_binary_refnis,
            )
            if group_eval_summary:
                mlflow.log_metric("group_rate_msqe_weighted", group_eval_summary["mse_weighted"])
                if "mse_weighted_thresholded" in group_eval_summary:
                    mlflow.log_metric("group_rate_msqe_weighted_thresholded", group_eval_summary["mse_weighted_thresholded"])
                if "rmse_weighted_thresholded" in group_eval_summary:
                    mlflow.log_metric("group_rate_rmsqe_weighted_thresholded", group_eval_summary["rmse_weighted_thresholded"])
                if "mae_weighted_thresholded" in group_eval_summary:
                    mlflow.log_metric("group_rate_mae_weighted_thresholded", group_eval_summary["mae_weighted_thresholded"])
                mlflow.log_metric("group_rate_mae_weighted", group_eval_summary["mae_weighted"])
        else:
            print("⚠️  Skipping semi-aggregated evaluation: required columns missing")

        # Compute metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        recall_optimal = recall_score(y_test, y_pred_optimal)
        prec_optimal = precision_score(y_test, y_pred_optimal)
        f1_optimal = f1_score(y_test, y_pred_optimal)

        # Also compute metrics at default 0.5 threshold for comparison
        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        recall_default = recall_score(y_test, y_pred_default)
        prec_default = precision_score(y_test, y_pred_default)
        f1_default = f1_score(y_test, y_pred_default)

        print(f"\n📈 Metrics comparison:")
        print(f"  Default threshold (0.5):")
        print(f"    Precision: {prec_default:.4f} | Recall: {recall_default:.4f} | F1: {f1_default:.4f}")
        print(f"  Optimal threshold ({optimal_threshold:.3f}):")
        print(f"    Precision: {prec_optimal:.4f} | Recall: {recall_optimal:.4f} | F1: {f1_optimal:.4f}")
        try:
            f1_improvement = (f1_optimal - f1_default) / f1_default * 100
            print(f"  F1 improvement: {f1_improvement:+.1f}%")
        except ZeroDivisionError:
            print("  F1 improvement: N/A (default F1 is 0)")

        # Log final metrics to MLflow (using optimal threshold)
        mlflow.log_metric("final_auc_roc", auc)
        mlflow.log_metric("final_auc_pr", auc_pr)
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        mlflow.log_metric("final_f1_score", f1_optimal)
        mlflow.log_metric("final_precision", prec_optimal)
        mlflow.log_metric("final_recall", recall_optimal)
        mlflow.log_metric("final_f1_score_default", f1_default)
        mlflow.log_metric("final_precision_default", prec_default)
        mlflow.log_metric("final_recall_default", recall_default)
        mlflow.log_metric("final_trained_rows", total_trained)
        mlflow.log_metric("final_total_trees", model.num_boosted_rounds())

        # Log model
        mlflow.xgboost.log_model(model, "model")

        # Save local checkpoint with feature importance
        import os
        from datetime import datetime
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        booster_name = params.get('booster', 'gbtree')
        checkpoint_path = os.path.join(checkpoint_dir, f"model_incremental_{booster_name}_{timestamp}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model
        model.save_model(os.path.join(checkpoint_path, "model.json"))

        # Save feature importance
        importance_types = ['weight', 'gain', 'cover']
        for imp_type in importance_types:
            try:
                importance = model.get_score(importance_type=imp_type)
                if importance:
                    imp_df = pd.DataFrame([
                        {'feature': k, 'importance': v}
                        for k, v in importance.items()
                    ]).sort_values('importance', ascending=False)
                    imp_df.to_csv(
                        os.path.join(checkpoint_path, f"feature_importance_{imp_type}.csv"),
                        index=False
                    )
            except:
                pass

        # Save semi-aggregated transition rates
        if group_eval_df is not None and not group_eval_df.empty:
            group_eval_df.to_csv(
                os.path.join(checkpoint_path, "transition_rates_by_group.csv"),
                index=False
            )
        if thresholds_by_refnis is not None and not thresholds_by_refnis.empty:
            thresholds_by_refnis.to_csv(
                os.path.join(checkpoint_path, "thresholds_by_refnis.csv"),
                index=False
            )

        # Rolling window feature importance for stability
        rolling_summary = None
        rolling_detail = None
        rolling_metrics = None
        rolling_trend = None
        rolling_windows = 0
        if rolling_importance:
            print("\n🧭 Rolling window feature importance...")
            rolling_cfg = {
                "time_col": "year",
                "label_col": "y_moved",
                "train_years": 8,
                "test_years": 1,
                "step_years": 1,
                "min_train_rows": 20000,
                "min_test_rows": 5000,
                "sample_fraction": 0.5,
                "max_windows": None,
                "importance_types": ["gain", "weight", "cover"],
                "random_state": 42,
                "external_memory": False,
                "external_memory_dir": None,
                "device": None,
                "tree_method": None,
            }
            if config is not None:
                rolling_model_params = config['model']['params'].copy()
                rolling_model_params.pop('verbose_eval', None)
                if "n_estimators" not in rolling_model_params:
                    rolling_model_params["n_estimators"] = 200
            else:
                rolling_model_params = {
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'n_estimators': 200,
                    'tree_method': 'hist',
                    'device': 'cuda',
                    'objective': 'binary:logistic',
                    'eval_metric': ['aucpr', 'logloss'],
                    'random_state': 42
                }
            if rolling_importance_config:
                rolling_cfg.update(rolling_importance_config)

            if rolling_cfg.get("device"):
                rolling_model_params["device"] = rolling_cfg["device"]
            if rolling_cfg.get("tree_method"):
                rolling_model_params["tree_method"] = rolling_cfg["tree_method"]

            rolling_results = rolling_window_feature_importance_from_parquet(
                parquet_path=parquet_path,
                feature_cols=feature_cols,
                model_params=rolling_model_params,
                time_col=rolling_cfg["time_col"],
                label_col=rolling_cfg["label_col"],
                train_years=rolling_cfg["train_years"],
                test_years=rolling_cfg["test_years"],
                step_years=rolling_cfg["step_years"],
                min_train_rows=rolling_cfg["min_train_rows"],
                min_test_rows=rolling_cfg["min_test_rows"],
                sample_fraction=rolling_cfg["sample_fraction"],
                max_windows=rolling_cfg["max_windows"],
                importance_types=rolling_cfg["importance_types"],
                random_state=rolling_cfg["random_state"],
                output_dir=checkpoint_path,
                stream_write=True,
                external_memory=rolling_cfg["external_memory"],
                external_memory_dir=rolling_cfg["external_memory_dir"],
            )

            if rolling_results:
                rolling_summary = rolling_results["importance_summary"]
                rolling_detail = rolling_results["importance_detail"]
                rolling_metrics = rolling_results["window_metrics"]
                rolling_trend = rolling_results["importance_trend"]
                rolling_windows = rolling_results["windows"]

                rolling_summary.to_csv(
                    os.path.join(checkpoint_path, "feature_importance_rolling_summary.csv"),
                    index=False
                )
                if rolling_detail is not None and not rolling_results.get("stream_write"):
                    rolling_detail.to_csv(
                        os.path.join(checkpoint_path, "feature_importance_rolling_detail.csv"),
                        index=False
                    )
                if rolling_metrics is not None and not rolling_results.get("stream_write"):
                    rolling_metrics.to_csv(
                        os.path.join(checkpoint_path, "feature_importance_rolling_metrics.csv"),
                        index=False
                    )
                if rolling_trend is not None and not rolling_trend.empty:
                    rolling_trend.to_csv(
                        os.path.join(checkpoint_path, "feature_importance_rolling_trend.csv"),
                        index=False
                    )

                print(f"  ✓ Rolling windows: {rolling_windows}")
            else:
                print("  ⚠️  Rolling window importance not produced")

        # Save metadata
        import json
        metadata = {
            'timestamp': timestamp,
            'model_params': params,
            'training_mode': 'incremental',
            'metrics': {
                'auc_roc': float(auc),
                'auc_pr': float(auc_pr),
                'f1_score': float(f1_optimal),
                'precision': float(prec_optimal),
                'recall': float(recall_optimal)
            },
            'data': {
                'total_trained_rows': total_trained,
                'test_rows': len(y_test),
                'n_features': len(feature_cols),
                'total_trees': int(model.num_boosted_rounds()),
                'train_years': '2011-2023',
                'test_years': '2024-2025'
            },
            'semi_aggregated_evaluation': {
                'enabled': bool(group_eval_summary),
                'group_cols': group_cols if len(group_cols) == 3 else [],
                'summary': group_eval_summary or {},
            },
            'rolling_importance': {
                'enabled': bool(rolling_importance),
                'windows': int(rolling_windows),
            },
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "="*60)
        print("📊 RESULTS (Full Dataset)")
        print("="*60)
        print(f"  Trained on: {total_trained:,} rows")
        print(f"  Tested on:  {len(y_test):,} rows")
        print(f"  Total trees: {model.num_boosted_rounds()}")
        print(f"  Precision: {prec_optimal:.4f}")
        print(f"  Recall:    { recall_optimal:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  AUC-PR:    {auc_pr:.4f}")
        print(f"  F1 Score:  {f1_optimal:.4f}")
        print("="*60)
        print(f"\n📊 MLflow: Run logged to experiment 'demographic_forecasts_training'")
        print(f"💾 Checkpoint saved to: {checkpoint_path}")

        return model, (auc, auc_pr, f1_optimal)


if __name__ == "__main__":
    import sys

    argparser=argparse.ArgumentParser(description="Run feature engineering and model training pipeline with optional hyperparameter tuning.")

    # Show help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
Usage: python run_test.py [OPTIONS]

Options:
  --reuse, -r           Reuse existing processed_features.parquet (skip feature engineering)
  --max-rows=N          Limit dataset to N rows during feature engineering
  --max=N               Shorthand for --max-rows
  --config=PATH         Load model config from YAML file (e.g., configs/models/xgboost_linear.yaml)
  --features=PATH       Load feature selection config from YAML (e.g., configs/data/features_mixed.yaml)
  --tune                Run hyperparameter tuning with Optuna
  --n-trials=N          Number of tuning trials (default: 50)
  --rolling-importance  Enable rolling window feature importance (stable importance)
  --rolling-train-years=N    Rolling train window size (default: 8)
  --rolling-test-years=N     Rolling test window size (default: 1)
  --rolling-step-years=N     Rolling step size (default: 1)
  --rolling-sample-fraction=F Rolling sample fraction (default: 0.5)
  --rolling-max-windows=N    Max rolling windows (default: no limit)
  --rolling-min-train-rows=N Minimum rows in rolling train window (default: 20000)
  --rolling-min-test-rows=N  Minimum rows in rolling test window (default: 5000)
  --rolling-external-memory  Use external-memory (disk) for rolling windows
  --rolling-external-memory-dir=PATH  Directory for rolling window cache files
  --rolling-device=DEVICE    Device for rolling models (cpu or cuda)
  --rolling-tree-method=METHOD Tree method for rolling models (e.g., hist)
  --target-batch-rows=N Target rows per incremental batch (default: 10000000)
  --help, -h            Show this help message

Training Strategies:
  < 5M rows:            In-memory training (fast, all data)
  5M - 10M rows:        In-memory training with sampling to 5M
  > 10M rows:           Incremental training (all data, ~30-60 min)

Examples:
  # Quick test with 500K rows (2 minutes)
  python run_test.py --max-rows=500000

  # Train with XGBoost linear model
  python run_test.py --reuse --config=configs/models/xgboost_linear.yaml

  # Run hyperparameter tuning (tree-based)
  python run_test.py --reuse --tune --n-trials=100

  # Run hyperparameter tuning (linear model)
  python run_test.py --reuse --config=configs/models/xgboost_linear.yaml --tune

  # Full dataset incremental training with config
  python run_test.py --config=configs/models/xgboost_classifier.yaml

  # Reuse existing features for experiments (skip feature engineering)
  python run_test.py --reuse

Workflow:
  1. First run: Create features
     python run_test.py --max-rows=5000000

  2. Tune hyperparameters
     python run_test.py --reuse --tune --n-trials=50

  3. Train with best config
     python run_test.py --reuse --config=configs/models/xgboost_linear.yaml
        """)
        sys.exit(0)

    # Parse command line arguments
    reuse = "--reuse" in sys.argv or "-r" in sys.argv
    tune = "--tune" in sys.argv
    rolling_importance = "--rolling-importance" in sys.argv
    target_batch_rows = None
    rolling_importance_config = {}
    max_rows = None
    config_path = None
    feature_config_path = None
    n_trials = 50

    # Check for arguments
    for arg in sys.argv:
        if arg.startswith("--max-rows="):
            max_rows = int(arg.split("=")[1])
        elif arg.startswith("--max="):
            max_rows = int(arg.split("=")[1])
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("--features="):
            feature_config_path = arg.split("=", 1)[1]
        elif arg.startswith("--n-trials="):
            n_trials = int(arg.split("=")[1])
        elif arg.startswith("--target-batch-rows="):
            target_batch_rows = int(arg.split("=")[1])
        elif arg.startswith("--rolling-train-years="):
            rolling_importance_config["train_years"] = int(arg.split("=")[1])
        elif arg.startswith("--rolling-test-years="):
            rolling_importance_config["test_years"] = int(arg.split("=")[1])
        elif arg.startswith("--rolling-step-years="):
            rolling_importance_config["step_years"] = int(arg.split("=")[1])
        elif arg.startswith("--rolling-sample-fraction="):
            rolling_importance_config["sample_fraction"] = float(arg.split("=")[1])
        elif arg.startswith("--rolling-max-windows="):
            rolling_importance_config["max_windows"] = int(arg.split("=")[1])
        elif arg.startswith("--rolling-min-train-rows="):
            rolling_importance_config["min_train_rows"] = int(arg.split("=")[1])
        elif arg.startswith("--rolling-min-test-rows="):
            rolling_importance_config["min_test_rows"] = int(arg.split("=")[1])
        elif arg == "--rolling-external-memory":
            rolling_importance_config["external_memory"] = True
        elif arg.startswith("--rolling-external-memory-dir="):
            rolling_importance_config["external_memory_dir"] = arg.split("=", 1)[1]
        elif arg.startswith("--rolling-device="):
            rolling_importance_config["device"] = arg.split("=", 1)[1]
        elif arg.startswith("--rolling-tree-method="):
            rolling_importance_config["tree_method"] = arg.split("=", 1)[1]

    # Run main
    if reuse:
        print("🔄 Reuse flag detected - will skip feature engineering if possible")

    if max_rows:
        print(f"📊 Max rows set to: {max_rows:,}")

    if config_path:
        print(f"📝 Model config: {config_path}")

    if feature_config_path:
        print(f"🎯 Feature selection: {feature_config_path}")

    if tune:
        print(f"🔍 Tuning mode enabled ({n_trials} trials)")
    if rolling_importance:
        print("🧭 Rolling window feature importance enabled")
    if target_batch_rows:
        print(f"📦 Target batch rows: {target_batch_rows:,}")
    if rolling_importance and rolling_importance_config:
        print(f"🧭 Rolling config: {rolling_importance_config}")

    main(
        reuse_processed=reuse,
        max_rows=max_rows,
        config_path=config_path,
        tune=tune,
        n_trials=n_trials,
        feature_config_path=feature_config_path,
        rolling_importance=rolling_importance,
        rolling_importance_config=rolling_importance_config or None,
        target_batch_rows=target_batch_rows,
    )
