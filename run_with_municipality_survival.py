"""
Survival analysis pipeline with municipality-level contextual features.

Predicts time-to-event for all life events (moving, birth, divorce, partnership)
using XGBoost AFT or Cox models. Outputs P(event within 1, 3, 5 years) for
each individual.

Reuses the same processed features from the classifier pipeline via --reuse flag,
then restructures data into survival format (duration + censoring indicator).

Usage:
    # Quick test run
    python run_with_municipality_survival.py --reuse --max-rows=100000

    # Full run with all events
    python run_with_municipality_survival.py --reuse --config configs/models/xgboost_survival.yaml

    # Specific events only
    python run_with_municipality_survival.py --reuse --events y_moved,divorce_event

    # With feature selection
    python run_with_municipality_survival.py --reuse --features configs/data/survival_features.yaml

    # Hyperparameter tuning
    python run_with_municipality_survival.py --reuse --tune --n-trials 50
"""
import sys
import os
import argparse
import gc
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
# Setup logging
from src.utils.logging_setup import (
    setup_logging, log_stage_start, log_stage_complete, log_memory_usage
)

# Reuse existing functions from the classifier pipeline
from run_test import (
    load_model_config, fix_dtypes, get_leaky_columns,
)
from src.utils.feature_selection import filter_features_by_config

# Survival-specific modules
from src.survival.data_prep import (
    create_survival_dataset,
    create_all_survival_datasets,
    create_aft_labels,
    create_cox_labels,
    get_horizon_labels,
    create_survival_labels_chunked,
)
from src.survival.models import (
    train_survival_model,
    train_all_events,
    train_incremental_survival,
)
from src.survival.evaluation import (
    evaluate_survival_model,
    evaluate_aft_shared_metrics,
    evaluate_by_group,
    StreamingGroupEvaluator,
    concordance_index,
)
from src.survival.predict import (
    predict_event_probabilities,
    predict_all_events,
    predictions_to_dataframe,
    estimate_baseline_hazard,
)

import mlflow

# Default events to model
DEFAULT_EVENTS = [
    'y_moved',
    'birth1_event',
    'birth2_event',
    'divorce_event',
    'getalifeother_event',
]

DEFAULT_HORIZONS = [1, 3, 5]


def main_survival(
    reuse_processed=False,
    max_rows=None,
    config_path=None,
    feature_config_path=None,
    events=None,
    horizons=None,
    tune=False,
    n_trials=50,
    target_batch_rows=None,
):
    """
    Main survival analysis pipeline.

    Args:
        reuse_processed: Reuse existing processed features
        max_rows: Limit dataset size
        config_path: Path to survival model config YAML
        feature_config_path: Path to feature selection config
        events: List of event column names to model (default: all 5)
        horizons: Prediction horizons in years (default: [1, 3, 5])
        tune: Run hyperparameter tuning
        n_trials: Number of tuning trials
        target_batch_rows: Target rows per incremental batch
    """
    from datetime import datetime
    # Setup logging
    log_file = setup_logging(
        log_file=f"survival_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        log_dir='.'
    )
    print(f"Logging to: {log_file}")
    log_memory_usage()

    # Configuration
    if events is None:
        events = DEFAULT_EVENTS
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    config = None
    model_type = 'aft'
    distribution = 'normal'
    sigma = 1.0

    if config_path:
        config = load_model_config(config_path)
        model_type = config.get('model', {}).get('type', 'aft')
        params = config.get('model', {}).get('params', {})
        distribution = params.get('aft_loss_distribution', 'normal')
        sigma = params.get('aft_loss_distribution_scale', 1.0)
        events = config.get('events', events)
        horizons = config.get('horizons', horizons)

    print("=" * 60)
    print("SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"  Model type: {model_type.upper()}")
    print(f"  Events: {', '.join(events)}")
    print(f"  Horizons: {horizons} years")
    if model_type == 'aft':
        print(f"  Distribution: {distribution} (sigma={sigma})")
    print("=" * 60)

    # ================================================================
    # STAGE 1: Load processed features
    # ================================================================
    output_path = "data/processed_features_with_municipality"

    success_file = os.path.join(output_path, "_SUCCESS")
    can_reuse = reuse_processed and os.path.isdir(output_path) and os.path.exists(success_file)

    if not can_reuse:
        print("\nProcessed features not found. Running feature engineering first...")
        print("Please run the classifier pipeline first:")
        print("  python run_with_municipality_features.py")
        print("Then re-run this script with --reuse")
        print("\nAlternatively, if you have processed features at a different path,")
        print("ensure there is a _SUCCESS marker file in the directory.")
        sys.exit(1)

    log_stage_start("Loading Processed Features")
    print(f"\nLoading processed features from {output_path}...")

    parquet_dataset = pq.ParquetDataset(output_path)
    total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
    print(f"  Total rows: {total_rows:,}")

    # Determine loading strategy
    safe_in_memory_limit = 5_000_000
    use_incremental = total_rows > safe_in_memory_limit and max_rows is None

    if use_incremental:
        print(f"  Dataset is large ({total_rows:,} rows)")
        print(f"  Will use INCREMENTAL training")
        # For incremental: we need to add survival columns to parquet first
        # Load all data (chunked) to add survival labels, then save
        log_stage_complete("Loading Processed Features")
        return _run_incremental_pipeline(
            output_path, total_rows, events, horizons,
            config, model_type, distribution, sigma,
            feature_config_path, target_batch_rows,
        )

    # In-memory path
    if max_rows is not None and total_rows > max_rows:
        print(f"  Sampling to {max_rows:,} rows...")
        # Load in chunks and sample
        chunks = []
        rows_loaded = 0
        sample_fraction = max_rows / total_rows

        for fragment in parquet_dataset.fragments:
            for batch in fragment.to_batches(batch_size=500_000):
                df_chunk = batch.to_pandas()
                n_sample = max(1, int(len(df_chunk) * sample_fraction))
                chunks.append(df_chunk.sample(n=n_sample, random_state=42))
                rows_loaded += len(df_chunk)
                if rows_loaded >= total_rows:
                    break
            if rows_loaded >= total_rows:
                break

        df = pd.concat(chunks, ignore_index=True)
        del chunks
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        print(f"  Loaded {len(df):,} rows (sampled)")
    else:
        print(f"  Loading all {total_rows:,} rows...")
        if total_rows > 2_000_000:
            chunks = []
            for fragment in parquet_dataset.fragments:
                for batch in fragment.to_batches(batch_size=500_000):
                    chunks.append(batch.to_pandas())
            df = pd.concat(chunks, ignore_index=True)
            del chunks
        else:
            df = pd.read_parquet(output_path)
        print(f"  Loaded {len(df):,} rows")

    log_stage_complete("Loading Processed Features")
    log_memory_usage()

    # ================================================================
    # STAGE 2: Convert to survival format
    # ================================================================
    log_stage_start("Survival Data Conversion")
    print(f"\nConverting to survival format for {len(events)} events...")

    # Verify event columns exist
    missing_events = [e for e in events if e not in df.columns]
    if missing_events:
        print(f"  WARNING: Missing event columns: {missing_events}")
        events = [e for e in events if e in df.columns]
        if not events:
            print("  ERROR: No valid event columns found!")
            sys.exit(1)

    df = create_all_survival_datasets(
        df, events, id_col='sid', time_col='year'
    )

    # Summary
    print(f"\nSurvival labels created:")
    for event in events:
        duration_col = f"{event}_duration"
        observed_col = f"{event}_event_observed"
        n_events = df[observed_col].sum()
        median_duration = df[duration_col].median()
        print(f"  {event}: median duration={median_duration:.1f}yr, "
              f"events={n_events:,} ({n_events / len(df):.1%})")

    log_stage_complete("Survival Data Conversion")
    log_memory_usage()

    # ================================================================
    # STAGE 3: Prepare features
    # ================================================================
    log_stage_start("Feature Preparation")

    # Identify feature columns (exclude identifiers, targets, survival labels)
    drop_cols = ['sid', 'year', 'refnis', 'y_moved']
    survival_label_cols = []
    for event in events:
        survival_label_cols.extend([
            f"{event}_duration", f"{event}_event_observed"
        ])
    leak_cols = get_leaky_columns(df.columns)

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
        and c not in leak_cols
        and c not in survival_label_cols
        and c not in events  # exclude raw event columns (leaky)
    ]

    print(f"  Filtered out {len(leak_cols)} leaky columns")
    print(f"  Filtered out {len(survival_label_cols)} survival label columns")

    # Apply feature selection config
    if feature_config_path is not None:
        feature_cols = filter_features_by_config(
            feature_cols,
            feature_config_path=feature_config_path,
            verbose=True,
        )
    else:
        print(f"  No feature config provided, using all {len(feature_cols)} features")

    print(f"  Final feature count: {len(feature_cols)}")

    # Fix data types
    df = fix_dtypes(df, feature_cols)

    # Impute NULLs
    print("\n  Imputing NULL values...")
    null_count_before = df[feature_cols].isnull().sum().sum()
    if null_count_before > 0:
        for col in feature_cols:
            if df[col].isnull().any():
                if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                elif df[col].dtype == 'bool':
                    df[col] = df[col].fillna(False)
        null_count_after = df[feature_cols].isnull().sum().sum()
        print(f"  Imputed {null_count_before - null_count_after:,} NULLs "
              f"(remaining: {null_count_after:,})")
    else:
        print("  No NULL values found")

    log_stage_complete("Feature Preparation")
    log_memory_usage()

    # ================================================================
    # STAGE 4: Time-based train/test split
    # ================================================================
    log_stage_start("Train/Test Split")

    if 'year' in df.columns:
        train_df = df[df['year'] < 2023].copy()
        test_df = df[df['year'] >= 2023].copy()
        print(f"  Train: years {train_df['year'].min()}-{train_df['year'].max()}, "
              f"{len(train_df):,} rows")
        print(f"  Test: years {test_df['year'].min()}-{test_df['year'].max()}, "
              f"{len(test_df):,} rows")
    else:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    del df
    gc.collect()

    log_stage_complete("Train/Test Split")
    log_memory_usage()

    # ================================================================
    # STAGE 5: Hyperparameter tuning (optional)
    # ================================================================
    if tune:
        log_stage_start("Hyperparameter Tuning")
        result = _run_hyperparameter_tuning(
            train_df, test_df, feature_cols, events,
            config, model_type, n_trials, horizons,
        )
        log_stage_complete("Hyperparameter Tuning")
        return result

    # ================================================================
    # STAGE 6: Train survival models
    # ================================================================
    log_stage_start("Model Training")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("demographic_forecasts_survival")

    with mlflow.start_run(run_name=f"survival_{model_type}_{'_'.join(events)}"):
        # Log configuration
        if config:
            mlflow.log_params({
                k: str(v) for k, v in config.get('model', {}).get('params', {}).items()
            })
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("events", ",".join(events))
        mlflow.log_param("horizons", str(horizons))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))

        # Train one model per event
        models = train_all_events(
            train_df, test_df, feature_cols, events,
            config=config, model_type=model_type,
        )

        log_stage_complete("Model Training")
        log_memory_usage()

        # ================================================================
        # STAGE 7: Generate predictions
        # ================================================================
        log_stage_start("Prediction Generation")

        X_test = test_df[feature_cols]

        # For Cox models, estimate baseline hazard from training data
        baseline_hazards = None
        if model_type == 'cox':
            baseline_hazards = {}
            for event in events:
                duration_col = f"{event}_duration"
                observed_col = f"{event}_event_observed"
                baseline_hazards[event] = estimate_baseline_hazard(
                    train_df[duration_col].values,
                    train_df[observed_col].values,
                )

        all_predictions = predict_all_events(
            models, X_test,
            horizons=horizons,
            model_type=model_type,
            distribution=distribution,
            sigma=sigma,
            baseline_hazards=baseline_hazards,
        )

        # Convert to DataFrame
        pred_df = predictions_to_dataframe(
            all_predictions, horizons=horizons, index=test_df.index,
        )
        print(f"\n  Prediction DataFrame: {pred_df.shape}")

        log_stage_complete("Prediction Generation")
        log_memory_usage()

        # ================================================================
        # STAGE 8: Evaluate models
        # ================================================================
        log_stage_start("Model Evaluation")

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        all_metrics = {}
        print(f'events:{str(events)}' )
        for event in events:
            from datetime import datetime
            event_start = datetime.now()

            duration_col = f"{event}_duration"
            observed_col = f"{event}_event_observed"

            print(f"\n--- {event} ---")
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Evaluating {len(test_df):,} test samples...")

            # Prepare predicted probabilities dict
            t0 = datetime.now()
            event_proba = {}
            for h in horizons:
                key = f'prob_{h}yr'
                if key in all_predictions[event]:
                    event_proba[h] = all_predictions[event][key]
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Prepared predictions (+{(datetime.now() - t0).total_seconds():.1f}s)")

            # Evaluate (with subsampling for expensive metrics)
            t0 = datetime.now()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Computing metrics (train: {len(train_df):,}, test: {len(test_df):,})...")
            metrics = evaluate_survival_model(
                predicted_risk=all_predictions[event]['risk_score'],
                predicted_proba=event_proba,
                duration_test=test_df[duration_col].values,
                event_test=test_df[observed_col].values,
                horizons=horizons,
                duration_train=train_df[duration_col].values,
                event_train=train_df[observed_col].values,
                event_name=event,
            )
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Metrics computed (+{(datetime.now() - t0).total_seconds():.1f}s)")

            all_metrics[event] = metrics

            # Log to MLflow
            mlflow.log_metric(f"{event}_c_index", metrics['c_index'])
            for h in horizons:
                auc_key = f'auc_{h}yr'
                brier_key = f'brier_{h}yr'
                if auc_key in metrics and not np.isnan(metrics[auc_key]):
                    mlflow.log_metric(f"{event}_{auc_key}", metrics[auc_key])
                if brier_key in metrics and not np.isnan(metrics[brier_key]):
                    mlflow.log_metric(f"{event}_{brier_key}", metrics[brier_key])

        # Shared AFT metrics (CRPS, IBS, TD-AUC) — comparable with GRU sequence
        if model_type == 'aft':
            print(f"\n  Computing shared AFT metrics (distribution={distribution})...")
            shared_results = evaluate_aft_shared_metrics(
                all_predictions, test_df, events,
                horizons=horizons,
                distribution=distribution,
                sigma=sigma,
            )
            agg = shared_results['aggregate']
            print(f"  Shared AFT aggregate: "
                  f"C-index={agg['mean_c_index']:.4f}, "
                  f"CRPS={agg['mean_crps']:.4f}, "
                  f"IBS={agg['mean_ibs']:.4f}, "
                  f"TD-AUC={agg['mean_td_auc']:.4f}")
            for k, v in agg.items():
                if not np.isnan(v):
                    mlflow.log_metric(f"shared_{k}", v)
            for event, evt_m in shared_results.get('per_event', {}).items():
                for k, v in evt_m.items():
                    mlflow.log_metric(f"shared_{event}_{k}", float(v))

        # Per-event F1, AP, CRPSS — comparable with GRU sequence
        from sklearn.metrics import f1_score, average_precision_score
        from src.sequence.evaluation import _fast_crps, _targets_to_survival

        print(f"\n  Computing F1, AP, CRPSS per event...")
        all_f1, all_ap, all_crpss = [], [], []
        n_horizons = len(horizons)
        for ei, event in enumerate(events):
            duration_col = f"{event}_duration"
            observed_col = f"{event}_event_observed"

            # F1 and AP per horizon
            event_f1s, event_aps = [], []
            for h in horizons:
                y_true = ((test_df[observed_col].values == 1) &
                          (test_df[duration_col].values <= h)).astype(int)
                y_prob = all_predictions[event].get(f'prob_{h}yr')
                if y_prob is None or y_true.sum() < 5:
                    continue
                # AP
                ap = float(average_precision_score(y_true, y_prob))
                event_aps.append(ap)
                mlflow.log_metric(f"{event}_ap_{h}yr", ap)
                # F1 with optimal threshold
                base_rate = float(y_true.mean())
                candidates = np.unique(np.concatenate([
                    np.linspace(max(0.005, base_rate * 0.2),
                                min(0.95, base_rate * 5), 30),
                    np.array([0.5, base_rate]),
                ]))
                best_f1 = 0.0
                for thr in candidates:
                    _f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
                    if _f1 > best_f1:
                        best_f1 = _f1
                f1_val = float(best_f1)
                event_f1s.append(f1_val)
                mlflow.log_metric(f"{event}_f1_{h}yr", f1_val)

            if event_f1s:
                mean_f1_e = float(np.mean(event_f1s))
                mean_ap_e = float(np.mean(event_aps))
                all_f1.append(mean_f1_e)
                all_ap.append(mean_ap_e)
                print(f"  {event}: mean_F1={mean_f1_e:.4f}, mean_AP={mean_ap_e:.4f}")
                mlflow.log_metric(f"{event}_mean_f1", mean_f1_e)
                mlflow.log_metric(f"{event}_mean_ap", mean_ap_e)

            # CRPSS per event (using XGBoost's predicted_log_time as mu)
            if model_type == 'aft' and 'predicted_log_time' in all_predictions[event]:
                mu_e = all_predictions[event]['predicted_log_time']
                sigma_e = np.full_like(mu_e, sigma)
                # Build duration/event from targets
                dur = test_df[duration_col].values.astype(np.float64)
                evt = test_df[observed_col].values.astype(np.float64)
                max_h = float(max(horizons))
                crps_e, crps_naive_e, skill_e = _fast_crps(
                    mu_e, sigma_e, dur, evt,
                    max_horizon=max_h,
                    distribution=distribution if distribution != 'normal' else 'normal',
                    return_skill=True,
                )
                all_crpss.append(skill_e)
                print(f"  {event}: CRPS={crps_e:.4f}, CRPSS={skill_e:.4f} (naive={crps_naive_e:.4f})")
                mlflow.log_metric(f"{event}_crps", crps_e)
                mlflow.log_metric(f"{event}_crpss", skill_e)

        if all_f1:
            mean_f1 = float(np.mean(all_f1))
            mean_ap = float(np.mean(all_ap))
            print(f"\n  Overall: mean_F1={mean_f1:.4f}, mean_AP={mean_ap:.4f}")
            mlflow.log_metric("mean_f1", mean_f1)
            mlflow.log_metric("mean_ap", mean_ap)
        if all_crpss:
            mean_crpss = float(np.mean(np.clip(all_crpss, 0, None)))
            print(f"  Overall: mean_CRPSS={mean_crpss:.4f}")
            mlflow.log_metric("mean_crpss", mean_crpss)

        # Per-event sigma calibration (AFT only)
        if model_type == 'aft':
            print(f"\n  Calibrating per-event sigma to minimise CRPS...")
            sigma_candidates = np.geomspace(0.3, 3.0, 30) * sigma
            for event in events:
                duration_col = f"{event}_duration"
                observed_col = f"{event}_event_observed"
                mu_e = all_predictions[event]['predicted_log_time']
                dur = test_df[duration_col].values.astype(np.float64)
                evt = test_df[observed_col].values.astype(np.float64)
                max_h = float(max(horizons))

                best_crps, best_sigma = float('inf'), sigma
                for s_cand in sigma_candidates:
                    sigma_arr = np.full_like(mu_e, s_cand)
                    crps_s = _fast_crps(
                        mu_e, sigma_arr, dur, evt,
                        max_horizon=max_h, distribution=distribution,
                    )
                    if crps_s < best_crps:
                        best_crps = crps_s
                        best_sigma = float(s_cand)
                _, _, skill_cal = _fast_crps(
                    mu_e, np.full_like(mu_e, best_sigma), dur, evt,
                    max_horizon=max_h, distribution=distribution,
                    return_skill=True,
                )
                print(f"  {event}: optimal_sigma={best_sigma:.3f} "
                      f"(was {sigma:.3f}), CRPS={best_crps:.4f}, CRPSS={skill_cal:.4f}")
                mlflow.log_metric(f"{event}_sigma_calibrated", best_sigma)
                mlflow.log_metric(f"{event}_crps_calibrated", best_crps)
                mlflow.log_metric(f"{event}_crpss_calibrated", skill_cal)

        # Event ordering (AFT only) — comparable with GRU sequence
        if model_type == 'aft':
            print(f"\n  Computing event ordering metrics...")
            try:
                from src.sequence.evaluation import evaluate_event_ordering

                n_ev = len(events)
                n_samples = len(test_df)
                pred_mu = np.column_stack([
                    all_predictions[event]['predicted_log_time'] for event in events
                ])
                # Build binary horizon targets
                targets = np.zeros((n_samples, n_ev * n_horizons), dtype=np.float32)
                for ei, event in enumerate(events):
                    dur = test_df[f"{event}_duration"].values
                    evt = test_df[f"{event}_event_observed"].values
                    for hi, h in enumerate(horizons):
                        col = ei * n_horizons + hi
                        targets[:, col] = ((evt == 1) & (dur <= h)).astype(np.float32)

                ordering = evaluate_event_ordering(
                    predicted_mu=pred_mu,
                    targets=targets,
                    events=events,
                    horizons=horizons,
                    min_events=2,
                    top_k=3,
                )
                n_elig = ordering['n_eligible']
                print(f"  Persons with 2+ events: {n_elig} "
                      f"({100*n_elig/n_samples:.1f}% of test set)")
                if n_elig > 0:
                    print(f"  Pairwise accuracy:     {ordering['pairwise_accuracy']:.4f}")
                    print(f"  Top-1 accuracy:        {ordering['top1_accuracy']:.4f}")
                    print(f"  Top-3 accuracy:        {ordering['topk_accuracy']:.4f}")
                    print(f"  Mean reciprocal rank:  {ordering['mean_reciprocal_rank']:.4f}")
                    print(f"  Kendall's tau:         {ordering['kendall_tau']:.4f}")
                    for k, v in ordering.items():
                        if isinstance(v, float) and not np.isnan(v):
                            mlflow.log_metric(f"ordering_{k}", v)
                    print("\n  Observed first-event distribution:")
                    for event_name, rate in ordering['per_event_first_rate'].items():
                        pred_rate = ordering['per_event_pred_first_rate'].get(event_name, 0)
                        print(f"    {event_name}: observed={rate:.3f}, predicted={pred_rate:.3f}")
                        mlflow.log_metric(f"ordering_obs_first_{event_name}", rate)
                        mlflow.log_metric(f"ordering_pred_first_{event_name}", pred_rate)
            except Exception as e:
                print(f"  Event ordering failed: {e}")
                import traceback; traceback.print_exc()

        # Prediction intervals (AFT only) — comparable with GRU sequence
        if model_type == 'aft':
            print(f"\n  Computing prediction intervals...")
            for event in events:
                mu_e = all_predictions[event]['predicted_log_time']
                med_time = np.exp(mu_e)
                # Confidence intervals from AFT distribution
                from scipy.stats import norm, logistic as logistic_dist
                if distribution == 'normal':
                    dist_fn = norm
                elif distribution == 'logistic':
                    dist_fn = logistic_dist
                else:
                    dist_fn = norm  # fallback

                for level in [0.5, 0.8, 0.9]:
                    alpha = 1.0 - level
                    lower_q = dist_fn.ppf(alpha / 2, loc=mu_e, scale=sigma)
                    upper_q = dist_fn.ppf(1 - alpha / 2, loc=mu_e, scale=sigma)
                    ci_lower = np.exp(lower_q)
                    ci_upper = np.exp(upper_q)
                    level_pct = int(level * 100)
                    print(f"  {event}: median={np.median(med_time):.2f}yr, "
                          f"{level_pct}%CI=[{np.median(ci_lower):.2f}, {np.median(ci_upper):.2f}]yr")
                    mlflow.log_metric(f"aft_{event}_median_tte", float(np.median(med_time)))
                    mlflow.log_metric(f"aft_{event}_ci{level_pct}_width",
                                      float(np.median(ci_upper - ci_lower)))

        # LM-style metrics (comparable with GRU sequence)
        print(f"\n  Computing LM-style metrics (top-k accuracy, MRR, perplexity)...")
        try:
            from src.survival.evaluation import stack_survival_predictions
            from src.sequence.evaluation import evaluate_lm_metrics, evaluate_grouped_metrics

            stacked_probs, stacked_targets = stack_survival_predictions(
                all_predictions, test_df, events, horizons=horizons,
            )

            lm_results = evaluate_lm_metrics(
                stacked_probs, stacked_targets, events, horizons, top_k=3,
            )
            print(f"  LM next-event top-1 accuracy: {lm_results['next_event_top1_accuracy']:.4f}")
            print(f"  LM next-event top-3 accuracy: {lm_results['next_event_topk_accuracy']:.4f}")
            print(f"  LM MRR:                       {lm_results['next_event_mrr']:.4f}")
            print(f"  LM event perplexity:          {lm_results['event_perplexity']:.4f}")
            print(f"  LM temporal consistency:       {lm_results['temporal_consistency']:.4f}")
            print(f"  LM cross-horizon rank stab.:   {lm_results['cross_horizon_rank_stability']:.4f}")

            for k, v in lm_results.items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(f"lm_{k}", v)
                elif isinstance(v, dict) and k == 'per_horizon':
                    for h_name, h_metrics in v.items():
                        for mk, mv in h_metrics.items():
                            if isinstance(mv, float) and not np.isnan(mv):
                                mlflow.log_metric(f"lm_{h_name}_{mk}", mv)
                elif isinstance(v, dict) and k in ('per_event_recall', 'per_event_precision'):
                    for ev_name, ev_val in v.items():
                        if isinstance(ev_val, float) and not np.isnan(ev_val):
                            mlflow.log_metric(f"lm_{k}_{ev_name}", ev_val)
        except Exception as e:
            print(f"  LM metrics failed: {e}")
            import traceback; traceback.print_exc()

        # Grouped evaluation (LM + classification + MAE per age-group / municipality)
        age_col = "age_group" if "age_group" in test_df.columns else None
        muni_col = "refnis" if "refnis" in test_df.columns else None
        grouped_cols = {c: c for c in [age_col, muni_col] if c is not None}

        if grouped_cols:
            print(f"\n  Computing grouped evaluation by {list(grouped_cols.keys())}...")
            try:
                for grp_name, grp_col in grouped_cols.items():
                    group_labels = test_df[grp_col].values

                    grp_result = evaluate_grouped_metrics(
                        stacked_probs, stacked_targets, group_labels,
                        events, horizons,
                        group_name=grp_name,
                        min_group_size=50,
                        calibrate_thresholds=True,
                    )

                    gt = grp_result['group_table']
                    summary = grp_result['summary']
                    lm_sum = grp_result['lm_summary']

                    print(f"\n  --- Grouped by {grp_name} ({summary.get('n_groups', 0)} groups) ---")

                    # Print classification/MAE summary
                    for ei, event in enumerate(events):
                        for hi, h in enumerate(horizons):
                            prefix = f'{event}_{h}yr'
                            mae_w = summary.get(f'{prefix}_mae_weighted')
                            rmse_w = summary.get(f'{prefix}_rmse_weighted')
                            if mae_w is not None:
                                print(f"    {prefix}: MAE_w={mae_w:.4f}  RMSE_w={rmse_w:.4f}")
                                mlflow.log_metric(f"grp_{grp_name}_{prefix}_mae_w", mae_w)
                                mlflow.log_metric(f"grp_{grp_name}_{prefix}_rmse_w", rmse_w)
                            cal_mae_w = summary.get(f'{prefix}_cal_mae_weighted')
                            if cal_mae_w is not None:
                                mlflow.log_metric(f"grp_{grp_name}_{prefix}_cal_mae_w", cal_mae_w)

                    # Print LM summary
                    if lm_sum:
                        print(f"    LM top-1 weighted: {lm_sum.get('lm_top1_acc_weighted', float('nan')):.4f}")
                        print(f"    LM top-3 weighted: {lm_sum.get('lm_top3_acc_weighted', float('nan')):.4f}")
                        print(f"    LM MRR weighted:   {lm_sum.get('lm_mrr_weighted', float('nan')):.4f}")
                        for lk, lv in lm_sum.items():
                            if isinstance(lv, float) and not np.isnan(lv):
                                mlflow.log_metric(f"grp_{grp_name}_{lk}", lv)

                    # Save group table as artifact
                    if not gt.empty:
                        grp_csv = f"group_eval_{grp_name}.csv"
                        gt.to_csv(grp_csv, index=False)
                        mlflow.log_artifact(grp_csv)
                        print(f"    Saved {grp_csv} ({len(gt)} groups)")

            except Exception as e:
                print(f"  Grouped evaluation failed: {e}")
                import traceback; traceback.print_exc()

        # Group-level evaluation (legacy per-event)
        sex_col = "gender" if "gender" in test_df.columns else None
        age_col_legacy = "age_group" if "age_group" in test_df.columns else None
        muni_col_legacy = "refnis" if "refnis" in test_df.columns else None
        group_cols = [c for c in [sex_col, age_col_legacy, muni_col_legacy] if c is not None]

        group_eval_dfs = {}
        group_eval_summaries = {}
        if group_cols:
            print(f"\nGroup-level evaluation by {group_cols}...")
            for event in events:
                duration_col = f"{event}_duration"
                observed_col = f"{event}_event_observed"

                event_proba = {}
                for h in horizons:
                    key = f'prob_{h}yr'
                    if key in all_predictions[event]:
                        event_proba[h] = all_predictions[event][key]

                group_df, group_summary = evaluate_by_group(
                    test_df,
                    all_predictions[event]['risk_score'],
                    event_proba,
                    duration_col, observed_col,
                    group_cols, horizons=horizons,
                )
                group_eval_dfs[event] = group_df
                group_eval_summaries[event] = group_summary

                if group_summary:
                    for h in horizons:
                        rmse_w = group_summary.get(f'rmse_weighted_{h}yr')
                        rmse_u = group_summary.get(f'rmse_unweighted_{h}yr')
                        mae_w = group_summary.get(f'mae_weighted_{h}yr')
                        if rmse_w is not None:
                            print(f"  {event} @{h}yr: RMSE_w={rmse_w:.4f}  RMSE_u={rmse_u:.4f}  MAE_w={mae_w:.4f}")
                            mlflow.log_metric(f"{event}_group_rmse_weighted_{h}yr", rmse_w)
                            mlflow.log_metric(f"{event}_group_rmse_unweighted_{h}yr", rmse_u)
                            mlflow.log_metric(f"{event}_group_mae_weighted_{h}yr", mae_w)

        log_stage_complete("Model Evaluation")
        log_memory_usage()

        # ================================================================
        # STAGE 9: Save artifacts
        # ================================================================
        log_stage_start("Saving Artifacts")

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"survival_{model_type}_{timestamp}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save models
        for event, model in models.items():
            model_file = os.path.join(checkpoint_path, f"model_{event}.json")
            model.save_model(model_file)
            print(f"  Saved model: {model_file}")

        # Save predictions
        pred_path = os.path.join(checkpoint_path, "predictions.parquet")
        pred_df.to_parquet(pred_path)
        print(f"  Saved predictions: {pred_path}")

        # Save feature importance
        for event, model in models.items():
            for imp_type in ['weight', 'gain', 'cover']:
                try:
                    importance = model.get_score(importance_type=imp_type)
                    if importance:
                        imp_df = pd.DataFrame([
                            {'feature': k, 'importance': v}
                            for k, v in importance.items()
                        ]).sort_values('importance', ascending=False)
                        imp_df.to_csv(
                            os.path.join(
                                checkpoint_path,
                                f"feature_importance_{event}_{imp_type}.csv"
                            ),
                            index=False,
                        )
                except Exception:
                    pass

        # Save group evaluation
        for event, group_df in group_eval_dfs.items():
            if not group_df.empty:
                group_df.to_csv(
                    os.path.join(checkpoint_path, f"group_eval_{event}.csv"),
                    index=False,
                )

        # Save calibration tables
        for event in events:
            cal = all_metrics[event].get('calibration', {})
            for h, cal_df in cal.items():
                cal_df.to_csv(
                    os.path.join(
                        checkpoint_path,
                        f"calibration_{event}_{h}yr.csv"
                    ),
                    index=False,
                )

        # Save metadata
        # Strip non-serializable items from metrics
        serializable_metrics = {}
        for event, m in all_metrics.items():
            serializable_metrics[event] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in m.items()
                if k != 'calibration'
            }

        metadata = {
            'timestamp': timestamp,
            'model_type': model_type,
            'events': events,
            'horizons': horizons,
            'distribution': distribution,
            'sigma': sigma,
            'n_features': len(feature_cols),
            'feature_cols': feature_cols,
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'metrics': serializable_metrics,
            'group_eval_summary': group_eval_summaries,
            'config_path': config_path,
            'feature_config_path': feature_config_path,
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Log model artifacts to MLflow
        mlflow.log_artifacts(checkpoint_path, artifact_path="survival_checkpoint")

        log_stage_complete("Saving Artifacts")

    # ================================================================
    # Final summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SURVIVAL ANALYSIS COMPLETE")
    print("=" * 60)
    for event in events:
        m = all_metrics[event]
        c_idx = m['c_index']
        aucs = [m.get(f'auc_{h}yr', np.nan) for h in horizons]
        auc_str = ", ".join(
            f"{h}yr={a:.3f}" if not np.isnan(a) else f"{h}yr=N/A"
            for h, a in zip(horizons, aucs)
        )
        print(f"  {event}: C-index={c_idx:.4f} | AUC: {auc_str}")
    print(f"\n  Checkpoint: {checkpoint_path}")
    print("=" * 60)

    return models, all_metrics, pred_df


def _run_incremental_pipeline(
    output_path, total_rows, events, horizons,
    config, model_type, distribution, sigma,
    feature_config_path, target_batch_rows,
):
    """
    Incremental training pipeline for large datasets.

    Adds survival labels year-by-year, then trains incrementally.
    """
    print("\nINCREMENTAL SURVIVAL PIPELINE")
    print(f"  Total rows: {total_rows:,}")

    # Step 1: Add survival labels to the parquet data
    # We need to process the full dataset to compute durations correctly
    # (need to know future event times for each person)
    log_stage_start("Adding Survival Labels")

    survival_parquet_path = "data/processed_features_survival"
    survival_success = os.path.join(survival_parquet_path, "_SUCCESS")

    if os.path.exists(survival_success):
        print(f"  Survival labels already exist at {survival_parquet_path}")
    else:
        create_survival_labels_chunked(
            parquet_path=output_path,
            output_path=survival_parquet_path,
            event_cols=events,
            id_col='sid',
            time_col='year',
        )

    log_stage_complete("Adding Survival Labels")
    log_memory_usage()

    # Step 2: Determine feature columns
    sample_df = pd.read_parquet(survival_parquet_path,
                                filters=[('year', '==', 2015)])
    if len(sample_df) == 0:
        # Try reading a small sample without filter
        sample_df = pd.read_parquet(survival_parquet_path).head(1000)

    drop_cols = ['sid', 'year', 'refnis', 'y_moved']
    survival_label_cols = []
    for event in events:
        survival_label_cols.extend([
            f"{event}_duration", f"{event}_event_observed"
        ])
    leak_cols = get_leaky_columns(sample_df.columns)

    feature_cols = [
        c for c in sample_df.columns
        if c not in drop_cols
        and c not in leak_cols
        and c not in survival_label_cols
        and c not in events
    ]

    if feature_config_path is not None:
        feature_cols = filter_features_by_config(
            feature_cols, feature_config_path=feature_config_path, verbose=True
        )

    print(f"  Using {len(feature_cols)} features")
    del sample_df

    # Step 3: Fix dtypes on a reference batch (for categorical encoding)
    ref_batch = pd.read_parquet(survival_parquet_path,
                                filters=[('year', '==', 2015)])
    if len(ref_batch) == 0:
        ref_batch = pd.read_parquet(survival_parquet_path).head(10000)
    ref_batch = fix_dtypes(ref_batch, feature_cols)
    del ref_batch
    gc.collect()

    # Step 4: Train incrementally per event
    log_stage_start("Incremental Training")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("demographic_forecasts_survival")

    with mlflow.start_run(run_name=f"survival_{model_type}_incremental"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("events", ",".join(events))
        mlflow.log_param("training_mode", "incremental")
        mlflow.log_param("n_features", len(feature_cols))

        models = {}
        for event in events:
            print(f"\nTraining {model_type.upper()} for {event} (incremental)...")
            model = train_incremental_survival(
                survival_parquet_path, total_rows,
                feature_cols, event,
                config=config, model_type=model_type,
                target_batch_rows=target_batch_rows,
            )
            models[event] = model
            print(f"  {event}: {model.num_boosted_rounds()} trees")

        log_stage_complete("Incremental Training")
        log_memory_usage()

        # Step 5: Streaming evaluation on test set
        log_stage_start("Evaluation (streaming)")

        import xgboost as xgb

        # Determine group columns from a small sample
        sample_batch = pd.read_parquet(
            survival_parquet_path,
            filters=[('year', '==', 2023)],
        ).head(10)
        sex_col = "gender" if "gender" in sample_batch.columns else None
        age_col = "age_group" if "age_group" in sample_batch.columns else None
        muni_col = "refnis" if "refnis" in sample_batch.columns else None
        group_cols = [c for c in [sex_col, age_col, muni_col] if c is not None]
        del sample_batch

        # Cox baseline hazards (need training data, loaded in chunks)
        baseline_hazards = None
        if model_type == 'cox':
            print("  Estimating baseline hazards from training data...")
            baseline_hazards = {}
            for event in events:
                dur_chunks, evt_chunks = [], []
                for fragment in pq.ParquetDataset(survival_parquet_path).fragments:
                    for batch in fragment.to_batches(batch_size=500_000):
                        chunk = batch.to_pandas()
                        chunk = chunk[chunk['year'] < 2023]
                        if len(chunk) > 0:
                            dur_chunks.append(chunk[f"{event}_duration"].values)
                            evt_chunks.append(chunk[f"{event}_event_observed"].values)
                        del chunk
                all_dur = np.concatenate(dur_chunks)
                all_evt = np.concatenate(evt_chunks)
                # Subsample if very large
                if len(all_dur) > 500_000:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(all_dur), 500_000, replace=False)
                    all_dur, all_evt = all_dur[idx], all_evt[idx]
                baseline_hazards[event] = estimate_baseline_hazard(all_dur, all_evt)
                del dur_chunks, evt_chunks, all_dur, all_evt
            gc.collect()

        # Initialize streaming evaluators per event
        stream_evaluators = {}
        if group_cols:
            for event in events:
                stream_evaluators[event] = StreamingGroupEvaluator(group_cols, horizons)

        rng = np.random.RandomState(42)
        max_cindex_samples = 200_000

        def _update_cindex_sample(acc, risk, dur, evt):
            risk = np.asarray(risk, dtype=np.float64)
            dur = np.asarray(dur, dtype=np.float64)
            evt = np.asarray(evt, dtype=np.int32)

            if acc['risk'].size == 0:
                if risk.size > max_cindex_samples:
                    idx = rng.choice(risk.size, size=max_cindex_samples, replace=False)
                    risk = risk[idx]
                    dur = dur[idx]
                    evt = evt[idx]
                acc['risk'] = risk
                acc['dur'] = dur
                acc['evt'] = evt
                return

            combined_risk = np.concatenate([acc['risk'], risk])
            combined_dur = np.concatenate([acc['dur'], dur])
            combined_evt = np.concatenate([acc['evt'], evt])

            if combined_risk.size > max_cindex_samples:
                idx = rng.choice(combined_risk.size, size=max_cindex_samples, replace=False)
                combined_risk = combined_risk[idx]
                combined_dur = combined_dur[idx]
                combined_evt = combined_evt[idx]

            acc['risk'] = combined_risk
            acc['dur'] = combined_dur
            acc['evt'] = combined_evt

        # Accumulators for C-index (bounded sample to avoid OOM)
        c_index_accum = {
            event: {
                'risk': np.array([], dtype=np.float64),
                'dur': np.array([], dtype=np.float64),
                'evt': np.array([], dtype=np.int32),
            }
            for event in events
        }

        # Stream test data in batches
        eval_batch_size = 200_000
        test_years = [2023, 2024, 2025]
        total_test_rows = 0

        print(f"\n  Streaming evaluation over test years {test_years}...")
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"survival_{model_type}_incr_{timestamp}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save models first
        for event, model in models.items():
            model.save_model(os.path.join(checkpoint_path, f"model_{event}.json"))

        for year in test_years:
            try:
                year_df = pd.read_parquet(
                    survival_parquet_path,
                    filters=[('year', '==', year)],
                )
            except Exception:
                continue
            if len(year_df) == 0:
                continue

            # Process in sub-batches
            n_batches = max(1, (len(year_df) + eval_batch_size - 1) // eval_batch_size)

            for batch_i in range(n_batches):
                start = batch_i * eval_batch_size
                end = min(start + eval_batch_size, len(year_df))
                batch_df = year_df.iloc[start:end].copy().reset_index(drop=True)

                batch_df = fix_dtypes(batch_df, feature_cols)
                # Impute NULLs
                for col in feature_cols:
                    if batch_df[col].isnull().any():
                        if batch_df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                            batch_df[col] = batch_df[col].fillna(batch_df[col].median())
                        elif batch_df[col].dtype == 'bool':
                            batch_df[col] = batch_df[col].fillna(False)

                X_batch = batch_df[feature_cols]
                dmatrix = xgb.DMatrix(X_batch)

                # Predict + evaluate per event (one at a time to save memory)
                for event in events:
                    duration_col = f"{event}_duration"
                    observed_col = f"{event}_event_observed"
                    model = models[event]

                    raw_preds = model.predict(dmatrix)

                    # Compute risk + horizon probabilities
                    if model_type == 'aft':
                        risk_score = -raw_preds
                        batch_proba = {}
                        for h in horizons:
                            log_h = np.log(h) if h > 0 else -np.inf
                            z = (log_h - raw_preds) / sigma
                            if distribution == 'logistic':
                                from scipy.stats import logistic as _logistic
                                prob = _logistic.cdf(z)
                            elif distribution in ('extreme', 'extreme_value'):
                                prob = 1.0 - np.exp(-np.exp(z))
                            else:  # normal (default)
                                from scipy.stats import norm as _norm
                                prob = _norm.cdf(z)
                            batch_proba[h] = np.clip(prob, 0.0, 1.0)
                    else:
                        risk_score = raw_preds
                        bh = baseline_hazards[event] if baseline_hazards else None
                        batch_proba = {}
                        for h in horizons:
                            if bh is not None:
                                nearest_idx = (bh.index - h).abs().argmin()
                                H0_t = bh.iloc[nearest_idx]
                                survival = np.exp(-H0_t * np.exp(raw_preds))
                                batch_proba[h] = np.clip(1.0 - survival, 0.0, 1.0)

                    # Accumulate for C-index (bounded sampling)
                    _update_cindex_sample(
                        c_index_accum[event],
                        risk_score,
                        batch_df[duration_col].values,
                        batch_df[observed_col].values,
                    )

                    # Feed streaming group evaluator
                    if event in stream_evaluators:
                        stream_evaluators[event].add_batch(
                            batch_df, batch_proba, duration_col, observed_col,
                        )

                    del raw_preds, risk_score, batch_proba

                total_test_rows += len(batch_df)
                del batch_df, X_batch, dmatrix
                gc.collect()

            del year_df
            gc.collect()
            print(f"    Year {year} done, cumulative test rows: {total_test_rows:,}")

        # Compute final metrics
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        all_metrics = {}
        for event in events:
            print(f"\n--- {event} ---")
            risk_all = c_index_accum[event]['risk']
            dur_all = c_index_accum[event]['dur']
            evt_all = c_index_accum[event]['evt']

            c_idx = concordance_index(dur_all, evt_all, risk_all)
            print(f"  C-index: {c_idx:.4f}")
            all_metrics[event] = {'c_index': c_idx}

            mlflow.log_metric(f"{event}_c_index", c_idx)

            del risk_all, dur_all, evt_all
            # Free accumulators for this event
            c_index_accum[event] = None
        del c_index_accum
        gc.collect()

        # Finalize group evaluations
        group_eval_dfs = {}
        group_eval_summaries = {}
        if group_cols:
            print(f"\nGroup-level RMSE evaluation ({total_test_rows:,} test rows)...")
            for event in events:
                group_df, group_summary = stream_evaluators[event].finalize(min_group_size=100)
                group_eval_dfs[event] = group_df
                group_eval_summaries[event] = group_summary

                if group_summary:
                    for h in horizons:
                        rmse_w = group_summary.get(f'rmse_weighted_{h}yr')
                        rmse_u = group_summary.get(f'rmse_unweighted_{h}yr')
                        mae_w = group_summary.get(f'mae_weighted_{h}yr')
                        if rmse_w is not None:
                            print(f"  {event} @{h}yr: RMSE_w={rmse_w:.4f}  RMSE_u={rmse_u:.4f}  MAE_w={mae_w:.4f}")
                            mlflow.log_metric(f"{event}_group_rmse_weighted_{h}yr", rmse_w)
                            mlflow.log_metric(f"{event}_group_rmse_unweighted_{h}yr", rmse_u)
                            mlflow.log_metric(f"{event}_group_mae_weighted_{h}yr", mae_w)

        del stream_evaluators
        gc.collect()

        log_stage_complete("Evaluation (streaming)")

        # Save artifacts
        # Save group evaluation CSVs
        for event, gdf in group_eval_dfs.items():
            if not gdf.empty:
                gdf.to_csv(
                    os.path.join(checkpoint_path, f"group_eval_{event}.csv"),
                    index=False,
                )

        serializable_metrics = {}
        for event, m in all_metrics.items():
            serializable_metrics[event] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in m.items()
                if k != 'calibration'
            }

        metadata = {
            'timestamp': timestamp,
            'model_type': model_type,
            'events': events,
            'horizons': horizons,
            'training_mode': 'incremental',
            'total_test_rows': total_test_rows,
            'metrics': serializable_metrics,
            'group_eval_summary': group_eval_summaries,
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        mlflow.log_artifacts(checkpoint_path, artifact_path="survival_checkpoint")

        print(f"\n  Checkpoint: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("SURVIVAL ANALYSIS COMPLETE (INCREMENTAL)")
    print("=" * 60)

    return models, all_metrics


def _run_hyperparameter_tuning(
    train_df, test_df, feature_cols, events,
    config, model_type, n_trials, horizons,
):
    """
    Hyperparameter tuning for survival models using Optuna.
    """
    import optuna
    from optuna.samplers import TPESampler
    from xgboost import XGBClassifier

    print("=" * 60)
    print("HYPERPARAMETER TUNING (Survival)")
    print("=" * 60)
    print(f"  Trials: {n_trials}")
    print(f"  Metric: C-index (discrimination)")

    # Use first event for tuning (typically y_moved)
    tune_event = events[0]
    duration_col = f"{tune_event}_duration"
    observed_col = f"{tune_event}_event_observed"

    print(f"  Tuning on: {tune_event}")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    duration_train = train_df[duration_col].values
    event_train = train_df[observed_col].values
    duration_test = test_df[duration_col].values
    event_test = test_df[observed_col].values

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("demographic_forecasts_survival_tuning")

    def objective(trial):
        try:
            if model_type == 'aft':
                params = {
                    'objective': 'survival:aft',
                    'eval_metric': 'aft-nloglik',
                    'aft_loss_distribution': trial.suggest_categorical(
                        'aft_loss_distribution', ['normal', 'logistic', 'extreme']
                    ),
                    'aft_loss_distribution_scale': trial.suggest_float(
                        'aft_loss_distribution_scale', 0.5, 2.0
                    ),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                    'tree_method': 'hist',
                    'seed': 42,
                }
            else:
                params = {
                    'objective': 'survival:cox',
                    'eval_metric': 'cox-nloglik',
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                    'tree_method': 'hist',
                    'seed': 42,
                }

            model = train_survival_model(
                X_train, duration_train, event_train,
                X_test, duration_test, event_test,
                config={'model': {'type': model_type, 'params': {**params, 'n_estimators': 300, 'early_stopping_rounds': 20, 'verbose_eval': 0}}},
                model_type=model_type,
            )

            # Compute C-index
            dtest = xgb.DMatrix(X_test)
            raw_preds = model.predict(dtest)

            if model_type == 'aft':
                risk = -raw_preds  # Lower predicted time = higher risk
            else:
                risk = raw_preds  # Higher log HR = higher risk

            c_idx = concordance_index(duration_test, event_test, risk)
            trial.set_user_attr('c_index', c_idx)

            del model, dtest, raw_preds
            gc.collect()

            return c_idx

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            gc.collect()
            return 0.5

    from src.survival.evaluation import concordance_index

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best = study.best_trial
    print(f"\nBest C-index: {best.value:.4f}")
    print(f"Best parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Save results
    results_path = "tuning_results_survival.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best.params,
            'best_c_index': best.value,
            'n_trials': n_trials,
            'event': tune_event,
            'model_type': model_type,
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run survival analysis pipeline with municipality features"
    )
    parser.add_argument('--reuse', '-r', action='store_true',
                        help='Reuse existing processed features')
    parser.add_argument('--max-rows', type=int,
                        help='Limit dataset to N rows')
    parser.add_argument('--config', type=str,
                        help='Path to survival model config YAML')
    parser.add_argument('--features', type=str,
                        help='Path to feature selection config YAML')
    parser.add_argument('--events', type=str,
                        help='Comma-separated event names (default: all)')
    parser.add_argument('--horizons', type=str, default='1,3,5',
                        help='Comma-separated prediction horizons in years')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of tuning trials')
    parser.add_argument('--target-batch-rows', type=int,
                        help='Target rows per incremental batch')

    args = parser.parse_args()

    # Parse events
    events = None
    if args.events:
        events = [e.strip() for e in args.events.split(',')]

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(',')]

    print("=" * 60)
    print("SURVIVAL ANALYSIS PIPELINE")
    print("=" * 60)
    if args.reuse:
        print("  Reuse: Will use existing processed features")
    if args.max_rows:
        print(f"  Max rows: {args.max_rows:,}")
    if args.config:
        print(f"  Config: {args.config}")
    if args.features:
        print(f"  Features: {args.features}")
    if events:
        print(f"  Events: {', '.join(events)}")
    print(f"  Horizons: {horizons} years")
    if args.tune:
        print(f"  Tuning: {args.n_trials} trials")
    print("=" * 60)
    print()

    main_survival(
        reuse_processed=args.reuse,
        max_rows=args.max_rows,
        config_path=args.config,
        feature_config_path=args.features,
        events=events,
        horizons=horizons,
        tune=args.tune,
        n_trials=args.n_trials,
        target_batch_rows=args.target_batch_rows,
    )
