"""
Sequence model pipeline for demographic event prediction.

Alternative to run_with_municipality_survival.py that uses an LLM-style
approach: converts person histories into token sequences and trains an
encoder (LSTM/GRU/Transformer) to predict event probabilities at horizons
of 1, 3, and 5 years.

Reuses the same processed features parquet as the survival pipeline.

Usage:
    # LSTM model (default)
    python run_with_municipality_sequence.py --reuse --config configs/models/pytorch_seq_lstm.yaml

    # GRU model
    python run_with_municipality_sequence.py --reuse --config configs/models/pytorch_seq_gru.yaml

    # Transformer model
    python run_with_municipality_sequence.py --reuse --config configs/models/pytorch_seq_transformer.yaml

    # With options
    python run_with_municipality_sequence.py --reuse --config configs/models/pytorch_seq_lstm.yaml \\
        --max-rows 100000 --events y_moved,divorce_event
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

from src.utils.logging_setup import (
    setup_logging, log_stage_start, log_stage_complete, log_memory_usage
)
from run_test import load_model_config

from src.sequence.vocabulary import LifeEventVocabulary, MUNICIPALITY_FEATURE_MAP
from src.sequence.dataset import (
    SequenceDataset, StreamingSequenceDataset, CachedSequenceDataset,
    CachedRollingWindowDataset,
)
from src.sequence.estimator import PyTorchSequenceEstimator
from src.sequence.evaluation import evaluate_sequence_predictions

import mlflow

DEFAULT_EVENTS = [
    'y_moved',
    'birth1_event',
    'birth2_event',
    'divorce_event',
    'getalifeother_event',
]

DEFAULT_HORIZONS = [1, 3, 5]

def _scan_unique_values(parquet_path: str, columns: List[str], chunk_size: int = 500_000) -> Dict[str, set]:
    dataset = pq.ParquetDataset(parquet_path)
    values = {col: set() for col in columns}

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=columns):
            df = batch.to_pandas()
            for col in columns:
                if col in df.columns:
                    col_values = df[col].dropna().unique()
                    values[col].update(col_values.tolist())
            del df
    return values


def _build_vocab_from_parquet(parquet_path: str, vocab_path: str) -> LifeEventVocabulary:
    # Categorical columns: scan unique values
    cat_columns = [
        'age_group',
        'age',
        'gender',
        'coupled',
        'eerste_nationaliteit',
        'hh_pos',
        'income_quintile',
    ]
    values = _scan_unique_values(parquet_path, columns=cat_columns)
    max_len = max((len(v) for v in values.values()), default=0)

    data = {}
    for col, vals in values.items():
        if not vals:
            continue
        sorted_vals = sorted(vals)
        if len(sorted_vals) < max_len:
            sorted_vals.extend([None] * (max_len - len(sorted_vals)))
        data[col] = sorted_vals

    df_cat = pd.DataFrame(data)

    # Municipality features: collect all values for quantile computation
    muni_columns = list(MUNICIPALITY_FEATURE_MAP.keys())
    muni_values = _scan_muni_values(parquet_path, muni_columns)

    # Build a DataFrame with municipality values for quantile bin computation
    if muni_values:
        max_muni_len = max(len(v) for v in muni_values.values())
        muni_data = {}
        for col, vals in muni_values.items():
            arr = list(vals)
            if len(arr) < max_muni_len:
                arr.extend([None] * (max_muni_len - len(arr)))
            muni_data[col] = arr
        df_muni = pd.DataFrame(muni_data)
        # Combine with categorical columns
        # Pad the shorter one to match
        if len(df_cat) < len(df_muni):
            df_cat = df_cat.reindex(range(len(df_muni)))
        elif len(df_muni) < len(df_cat):
            df_muni = df_muni.reindex(range(len(df_cat)))
        df = pd.concat([df_cat, df_muni], axis=1)
    else:
        df = df_cat

    vocabulary = LifeEventVocabulary()
    vocabulary.build_from_dataframe(df)
    vocabulary.save(vocab_path)
    return vocabulary


def _scan_muni_values(
    parquet_path: str,
    columns: List[str],
    chunk_size: int = 500_000,
    sample_fraction: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Collect municipality feature values from parquet for quantile computation.

    Samples a fraction of rows to avoid loading all data into memory.
    Returns arrays of sampled values per column.
    """
    dataset = pq.ParquetDataset(parquet_path)
    # Check which columns actually exist
    schema_cols = set(dataset.schema.names)
    available = [c for c in columns if c in schema_cols]
    if not available:
        return {}

    collected = {col: [] for col in available}
    rng = np.random.RandomState(42)

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=available):
            df = batch.to_pandas()
            # Sample to keep memory manageable
            if len(df) > 1000:
                df = df.sample(frac=sample_fraction, random_state=rng)
            for col in available:
                vals = df[col].dropna().values
                if len(vals) > 0:
                    collected[col].append(vals)
            del df

    result = {}
    for col, arrays in collected.items():
        if arrays:
            result[col] = np.concatenate(arrays)
    return result


def _collect_valid_persons(
    parquet_path: str,
    cutoff_year: int,
    chunk_size: int = 500_000,
) -> set:
    dataset = pq.ParquetDataset(parquet_path)
    history = set()
    future = set()

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=['sid', 'year']):
            df = batch.to_pandas()
            history.update(df.loc[df['year'] <= cutoff_year, 'sid'].unique().tolist())
            future.update(df.loc[df['year'] > cutoff_year, 'sid'].unique().tolist())
            del df

    return history & future


def _scan_year_range(parquet_path: str, chunk_size: int = 500_000):
    """Scan parquet to find min and max year in the dataset."""
    dataset = pq.ParquetDataset(parquet_path)
    min_year = float('inf')
    max_year = float('-inf')
    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=['year']):
            years = batch.to_pandas()['year']
            min_year = min(min_year, int(years.min()))
            max_year = max(max_year, int(years.max()))
    return min_year, max_year


def _collect_valid_persons_windowed(
    parquet_path: str,
    cutoff_year: int,
    min_history_year: int,
    max_horizon: int,
    chunk_size: int = 500_000,
) -> set:
    """Find persons with history in (min_history_year, cutoff] AND future in (cutoff, cutoff+max_horizon]."""
    dataset = pq.ParquetDataset(parquet_path)
    history = set()
    future = set()

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=['sid', 'year']):
            df = batch.to_pandas()
            hist_mask = (df['year'] > min_history_year) & (df['year'] <= cutoff_year)
            future_mask = (df['year'] > cutoff_year) & (df['year'] <= cutoff_year + max_horizon)
            if hist_mask.any():
                history.update(df.loc[hist_mask, 'sid'].unique().tolist())
            if future_mask.any():
                future.update(df.loc[future_mask, 'sid'].unique().tolist())
            del df

    return history & future


def _build_predictions_from_probs(
    probs: np.ndarray,
    events: List[str],
    horizons: List[int],
) -> Dict[str, Dict[str, np.ndarray]]:
    n_horizons = len(horizons)
    result = {}
    for ei, event in enumerate(events):
        event_probs = {}
        for hi, horizon in enumerate(horizons):
            col_idx = ei * n_horizons + hi
            event_probs[f'prob_{horizon}yr'] = probs[:, col_idx]

        longest_horizon_prob = probs[:, ei * n_horizons + (n_horizons - 1)]
        event_probs['risk_score'] = longest_horizon_prob
        result[event] = event_probs
    return result


def _compute_labels_streaming(
    parquet_path: str,
    sids: np.ndarray,
    events: List[str],
    horizons: List[int],
    cutoff_year: int,
    chunk_size: int = 500_000,
) -> np.ndarray:
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    n_horizons = len(horizons)
    y_true = np.zeros((len(sids), len(events) * n_horizons), dtype=np.float32)
    max_year = cutoff_year + max(horizons)

    dataset = pq.ParquetDataset(parquet_path)
    for fragment in dataset.fragments:
        for batch in fragment.to_batches(
            batch_size=chunk_size,
            columns=['sid', 'year'] + events,
        ):
            df = batch.to_pandas()
            df = df[df['year'] <= max_year]
            if df.empty:
                del df
                continue

            idx = df['sid'].map(sid_to_idx)
            df = df[idx.notna()].copy()
            if df.empty:
                del df
                continue

            idx = idx[idx.notna()].astype(int).values
            years = df['year'].values

            for ei, event in enumerate(events):
                if event not in df.columns:
                    continue
                event_mask = df[event].fillna(0).astype(int).values == 1
                if not event_mask.any():
                    continue
                event_idx = idx[event_mask]
                event_years = years[event_mask]
                for hi, horizon in enumerate(horizons):
                    within = event_years <= (cutoff_year + horizon)
                    if within.any():
                        y_true[event_idx[within], ei * n_horizons + hi] = 1.0

            del df

    return y_true


def _evaluate_from_arrays(
    all_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    events: List[str],
    horizons: List[int],
) -> Dict[str, Any]:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        brier_score_loss,
    )

    results = {}
    n_horizons = len(horizons)

    for ei, event in enumerate(events):
        if event not in all_predictions:
            continue

        event_results = {}
        for hi, horizon in enumerate(horizons):
            prob_key = f'prob_{horizon}yr'
            if prob_key not in all_predictions[event]:
                continue

            y_prob = all_predictions[event][prob_key]
            y_event = y_true[:, ei * n_horizons + hi]

            metrics = {}
            n_pos = int(y_event.sum())
            n_neg = len(y_event) - n_pos

            if n_pos > 0 and n_neg > 0:
                metrics['auc'] = float(roc_auc_score(y_event, y_prob))
                metrics['ap'] = float(average_precision_score(y_event, y_prob))

                best_f1, best_thresh = 0.0, 0.5
                for thresh in np.linspace(0.01, 0.99, 100):
                    y_pred = (y_prob >= thresh).astype(int)
                    f1 = f1_score(y_event, y_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh

                metrics['f1'] = float(best_f1)
                metrics['threshold'] = float(best_thresh)
                metrics['brier'] = float(brier_score_loss(y_event, y_prob))
            else:
                metrics['auc'] = float('nan')
                metrics['ap'] = float('nan')
                metrics['f1'] = float('nan')
                metrics['brier'] = float('nan')

            metrics['n_pos'] = n_pos
            metrics['n_total'] = len(y_event)
            metrics['prevalence'] = n_pos / len(y_event) if len(y_event) > 0 else 0

            event_results[f'{horizon}yr'] = metrics

        results[event] = event_results

    all_aucs = []
    all_aps = []
    for event_metrics in results.values():
        for h_metrics in event_metrics.values():
            if not np.isnan(h_metrics.get('auc', float('nan'))):
                all_aucs.append(h_metrics['auc'])
            if not np.isnan(h_metrics.get('ap', float('nan'))):
                all_aps.append(h_metrics['ap'])

    aggregate = {
        'mean_auc': float(np.mean(all_aucs)) if all_aucs else float('nan'),
        'mean_ap': float(np.mean(all_aps)) if all_aps else float('nan'),
    }

    return {
        'per_event': results,
        'aggregate': aggregate,
    }


def _load_group_features_streaming(
    parquet_path: str,
    sids: np.ndarray,
    cutoff_year: int,
    group_cols: List[str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Load demographic features for evaluated persons from parquet.

    Takes the most recent observation at or before cutoff_year for each person.
    Returns a DataFrame with sid + group columns, one row per person,
    ordered to match the sids array.
    """
    sid_set = set(sids.tolist()) if hasattr(sids, 'tolist') else set(sids)
    columns = ['sid', 'year'] + group_cols

    dataset = pq.ParquetDataset(parquet_path)
    schema_cols = set(dataset.schema.names)
    columns = [c for c in columns if c in schema_cols]
    available_group_cols = [c for c in group_cols if c in schema_cols]

    # Track best (most recent) year per person — accumulate deduplicated
    best_df = None

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=columns):
            df = batch.to_pandas()
            df = df[(df['year'] <= cutoff_year) & (df['sid'].isin(sid_set))]
            if df.empty:
                del df
                continue

            if best_df is None:
                best_df = df
            else:
                best_df = pd.concat([best_df, df], ignore_index=True)
            del df

            # Periodically deduplicate to keep memory bounded
            if best_df is not None and len(best_df) > len(sid_set) * 2:
                idx = best_df.groupby('sid')['year'].idxmax()
                best_df = best_df.loc[idx].reset_index(drop=True)

    if best_df is None or best_df.empty:
        result = pd.DataFrame({'sid': sids})
        for c in available_group_cols:
            result[c] = None
        return result

    # Final dedup: keep most recent year per person
    idx = best_df.groupby('sid')['year'].idxmax()
    best_df = best_df.loc[idx, ['sid'] + available_group_cols].reset_index(drop=True)

    # Align to sids array order
    sid_order = pd.DataFrame({'sid': sids, '_order': range(len(sids))})
    result = sid_order.merge(best_df, on='sid', how='left').sort_values('_order').drop(columns='_order').reset_index(drop=True)
    del best_df, sid_order
    return result


def _evaluate_groups(
    group_df: pd.DataFrame,
    all_predictions: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    events: List[str],
    horizons: List[int],
    group_cols: List[str],
    min_group_size: int = 100,
    group_thresholds: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Compute group-level observed vs predicted rates per event-horizon.

    Returns dict with per-event group DataFrames and RMSE/MAE summaries.
    If *group_thresholds* is provided (from ``calibrate_group_thresholds``),
    also computes thresholded group rates for comparison.
    """
    n_horizons = len(horizons)
    results = {}

    for ei, event in enumerate(events):
        if event not in all_predictions:
            continue

        rows = []
        for h in horizons:
            prob_key = f'prob_{h}yr'
            if prob_key not in all_predictions[event]:
                continue
            col_idx = ei * n_horizons + horizons.index(h)
            group_df[f'pred_{h}yr'] = all_predictions[event][prob_key]
            group_df[f'true_{h}yr'] = y_true[:, col_idx]

        grouped = group_df.groupby(group_cols, dropna=False)
        group_rows = []
        for group_key, group_data in grouped:
            if len(group_data) < min_group_size:
                continue
            row = {}
            if len(group_cols) == 1:
                row[group_cols[0]] = group_key
                gval = group_key
            else:
                for i, col in enumerate(group_cols):
                    row[col] = group_key[i]
                gval = group_key
            row['count'] = len(group_data)

            for hi, h in enumerate(horizons):
                pred_col = f'pred_{h}yr'
                true_col = f'true_{h}yr'
                if pred_col in group_data.columns:
                    obs_rate = float(group_data[true_col].mean())
                    pred_rate = float(group_data[pred_col].mean())
                    row[f'observed_rate_{h}yr'] = obs_rate
                    row[f'predicted_rate_{h}yr'] = pred_rate
                    row[f'abs_error_{h}yr'] = abs(pred_rate - obs_rate)
                    row[f'sq_error_{h}yr'] = (pred_rate - obs_rate) ** 2

                    # Per-group thresholded rate
                    if group_thresholds is not None:
                        col_idx = ei * n_horizons + hi
                        g_thresh = group_thresholds.get(gval, {})
                        thr = g_thresh.get(col_idx, 0.5)
                        thr_rate = float((group_data[pred_col] >= thr).mean())
                        row[f'predicted_rate_thr_{h}yr'] = thr_rate
                        row[f'abs_error_thr_{h}yr'] = abs(thr_rate - obs_rate)
                        row[f'sq_error_thr_{h}yr'] = (thr_rate - obs_rate) ** 2

            group_rows.append(row)

        if not group_rows:
            results[event] = {'group_df': pd.DataFrame(), 'summary': {}}
            continue

        event_group_df = pd.DataFrame(group_rows)

        # Compute summary RMSE/MAE
        summary = {'groups': len(event_group_df), 'min_group_size': min_group_size}
        counts = event_group_df['count'].values
        total = counts.sum()

        for h in horizons:
            ae_col = f'abs_error_{h}yr'
            se_col = f'sq_error_{h}yr'
            if ae_col not in event_group_df.columns:
                continue
            ae = event_group_df[ae_col].values
            se = event_group_df[se_col].values
            weights = counts / total

            summary[f'mae_weighted_{h}yr'] = float(np.average(ae, weights=weights))
            summary[f'mae_unweighted_{h}yr'] = float(ae.mean())
            summary[f'rmse_weighted_{h}yr'] = float(np.sqrt(np.average(se, weights=weights)))
            summary[f'rmse_unweighted_{h}yr'] = float(np.sqrt(se.mean()))

            # Thresholded summary
            ae_thr_col = f'abs_error_thr_{h}yr'
            se_thr_col = f'sq_error_thr_{h}yr'
            if ae_thr_col in event_group_df.columns:
                ae_t = event_group_df[ae_thr_col].values
                se_t = event_group_df[se_thr_col].values
                summary[f'mae_weighted_thr_{h}yr'] = float(np.average(ae_t, weights=weights))
                summary[f'rmse_weighted_thr_{h}yr'] = float(np.sqrt(np.average(se_t, weights=weights)))

        results[event] = {'group_df': event_group_df, 'summary': summary}

        # Clean up temp columns
        for h in horizons:
            group_df.drop(columns=[f'pred_{h}yr', f'true_{h}yr'], errors='ignore', inplace=True)

    return results


def _run_hyperparameter_tuning(
    output_path: str,
    total_rows: int,
    events: List[str],
    horizons: List[int],
    config: Optional[Dict[str, Any]],
    encoder_type: str,
    cutoff_year: int,
    reuse_processed: bool,
    config_path: Optional[str],
    sample_fraction: Optional[float],
    n_trials: int = 50,
    tuning_epochs: int = 15,
    search_space: str = "conservative",
    tuning_seed: int = 42,
    tuning_workers: int = 1,
    tuning_fraction: float = 1.0,
):
    """Hyperparameter tuning for sequence models using Optuna."""
    from src.tuning.sequence_tuning import SequenceModelTuner

    print("=" * 60)
    print("HYPERPARAMETER TUNING (Sequence)")
    print("=" * 60)
    print(f"  Encoder: {encoder_type}")
    print(f"  Trials: {n_trials}")
    print(f"  Epochs per trial: {tuning_epochs}")
    print(f"  Search space: {search_space}")
    print(f"  Metric: composite (0.4*AP + 0.3*F1 + 0.3*AUC, maximize)")

    # ---- Build vocabulary (reused across all trials) ----
    log_stage_start("Building Vocabulary")

    vocab_path = "checkpoints/sequence_vocab.joblib"
    os.makedirs("checkpoints", exist_ok=True)

    if reuse_processed and os.path.exists(vocab_path):
        print(f"  Loading existing vocabulary from {vocab_path}...")
        vocabulary = LifeEventVocabulary.load(vocab_path)
        missing_tokens = [
            f"{prefix}_Q1"
            for prefix in MUNICIPALITY_FEATURE_MAP.values()
            if vocabulary.token_to_id(f"{prefix}_Q1") == vocabulary.UNK
        ]
        if missing_tokens:
            print("  Vocabulary missing municipality tokens; rebuilding...")
            vocabulary = _build_vocab_from_parquet(output_path, vocab_path)
    else:
        print("  Building vocabulary from parquet (streaming)...")
        vocabulary = _build_vocab_from_parquet(output_path, vocab_path)

    print(f"  Vocabulary size: {vocabulary.vocab_size} tokens")
    log_stage_complete("Building Vocabulary")

    # ---- Load datasets from existing caches ----
    # Reuse the full cached datasets from training runs. If sample_fraction
    # is set, we subsample via sampler in the training loop — no need for
    # separate fraction-specific caches.
    log_stage_start("Loading Datasets for Tuning")

    max_seq_len = config.get('model', {}).get('params', {}).get('max_seq_len', 256) if config else 256
    val_cutoff = cutoff_year - 2

    cache_dir = "checkpoints/sequence_cache"
    os.makedirs(cache_dir, exist_ok=True)

    def _cache_exists(cp):
        return os.path.exists(cp) or os.path.isdir(cp.replace('.pt', '_chunks'))

    # Try full caches first, then sample-fraction-specific caches
    train_cache_full = os.path.join(cache_dir, f"train_cut{cutoff_year}.pt")
    val_cache_full = os.path.join(cache_dir, f"val_cut{val_cutoff}.pt")
    sf_tag = f"_sf{sample_fraction}" if sample_fraction else ""
    train_cache_sf = os.path.join(cache_dir, f"train_cut{cutoff_year}{sf_tag}.pt")
    val_cache_sf = os.path.join(cache_dir, f"val_cut{val_cutoff}{sf_tag}.pt")

    # Prefer full cache, fall back to sample-fraction cache, then build
    if _cache_exists(train_cache_full):
        train_cache = train_cache_full
        print(f"  Reusing full train cache")
    elif sf_tag and _cache_exists(train_cache_sf):
        train_cache = train_cache_sf
        print(f"  Reusing sample-fraction train cache ({sf_tag})")
    else:
        train_cache = train_cache_sf if sf_tag else train_cache_full
        print(f"  No train cache found — will build from parquet")

    if _cache_exists(val_cache_full):
        val_cache = val_cache_full
        print(f"  Reusing full val cache")
    elif sf_tag and _cache_exists(val_cache_sf):
        val_cache = val_cache_sf
        print(f"  Reusing sample-fraction val cache ({sf_tag})")
    else:
        val_cache = val_cache_sf if sf_tag else val_cache_full
        print(f"  No val cache found — will build from parquet")

    # Only scan parquet if we actually need to build a cache
    valid_persons = None
    val_persons = None
    if not _cache_exists(train_cache) or not _cache_exists(val_cache):
        print("  Building missing caches from parquet...")
        if not _cache_exists(train_cache):
            valid_persons = _collect_valid_persons(output_path, cutoff_year)
            print(f"  Train persons: {len(valid_persons):,}")
            if sample_fraction and not _cache_exists(train_cache_full):
                rng = np.random.RandomState(42)
                n_train = max(1, int(len(valid_persons) * sample_fraction))
                valid_persons = set(rng.choice(list(valid_persons), n_train, replace=False))
                print(f"  Subsampled to: {len(valid_persons):,}")
        if not _cache_exists(val_cache):
            val_persons = _collect_valid_persons(output_path, val_cutoff)
            print(f"  Val persons: {len(val_persons):,}")
            if sample_fraction and not _cache_exists(val_cache_full):
                rng = np.random.RandomState(42)
                n_val = max(1, int(len(val_persons) * sample_fraction))
                val_persons = set(rng.choice(list(val_persons), n_val, replace=False))
                print(f"  Subsampled to: {len(val_persons):,}")

    train_dataset = CachedSequenceDataset(
        parquet_path=output_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=cutoff_year,
        allowed_sids=valid_persons,
        cache_path=train_cache,
    )
    print(f"  Train dataset: {len(train_dataset):,} persons")

    val_dataset = CachedSequenceDataset(
        parquet_path=output_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=val_cutoff,
        allowed_sids=val_persons,
        cache_path=val_cache,
    )
    print(f"  Val dataset: {len(val_dataset):,} persons")
    del valid_persons, val_persons
    gc.collect()

    # If we loaded the full cache but want to use a fraction for tuning,
    # select a subset of chunks (not random indices!) to keep sequential
    # chunk access fast. For non-chunked datasets, use Subset.
    if sample_fraction is not None and sample_fraction < 1.0:
        rng = np.random.RandomState(42)

        def _subsample_dataset(ds, frac, label):
            if getattr(ds, '_using_chunks', False):
                # Select a fraction of chunks — keeps chunk-sequential access
                n_chunks = len(ds._chunk_sizes)
                n_use = max(1, int(n_chunks * frac))
                selected = sorted(rng.choice(n_chunks, n_use, replace=False))
                # Restrict chunk metadata so ChunkShuffledSampler only sees these
                ds._chunk_paths = [ds._chunk_paths[i] for i in selected]
                ds._chunk_sizes = [ds._chunk_sizes[i] for i in selected]
                ds._chunk_offsets = []
                offset = 0
                for s in ds._chunk_sizes:
                    ds._chunk_offsets.append(offset)
                    offset += s
                ds._chunk_offsets.append(offset)  # sentinel
                ds._total_len = offset
                ds._chunk_cache = {}
                ds._chunk_cache_order = []
                n_persons = sum(ds._chunk_sizes)
                print(f"  {label}: using {n_use}/{n_chunks} chunks ({n_persons:,} persons, {frac:.0%})")
            else:
                # Non-chunked: random subset via Subset wrapper
                from torch.utils.data import Subset

                class _SubsetWithAttrs(Subset):
                    def get_pos_weights(self, max_samples=None):
                        return self.dataset.get_pos_weights(max_samples=max_samples)

                n = max(1, int(len(ds) * frac))
                idx = rng.choice(len(ds), n, replace=False).tolist()
                ds = _SubsetWithAttrs(ds, idx)
                print(f"  {label}: {len(ds):,} persons ({frac:.0%})")
            return ds

        train_dataset = _subsample_dataset(train_dataset, sample_fraction, "Train subset")
        val_dataset = _subsample_dataset(val_dataset, sample_fraction, "Val subset")

    log_stage_complete("Loading Datasets for Tuning")
    log_memory_usage()

    # ---- Run Optuna tuning ----
    log_stage_start("Optuna Hyperparameter Tuning")

    tuner = SequenceModelTuner(
        base_config=config or {'model': {'type': f'seq_{encoder_type}', 'params': {'encoder_type': encoder_type}}},
        vocabulary=vocabulary,
        events=events,
        horizons=horizons,
        objective_metric="composite",  # 0.4*AP + 0.3*F1 + 0.3*AUC
        direction="maximize",
        tuning_epochs=tuning_epochs,
        seed=tuning_seed,
    )

    results = tuner.tune(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        n_trials=n_trials,
        search_space=search_space,
        n_jobs=tuning_workers,
        tuning_fraction=tuning_fraction,
    )

    log_stage_complete("Optuna Hyperparameter Tuning")

    # ---- Save results ----
    results_path = f"tuning_results_sequence_{encoder_type}.json"
    SequenceModelTuner.save_results(results, results_path)
    print(f"  Results saved to {results_path}")

    if config_path and config:
        tuned_config_path = config_path.replace('.yaml', '_tuned.yaml')
        SequenceModelTuner.save_tuned_config(results, config, tuned_config_path)

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"  Best val_mean_auc: {results['best_value']:.4f}")
    print(f"  Trials: {results['n_complete']} complete, "
          f"{results['n_pruned']} pruned, {results['n_failed']} failed")
    print(f"  Best parameters:")
    for k, v in results['best_params'].items():
        print(f"    {k}: {v}")
    print(f"  Results: {results_path}")
    print("=" * 60)

    return results


def _run_streaming_sequence_pipeline(
    output_path: str,
    total_rows: int,
    events: List[str],
    horizons: List[int],
    config: Optional[Dict[str, Any]],
    encoder_type: str,
    cutoff_year: int,
    reuse_processed: bool,
    config_path: Optional[str],
    sample_fraction: Optional[float],
    eval_fraction: Optional[float],
):
    print(f"  Dataset is large ({total_rows:,} rows)")
    print("  Using STREAMING mode")

    log_stage_start("Building Vocabulary")

    vocab_path = "checkpoints/sequence_vocab.joblib"
    os.makedirs("checkpoints", exist_ok=True)

    if reuse_processed and os.path.exists(vocab_path):
        print(f"  Loading existing vocabulary from {vocab_path}...")
        vocabulary = LifeEventVocabulary.load(vocab_path)
        missing_tokens = [
            f"{prefix}_Q1"
            for prefix in MUNICIPALITY_FEATURE_MAP.values()
            if vocabulary.token_to_id(f"{prefix}_Q1") == vocabulary.UNK
        ]
        if missing_tokens:
            print("  Vocabulary missing municipality tokens; rebuilding...")
            vocabulary = _build_vocab_from_parquet(output_path, vocab_path)
    else:
        print("  Building vocabulary from parquet (streaming)...")
        vocabulary = _build_vocab_from_parquet(output_path, vocab_path)

    print(f"  Vocabulary size: {vocabulary.vocab_size} tokens")
    log_stage_complete("Building Vocabulary")

    log_stage_start("Creating Sequence Datasets (Streaming)")

    parquet_dataset = pq.ParquetDataset(output_path)
    schema_cols = set(parquet_dataset.schema.names)
    missing_events = [e for e in events if e not in schema_cols]
    if missing_events:
        print(f"  WARNING: Missing event columns: {missing_events}")
        events = [e for e in events if e in schema_cols]
        if not events:
            print("  ERROR: No valid event columns found!")
            sys.exit(1)

    required_cols = ['sid', 'year']
    for col in required_cols:
        if col not in schema_cols:
            print(f"  ERROR: Required column '{col}' not found!")
            sys.exit(1)

    max_seq_len = config.get('model', {}).get('params', {}).get('max_seq_len', 256) if config else 256

    val_cutoff = cutoff_year - 2

    # Use CachedSequenceDataset: tokenize once, train from memory
    cache_dir = "checkpoints/sequence_cache"
    os.makedirs(cache_dir, exist_ok=True)

    sf_tag = f"_sf{sample_fraction}" if sample_fraction else ""
    ef_tag = f"_ef{eval_fraction}" if eval_fraction else ""
    train_cache = os.path.join(cache_dir, f"train_cut{cutoff_year}{sf_tag}.pt")
    val_cache = os.path.join(cache_dir, f"val_cut{val_cutoff}{sf_tag}.pt")
    test_cache = os.path.join(cache_dir, f"test_cut{cutoff_year}{sf_tag}{ef_tag}.pt")

    def _cache_exists(cp):
        """Check if cache exists as .pt file or _chunks directory."""
        return os.path.exists(cp) or os.path.isdir(cp.replace('.pt', '_chunks'))

    has_test = _cache_exists(test_cache)
    train_cached = _cache_exists(train_cache)
    val_cached = _cache_exists(val_cache)

    # Only scan parquet for valid persons if caches need to be built
    valid_persons = None
    val_persons = None
    eval_persons = None

    if not train_cached or not val_cached or (has_test and not _cache_exists(test_cache)):
        print("  Some caches missing - scanning parquet for valid persons...")
        valid_persons = _collect_valid_persons(output_path, cutoff_year)
        print(f"  Persons with both history and future: {len(valid_persons):,}")

        val_persons = _collect_valid_persons(output_path, val_cutoff)
        print(f"  Validation persons: {len(val_persons):,} (cutoff={val_cutoff})")

        if sample_fraction is not None:
            if not (0 < sample_fraction <= 1.0):
                print("  ERROR: --sample-fraction must be in (0, 1]")
                sys.exit(1)
            rng = np.random.RandomState(42)
            if valid_persons:
                n_train = max(1, int(len(valid_persons) * sample_fraction))
                valid_persons = set(rng.choice(list(valid_persons), n_train, replace=False))
            if val_persons:
                n_val = max(1, int(len(val_persons) * sample_fraction))
                val_persons = set(rng.choice(list(val_persons), n_val, replace=False))
            print(f"  Subsampled train persons: {len(valid_persons):,}")
            print(f"  Subsampled val persons: {len(val_persons):,}")

        eval_persons = valid_persons
        if eval_fraction is not None:
            if not (0 < eval_fraction <= 1.0):
                print("  ERROR: --eval-fraction must be in (0, 1]")
                sys.exit(1)
            rng = np.random.RandomState(43)
            if valid_persons:
                n_eval = max(1, int(len(valid_persons) * eval_fraction))
                eval_persons = set(rng.choice(list(valid_persons), n_eval, replace=False))
            print(f"  Evaluation persons: {len(eval_persons):,}")
    else:
        print(f"  All caches exist - skipping parquet scan")

    all_caches = [train_cache, val_cache] + ([test_cache] if has_test else [])
    n_cached = sum(1 for cp in all_caches if _cache_exists(cp))
    n_total = len(all_caches)
    if n_cached == n_total:
        print(f"  Reusing all {n_total} cached datasets from {cache_dir}")
    elif n_cached > 0:
        print(f"  Reusing {n_cached}/{n_total} cached datasets, building the rest...")
    else:
        print("  Pre-tokenizing datasets (will cache to disk)...")
    if not has_test:
        print("  (No test cache found - skipping test evaluation)")

    train_dataset = CachedSequenceDataset(
        parquet_path=output_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=cutoff_year,
        allowed_sids=valid_persons,
        cache_path=train_cache,
    )
    print(f"  Train dataset: {len(train_dataset):,} persons")

    val_dataset = CachedSequenceDataset(
        parquet_path=output_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=val_cutoff,
        allowed_sids=val_persons,
        cache_path=val_cache,
    )
    print(f"  Val dataset: {len(val_dataset):,} persons")
    # Free person sets after dataset creation
    del valid_persons, val_persons
    gc.collect()

    test_dataset = None
    if has_test:
        test_dataset = CachedSequenceDataset(
            parquet_path=output_path,
            vocabulary=vocabulary,
            max_seq_len=max_seq_len,
            events=events,
            horizons=horizons,
            cutoff_year=cutoff_year,
            allowed_sids=eval_persons,
            return_ids=True,
            cache_path=test_cache,
        )
        print(f"  Test dataset: {len(test_dataset):,} persons")
    else:
        print("  Test dataset: skipped (no cache)")
    del eval_persons
    gc.collect()

    log_stage_complete("Creating Sequence Datasets")
    log_memory_usage()

    log_stage_start("Model Training")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("demographic_forecasts_sequence_aft")

    model_config = config.get('model', {}) if config else {
        'type': 'seq_lstm',
        'params': {'encoder_type': 'lstm', 'embed_dim': 128}
    }

    estimator = PyTorchSequenceEstimator(
        model_config=model_config,
        device_config=None,
    )

    with mlflow.start_run(run_name=f"sequence_{encoder_type}_{'_'.join(events)}"):
        mlflow.log_params({
            'encoder_type': encoder_type,
            'embed_dim': model_config.get('params', {}).get('embed_dim', 128),
            'max_seq_len': model_config.get('params', {}).get('max_seq_len', 256),
            'vocab_size': vocabulary.vocab_size,
            'n_events': len(events),
            'events': ','.join(events),
            'horizons': str(horizons),
            'cutoff_year': cutoff_year,
            'train_persons': len(train_dataset),
            'val_persons': len(val_dataset),
            'eval_persons': len(test_dataset) if test_dataset else 0,
            'training_mode': 'cached',
            'has_test_set': has_test,
        })

        result = estimator.fit(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            vocabulary=vocabulary,
            lr_find_plot_path=os.path.join("checkpoints", "lr_finder.png"),
        )

        log_stage_complete("Model Training")
        log_memory_usage()

        metrics = None
        agg = None
        all_predictions = None
        sids = None
        pred_df = None
        group_eval_results = {}

        if test_dataset is not None:
            log_stage_start("Prediction Generation")

            proba_result = estimator.predict_proba(dataset=test_dataset)
            probs = proba_result.probabilities
            sids = proba_result.metadata.get('sids')
            if sids is None and hasattr(test_dataset, '_sids'):
                sids = test_dataset._sids
            if sids is None:
                print("  ERROR: Missing person IDs from prediction")
                sys.exit(1)

            all_predictions = _build_predictions_from_probs(probs, events, horizons)

            print(f"\n  Generated predictions for {len(events)} events "
                  f"at {len(horizons)} horizons")
            for event in events:
                if event in all_predictions:
                    for h in horizons:
                        key = f'prob_{h}yr'
                        if key in all_predictions[event]:
                            event_probs = all_predictions[event][key]
                            print(f"    {event} @{h}yr: mean={event_probs.mean():.4f}, "
                                  f"median={np.median(event_probs):.4f}, "
                                  f"P>0.5={(event_probs > 0.5).mean():.3%}")

            log_stage_complete("Prediction Generation")
            log_memory_usage()

            log_stage_start("Model Evaluation")

            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)

            y_true = _compute_labels_streaming(
                parquet_path=output_path,
                sids=sids,
                events=events,
                horizons=horizons,
                cutoff_year=cutoff_year,
            )

            metrics = _evaluate_from_arrays(
                all_predictions=all_predictions,
                y_true=y_true,
                events=events,
                horizons=horizons,
            )

            for event, event_metrics in metrics['per_event'].items():
                print(f"\n--- {event} ---")
                for horizon_key, h_metrics in event_metrics.items():
                    auc = h_metrics.get('auc', float('nan'))
                    ap = h_metrics.get('ap', float('nan'))
                    f1 = h_metrics.get('f1', float('nan'))
                    brier = h_metrics.get('brier', float('nan'))
                    prev = h_metrics.get('prevalence', 0)

                    print(f"  @{horizon_key}: AUC={auc:.4f}  AP={ap:.4f}  "
                          f"F1={f1:.4f}  Brier={brier:.4f}  "
                          f"prevalence={prev:.3%}")

                    if not np.isnan(auc):
                        mlflow.log_metric(f"{event}_{horizon_key}_auc", auc)
                    if not np.isnan(ap):
                        mlflow.log_metric(f"{event}_{horizon_key}_ap", ap)
                    if not np.isnan(f1):
                        mlflow.log_metric(f"{event}_{horizon_key}_f1", f1)
                    if not np.isnan(brier):
                        mlflow.log_metric(f"{event}_{horizon_key}_brier", brier)

            agg = metrics['aggregate']
            print(f"\n  Aggregate: mean_AUC={agg['mean_auc']:.4f}, "
                  f"mean_AP={agg['mean_ap']:.4f}")
            mlflow.log_metric("aggregate_mean_auc", agg['mean_auc'])
            mlflow.log_metric("aggregate_mean_ap", agg['mean_ap'])

            # Group-level evaluation
            log_stage_start("Group-Level Evaluation")

            group_cols = []
            for col_name in ['gender', 'age_group', 'refnis']:
                if col_name in pq.ParquetDataset(output_path).schema.names:
                    group_cols.append(col_name)

            group_eval_results = {}
            if group_cols:
                print(f"\n  Loading group features ({', '.join(group_cols)}) from parquet...")
                group_features_df = _load_group_features_streaming(
                    parquet_path=output_path,
                    sids=sids,
                    cutoff_year=cutoff_year,
                    group_cols=group_cols,
                )
                print(f"  Loaded features for {len(group_features_df):,} persons")

                # Per-group threshold calibration (minimise group-rate MSE)
                g_thresholds = None
                if 'refnis' in group_cols:
                    print(f"  Calibrating per-refnis thresholds on test set...")
                    g_thresholds = estimator.calibrate_group_thresholds(
                        dataset=test_dataset,
                        group_series=group_features_df['refnis'].values,
                        min_group_size=100,
                    )
                    print(f"  Per-group thresholds calibrated for {len(g_thresholds)} groups")

                group_eval_results = _evaluate_groups(
                    group_df=group_features_df,
                    all_predictions=all_predictions,
                    y_true=y_true,
                    events=events,
                    horizons=horizons,
                    group_cols=group_cols,
                    min_group_size=100,
                    group_thresholds=g_thresholds,
                )

                print("\n" + "=" * 60)
                print("GROUP-LEVEL EVALUATION")
                print("=" * 60)

                for event, eres in group_eval_results.items():
                    summary = eres.get('summary', {})
                    if not summary:
                        continue
                    print(f"\n--- {event} ({summary.get('groups', 0)} groups) ---")
                    for h in horizons:
                        rmse_w = summary.get(f'rmse_weighted_{h}yr')
                        rmse_u = summary.get(f'rmse_unweighted_{h}yr')
                        mae_w = summary.get(f'mae_weighted_{h}yr')
                        mae_t = summary.get(f'mae_weighted_thr_{h}yr')
                        rmse_t = summary.get(f'rmse_weighted_thr_{h}yr')
                        if rmse_w is not None:
                            line = f"  @{h}yr: RMSE_w={rmse_w:.4f}  MAE_w={mae_w:.4f}"
                            if rmse_t is not None:
                                line += f"  | thr: RMSE_w={rmse_t:.4f}  MAE_w={mae_t:.4f}"
                            print(line)
                            mlflow.log_metric(f"{event}_group_rmse_weighted_{h}yr", rmse_w)
                            mlflow.log_metric(f"{event}_group_rmse_unweighted_{h}yr", rmse_u)
                            mlflow.log_metric(f"{event}_group_mae_weighted_{h}yr", mae_w)
                            if mae_t is not None:
                                mlflow.log_metric(f"{event}_group_mae_weighted_thr_{h}yr", mae_t)
                                mlflow.log_metric(f"{event}_group_rmse_weighted_thr_{h}yr", rmse_t)

                del group_features_df
                gc.collect()
            else:
                print("  No group columns found in parquet - skipping group evaluation")

            log_stage_complete("Group-Level Evaluation")
            log_memory_usage()

            log_stage_complete("Model Evaluation")
            log_memory_usage()
        else:
            print("\n  Skipping standard prediction & evaluation (no test set)")

        # =============================================================
        # AFT survival metrics — always run for AFT models
        # Uses test_dataset if available, otherwise val_dataset
        # =============================================================
        prediction_intervals = None
        loss_type = config.get('model', {}).get('params', {}).get('loss_type', 'bce') if config else 'bce'
        print(f"\n  [DEBUG] loss_type={loss_type}, test_dataset={'present' if test_dataset is not None else 'None'}")

        if loss_type in ('aft', 'deephit'):
            aft_eval_dataset = test_dataset if test_dataset is not None else val_dataset
            aft_eval_label = "test" if test_dataset is not None else "val"
            print(f"\n  Computing {loss_type.upper()} metrics on {aft_eval_label} set ({len(aft_eval_dataset):,} persons)")

            log_stage_start(f"{loss_type.upper()} Survival Metrics")
            print(f"  Computing {loss_type}-native survival metrics (C-index, TD-AUC)...")
            try:
                from src.sequence.evaluation import evaluate_aft_survival_metrics, _targets_to_survival, _fast_c_index
                from src.sequence.dataset import sequence_collate_fn as _collate
                from torch.utils.data import DataLoader as _DL
                from sklearn.metrics import roc_auc_score as _roc_auc

                _loader = _DL(
                    aft_eval_dataset, batch_size=estimator.batch_size * 2,
                    shuffle=False, collate_fn=_collate, drop_last=False,
                )
                _tgt = []
                for _b in _loader:
                    _tgt.append(_b['targets'].numpy())
                aft_targets = np.concatenate(_tgt, axis=0)

                if loss_type == 'aft':
                    aft_result = estimator.predict_aft_params(dataset=aft_eval_dataset)
                    aft_surv_metrics = evaluate_aft_survival_metrics(
                        aft_params=aft_result.probabilities,
                        targets=aft_targets,
                        events=events,
                        horizons=horizons,
                    )

                    if aft_surv_metrics and 'per_event' in aft_surv_metrics:
                        print(f"\n  AFT survival metrics ({aft_eval_label} set, distribution-native):")
                        for event_name, ev_metrics in aft_surv_metrics['per_event'].items():
                            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ev_metrics.items())
                            print(f"    {event_name}: {metrics_str}")
                            for k, v in ev_metrics.items():
                                if isinstance(v, float) and not np.isnan(v):
                                    mlflow.log_metric(f"aft_{event_name}_{k}", v)
                        aft_agg = aft_surv_metrics.get('aggregate', {})
                        if aft_agg:
                            print(f"\n  AFT aggregate: "
                                  f"C-index={aft_agg.get('mean_c_index', float('nan')):.4f}, "
                                  f"CRPS={aft_agg.get('mean_crps', float('nan')):.4f}, "
                                  f"IBS={aft_agg.get('mean_ibs', float('nan')):.4f}, "
                                  f"TD-AUC={aft_agg.get('mean_td_auc', float('nan')):.4f}")
                            for k, v in aft_agg.items():
                                if not np.isnan(v):
                                    mlflow.log_metric(f"aft_{k}", v)
                    else:
                        print("  WARNING: evaluate_aft_survival_metrics returned empty results")

                elif loss_type == 'deephit':
                    # DeepHit: compute C-index and TD-AUC from learned CDF
                    result = estimator.predict_proba(dataset=aft_eval_dataset)
                    probs = result.probabilities  # (n, n_events * n_horizons)
                    n_ev = len(events)
                    n_h = len(horizons)
                    probs_3d = probs.reshape(-1, n_ev, n_h)

                    print(f"\n  DeepHit survival metrics ({aft_eval_label} set):")
                    c_indices = []
                    td_aucs = []
                    for ei, event in enumerate(events):
                        cols = [ei * n_h + hi for hi in range(n_h)]
                        event_tgt = aft_targets[:, cols]
                        duration, event_ind = _targets_to_survival(event_tgt, horizons)
                        cdf_e = probs_3d[:, ei, :]

                        n_pos = int(event_ind.sum())
                        if n_pos < 5 or (len(event_ind) - n_pos) < 5:
                            print(f"    {event}: skipped (too few events)")
                            continue

                        risk = cdf_e[:, -1]
                        c_idx = _fast_c_index(event_ind, duration, risk)
                        c_indices.append(c_idx)

                        ev_td = []
                        for hi, h in enumerate(horizons):
                            y_h = event_tgt[:, hi]
                            n_p = int(y_h.sum())
                            if 0 < n_p < len(y_h):
                                auc_h = float(_roc_auc(y_h, cdf_e[:, hi]))
                                ev_td.append(auc_h)
                                td_aucs.append(auc_h)
                                mlflow.log_metric(f"deephit_td_auc_{event}_{h}yr", auc_h)

                        td_str = ", ".join(f"{h}yr={a:.4f}" for h, a in zip(horizons, ev_td)) if ev_td else "N/A"
                        print(f"    {event}: C-index={c_idx:.4f}, TD-AUC: {td_str}")
                        mlflow.log_metric(f"deephit_{event}_c_index", c_idx)

                    if c_indices:
                        mean_c = float(np.mean(c_indices))
                        mean_td = float(np.mean(td_aucs)) if td_aucs else float('nan')
                        print(f"\n  DeepHit aggregate: C-index={mean_c:.4f}, TD-AUC={mean_td:.4f}")
                        mlflow.log_metric("deephit_mean_c_index", mean_c)
                        if not np.isnan(mean_td):
                            mlflow.log_metric("deephit_mean_td_auc", mean_td)

            except Exception as e:
                print(f"  {loss_type.upper()} survival metrics FAILED: {e}")
                import traceback; traceback.print_exc()
            log_stage_complete(f"{loss_type.upper()} Survival Metrics")

            if loss_type == 'aft':
                log_stage_start("Prediction Intervals")
                print("  Computing prediction intervals from AFT distribution...")
                try:
                    prediction_intervals = estimator.predict_intervals(
                        dataset=aft_eval_dataset,
                        confidence_levels=[0.5, 0.8, 0.9],
                    )
                    for event_name in events:
                        if event_name in prediction_intervals:
                            ei = prediction_intervals[event_name]
                            med = ei['median']
                            ci80_lo = ei['ci_80_lower']
                            ci80_hi = ei['ci_80_upper']
                            print(f"    {event_name}: median={np.median(med):.2f}yr, "
                                  f"80%CI=[{np.median(ci80_lo):.2f}, {np.median(ci80_hi):.2f}]yr")
                            mlflow.log_metric(f"aft_{event_name}_median_tte",
                                              float(np.median(med)))
                            mlflow.log_metric(f"aft_{event_name}_ci80_width",
                                              float(np.median(ci80_hi - ci80_lo)))
                except Exception as e:
                    print(f"  Prediction intervals FAILED: {e}")
                    import traceback; traceback.print_exc()
                log_stage_complete("Prediction Intervals")
        else:
            print(f"  Skipping survival metrics (loss_type={loss_type})")

        log_stage_start("Saving Artifacts")

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"sequence_{encoder_type}_{timestamp}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        estimator.save(checkpoint_path)
        print(f"  Saved model: {checkpoint_path}")

        if all_predictions is not None and sids is not None:
            pred_records = []
            for i, sid in enumerate(sids):
                record = {'sid': sid}
                for event in events:
                    if event in all_predictions:
                        for h in horizons:
                            key = f'prob_{h}yr'
                            if key in all_predictions[event]:
                                record[f'{event}_prob_{h}yr'] = float(
                                    all_predictions[event][key][i]
                                )
                    # Add prediction intervals
                    if prediction_intervals and event in prediction_intervals:
                        ei = prediction_intervals[event]
                        record[f'{event}_median_tte'] = float(ei['median'][i])
                        for level in [50, 80, 90]:
                            lo_key = f'ci_{level}_lower'
                            hi_key = f'ci_{level}_upper'
                            if lo_key in ei:
                                record[f'{event}_ci{level}_lower'] = float(ei[lo_key][i])
                                record[f'{event}_ci{level}_upper'] = float(ei[hi_key][i])
                pred_records.append(record)

            pred_df = pd.DataFrame(pred_records)
            pred_path = os.path.join(checkpoint_path, "predictions.parquet")
            pred_df.to_parquet(pred_path)
            print(f"  Saved predictions: {pred_path}")

        serializable_metrics = {}
        if metrics is not None:
            for event, em in metrics['per_event'].items():
                serializable_metrics[event] = {}
                for hk, hm in em.items():
                    serializable_metrics[event][hk] = {
                        k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in hm.items()
                    }

        # Save group eval CSVs
        group_eval_summaries_serial = {}
        if group_eval_results:
            for event, eres in group_eval_results.items():
                gdf = eres.get('group_df')
                if gdf is not None and len(gdf) > 0:
                    csv_path = os.path.join(checkpoint_path, f"group_eval_{event}.csv")
                    gdf.to_csv(csv_path, index=False)
                    print(f"  Saved group eval: {csv_path}")
                summary = eres.get('summary', {})
                if summary:
                    group_eval_summaries_serial[event] = {
                        k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in summary.items()
                    }

        metadata = {
            'timestamp': timestamp,
            'model_type': 'sequence',
            'encoder_type': encoder_type,
            'events': events,
            'horizons': horizons,
            'cutoff_year': cutoff_year,
            'vocab_size': vocabulary.vocab_size,
            'train_persons': len(train_dataset),
            'val_persons': len(val_dataset),
            'eval_persons': len(test_dataset) if test_dataset else 0,
            'epochs_trained': result.metadata.get('epochs_trained', 0),
            'n_params': result.metadata.get('n_params', 0),
            'metrics': serializable_metrics,
            'aggregate': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in agg.items()
            } if agg else {},
            'group_eval': group_eval_summaries_serial,
            'config_path': config_path,
            'training_mode': 'cached',
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        mlflow.log_artifacts(checkpoint_path, artifact_path="sequence_checkpoint")

        log_stage_complete("Saving Artifacts")

    print("\n" + "=" * 60)
    print("SEQUENCE MODEL COMPLETE")
    print("=" * 60)
    if metrics is not None:
        for event in events:
            if event in metrics['per_event']:
                aucs = []
                for hk, hm in metrics['per_event'][event].items():
                    a = hm.get('auc', float('nan'))
                    aucs.append(f"{hk}={a:.3f}" if not np.isnan(a) else f"{hk}=N/A")
                print(f"  {event}: AUC: {', '.join(aucs)}")
        print(f"\n  Aggregate: mean_AUC={agg['mean_auc']:.4f}")
    else:
        print("  (No test evaluation - only training completed)")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)

    return estimator, metrics, pred_df


def _run_rolling_window_pipeline(
    output_path: str,
    total_rows: int,
    events: List[str],
    horizons: List[int],
    config: Optional[Dict[str, Any]],
    encoder_type: str,
    reuse_processed: bool,
    config_path: Optional[str],
    sample_fraction: Optional[float],
    eval_fraction: Optional[float],
    history_len: int = 5,
    rolling_cutoffs: Optional[List[int]] = None,
    train_cutoffs: Optional[List[int]] = None,
    val_cutoff: Optional[int] = None,
    test_cutoff: Optional[int] = None,
    existing_cache: Optional[str] = None,
):
    """Run sequence model pipeline with rolling window validation.

    Creates multiple training samples per person using different cutoff years.
    Each window has `history_len` years of history and up to max(horizons)
    years of future for targets.
    """
    max_horizon = max(horizons)
    max_seq_len = config.get('model', {}).get('params', {}).get('max_seq_len', 256) if config else 256

    # ================================================================
    # Step 1: Determine cutoff years
    # ================================================================
    if rolling_cutoffs is None:
        print("  Auto-detecting year range from data...")
        min_year, max_year = _scan_year_range(output_path)
        print(f"  Data year range: {min_year}-{max_year}")
        first_cutoff = min_year + history_len
        last_cutoff = max_year - max_horizon
        rolling_cutoffs = list(range(first_cutoff, last_cutoff + 1))

    if train_cutoffs is None:
        train_cutoffs = rolling_cutoffs[:-2]
    if val_cutoff is None:
        val_cutoff = rolling_cutoffs[-2]
    if test_cutoff is None:
        test_cutoff = rolling_cutoffs[-1]

    print(f"\n  Rolling window configuration:")
    print(f"    History length: {history_len} years")
    print(f"    All cutoffs: {rolling_cutoffs}")
    print(f"    Train cutoffs: {train_cutoffs}")
    print(f"    Val cutoff: {val_cutoff}")
    print(f"    Test cutoff: {test_cutoff}")

    for cutoff in rolling_cutoffs:
        min_h = cutoff - history_len
        max_f = cutoff + max_horizon
        role = "TRAIN" if cutoff in train_cutoffs else ("VAL" if cutoff == val_cutoff else "TEST")
        print(f"    W cutoff={cutoff} [{role}]: history ({min_h}, {cutoff}] -> targets ({cutoff}, {max_f}]")
    print()

    # ================================================================
    # Step 2: Build vocabulary
    # ================================================================
    log_stage_start("Building Vocabulary")

    vocab_path = "checkpoints/sequence_vocab.joblib"
    os.makedirs("checkpoints", exist_ok=True)

    if reuse_processed and os.path.exists(vocab_path):
        print(f"  Loading existing vocabulary from {vocab_path}...")
        vocabulary = LifeEventVocabulary.load(vocab_path)
        missing_tokens = [
            f"{prefix}_Q1"
            for prefix in MUNICIPALITY_FEATURE_MAP.values()
            if vocabulary.token_to_id(f"{prefix}_Q1") == vocabulary.UNK
        ]
        if missing_tokens:
            print("  Vocabulary missing municipality tokens; rebuilding...")
            vocabulary = _build_vocab_from_parquet(output_path, vocab_path)
    else:
        print("  Building vocabulary from parquet...")
        vocabulary = _build_vocab_from_parquet(output_path, vocab_path)

    print(f"  Vocabulary size: {vocabulary.vocab_size} tokens")
    log_stage_complete("Building Vocabulary")

    # ================================================================
    # Step 3: Build/load per-window caches
    # ================================================================
    log_stage_start("Creating Rolling Window Datasets")

    cache_dir = "checkpoints/sequence_cache"
    os.makedirs(cache_dir, exist_ok=True)

    sf_tag = f"_sf{sample_fraction}" if sample_fraction else ""

    def _cache_path_for(cutoff):
        return os.path.join(cache_dir, f"rolling_cut{cutoff}_h{history_len}{sf_tag}.pt")

    def _cache_exists(cp):
        return os.path.exists(cp) or os.path.isdir(cp.replace('.pt', '_chunks'))

    # Build any missing caches
    all_cutoffs = sorted(set(train_cutoffs + [val_cutoff, test_cutoff]))
    missing_cutoffs = [c for c in all_cutoffs if not _cache_exists(_cache_path_for(c))]
    for c in all_cutoffs:
        if c not in missing_cutoffs:
            print(f"  Cache exists for cutoff={c}")

    if missing_cutoffs and existing_cache:
        # Fast path: build all missing caches from existing shards
        from build_sequence_cache import _build_rolling_from_existing_cache
        print(f"  Building {len(missing_cutoffs)} missing caches from existing shards: {existing_cache}")
        _build_rolling_from_existing_cache(
            existing_cache_dir=existing_cache,
            vocabulary=vocabulary,
            max_seq_len=max_seq_len,
            events=events,
            horizons=horizons,
            cutoff_years=missing_cutoffs,
            history_len=history_len,
            cache_dir=cache_dir,
            sample_fraction=sample_fraction,
        )
    elif missing_cutoffs:
        # Slow path: build from raw parquet one by one
        for cutoff in missing_cutoffs:
            cp = _cache_path_for(cutoff)
            min_hist_year = cutoff - history_len
            print(f"  Building cache for cutoff={cutoff} (history ({min_hist_year}, {cutoff}])...")

            valid = _collect_valid_persons_windowed(
                output_path, cutoff, min_hist_year, max_horizon,
            )
            print(f"    Valid persons: {len(valid):,}")

            if sample_fraction is not None:
                rng = np.random.RandomState(42 + cutoff)
                n_sample = max(1, int(len(valid) * sample_fraction))
                valid = set(rng.choice(list(valid), n_sample, replace=False))
                print(f"    Subsampled to {sample_fraction:.2%}: {len(valid):,} persons")

            CachedSequenceDataset(
                parquet_path=output_path,
                vocabulary=vocabulary,
                max_seq_len=max_seq_len,
                events=events,
                horizons=horizons,
                cutoff_year=cutoff,
                allowed_sids=valid,
                cache_path=cp,
                min_history_year=min_hist_year,
            )
            print(f"    Cache built for cutoff={cutoff}")

    # Create combined training dataset
    train_cache_paths = [_cache_path_for(c) for c in train_cutoffs]
    train_dataset = CachedRollingWindowDataset(
        cache_paths=train_cache_paths,
        cutoff_years=train_cutoffs,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
    )
    print(f"  Rolling train dataset: {len(train_dataset):,} samples from {len(train_cutoffs)} windows")

    # Create single-window val and test datasets
    val_cp = _cache_path_for(val_cutoff)
    val_dataset = CachedSequenceDataset(
        parquet_path=output_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=val_cutoff,
        cache_path=val_cp,
        min_history_year=val_cutoff - history_len,
    )
    print(f"  Val dataset: {len(val_dataset):,} persons (cutoff={val_cutoff})")

    test_cp = _cache_path_for(test_cutoff)
    test_dataset = CachedSequenceDataset(
        parquet_path=output_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=test_cutoff,
        return_ids=True,
        cache_path=test_cp,
        min_history_year=test_cutoff - history_len,
    )
    print(f"  Test dataset: {len(test_dataset):,} persons (cutoff={test_cutoff})")

    log_stage_complete("Creating Rolling Window Datasets")
    log_memory_usage()

    # ================================================================
    # Step 4: Train model
    # ================================================================
    log_stage_start("Model Training")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("demographic_forecasts_sequence_aft")

    model_config = config.get('model', {}) if config else {
        'type': 'seq_lstm',
        'params': {'encoder_type': 'lstm', 'embed_dim': 128}
    }

    estimator = PyTorchSequenceEstimator(
        model_config=model_config,
        device_config=None,
    )

    with mlflow.start_run(run_name=f"rolling_{encoder_type}_h{history_len}_{'_'.join(events)}"):
        mlflow.log_params({
            'encoder_type': encoder_type,
            'embed_dim': model_config.get('params', {}).get('embed_dim', 128),
            'max_seq_len': max_seq_len,
            'vocab_size': vocabulary.vocab_size,
            'n_events': len(events),
            'events': ','.join(events),
            'horizons': str(horizons),
            'training_mode': 'rolling_window',
            'history_len': history_len,
            'train_cutoffs': str(train_cutoffs),
            'val_cutoff': val_cutoff,
            'test_cutoff': test_cutoff,
            'train_samples': len(train_dataset),
            'val_persons': len(val_dataset),
            'test_persons': len(test_dataset),
        })

        result = estimator.fit(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            vocabulary=vocabulary,
            lr_find_plot_path=os.path.join("checkpoints", "lr_finder_rolling.png"),
        )

        log_stage_complete("Model Training")
        log_memory_usage()

        # ================================================================
        # Step 5: Evaluation on test window
        # ================================================================
        metrics = None
        agg = None
        all_predictions = None
        sids = None
        pred_df = None
        group_eval_results = {}

        log_stage_start("Prediction Generation")

        proba_result = estimator.predict_proba(dataset=test_dataset)
        probs = proba_result.probabilities
        sids = proba_result.metadata.get('sids')
        if sids is None and hasattr(test_dataset, '_sids'):
            sids = test_dataset._sids

        all_predictions = _build_predictions_from_probs(probs, events, horizons)

        print(f"\n  Generated predictions for {len(events)} events "
              f"at {len(horizons)} horizons (test cutoff={test_cutoff})")
        for event in events:
            if event in all_predictions:
                for h in horizons:
                    key = f'prob_{h}yr'
                    if key in all_predictions[event]:
                        event_probs = all_predictions[event][key]
                        print(f"    {event} @{h}yr: mean={event_probs.mean():.4f}, "
                              f"median={np.median(event_probs):.4f}, "
                              f"P>0.5={(event_probs > 0.5).mean():.3%}")

        log_stage_complete("Prediction Generation")

        # Compute test labels
        log_stage_start("Model Evaluation")
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (Rolling Window)")
        print("=" * 60)

        y_true = _compute_labels_streaming(
            parquet_path=output_path,
            sids=sids,
            events=events,
            horizons=horizons,
            cutoff_year=test_cutoff,
        )

        metrics = _evaluate_from_arrays(
            all_predictions=all_predictions,
            y_true=y_true,
            events=events,
            horizons=horizons,
        )

        for event, event_metrics in metrics['per_event'].items():
            print(f"\n--- {event} ---")
            for horizon_key, h_metrics in event_metrics.items():
                auc = h_metrics.get('auc', float('nan'))
                ap = h_metrics.get('ap', float('nan'))
                f1 = h_metrics.get('f1', float('nan'))
                brier = h_metrics.get('brier', float('nan'))
                prev = h_metrics.get('prevalence', 0)
                print(f"  @{horizon_key}: AUC={auc:.4f}  AP={ap:.4f}  "
                      f"F1={f1:.4f}  Brier={brier:.4f}  "
                      f"prevalence={prev:.3%}")
                if not np.isnan(auc):
                    mlflow.log_metric(f"{event}_{horizon_key}_auc", auc)
                if not np.isnan(ap):
                    mlflow.log_metric(f"{event}_{horizon_key}_ap", ap)
                if not np.isnan(f1):
                    mlflow.log_metric(f"{event}_{horizon_key}_f1", f1)

        agg = metrics['aggregate']
        print(f"\n  Aggregate: mean_AUC={agg['mean_auc']:.4f}, "
              f"mean_AP={agg['mean_ap']:.4f}")
        mlflow.log_metric("aggregate_mean_auc", agg['mean_auc'])
        mlflow.log_metric("aggregate_mean_ap", agg['mean_ap'])

        log_stage_complete("Model Evaluation")

        # ================================================================
        # AFT survival metrics
        # ================================================================
        prediction_intervals = None
        loss_type = config.get('model', {}).get('params', {}).get('loss_type', 'bce') if config else 'bce'

        if loss_type in ('aft', 'deephit'):
            log_stage_start(f"{loss_type.upper()} Survival Metrics")
            print(f"\n  Computing {loss_type.upper()} metrics on test set ({len(test_dataset):,} persons)")
            try:
                from src.sequence.evaluation import evaluate_aft_survival_metrics, _targets_to_survival, _fast_c_index
                from src.sequence.dataset import sequence_collate_fn as _collate
                from torch.utils.data import DataLoader as _DL
                from sklearn.metrics import roc_auc_score as _roc_auc

                _loader = _DL(
                    test_dataset, batch_size=estimator.batch_size * 2,
                    shuffle=False, collate_fn=_collate, drop_last=False,
                )
                _tgt = []
                for _b in _loader:
                    _tgt.append(_b['targets'].numpy())
                surv_targets = np.concatenate(_tgt, axis=0)

                if loss_type == 'aft':
                    aft_result = estimator.predict_aft_params(dataset=test_dataset)
                    aft_surv_metrics = evaluate_aft_survival_metrics(
                        aft_params=aft_result.probabilities,
                        targets=surv_targets, events=events, horizons=horizons,
                    )
                    if aft_surv_metrics and 'per_event' in aft_surv_metrics:
                        print(f"\n  AFT survival metrics (test set, cutoff={test_cutoff}):")
                        for event_name, ev_metrics in aft_surv_metrics['per_event'].items():
                            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ev_metrics.items())
                            print(f"    {event_name}: {metrics_str}")
                            for k, v in ev_metrics.items():
                                if isinstance(v, float) and not np.isnan(v):
                                    mlflow.log_metric(f"aft_{event_name}_{k}", v)
                        aft_agg = aft_surv_metrics.get('aggregate', {})
                        if aft_agg:
                            print(f"\n  AFT aggregate: "
                                  f"C-index={aft_agg.get('mean_c_index', float('nan')):.4f}, "
                                  f"CRPS={aft_agg.get('mean_crps', float('nan')):.4f}, "
                                  f"IBS={aft_agg.get('mean_ibs', float('nan')):.4f}, "
                                  f"TD-AUC={aft_agg.get('mean_td_auc', float('nan')):.4f}")
                            for k, v in aft_agg.items():
                                if not np.isnan(v):
                                    mlflow.log_metric(f"aft_{k}", v)

                elif loss_type == 'deephit':
                    result = estimator.predict_proba(dataset=test_dataset)
                    probs = result.probabilities
                    n_ev, n_h = len(events), len(horizons)
                    probs_3d = probs.reshape(-1, n_ev, n_h)

                    print(f"\n  DeepHit survival metrics (test set, cutoff={test_cutoff}):")
                    c_indices, td_aucs = [], []
                    for ei, event in enumerate(events):
                        cols = [ei * n_h + hi for hi in range(n_h)]
                        event_tgt = surv_targets[:, cols]
                        duration, event_ind = _targets_to_survival(event_tgt, horizons)
                        cdf_e = probs_3d[:, ei, :]
                        n_pos = int(event_ind.sum())
                        if n_pos < 5 or (len(event_ind) - n_pos) < 5:
                            continue
                        risk = cdf_e[:, -1]
                        c_idx = _fast_c_index(event_ind, duration, risk)
                        c_indices.append(c_idx)
                        ev_td = []
                        for hi, h in enumerate(horizons):
                            y_h = event_tgt[:, hi]
                            n_p = int(y_h.sum())
                            if 0 < n_p < len(y_h):
                                auc_h = float(_roc_auc(y_h, cdf_e[:, hi]))
                                ev_td.append(auc_h)
                                td_aucs.append(auc_h)
                        td_str = ", ".join(f"{h}yr={a:.4f}" for h, a in zip(horizons, ev_td))
                        print(f"    {event}: C-index={c_idx:.4f}, TD-AUC: {td_str}")
                    if c_indices:
                        print(f"\n  DeepHit aggregate: C-index={np.mean(c_indices):.4f}, "
                              f"TD-AUC={np.mean(td_aucs):.4f}")

            except Exception as e:
                print(f"  {loss_type.upper()} survival metrics FAILED: {e}")
                import traceback; traceback.print_exc()
            log_stage_complete(f"{loss_type.upper()} Survival Metrics")

            log_stage_start("Prediction Intervals")
            try:
                prediction_intervals = estimator.predict_intervals(
                    dataset=test_dataset,
                    confidence_levels=[0.5, 0.8, 0.9],
                )
                for event_name in events:
                    if event_name in prediction_intervals:
                        ei = prediction_intervals[event_name]
                        med = ei['median']
                        ci80_lo = ei['ci_80_lower']
                        ci80_hi = ei['ci_80_upper']
                        print(f"    {event_name}: median={np.median(med):.2f}yr, "
                              f"80%CI=[{np.median(ci80_lo):.2f}, {np.median(ci80_hi):.2f}]yr")
                        mlflow.log_metric(f"aft_{event_name}_median_tte",
                                          float(np.median(med)))
                        mlflow.log_metric(f"aft_{event_name}_ci80_width",
                                          float(np.median(ci80_hi - ci80_lo)))
            except Exception as e:
                print(f"  Prediction intervals FAILED: {e}")
                import traceback; traceback.print_exc()
            log_stage_complete("Prediction Intervals")

        # ================================================================
        # Per-window evaluation (stability analysis)
        # ================================================================
        log_stage_start("Per-Window Stability Analysis")
        print("\n  Evaluating model on each individual window:")
        for cutoff in rolling_cutoffs:
            cp = _cache_path_for(cutoff)
            if not _cache_exists(cp):
                continue
            try:
                window_ds = CachedSequenceDataset(
                    parquet_path=output_path,
                    vocabulary=vocabulary,
                    max_seq_len=max_seq_len,
                    events=events,
                    horizons=horizons,
                    cutoff_year=cutoff,
                    cache_path=cp,
                    min_history_year=cutoff - history_len,
                    return_ids=True,
                )
                window_proba = estimator.predict_proba(dataset=window_ds)
                window_preds = _build_predictions_from_probs(
                    window_proba.probabilities, events, horizons,
                )
                window_sids = window_proba.metadata.get('sids')
                if window_sids is None and hasattr(window_ds, '_sids') and window_ds._sids is not None:
                    window_sids = window_ds._sids
                if window_sids is None:
                    raise ValueError(
                        f"No sids available for cutoff={cutoff}. "
                        f"Ensure the dataset is created with return_ids=True "
                        f"and that cached chunks contain 'sids'."
                    )

                window_labels = _compute_labels_streaming(
                    parquet_path=output_path,
                    sids=window_sids,
                    events=events,
                    horizons=horizons,
                    cutoff_year=cutoff,
                )
                window_metrics = _evaluate_from_arrays(
                    all_predictions=window_preds,
                    y_true=window_labels,
                    events=events,
                    horizons=horizons,
                )
                w_agg = window_metrics['aggregate']
                role = "TRAIN" if cutoff in train_cutoffs else ("VAL" if cutoff == val_cutoff else "TEST")
                print(f"    W cutoff={cutoff} [{role}]: mean_AUC={w_agg['mean_auc']:.4f}, "
                      f"mean_AP={w_agg['mean_ap']:.4f}")
                mlflow.log_metric(f"window_{cutoff}_mean_auc", w_agg['mean_auc'])
                mlflow.log_metric(f"window_{cutoff}_mean_ap", w_agg['mean_ap'])
                del window_ds, window_proba, window_preds, window_labels
                gc.collect()
            except Exception as e:
                print(f"    W cutoff={cutoff}: FAILED ({e})")
        log_stage_complete("Per-Window Stability Analysis")

        # ================================================================
        # Save artifacts
        # ================================================================
        log_stage_start("Saving Artifacts")

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"rolling_{encoder_type}_{timestamp}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        estimator.save(checkpoint_path)
        print(f"  Saved model: {checkpoint_path}")

        if all_predictions is not None and sids is not None:
            pred_records = []
            for i, sid in enumerate(sids):
                record = {'sid': sid}
                for event in events:
                    if event in all_predictions:
                        for h in horizons:
                            key = f'prob_{h}yr'
                            if key in all_predictions[event]:
                                record[f'{event}_prob_{h}yr'] = float(
                                    all_predictions[event][key][i]
                                )
                    if prediction_intervals and event in prediction_intervals:
                        ei = prediction_intervals[event]
                        record[f'{event}_median_tte'] = float(ei['median'][i])
                        for level in [50, 80, 90]:
                            lo_key = f'ci_{level}_lower'
                            hi_key = f'ci_{level}_upper'
                            if lo_key in ei:
                                record[f'{event}_ci{level}_lower'] = float(ei[lo_key][i])
                                record[f'{event}_ci{level}_upper'] = float(ei[hi_key][i])
                pred_records.append(record)

            pred_df = pd.DataFrame(pred_records)
            pred_path = os.path.join(checkpoint_path, "predictions.parquet")
            pred_df.to_parquet(pred_path)
            print(f"  Saved predictions: {pred_path}")

        serializable_metrics = {}
        if metrics is not None:
            for event, em in metrics['per_event'].items():
                serializable_metrics[event] = {}
                for hk, hm in em.items():
                    serializable_metrics[event][hk] = {
                        k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in hm.items()
                    }

        metadata = {
            'timestamp': timestamp,
            'model_type': 'sequence',
            'encoder_type': encoder_type,
            'events': events,
            'horizons': horizons,
            'training_mode': 'rolling_window',
            'history_len': history_len,
            'train_cutoffs': train_cutoffs,
            'val_cutoff': val_cutoff,
            'test_cutoff': test_cutoff,
            'vocab_size': vocabulary.vocab_size,
            'train_samples': len(train_dataset),
            'val_persons': len(val_dataset),
            'test_persons': len(test_dataset),
            'epochs_trained': result.metadata.get('epochs_trained', 0),
            'n_params': result.metadata.get('n_params', 0),
            'metrics': serializable_metrics,
            'aggregate': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in agg.items()
            } if agg else {},
            'config_path': config_path,
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        mlflow.log_artifacts(checkpoint_path, artifact_path="rolling_checkpoint")

        log_stage_complete("Saving Artifacts")

    print("\n" + "=" * 60)
    print("ROLLING WINDOW PIPELINE COMPLETE")
    print("=" * 60)
    if metrics is not None:
        for event in events:
            if event in metrics['per_event']:
                aucs = []
                for hk, hm in metrics['per_event'][event].items():
                    a = hm.get('auc', float('nan'))
                    aucs.append(f"{hk}={a:.3f}" if not np.isnan(a) else f"{hk}=N/A")
                print(f"  {event}: AUC: {', '.join(aucs)}")
        print(f"\n  Aggregate: mean_AUC={agg['mean_auc']:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)

    return estimator, metrics, pred_df


def main_sequence(
    reuse_processed=False,
    max_rows=None,
    config_path=None,
    events=None,
    horizons=None,
    cutoff_year=2022,
    streaming=False,
    sample_fraction=None,
    eval_fraction=None,
    tune=False,
    n_trials=50,
    tuning_epochs=15,
    search_space='conservative',
    tuning_seed=42,
    tuning_workers=1,
    tuning_fraction=1.0,
    rolling=False,
    history_len=5,
    rolling_cutoffs=None,
    rolling_train_cutoffs=None,
    rolling_val_cutoff=None,
    rolling_test_cutoff=None,
    existing_cache=None,
):
    """
    Main sequence model pipeline.

    Args:
        reuse_processed: Reuse existing processed features.
        max_rows: Limit dataset size.
        config_path: Path to sequence model config YAML.
        events: List of event column names to predict.
        horizons: Prediction horizons in years.
        cutoff_year: Use history up to this year for input,
                     predict events after this year.
        streaming: Force streaming mode for large datasets.
        sample_fraction: Use a fraction of persons for faster prototyping.
        eval_fraction: Use a fraction of persons for evaluation/prediction.
        rolling: Use rolling window validation.
        history_len: History window length in years for rolling mode.
        rolling_cutoffs: List of cutoff years for rolling windows.
        rolling_train_cutoffs: Subset of cutoffs to use for training.
        rolling_val_cutoff: Cutoff year for validation.
        rolling_test_cutoff: Cutoff year for test.
    """
    log_file = setup_logging(
        log_file=f"sequence_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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
    encoder_type = 'lstm'

    if config_path:
        config = load_model_config(config_path)
        params = config.get('model', {}).get('params', {})
        encoder_type = params.get('encoder_type', 'lstm')
        events = config.get('events', events)
        horizons = config.get('horizons', horizons)

    print("=" * 60)
    print("SEQUENCE MODEL PIPELINE")
    print("=" * 60)
    print(f"  Encoder: {encoder_type.upper()}")
    print(f"  Events: {', '.join(events)}")
    print(f"  Horizons: {horizons} years")
    print(f"  Cutoff year: {cutoff_year}")
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
        sys.exit(1)

    log_stage_start("Loading Processed Features")
    print(f"\nLoading processed features from {output_path}...")

    parquet_dataset = pq.ParquetDataset(output_path)
    total_rows = sum(piece.metadata.num_rows for piece in parquet_dataset.fragments)
    print(f"  Total rows: {total_rows:,}")

    # Hyperparameter tuning branch
    if tune:
        log_stage_complete("Loading Processed Features")
        return _run_hyperparameter_tuning(
            output_path=output_path,
            total_rows=total_rows,
            events=events,
            horizons=horizons,
            config=config,
            encoder_type=encoder_type,
            cutoff_year=cutoff_year,
            reuse_processed=reuse_processed,
            config_path=config_path,
            sample_fraction=sample_fraction,
            n_trials=n_trials,
            tuning_epochs=tuning_epochs,
            search_space=search_space,
            tuning_seed=tuning_seed,
            tuning_workers=tuning_workers,
            tuning_fraction=tuning_fraction,
        )

    # Rolling window branch
    if rolling:
        log_stage_complete("Loading Processed Features")
        return _run_rolling_window_pipeline(
            output_path=output_path,
            total_rows=total_rows,
            events=events,
            horizons=horizons,
            config=config,
            encoder_type=encoder_type,
            reuse_processed=reuse_processed,
            config_path=config_path,
            sample_fraction=sample_fraction,
            eval_fraction=eval_fraction,
            history_len=history_len,
            rolling_cutoffs=rolling_cutoffs,
            train_cutoffs=rolling_train_cutoffs,
            val_cutoff=rolling_val_cutoff,
            test_cutoff=rolling_test_cutoff,
            existing_cache=existing_cache,
        )

    # Determine loading strategy
    safe_in_memory_limit = 2_000_000
    use_streaming = streaming or (total_rows > safe_in_memory_limit and max_rows is None)

    if use_streaming and max_rows is not None:
        print("  Streaming mode ignored because --max-rows is set")
        use_streaming = False

    if use_streaming:
        log_stage_complete("Loading Processed Features")
        return _run_streaming_sequence_pipeline(
            output_path=output_path,
            total_rows=total_rows,
            events=events,
            horizons=horizons,
            config=config,
            encoder_type=encoder_type,
            cutoff_year=cutoff_year,
            reuse_processed=reuse_processed,
            config_path=config_path,
            sample_fraction=sample_fraction,
            eval_fraction=eval_fraction,
        )

    # Load data
    if max_rows is not None and total_rows > max_rows:
        print(f"  Sampling to {max_rows:,} rows...")
        chunks = []
        rows_loaded = 0
        sample_fraction = max_rows / total_rows

        for fragment in parquet_dataset.fragments:
            for batch in fragment.to_batches(batch_size=10_000):
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
                for batch in fragment.to_batches(batch_size=10_000):
                    chunks.append(batch.to_pandas())
            df = pd.concat(chunks, ignore_index=True)
            del chunks
        else:
            df = pd.read_parquet(output_path)
        print(f"  Loaded {len(df):,} rows")

    if sample_fraction is not None:
        if not (0 < sample_fraction <= 1.0):
            print("  ERROR: --sample-fraction must be in (0, 1]")
            sys.exit(1)
        print(f"  Subsampling to {sample_fraction:.2%} of rows...")
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        print(f"  Loaded {len(df):,} rows (fraction)")

    log_stage_complete("Loading Processed Features")
    log_memory_usage()

    # Verify event columns exist
    missing_events = [e for e in events if e not in df.columns]
    if missing_events:
        print(f"  WARNING: Missing event columns: {missing_events}")
        events = [e for e in events if e in df.columns]
        if not events:
            print("  ERROR: No valid event columns found!")
            sys.exit(1)

    # Verify required columns
    required_cols = ['sid', 'year']
    for col in required_cols:
        if col not in df.columns:
            print(f"  ERROR: Required column '{col}' not found!")
            sys.exit(1)

    # ================================================================
    # STAGE 2: Build vocabulary
    # ================================================================
    log_stage_start("Building Vocabulary")

    vocab_path = "checkpoints/sequence_vocab.joblib"
    os.makedirs("checkpoints", exist_ok=True)

    if reuse_processed and os.path.exists(vocab_path):
        print(f"  Loading existing vocabulary from {vocab_path}...")
        vocabulary = LifeEventVocabulary.load(vocab_path)
        missing_tokens = [
            f"{prefix}_Q1"
            for prefix in MUNICIPALITY_FEATURE_MAP.values()
            if vocabulary.token_to_id(f"{prefix}_Q1") == vocabulary.UNK
        ]
        if missing_tokens:
            print("  Vocabulary missing municipality tokens; rebuilding...")
            vocabulary = LifeEventVocabulary()
            vocabulary.build_from_dataframe(df)
            vocabulary.save(vocab_path)
    else:
        print("  Building vocabulary from data...")
        vocabulary = LifeEventVocabulary()
        vocabulary.build_from_dataframe(df)
        vocabulary.save(vocab_path)

    print(f"  Vocabulary size: {vocabulary.vocab_size} tokens")
    log_stage_complete("Building Vocabulary")

    # ================================================================
    # STAGE 3: Train/test split + create datasets
    # ================================================================
    log_stage_start("Creating Sequence Datasets")

    if 'year' in df.columns:
        train_df = df[df['year'] <= cutoff_year].copy()
        test_df = df[df['year'] > cutoff_year].copy()
        # But we need the FULL history for each person in training
        # (including years <= cutoff as input, and years > cutoff as targets)
        # So we pass the full dataframe to SequenceDataset
        print(f"  Train persons: history up to year {cutoff_year}")
        print(f"  Test: predict events in years {cutoff_year+1}-{df['year'].max()}")
    else:
        print("  ERROR: 'year' column required for time-based split!")
        sys.exit(1)

    # Get unique persons that have BOTH history and future data
    persons_with_history = set(df[df['year'] <= cutoff_year]['sid'].unique())
    persons_with_future = set(df[df['year'] > cutoff_year]['sid'].unique())
    valid_persons = persons_with_history & persons_with_future
    print(f"  Persons with history: {len(persons_with_history):,}")
    print(f"  Persons with future: {len(persons_with_future):,}")
    print(f"  Persons with both (used for training): {len(valid_persons):,}")

    # Filter to valid persons
    train_data = df[df['sid'].isin(valid_persons)].copy()
    del df
    gc.collect()

    # Create training dataset (uses all data, split handled by cutoff_year)
    train_dataset = SequenceDataset(
        df=train_data,
        vocabulary=vocabulary,
        max_seq_len=config.get('model', {}).get('params', {}).get('max_seq_len', 256) if config else 256,
        events=events,
        horizons=horizons,
        cutoff_year=cutoff_year,
    )
    print(f"  Train dataset: {len(train_dataset):,} persons")

    # Validation dataset: use a later cutoff to validate
    # Use cutoff_year-1 as validation cutoff, predicting into cutoff_year+
    val_cutoff = cutoff_year - 2
    persons_val_history = set(train_data[train_data['year'] <= val_cutoff]['sid'].unique())
    persons_val_future = set(train_data[train_data['year'] > val_cutoff]['sid'].unique())
    val_persons = persons_val_history & persons_val_future

    val_data = train_data[train_data['sid'].isin(val_persons)].copy()
    val_dataset = SequenceDataset(
        df=val_data,
        vocabulary=vocabulary,
        max_seq_len=config.get('model', {}).get('params', {}).get('max_seq_len', 256) if config else 256,
        events=events,
        horizons=horizons,
        cutoff_year=val_cutoff,
    )
    del val_data
    gc.collect()
    print(f"  Validation dataset: {len(val_dataset):,} persons (cutoff={val_cutoff})")

    eval_persons = valid_persons
    if eval_fraction is not None:
        if not (0 < eval_fraction <= 1.0):
            print("  ERROR: --eval-fraction must be in (0, 1]")
            sys.exit(1)
        rng = np.random.RandomState(42)
        n_eval = max(1, int(len(valid_persons) * eval_fraction))
        eval_persons = set(rng.choice(list(valid_persons), n_eval, replace=False))
        print(f"  Evaluation persons: {len(eval_persons):,} (fraction)")

    if eval_persons == valid_persons:
        # Test dataset: same persons, same cutoff as train, different purpose
        # (evaluation only)
        test_dataset = train_dataset
    else:
        eval_data = train_data[train_data['sid'].isin(eval_persons)].copy()
        test_dataset = SequenceDataset(
            df=eval_data,
            vocabulary=vocabulary,
            max_seq_len=config.get('model', {}).get('params', {}).get('max_seq_len', 256) if config else 256,
            events=events,
            horizons=horizons,
            cutoff_year=cutoff_year,
        )
        del eval_data
        gc.collect()

    log_stage_complete("Creating Sequence Datasets")
    log_memory_usage()

    # ================================================================
    # STAGE 4: Train model
    # ================================================================
    log_stage_start("Model Training")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("demographic_forecasts_sequence")

    model_config = config.get('model', {}) if config else {
        'type': 'seq_lstm',
        'params': {'encoder_type': 'lstm', 'embed_dim': 128}
    }

    estimator = PyTorchSequenceEstimator(
        model_config=model_config,
        device_config=None,
    )

    with mlflow.start_run(run_name=f"sequence_{encoder_type}_{'_'.join(events)}"):
        mlflow.log_params({
            'encoder_type': encoder_type,
            'embed_dim': model_config.get('params', {}).get('embed_dim', 128),
            'max_seq_len': model_config.get('params', {}).get('max_seq_len', 256),
            'vocab_size': vocabulary.vocab_size,
            'n_events': len(events),
            'events': ','.join(events),
            'horizons': str(horizons),
            'cutoff_year': cutoff_year,
            'train_persons': len(train_dataset),
            'val_persons': len(val_dataset),
        })

        result = estimator.fit(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            vocabulary=vocabulary,
            lr_find_plot_path=os.path.join("checkpoints", "lr_finder.png"),
        )

        log_stage_complete("Model Training")
        log_memory_usage()

        # ================================================================
        # STAGE 4b: Calibrate per-event/horizon thresholds on val set
        # ================================================================
        log_stage_start("Threshold Calibration")
        print("\n  Calibrating optimal F1 thresholds on validation set...")
        optimal_thresholds = estimator.calibrate_thresholds(
            dataset=val_dataset,
        )
        print(f"  Thresholds: mean={optimal_thresholds.mean():.3f}, "
              f"range=[{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}]")
        log_stage_complete("Threshold Calibration")

        # ================================================================
        # STAGE 4c: Probability calibration (isotonic/platt)
        # ================================================================
        cal_method = model_config.get('params', {}).get('calibration_method', None)
        if cal_method:
            log_stage_start("Probability Calibration")
            print(f"\n  Fitting {cal_method} probability calibrators on validation set...")
            calibrators = estimator.calibrate_probabilities(
                dataset=val_dataset, method=cal_method,
            )
            print(f"  Calibrated {len(calibrators)} event-horizon columns")
            log_stage_complete("Probability Calibration")
        else:
            print("\n  Probability calibration: skipped (calibration_method not set)")

        # ================================================================
        # STAGE 4d: Per-event sigma calibration (minimise CRPS)
        # ================================================================
        if estimator.loss_type == 'aft':
            log_stage_start("Sigma Calibration")
            print("\n  Calibrating per-event sigma scales to minimise CRPS...")
            sigma_scales = estimator.calibrate_sigma_per_event(
                dataset=val_dataset,
            )
            for ei, event in enumerate(events):
                s = sigma_scales.get(ei, 1.0)
                print(f"    {event}: sigma_scale = {s:.3f}")
            log_stage_complete("Sigma Calibration")

        # ================================================================
        # STAGE 5: Generate predictions
        # ================================================================
        log_stage_start("Prediction Generation")

        all_predictions = estimator.predict_as_dict(
            dataset=test_dataset,
        )

        print(f"\n  Generated predictions for {len(events)} events "
              f"at {len(horizons)} horizons")
        for event in events:
            if event in all_predictions:
                for h in horizons:
                    key = f'prob_{h}yr'
                    if key in all_predictions[event]:
                        probs = all_predictions[event][key]
                        print(f"    {event} @{h}yr: mean={probs.mean():.4f}, "
                              f"median={np.median(probs):.4f}, "
                              f"P>0.5={(probs > 0.5).mean():.3%}")

        log_stage_complete("Prediction Generation")
        log_memory_usage()

        # ================================================================
        # STAGE 6: Evaluate
        # ================================================================
        log_stage_start("Model Evaluation")

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Build test_df for evaluation (only future data for ground truth)
        test_eval_df = train_data[train_data['sid'].isin(eval_persons)].copy()

        metrics = evaluate_sequence_predictions(
            all_predictions=all_predictions,
            test_df=test_eval_df,
            events=events,
            horizons=horizons,
            cutoff_year=cutoff_year,
        )

        # Print and log results
        for event, event_metrics in metrics['per_event'].items():
            print(f"\n--- {event} ---")
            for horizon_key, h_metrics in event_metrics.items():
                auc = h_metrics.get('auc', float('nan'))
                ap = h_metrics.get('ap', float('nan'))
                f1 = h_metrics.get('f1', float('nan'))
                brier = h_metrics.get('brier', float('nan'))
                prev = h_metrics.get('prevalence', 0)

                print(f"  @{horizon_key}: AUC={auc:.4f}  AP={ap:.4f}  "
                      f"F1={f1:.4f}  Brier={brier:.4f}  "
                      f"prevalence={prev:.3%}")

                if not np.isnan(auc):
                    mlflow.log_metric(f"{event}_{horizon_key}_auc", auc)
                if not np.isnan(ap):
                    mlflow.log_metric(f"{event}_{horizon_key}_ap", ap)
                if not np.isnan(f1):
                    mlflow.log_metric(f"{event}_{horizon_key}_f1", f1)
                if not np.isnan(brier):
                    mlflow.log_metric(f"{event}_{horizon_key}_brier", brier)

        agg = metrics['aggregate']
        print(f"\n  Aggregate: mean_AUC={agg['mean_auc']:.4f}, "
              f"mean_AP={agg['mean_ap']:.4f}")
        mlflow.log_metric("aggregate_mean_auc", agg['mean_auc'])
        mlflow.log_metric("aggregate_mean_ap", agg['mean_ap'])

        log_stage_complete("Model Evaluation")
        log_memory_usage()

        # ================================================================
        # STAGE 6b: Survival metrics
        # ================================================================
        log_stage_start("Survival Metrics")

        # Collect targets once (shared by both metric types)
        from torch.utils.data import DataLoader
        from src.sequence.dataset import sequence_collate_fn

        test_loader = DataLoader(
            test_dataset, batch_size=estimator.batch_size * 2,
            shuffle=False, collate_fn=sequence_collate_fn, drop_last=False,
        )
        test_targets_list = []
        for batch in test_loader:
            test_targets_list.append(batch['targets'].numpy())
        test_targets = np.concatenate(test_targets_list, axis=0)

        # Survival metrics: AFT uses parametric distribution, DeepHit uses learned CDF
        loss_type = model_config.get('params', {}).get('loss_type', 'bce')
        if loss_type == 'aft':
            print("\n  Computing AFT-native survival metrics (C-index, CRPS, IBS, TD-AUC)...")
            try:
                from src.sequence.evaluation import evaluate_aft_survival_metrics

                aft_result = estimator.predict_aft_params(dataset=test_dataset)
                aft_surv_metrics = evaluate_aft_survival_metrics(
                    aft_params=aft_result.probabilities,
                    targets=test_targets,
                    events=events,
                    horizons=horizons,
                )

                if aft_surv_metrics and 'per_event' in aft_surv_metrics:
                    print("\n  AFT survival metrics (distribution-native):")
                    for event, ev_metrics in aft_surv_metrics['per_event'].items():
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ev_metrics.items())
                        print(f"    {event}: {metrics_str}")
                        for k, v in ev_metrics.items():
                            if isinstance(v, float) and not np.isnan(v):
                                mlflow.log_metric(f"aft_{event}_{k}", v)

                    aft_agg = aft_surv_metrics.get('aggregate', {})
                    if aft_agg:
                        print(f"\n  AFT aggregate: "
                              f"C-index={aft_agg.get('mean_c_index', float('nan')):.4f}, "
                              f"CRPS={aft_agg.get('mean_crps', float('nan')):.4f}, "
                              f"IBS={aft_agg.get('mean_ibs', float('nan')):.4f}, "
                              f"TD-AUC={aft_agg.get('mean_td_auc', float('nan')):.4f}")
                        for k, v in aft_agg.items():
                            if not np.isnan(v):
                                mlflow.log_metric(f"aft_{k}", v)
                else:
                    print("  No AFT survival metrics computed")
            except Exception as e:
                print(f"  AFT survival metrics failed: {e}")
                import traceback; traceback.print_exc()

        elif loss_type == 'deephit':
            print("\n  Computing DeepHit survival metrics (C-index, TD-AUC)...")
            try:
                import torch
                from src.sequence.evaluation import evaluate_survival_metrics, _targets_to_survival, _fast_c_index

                proba_result = estimator.predict_proba(dataset=test_dataset)
                probs = proba_result.probabilities  # (N, n_events * n_horizons) CDF values

                # Standard survival metrics (TD-AUC, IPCW-Brier)
                surv_metrics = evaluate_survival_metrics(
                    probabilities=probs,
                    targets=test_targets,
                    events=events,
                    horizons=horizons,
                )

                # DeepHit C-index: use CDF at max horizon as risk score
                n_events = len(events)
                n_horizons = len(horizons)
                probs_3d = probs.reshape(-1, n_events, n_horizons)  # (N, E, H)

                dh_per_event = {}
                c_indices = []
                for ei, event in enumerate(events):
                    risk_score = probs_3d[:, ei, -1]  # CDF at max horizon
                    durations, event_obs = _targets_to_survival(
                        test_targets[:, ei * n_horizons:(ei + 1) * n_horizons],
                        horizons,
                    )
                    ci = _fast_c_index(event_obs, durations, risk_score)
                    dh_per_event[event] = {'c_index': ci}
                    if not np.isnan(ci):
                        c_indices.append(ci)

                # Merge with standard survival metrics
                if surv_metrics and 'per_event' in surv_metrics:
                    for event in events:
                        if event in surv_metrics['per_event']:
                            dh_per_event.setdefault(event, {}).update(surv_metrics['per_event'][event])

                print("\n  DeepHit survival metrics:")
                for event, ev_metrics in dh_per_event.items():
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ev_metrics.items())
                    print(f"    {event}: {metrics_str}")
                    for k, v in ev_metrics.items():
                        if isinstance(v, float) and not np.isnan(v):
                            mlflow.log_metric(f"dh_{event}_{k}", v)

                mean_ci = np.mean(c_indices) if c_indices else float('nan')
                surv_agg = surv_metrics.get('aggregate', {}) if surv_metrics else {}
                print(f"\n  DeepHit aggregate: "
                      f"C-index={mean_ci:.4f}, "
                      f"mean_TD-AUC={surv_agg.get('mean_td_auc', float('nan')):.4f}")
                if not np.isnan(mean_ci):
                    mlflow.log_metric("dh_mean_c_index", mean_ci)
                for k, v in surv_agg.items():
                    if not np.isnan(v):
                        mlflow.log_metric(f"dh_{k}", v)

            except Exception as e:
                print(f"  DeepHit survival metrics failed: {e}")
                import traceback; traceback.print_exc()

        else:
            print("\n  Computing time-dependent survival metrics...")
            try:
                from src.sequence.evaluation import evaluate_survival_metrics

                proba_result = estimator.predict_proba(dataset=test_dataset)
                surv_metrics = evaluate_survival_metrics(
                    probabilities=proba_result.probabilities,
                    targets=test_targets,
                    events=events,
                    horizons=horizons,
                )

                if surv_metrics and 'per_event' in surv_metrics:
                    print("\n  Survival metrics:")
                    for event, ev_metrics in surv_metrics['per_event'].items():
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ev_metrics.items())
                        print(f"    {event}: {metrics_str}")
                        for k, v in ev_metrics.items():
                            if not np.isnan(v):
                                mlflow.log_metric(f"surv_{event}_{k}", v)

                    surv_agg = surv_metrics.get('aggregate', {})
                    if surv_agg:
                        print(f"\n  Survival aggregate: "
                              f"mean_TD-AUC={surv_agg.get('mean_td_auc', float('nan')):.4f}, "
                              f"mean_IPCW_Brier={surv_agg.get('mean_ipcw_brier', float('nan')):.4f}")
                        for k, v in surv_agg.items():
                            if not np.isnan(v):
                                mlflow.log_metric(f"surv_{k}", v)
                else:
                    print("  No survival metrics computed")
            except Exception as e:
                print(f"  Survival metrics failed: {e}")

        log_stage_complete("Survival Metrics")

        # ================================================================
        # STAGE 6c: MC Dropout uncertainty quantification
        # ================================================================
        mc_samples = int(model_config.get('params', {}).get('mc_dropout_samples', 0))
        if mc_samples > 0:
            log_stage_start("MC Dropout UQ")
            print(f"\n  Running MC Dropout with {mc_samples} forward passes...")
            mc_result = estimator.predict_proba_mc(
                dataset=test_dataset, n_samples=mc_samples,
            )
            mc_std = mc_result.metadata.get('mc_std')
            if mc_std is not None:
                print(f"  Mean uncertainty (std): {mc_std.mean():.4f}")
                print(f"  Per-column uncertainty range: "
                      f"[{mc_std.mean(axis=0).min():.4f}, {mc_std.mean(axis=0).max():.4f}]")
                mlflow.log_metric("mc_dropout_mean_std", float(mc_std.mean()))
            log_stage_complete("MC Dropout UQ")
        else:
            print("\n  MC Dropout: skipped (mc_dropout_samples=0)")

        # ================================================================
        # STAGE 6e: Prediction intervals (AFT only)
        # ================================================================
        prediction_intervals = None
        if loss_type == 'aft':
            log_stage_start("Prediction Intervals")
            print("\n  Computing prediction intervals from AFT distribution...")
            try:
                prediction_intervals = estimator.predict_intervals(
                    dataset=test_dataset,
                    confidence_levels=[0.5, 0.8, 0.9],
                )
                for event in events:
                    if event in prediction_intervals:
                        ei = prediction_intervals[event]
                        med = ei['median']
                        ci80_lo = ei['ci_80_lower']
                        ci80_hi = ei['ci_80_upper']
                        print(f"    {event}: median={np.median(med):.2f}yr, "
                              f"80%CI=[{np.median(ci80_lo):.2f}, {np.median(ci80_hi):.2f}]yr")
                        for h in horizons:
                            p = ei[f'prob_{h}yr']
                            print(f"      P(event<={h}yr): mean={p.mean():.4f}, "
                                  f"median={np.median(p):.4f}")
                        mlflow.log_metric(f"aft_{event}_median_tte",
                                          float(np.median(med)))
                        mlflow.log_metric(f"aft_{event}_ci80_width",
                                          float(np.median(ci80_hi - ci80_lo)))
            except Exception as e:
                print(f"  Prediction intervals failed: {e}")
                import traceback; traceback.print_exc()
            log_stage_complete("Prediction Intervals")
        else:
            print("\n  Prediction intervals: skipped (requires loss_type='aft')")

        # ================================================================
        # STAGE 6f: Event ordering evaluation (AFT or DeepHit)
        # ================================================================
        if loss_type in ('aft', 'deephit'):
            log_stage_start("Event Ordering")
            print("\n  Evaluating event ordering (pairwise, top-1/k, Kendall's tau)...")
            try:
                from src.sequence.evaluation import evaluate_event_ordering

                n_ev = len(events)
                n_h = len(horizons)

                if loss_type == 'aft':
                    aft_params = aft_result.probabilities
                    pred_mu = aft_params[:, :n_ev]
                else:
                    # DeepHit: derive pseudo-mu from learned CDF
                    # Higher CDF at max horizon → event expected sooner → lower pseudo-mu
                    if 'proba_result' not in dir():
                        proba_result = estimator.predict_proba(dataset=test_dataset)
                    probs_3d = proba_result.probabilities.reshape(-1, n_ev, n_h)
                    cdf_max = probs_3d[:, :, -1]  # (N, E) CDF at max horizon
                    # Negative log so higher CDF → lower mu (earlier event)
                    pred_mu = -np.log(np.clip(cdf_max, 1e-7, 1.0))

                ordering = evaluate_event_ordering(
                    predicted_mu=pred_mu,
                    targets=test_targets,
                    events=events,
                    horizons=horizons,
                    min_events=2,
                    top_k=3,
                )

                n_elig = ordering['n_eligible']
                print(f"  Persons with 2+ events: {n_elig} "
                      f"({100*n_elig/len(test_targets):.1f}% of test set)")
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
                    for event, rate in ordering['per_event_first_rate'].items():
                        pred_rate = ordering['per_event_pred_first_rate'].get(event, 0)
                        print(f"    {event}: observed={rate:.3f}, predicted={pred_rate:.3f}")
                        mlflow.log_metric(f"ordering_obs_first_{event}", rate)
                        mlflow.log_metric(f"ordering_pred_first_{event}", pred_rate)
                else:
                    print("  No persons with 2+ events — skipping ordering metrics")
            except Exception as e:
                print(f"  Event ordering evaluation failed: {e}")
                import traceback; traceback.print_exc()
            log_stage_complete("Event Ordering")
        else:
            print("\n  Event ordering: skipped (requires loss_type='aft' or 'deephit')")

        # ================================================================
        # STAGE 6g: Language-model-style metrics (all loss types)
        # ================================================================
        log_stage_start("LM Metrics")
        print("\n  Evaluating language-model-style metrics (next-event accuracy, perplexity, temporal)...")
        try:
            from src.sequence.evaluation import evaluate_lm_metrics

            if 'proba_result' not in dir():
                proba_result = estimator.predict_proba(dataset=test_dataset)

            lm_results = evaluate_lm_metrics(
                probabilities=proba_result.probabilities,
                targets=test_targets,
                events=events,
                horizons=horizons,
                top_k=3,
            )

            n_w = lm_results['n_with_events']
            n_t = lm_results['n_total']
            print(f"  Persons with events: {n_w} ({100*n_w/max(n_t,1):.1f}% of test set)")

            if n_w > 0:
                print(f"  Next-event top-1 accuracy:     {lm_results['next_event_top1_accuracy']:.4f}")
                print(f"  Next-event top-3 accuracy:     {lm_results['next_event_topk_accuracy']:.4f}")
                print(f"  Next-event MRR:                {lm_results['next_event_mrr']:.4f}")
                print(f"  Event perplexity:              {lm_results['event_perplexity']:.4f}")
                print(f"  Multi-label perplexity:        {lm_results['multi_label_perplexity']:.4f}")
                print(f"  Temporal consistency:           {lm_results['temporal_consistency']:.4f}")
                print(f"  Cross-horizon rank stability:  {lm_results['cross_horizon_rank_stability']:.4f}")

                # Log scalar metrics to MLflow
                for lm_key in ['next_event_top1_accuracy', 'next_event_topk_accuracy',
                               'next_event_mrr', 'event_perplexity', 'multi_label_perplexity',
                               'temporal_consistency', 'cross_horizon_rank_stability']:
                    v = lm_results[lm_key]
                    if isinstance(v, float) and not np.isnan(v):
                        mlflow.log_metric(f"lm_{lm_key}", v)

                # Per-horizon breakdown
                print("\n  Per-horizon accuracy:")
                for h_key, h_vals in lm_results['per_horizon'].items():
                    h_n = h_vals['n_with_events']
                    h_t1 = h_vals['top1_accuracy']
                    h_tk = h_vals['topk_accuracy']
                    if h_n > 0:
                        print(f"    {h_key}: top1={h_t1:.4f}, top3={h_tk:.4f} (n={h_n})")
                        mlflow.log_metric(f"lm_top1_{h_key}", h_t1)
                        mlflow.log_metric(f"lm_top3_{h_key}", h_tk)

                # Per-event recall/precision
                print("\n  Per-event next-event recall / precision:")
                for event in events:
                    rec = lm_results['per_event_recall'].get(event, float('nan'))
                    prec = lm_results['per_event_precision'].get(event, float('nan'))
                    rec_s = f"{rec:.3f}" if not np.isnan(rec) else "n/a"
                    prec_s = f"{prec:.3f}" if not np.isnan(prec) else "n/a"
                    print(f"    {event}: recall={rec_s}, precision={prec_s}")
                    if not np.isnan(rec):
                        mlflow.log_metric(f"lm_recall_{event}", rec)
                    if not np.isnan(prec):
                        mlflow.log_metric(f"lm_precision_{event}", prec)
            else:
                print("  No persons with events — skipping LM metrics")
        except Exception as e:
            print(f"  LM metrics evaluation failed: {e}")
            import traceback; traceback.print_exc()
        log_stage_complete("LM Metrics")

        # ================================================================
        # STAGE 6h: Grouped evaluation (per age-group / municipality)
        # ================================================================
        log_stage_start("Grouped Evaluation")
        print("\n  Evaluating metrics per demographic group (age, municipality)...")
        try:
            from src.sequence.evaluation import evaluate_grouped_metrics

            if 'proba_result' not in dir():
                proba_result = estimator.predict_proba(dataset=test_dataset)
            sids_g = proba_result.metadata.get('sids')
            if sids_g is None and hasattr(test_dataset, '_sids'):
                sids_g = test_dataset._sids

            # Detect available group columns
            schema_cols = set(pq.ParquetDataset(output_path).schema.names)
            available_group_cols = [c for c in ['age_group', 'refnis'] if c in schema_cols]

            if available_group_cols and sids_g is not None:
                group_features_df = _load_group_features_streaming(
                    parquet_path=output_path,
                    sids=sids_g,
                    cutoff_year=test_cutoff,
                    group_cols=available_group_cols,
                )
                print(f"  Loaded group features for {len(group_features_df):,} persons "
                      f"({', '.join(available_group_cols)})")

                for group_col in available_group_cols:
                    if group_col not in group_features_df.columns:
                        continue

                    print(f"\n{'=' * 60}")
                    print(f"GROUPED EVALUATION: {group_col}")
                    print(f"{'=' * 60}")

                    g_result = evaluate_grouped_metrics(
                        probabilities=proba_result.probabilities,
                        targets=test_targets,
                        group_labels=group_features_df[group_col].values,
                        events=events,
                        horizons=horizons,
                        group_name=group_col,
                        min_group_size=50 if group_col == 'age_group' else 100,
                        calibrate_thresholds=True,
                    )

                    gt = g_result['group_table']
                    if gt.empty:
                        print(f"  No groups with sufficient size — skipping")
                        continue

                    summary = g_result['summary']
                    print(f"  {summary['n_groups']} groups, {summary['total_persons']:,} persons")

                    # Print per-event MAE/RMSE summary
                    print(f"\n  Rate MAE (predicted mean vs observed rate):")
                    for ei, event in enumerate(events):
                        for h in horizons:
                            prefix = f'{event}_{h}yr'
                            mae_w = summary.get(f'{prefix}_mae_weighted')
                            cal_mae_w = summary.get(f'{prefix}_cal_mae_weighted')
                            if mae_w is not None:
                                line = f"    {prefix}: MAE={mae_w:.4f}"
                                if cal_mae_w is not None:
                                    line += f"  cal_MAE={cal_mae_w:.4f}"
                                print(line)
                                mlflow.log_metric(f"grp_{group_col}_{prefix}_mae", mae_w)
                                if cal_mae_w is not None:
                                    mlflow.log_metric(f"grp_{group_col}_{prefix}_cal_mae", cal_mae_w)

                    # LM metrics per group
                    lm_sum = g_result['lm_summary']
                    if lm_sum:
                        print(f"\n  LM metrics across {group_col} groups:")
                        for lm_key in ['lm_top1_acc', 'lm_top3_acc', 'lm_mrr', 'lm_perplexity']:
                            w = lm_sum.get(f'{lm_key}_weighted')
                            s = lm_sum.get(f'{lm_key}_std')
                            if w is not None:
                                print(f"    {lm_key}: {w:.4f} (std={s:.4f})")
                                mlflow.log_metric(f"grp_{group_col}_{lm_key}", w)

                    # Print group table (top/bottom groups by LM top-1)
                    if 'lm_top1_acc' in gt.columns and not gt['lm_top1_acc'].isna().all():
                        sorted_gt = gt.dropna(subset=['lm_top1_acc']).sort_values('lm_top1_acc')
                        n_show = min(5, len(sorted_gt))
                        if n_show > 0:
                            print(f"\n  Bottom {n_show} groups by LM top-1 accuracy:")
                            for _, r in sorted_gt.head(n_show).iterrows():
                                print(f"    {group_col}={r[group_col]}: n={int(r['count'])}, "
                                      f"top1={r['lm_top1_acc']:.3f}, top3={r['lm_top3_acc']:.3f}, "
                                      f"ppl={r['lm_perplexity']:.2f}")
                            print(f"\n  Top {n_show} groups by LM top-1 accuracy:")
                            for _, r in sorted_gt.tail(n_show).iterrows():
                                print(f"    {group_col}={r[group_col]}: n={int(r['count'])}, "
                                      f"top1={r['lm_top1_acc']:.3f}, top3={r['lm_top3_acc']:.3f}, "
                                      f"ppl={r['lm_perplexity']:.2f}")

                    # Save full group table as artifact
                    group_table_path = f"checkpoints/grouped_metrics_{group_col}.csv"
                    os.makedirs("checkpoints", exist_ok=True)
                    gt.to_csv(group_table_path, index=False)
                    mlflow.log_artifact(group_table_path)
                    print(f"\n  Full group table saved to {group_table_path}")

                del group_features_df
            else:
                print("  No group columns available or no sids — skipping grouped evaluation")
        except Exception as e:
            print(f"  Grouped evaluation failed: {e}")
            import traceback; traceback.print_exc()
        log_stage_complete("Grouped Evaluation")

        # ================================================================
        # STAGE 7: Save artifacts
        # ================================================================
        log_stage_start("Saving Artifacts")

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"sequence_{encoder_type}_{timestamp}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model
        estimator.save(checkpoint_path)
        print(f"  Saved model: {checkpoint_path}")

        # Save predictions as parquet (including intervals if available)
        pred_records = []
        person_ids = train_dataset._person_ids
        for i, sid in enumerate(person_ids):
            record = {'sid': sid}
            for event in events:
                if event in all_predictions:
                    for h in horizons:
                        key = f'prob_{h}yr'
                        if key in all_predictions[event]:
                            record[f'{event}_prob_{h}yr'] = float(
                                all_predictions[event][key][i]
                            )
                # Add prediction intervals
                if prediction_intervals and event in prediction_intervals:
                    ei = prediction_intervals[event]
                    record[f'{event}_median_tte'] = float(ei['median'][i])
                    for level in [50, 80, 90]:
                        lo_key = f'ci_{level}_lower'
                        hi_key = f'ci_{level}_upper'
                        if lo_key in ei:
                            record[f'{event}_ci{level}_lower'] = float(ei[lo_key][i])
                            record[f'{event}_ci{level}_upper'] = float(ei[hi_key][i])
            pred_records.append(record)

        pred_df = pd.DataFrame(pred_records)
        pred_path = os.path.join(checkpoint_path, "predictions.parquet")
        pred_df.to_parquet(pred_path)
        print(f"  Saved predictions: {pred_path}")

        # Save metadata
        serializable_metrics = {}
        for event, em in metrics['per_event'].items():
            serializable_metrics[event] = {}
            for hk, hm in em.items():
                serializable_metrics[event][hk] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in hm.items()
                }

        metadata = {
            'timestamp': timestamp,
            'model_type': 'sequence',
            'encoder_type': encoder_type,
            'events': events,
            'horizons': horizons,
            'cutoff_year': cutoff_year,
            'vocab_size': vocabulary.vocab_size,
            'train_persons': len(train_dataset),
            'val_persons': len(val_dataset),
            'epochs_trained': result.metadata.get('epochs_trained', 0),
            'n_params': result.metadata.get('n_params', 0),
            'metrics': serializable_metrics,
            'aggregate': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in agg.items()
            },
            'config_path': config_path,
        }
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        mlflow.log_artifacts(checkpoint_path, artifact_path="sequence_checkpoint")

        log_stage_complete("Saving Artifacts")

    # ================================================================
    # Final summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SEQUENCE MODEL COMPLETE")
    print("=" * 60)
    for event in events:
        if event in metrics['per_event']:
            aucs = []
            for hk, hm in metrics['per_event'][event].items():
                a = hm.get('auc', float('nan'))
                aucs.append(f"{hk}={a:.3f}" if not np.isnan(a) else f"{hk}=N/A")
            print(f"  {event}: AUC: {', '.join(aucs)}")
    print(f"\n  Aggregate: mean_AUC={agg['mean_auc']:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)

    return estimator, metrics, pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sequence model pipeline for demographic event prediction"
    )
    parser.add_argument('--reuse', '-r', action='store_true',
                        help='Reuse existing processed features')
    parser.add_argument('--max-rows', type=int,
                        help='Limit dataset to N rows')
    parser.add_argument('--config', type=str,
                        help='Path to sequence model config YAML')
    parser.add_argument('--events', type=str,
                        help='Comma-separated event names (default: all)')
    parser.add_argument('--horizons', type=str, default='1,3,5',
                        help='Comma-separated prediction horizons in years')
    parser.add_argument('--cutoff-year', type=int, default=2022,
                        help='Year cutoff for train/test split')
    parser.add_argument('--streaming', action='store_true',
                        help='Force streaming mode for large datasets')
    parser.add_argument('--sample-fraction', type=float,
                        help='Use a fraction of persons/rows for faster prototyping')
    parser.add_argument('--eval-fraction', type=float,
                        help='Use a fraction of persons for evaluation/prediction')
    parser.add_argument('--tune', action='store_true',
                        help='Run Optuna hyperparameter tuning instead of training')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna tuning trials (default: 50)')
    parser.add_argument('--tuning-epochs', type=int, default=15,
                        help='Max epochs per trial during tuning (default: 15)')
    parser.add_argument('--search-space', type=str, default='conservative',
                        choices=['full', 'conservative'],
                        help='Search space breadth for tuning (default: conservative)')
    parser.add_argument('--tuning-seed', type=int, default=42,
                        help='Random seed for Optuna TPE sampler (default: 42). '
                             'Use different seeds to run parallel jobs without overlap.')
    parser.add_argument('--tuning-workers', type=int, default=1,
                        help='Number of parallel Optuna workers within one job (default: 1). '
                             'Workers share the GPU, so use 2+ only with small models (GRU embed=16-32).')
    parser.add_argument('--tuning-fraction', type=float, default=0.3,
                        help='Fraction of training data per tuning trial (default: 0.3). '
                             'Each trial gets a different random subset. Reduces GPU memory '
                             'and speeds up trials, enabling parallel workers.')
    # Rolling window arguments
    parser.add_argument('--rolling', action='store_true',
                        help='Use rolling window validation (multiple cutoff years)')
    parser.add_argument('--history-len', type=int, default=5,
                        help='History window length in years for rolling mode (default: 5)')
    parser.add_argument('--rolling-cutoffs', type=str,
                        help='Comma-separated cutoff years for rolling windows. '
                             'Auto-detected from data if not specified.')
    parser.add_argument('--rolling-train-cutoffs', type=str,
                        help='Comma-separated cutoffs for training (default: all except last 2)')
    parser.add_argument('--rolling-val-cutoff', type=int,
                        help='Cutoff year for validation window (default: second-to-last)')
    parser.add_argument('--rolling-test-cutoff', type=int,
                        help='Cutoff year for test window (default: last)')
    parser.add_argument('--existing-cache', type=str,
                        help='Path to existing single-cutoff cache directory with '
                             'per-year shards. Reuses these to build rolling caches '
                             'without re-reading raw parquet.')

    args = parser.parse_args()

    events = None
    if args.events:
        events = [e.strip() for e in args.events.split(',')]

    horizons = [int(h.strip()) for h in args.horizons.split(',')]

    print("=" * 60)
    print("SEQUENCE MODEL PIPELINE")
    print("=" * 60)
    if args.reuse:
        print("  Reuse: Will use existing processed features")
    if args.max_rows:
        print(f"  Max rows: {args.max_rows:,}")
    if args.config:
        print(f"  Config: {args.config}")
    if events:
        print(f"  Events: {', '.join(events)}")
    if args.streaming:
        print("  Streaming: enabled")
    if args.sample_fraction is not None:
        print(f"  Sample fraction: {args.sample_fraction:.2%}")
    if args.eval_fraction is not None:
        print(f"  Eval fraction: {args.eval_fraction:.2%}")
    if args.tune:
        print(f"  Tuning: {args.n_trials} trials, {args.tuning_epochs} epochs/trial, {args.search_space} space, seed={args.tuning_seed}, fraction={args.tuning_fraction:.0%}, workers={args.tuning_workers}")
    if args.rolling:
        print(f"  Rolling: history_len={args.history_len}")
        if args.rolling_cutoffs:
            print(f"  Rolling cutoffs: {args.rolling_cutoffs}")
        if args.rolling_train_cutoffs:
            print(f"  Train cutoffs: {args.rolling_train_cutoffs}")
        if args.rolling_val_cutoff:
            print(f"  Val cutoff: {args.rolling_val_cutoff}")
        if args.rolling_test_cutoff:
            print(f"  Test cutoff: {args.rolling_test_cutoff}")
    print(f"  Horizons: {horizons} years")
    print(f"  Cutoff year: {args.cutoff_year}")
    print("=" * 60)
    print()

    # Parse rolling cutoff lists
    rolling_cutoffs = None
    if args.rolling_cutoffs:
        rolling_cutoffs = sorted([int(c.strip()) for c in args.rolling_cutoffs.split(',')])
    rolling_train_cutoffs = None
    if args.rolling_train_cutoffs:
        rolling_train_cutoffs = sorted([int(c.strip()) for c in args.rolling_train_cutoffs.split(',')])

    main_sequence(
        reuse_processed=args.reuse,
        max_rows=args.max_rows,
        config_path=args.config,
        events=events,
        horizons=horizons,
        cutoff_year=args.cutoff_year,
        streaming=args.streaming,
        sample_fraction=args.sample_fraction,
        eval_fraction=args.eval_fraction,
        tune=args.tune,
        n_trials=args.n_trials,
        tuning_epochs=args.tuning_epochs,
        search_space=args.search_space,
        tuning_seed=args.tuning_seed,
        tuning_workers=args.tuning_workers,
        tuning_fraction=args.tuning_fraction,
        rolling=args.rolling,
        history_len=args.history_len,
        rolling_cutoffs=rolling_cutoffs,
        rolling_train_cutoffs=rolling_train_cutoffs,
        rolling_val_cutoff=args.rolling_val_cutoff,
        rolling_test_cutoff=args.rolling_test_cutoff,
        existing_cache=args.existing_cache,
    )
