"""
Resume per-window evaluation from a saved checkpoint.

Loads a trained model from a checkpoint directory and runs the
per-window stability analysis that was skipped or failed during training.
Includes per-group (age_group, refnis) metrics per window.

Usage:
    python eval_from_checkpoint.py \
        --checkpoint checkpoints/rolling_gru_20260223_150054 \
        --config configs/models/pytorch_seq_gru_fast_deephit.yaml
"""
import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.sequence.estimator import PyTorchSequenceEstimator
from src.sequence.dataset import CachedSequenceDataset
from src.sequence.vocabulary import LifeEventVocabulary


def _build_predictions_from_probs(probs, events, horizons):
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


def _compute_labels_streaming(parquet_path, sids, events, horizons, cutoff_year, chunk_size=500_000):
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


def _evaluate_from_arrays(all_predictions, y_true, events, horizons):
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


def _load_group_features_streaming(parquet_path, sids, cutoff_year, group_cols,
                                    chunk_size=500_000):
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

    best_df = None

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=columns):
            df = batch.to_pandas()
            df = df[(df['sid'].isin(sid_set)) & (df['year'] <= cutoff_year)]
            if df.empty:
                del df
                continue

            if best_df is None:
                best_df = df
            else:
                best_df = pd.concat([best_df, df], ignore_index=True)
            del df

            # Periodically deduplicate to keep memory bounded
            if best_df is not None and len(best_df) > 2_000_000:
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
    result = sid_order.merge(best_df, on='sid', how='left').sort_values(
        '_order').drop(columns='_order').reset_index(drop=True)
    del best_df, sid_order
    return result


def _evaluate_groups(group_df, all_predictions, y_true, events, horizons,
                     group_cols, min_group_size=100):
    """Compute group-level observed vs predicted rates per event-horizon.

    Returns dict with per-event group DataFrames and RMSE/MAE summaries.
    """
    n_horizons = len(horizons)
    results = {}

    for ei, event in enumerate(events):
        if event not in all_predictions:
            continue

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
            else:
                for i, col in enumerate(group_cols):
                    row[col] = group_key[i]
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

            group_rows.append(row)

        if not group_rows:
            results[event] = {'group_df': pd.DataFrame(), 'summary': {}}
            # Clean up temp columns
            for h in horizons:
                group_df.drop(columns=[f'pred_{h}yr', f'true_{h}yr'],
                              errors='ignore', inplace=True)
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

        results[event] = {'group_df': event_group_df, 'summary': summary}

        # Clean up temp columns
        for h in horizons:
            group_df.drop(columns=[f'pred_{h}yr', f'true_{h}yr'],
                          errors='ignore', inplace=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model from checkpoint on rolling windows")
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint directory')
    parser.add_argument('--config', type=str, help='Path to model config YAML (for max_seq_len)')
    parser.add_argument('--data-path', type=str, default='data/processed_features_with_municipality',
                        help='Path to processed parquet data')
    parser.add_argument('--cache-dir', type=str, default='checkpoints/sequence_cache',
                        help='Path to cached sequence datasets')
    parser.add_argument('--history-len', type=int, default=5, help='History window length')
    parser.add_argument('--cutoffs', type=str, help='Comma-separated cutoff years to evaluate')
    parser.add_argument('--min-group-size', type=int, default=100,
                        help='Minimum group size for group-level evaluation')
    args = parser.parse_args()

    # Load metadata
    meta_path = os.path.join(args.checkpoint, 'metadata.json')
    with open(meta_path) as f:
        metadata = json.load(f)

    events = metadata['events']
    horizons = metadata['horizons']
    history_len = metadata.get('history_len', args.history_len)
    train_cutoffs = metadata.get('train_cutoffs', [])
    val_cutoff = metadata.get('val_cutoff')
    test_cutoff = metadata.get('test_cutoff')

    print("=" * 60)
    print("EVALUATION FROM CHECKPOINT")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Events: {events}")
    print(f"  Horizons: {horizons}")
    print(f"  History len: {history_len}")
    print(f"  Train cutoffs: {train_cutoffs}")
    print(f"  Val cutoff: {val_cutoff}")
    print(f"  Test cutoff: {test_cutoff}")

    # Determine cutoffs to evaluate
    if args.cutoffs:
        rolling_cutoffs = [int(c.strip()) for c in args.cutoffs.split(',')]
    else:
        rolling_cutoffs = sorted(set(train_cutoffs + ([val_cutoff] if val_cutoff else [])
                                     + ([test_cutoff] if test_cutoff else [])))

    print(f"  Evaluating cutoffs: {rolling_cutoffs}")
    print("=" * 60)

    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    estimator = PyTorchSequenceEstimator.load(args.checkpoint, device='cuda')
    vocabulary = estimator.vocabulary_
    print(f"  Model loaded: {estimator.encoder_type}, vocab_size={vocabulary.vocab_size}")

    # Get max_seq_len from config if provided
    max_seq_len = estimator.max_seq_len
    if args.config:
        from src.config_loader import load_yaml
        config = load_yaml(args.config)
        max_seq_len = config.get('model', {}).get('params', {}).get('max_seq_len', max_seq_len)
    print(f"  max_seq_len: {max_seq_len}")

    output_path = args.data_path

    # Detect available group columns
    schema_cols = set(pq.ParquetDataset(output_path).schema.names)
    group_cols = [c for c in ['gender', 'age_group', 'refnis'] if c in schema_cols]
    print(f"  Group columns: {group_cols if group_cols else 'none found'}")

    cache_dir = args.cache_dir

    # Evaluate each window
    print("\nPer-window evaluation:")
    print("-" * 60)
    all_window_metrics = {}

    for cutoff in rolling_cutoffs:
        # CachedSequenceDataset expects cache_path as a .pt path;
        # it automatically checks for the _chunks directory variant.
        cache_pt = os.path.join(cache_dir, f"rolling_cut{cutoff}_h{history_len}.pt")
        cache_chunks = cache_pt.replace('.pt', '_chunks')
        if not os.path.isfile(cache_pt) and not os.path.isdir(cache_chunks):
            print(f"  cutoff={cutoff}: No cache found, skipping")
            continue

        try:
            source = cache_pt if os.path.isfile(cache_pt) else cache_chunks
            print(f"\n  cutoff={cutoff}: Loading from {source}...")
            window_ds = CachedSequenceDataset(
                parquet_path=output_path,
                vocabulary=vocabulary,
                max_seq_len=max_seq_len,
                events=events,
                horizons=horizons,
                cutoff_year=cutoff,
                cache_path=cache_pt,
                min_history_year=cutoff - history_len,
                return_ids=True,
            )

            print(f"    Dataset: {len(window_ds):,} persons")

            window_proba = estimator.predict_proba(dataset=window_ds)
            window_preds = _build_predictions_from_probs(
                window_proba.probabilities, events, horizons,
            )
            window_sids = window_proba.metadata.get('sids')
            if window_sids is None and hasattr(window_ds, '_sids') and window_ds._sids is not None:
                window_sids = window_ds._sids

            if window_sids is None:
                print(f"    WARNING: No sids available, skipping label computation")
                continue

            print(f"    Computing ground truth labels...")
            window_labels = _compute_labels_streaming(
                parquet_path=output_path,
                sids=window_sids,
                events=events,
                horizons=horizons,
                cutoff_year=cutoff,
            )

            # ---- Aggregate metrics ----
            window_metrics = _evaluate_from_arrays(
                all_predictions=window_preds,
                y_true=window_labels,
                events=events,
                horizons=horizons,
            )
            w_agg = window_metrics['aggregate']
            role = "TRAIN" if cutoff in train_cutoffs else (
                "VAL" if cutoff == val_cutoff else "TEST")
            print(f"    [{role}] mean_AUC={w_agg['mean_auc']:.4f}, "
                  f"mean_AP={w_agg['mean_ap']:.4f}")

            # Per-event detail
            for event, em in window_metrics['per_event'].items():
                for hk, hm in em.items():
                    print(f"      {event}@{hk}: AUC={hm['auc']:.4f}, "
                          f"AP={hm['ap']:.4f}, F1={hm['f1']:.4f}")

            # ---- Group-level metrics ----
            group_eval = {}
            if group_cols:
                print(f"    Loading group features ({', '.join(group_cols)})...")
                group_features_df = _load_group_features_streaming(
                    parquet_path=output_path,
                    sids=window_sids,
                    cutoff_year=cutoff,
                    group_cols=group_cols,
                )
                print(f"    Loaded features for {len(group_features_df):,} persons")

                group_eval = _evaluate_groups(
                    group_df=group_features_df,
                    all_predictions=window_preds,
                    y_true=window_labels,
                    events=events,
                    horizons=horizons,
                    group_cols=group_cols,
                    min_group_size=args.min_group_size,
                )

                # Print group-level summaries
                print(f"\n    GROUP-LEVEL EVALUATION (cutoff={cutoff})")
                for event, eres in group_eval.items():
                    summary = eres.get('summary', {})
                    if not summary:
                        continue
                    n_groups = summary.get('groups', 0)
                    print(f"      {event} ({n_groups} groups):")
                    for h in horizons:
                        mae_w = summary.get(f'mae_weighted_{h}yr', float('nan'))
                        rmse_w = summary.get(f'rmse_weighted_{h}yr', float('nan'))
                        mae_u = summary.get(f'mae_unweighted_{h}yr', float('nan'))
                        rmse_u = summary.get(f'rmse_unweighted_{h}yr', float('nan'))
                        print(f"        @{h}yr: MAE(w)={mae_w:.4f} RMSE(w)={rmse_w:.4f} "
                              f"MAE(u)={mae_u:.4f} RMSE(u)={rmse_u:.4f}")

                # Save group CSVs per window
                for event, eres in group_eval.items():
                    gdf = eres.get('group_df')
                    if gdf is not None and len(gdf) > 0:
                        csv_path = os.path.join(
                            args.checkpoint,
                            f"group_eval_{event}_cut{cutoff}.csv",
                        )
                        gdf.to_csv(csv_path, index=False)
                        print(f"    Saved: {csv_path}")

                del group_features_df

            window_metrics['group_eval'] = {
                event: eres.get('summary', {})
                for event, eres in group_eval.items()
            }
            all_window_metrics[cutoff] = window_metrics

            del window_ds, window_proba, window_preds, window_labels
            gc.collect()

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if all_window_metrics:
        results_path = os.path.join(args.checkpoint, 'window_metrics.json')
        serializable = {}
        for cutoff, wm in all_window_metrics.items():
            serializable[str(cutoff)] = {
                'aggregate': wm['aggregate'],
                'per_event': {
                    event: {
                        hk: {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in hm.items()}
                        for hk, hm in em.items()
                    }
                    for event, em in wm['per_event'].items()
                },
                'group_eval': {
                    event: {
                        k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in summary.items()
                    }
                    for event, summary in wm.get('group_eval', {}).items()
                    if summary
                },
            }
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {results_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
