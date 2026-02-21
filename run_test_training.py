#!/usr/bin/env python3
"""Quick test: train sequence model using available chunks, with full evaluation."""
import os
import sys

import torch
import numpy as np

sys.path.insert(0, '/home/annelies/demographic_forecasts')

from src.sequence.vocabulary import LifeEventVocabulary
from src.sequence.dataset import CachedSequenceDataset
from src.sequence.estimator import PyTorchSequenceEstimator


def evaluate_predictions(y_true, y_prob, events, horizons):
    """Compute AUC, AP, F1, Brier for all event-horizon combinations."""
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        brier_score_loss,
    )

    n_horizons = len(horizons)
    results = {}

    for ei, event in enumerate(events):
        event_results = {}
        for hi, horizon in enumerate(horizons):
            col_idx = ei * n_horizons + hi
            y_e = y_true[:, col_idx]
            y_p = y_prob[:, col_idx]

            metrics = {}
            n_pos = int(y_e.sum())
            n_neg = len(y_e) - n_pos

            if n_pos > 0 and n_neg > 0:
                metrics['auc'] = float(roc_auc_score(y_e, y_p))
                metrics['ap'] = float(average_precision_score(y_e, y_p))

                best_f1, best_thresh = 0.0, 0.5
                for thresh in np.linspace(0.01, 0.99, 100):
                    y_pred = (y_p >= thresh).astype(int)
                    f1 = f1_score(y_e, y_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh

                metrics['f1'] = float(best_f1)
                metrics['threshold'] = float(best_thresh)
                metrics['brier'] = float(brier_score_loss(y_e, y_p))
            else:
                metrics['auc'] = float('nan')
                metrics['ap'] = float('nan')
                metrics['f1'] = float('nan')
                metrics['brier'] = float('nan')

            metrics['n_pos'] = n_pos
            metrics['n_total'] = len(y_e)
            metrics['prevalence'] = n_pos / len(y_e) if len(y_e) > 0 else 0

            event_results[f'{horizon}yr'] = metrics
        results[event] = event_results

    all_aucs = [m['auc'] for er in results.values() for m in er.values() if not np.isnan(m.get('auc', float('nan')))]
    all_aps = [m['ap'] for er in results.values() for m in er.values() if not np.isnan(m.get('ap', float('nan')))]

    return {
        'per_event': results,
        'aggregate': {
            'mean_auc': float(np.mean(all_aucs)) if all_aucs else float('nan'),
            'mean_ap': float(np.mean(all_aps)) if all_aps else float('nan'),
        },
    }


def main():
    train_chunk_dir = "checkpoints/sequence_cache/train_cut2022_chunks"
    val_chunk_dir = "checkpoints/sequence_cache/val_cut2020_chunks"

    train_chunks = sorted([f for f in os.listdir(train_chunk_dir) if f.endswith('.pt')])
    val_chunks = sorted([f for f in os.listdir(val_chunk_dir) if f.endswith('.pt')])
    print(f"Found {len(train_chunks)} train chunks, {len(val_chunks)} val chunks")

    # Use 50% of train chunks for training, symlink into a temp dir
    n_train = len(train_chunks) // 2
    train_subset_dir = "checkpoints/sequence_cache/train_half_chunks"
    os.makedirs(train_subset_dir, exist_ok=True)

    for f in train_chunks[:n_train]:
        dst = os.path.join(train_subset_dir, f)
        src = os.path.abspath(os.path.join(train_chunk_dir, f))
        if not os.path.exists(dst):
            os.symlink(src, dst)

    print(f"Using {n_train} train chunks ({n_train * 50000:,} persons)")
    print(f"Using {len(val_chunks)} val chunks")

    # Load vocabulary
    vocab = LifeEventVocabulary.load("checkpoints/sequence_vocab.joblib")
    print(f"Vocabulary size: {vocab.vocab_size}")

    events = ['y_moved', 'birth1_event', 'birth2_event', 'divorce_event', 'getalifeother_event']
    horizons = [1, 3, 5]

    # Create datasets - train from half the train chunks, val from all val chunks
    train_dataset = CachedSequenceDataset(
        parquet_path="dummy",
        vocabulary=vocab,
        max_seq_len=256,
        events=events,
        horizons=horizons,
        cutoff_year=2022,
        cache_path="checkpoints/sequence_cache/train_half.pt",
    )
    print(f"Train dataset: {len(train_dataset):,} persons")

    val_dataset = CachedSequenceDataset(
        parquet_path="dummy",
        vocabulary=vocab,
        max_seq_len=256,
        events=events,
        horizons=horizons,
        cutoff_year=2020,
        cache_path="checkpoints/sequence_cache/val_cut2020.pt",
    )
    print(f"Val dataset: {len(val_dataset):,} persons")

    # Event rates (sample 10 chunks max to avoid loading all 70)
    pos_weights = train_dataset.get_pos_weights(max_samples=500_000)
    n_horizons = len(horizons)
    print(f"\nEvent rates (train, sampled):")
    for ei, event in enumerate(events):
        for hi, horizon in enumerate(horizons):
            pw = pos_weights[ei * n_horizons + hi].item()
            rate = 1.0 / (1.0 + pw) if pw > 0 else 0
            print(f"  {event} @{horizon}yr: {rate:.3%} (pos_weight={pw:.1f})")

    # Train
    model_config = {
        'type': 'seq_gru',
        'params': {
            'encoder_type': 'gru',
            'embed_dim': 64,
            'max_seq_len': 256,
            'encoder': {
                'hidden_dim': 128,
                'num_layers': 1,
                'dropout': 0.2,
            },
            'head_hidden_dims': [64],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 2,
            'batch_size': 512,
            'early_stopping_patience': 3,
            'use_amp': False,
        }
    }

    estimator = PyTorchSequenceEstimator(
        model_config=model_config,
        device_config=None,
    )

    print(f"\nStarting training (2 epochs)...")
    result = estimator.fit(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        vocabulary=vocab,
    )
    print(f"\nTraining complete!")
    print(f"  Metrics: {result.metrics}")

    # Predict on val set (used as evaluation set)
    print(f"\nGenerating predictions on val set...")
    proba_result = estimator.predict_proba(dataset=val_dataset)
    probs = proba_result.probabilities
    print(f"  Predictions shape: {probs.shape}")

    # Print prediction summaries
    n_horizons = len(horizons)
    for ei, event in enumerate(events):
        for hi, horizon in enumerate(horizons):
            col_idx = ei * n_horizons + hi
            p = probs[:, col_idx]
            print(f"  {event} @{horizon}yr: mean={p.mean():.4f}, "
                  f"median={np.median(p):.4f}, P>0.5={(p > 0.5).mean():.3%}")

    # Get ground truth from val chunks
    print(f"\nComputing ground truth labels from val chunks...")
    y_true_parts = []
    for i in range(len(val_dataset._chunk_paths)):
        chunk = torch.load(val_dataset._chunk_paths[i], map_location='cpu', weights_only=False)
        y_true_parts.append(chunk['targets'].numpy())
        del chunk
    y_true = np.concatenate(y_true_parts)
    print(f"  y_true shape: {y_true.shape}")

    # Evaluate
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print('=' * 60)

    metrics = evaluate_predictions(y_true, probs, events, horizons)

    for event, event_metrics in metrics['per_event'].items():
        print(f"\n--- {event} ---")
        for horizon_key, h_metrics in event_metrics.items():
            auc = h_metrics.get('auc', float('nan'))
            ap = h_metrics.get('ap', float('nan'))
            f1 = h_metrics.get('f1', float('nan'))
            brier = h_metrics.get('brier', float('nan'))
            prev = h_metrics.get('prevalence', 0)
            n_pos = h_metrics.get('n_pos', 0)
            n_total = h_metrics.get('n_total', 0)

            print(f"  @{horizon_key}: AUC={auc:.4f}  AP={ap:.4f}  "
                  f"F1={f1:.4f}  Brier={brier:.4f}  "
                  f"prevalence={prev:.3%} ({n_pos}/{n_total})")

    agg = metrics['aggregate']
    print(f"\n  Aggregate: mean_AUC={agg['mean_auc']:.4f}, "
          f"mean_AP={agg['mean_ap']:.4f}")


if __name__ == '__main__':
    main()
