#!/usr/bin/env python3
"""
Pre-tokenize all sequence datasets and cache to disk.

Run this once (overnight if needed) to build all cache files.
Subsequent training runs will load from cache instantly.

Usage:
    python build_sequence_cache.py --config configs/models/pytorch_seq_gru.yaml
    python build_sequence_cache.py --sample-fraction 0.1  # For testing
"""
import os
import sys
import argparse
from datetime import datetime
import pyarrow.parquet as pq

from src.utils.logging_setup import setup_logging, log_stage_start, log_stage_complete
from src.sequence.vocabulary import LifeEventVocabulary, MUNICIPALITY_FEATURE_MAP
from src.sequence.dataset import CachedSequenceDataset
from run_test import load_model_config
from run_with_municipality_sequence import (
    _build_vocab_from_parquet,
    _collect_valid_persons,
    DEFAULT_EVENTS,
)


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize sequence datasets')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--sample-fraction', type=float,
                        help='Optional: build cache for a subset (for testing)')
    parser.add_argument('--cutoff-year', type=int, default=2022,
                        help='Year cutoff for train/test split')
    parser.add_argument('--data-path', type=str,
                        default='data/processed/features/with_municipality_features.parquet',
                        help='Path to input parquet file')

    args = parser.parse_args()

    log_file = setup_logging(
        log_file=f"build_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        log_dir='.'
    )
    print(f"Logging to: {log_file}\n")

    print("=" * 70)
    print("SEQUENCE DATASET CACHE BUILDER")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Data: {args.data_path}")
    print(f"Cutoff year: {args.cutoff_year}")
    if args.sample_fraction:
        print(f"Sample fraction: {args.sample_fraction:.2%} (subset mode)")
    else:
        print("Sample fraction: 100% (FULL DATASET)")
    print("=" * 70)
    print()

    # Load config
    config = load_model_config(args.config)
    events = config.get('events', DEFAULT_EVENTS)
    horizons = config.get('horizons', [1, 3, 5])
    max_seq_len = config.get('model', {}).get('params', {}).get('max_seq_len', 256)

    print(f"Events: {', '.join(events)}")
    print(f"Horizons: {horizons}")
    print(f"Max sequence length: {max_seq_len}")
    print()

    # Check data exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        sys.exit(1)

    dataset = pq.ParquetDataset(args.data_path)
    total_rows = sum(fragment.metadata.num_rows for fragment in dataset.fragments)
    print(f"Total rows in dataset: {total_rows:,}")
    print()

    # Build vocabulary
    log_stage_start("Building Vocabulary")

    vocab_path = "checkpoints/sequence_vocab.joblib"
    os.makedirs("checkpoints", exist_ok=True)

    if os.path.exists(vocab_path):
        print(f"Loading existing vocabulary from {vocab_path}...")
        vocabulary = LifeEventVocabulary.load(vocab_path)
        # Check if it has municipality tokens
        missing_tokens = [
            f"{prefix}_Q1"
            for prefix in MUNICIPALITY_FEATURE_MAP.values()
            if vocabulary.token_to_id(f"{prefix}_Q1") == vocabulary.UNK
        ]
        if missing_tokens:
            print("Vocabulary missing municipality tokens; rebuilding...")
            vocabulary = _build_vocab_from_parquet(args.data_path, vocab_path)
    else:
        print("Building vocabulary from parquet (this may take a few minutes)...")
        vocabulary = _build_vocab_from_parquet(args.data_path, vocab_path)

    print(f"Vocabulary size: {vocabulary.vocab_size} tokens")
    log_stage_complete("Building Vocabulary")
    print()

    # Collect valid persons
    log_stage_start("Collecting Valid Persons")

    cutoff_year = args.cutoff_year
    val_cutoff = cutoff_year - 2

    print(f"Scanning for persons with history before {cutoff_year} and future after...")
    valid_persons = _collect_valid_persons(args.data_path, cutoff_year)
    print(f"Train persons (cutoff={cutoff_year}): {len(valid_persons):,}")

    print(f"Scanning for persons with history before {val_cutoff} and future after...")
    val_persons = _collect_valid_persons(args.data_path, val_cutoff)
    print(f"Val persons (cutoff={val_cutoff}): {len(val_persons):,}")

    # Apply sampling if requested
    if args.sample_fraction is not None:
        if not (0 < args.sample_fraction <= 1.0):
            print("ERROR: --sample-fraction must be in (0, 1]")
            sys.exit(1)

        import numpy as np
        rng = np.random.RandomState(42)

        n_train = max(1, int(len(valid_persons) * args.sample_fraction))
        valid_persons = set(rng.choice(list(valid_persons), n_train, replace=False))

        n_val = max(1, int(len(val_persons) * args.sample_fraction))
        val_persons = set(rng.choice(list(val_persons), n_val, replace=False))

        print(f"\nSubsampled to {args.sample_fraction:.2%}:")
        print(f"  Train: {len(valid_persons):,} persons")
        print(f"  Val: {len(val_persons):,} persons")

    log_stage_complete("Collecting Valid Persons")
    print()

    # Build cache paths
    cache_dir = "checkpoints/sequence_cache"
    os.makedirs(cache_dir, exist_ok=True)

    sf_tag = f"_sf{args.sample_fraction}" if args.sample_fraction else ""
    train_cache = os.path.join(cache_dir, f"train_cut{cutoff_year}{sf_tag}.pt")
    val_cache = os.path.join(cache_dir, f"val_cut{val_cutoff}{sf_tag}.pt")
    test_cache = os.path.join(cache_dir, f"test_cut{cutoff_year}{sf_tag}.pt")

    print("Cache files:")
    print(f"  Train: {train_cache}")
    print(f"  Val:   {val_cache}")
    print(f"  Test:  {test_cache}")
    print()

    # Check what already exists
    existing = []
    if os.path.exists(train_cache):
        existing.append("train")
    if os.path.exists(val_cache):
        existing.append("val")
    if os.path.exists(test_cache):
        existing.append("test")

    if existing:
        print(f"WARNING: Found existing cache files: {', '.join(existing)}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
        print()

    # Build train cache
    log_stage_start("Building Train Cache")
    print(f"Tokenizing {len(valid_persons):,} train persons...")
    print("This will take time but only needs to run once!")
    print()

    train_dataset = CachedSequenceDataset(
        parquet_path=args.data_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=cutoff_year,
        allowed_sids=valid_persons,
        cache_path=train_cache,
    )
    print(f"✓ Train cache built: {len(train_dataset):,} persons")
    log_stage_complete("Building Train Cache")
    print()

    # Build val cache
    log_stage_start("Building Val Cache")
    print(f"Tokenizing {len(val_persons):,} val persons...")
    print()

    val_dataset = CachedSequenceDataset(
        parquet_path=args.data_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=val_cutoff,
        allowed_sids=val_persons,
        cache_path=val_cache,
    )
    print(f"✓ Val cache built: {len(val_dataset):,} persons")
    log_stage_complete("Building Val Cache")
    print()

    # Build test cache (same as train for now)
    log_stage_start("Building Test Cache")
    print(f"Tokenizing {len(valid_persons):,} test persons...")
    print()

    test_dataset = CachedSequenceDataset(
        parquet_path=args.data_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=cutoff_year,
        allowed_sids=valid_persons,
        cache_path=test_cache,
    )
    print(f"✓ Test cache built: {len(test_dataset):,} persons")
    log_stage_complete("Building Test Cache")
    print()

    print("=" * 70)
    print("CACHE BUILDING COMPLETE!")
    print("=" * 70)
    print()
    print("Cache files created:")
    for path in [train_cache, val_cache, test_cache]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {path} ({size_mb:.1f} MB)")
    print()
    print("You can now run training with instant startup:")
    print(f"  python run_with_municipality_sequence.py --reuse --config {args.config}")
    if args.sample_fraction:
        print(f"                                           --sample-fraction {args.sample_fraction}")
    print()


if __name__ == '__main__':
    main()
