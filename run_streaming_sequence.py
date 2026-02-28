#!/usr/bin/env python3
"""
Train GRU sequence model on streaming pipeline output.

This script works with the processed data from run_municipality_streaming.py,
which does NOT include municipality features (to save disk space).

Usage:
    # Build vocabulary + train
    python run_streaming_sequence.py --config configs/models/pytorch_seq_gru_fast_deephit.yaml

    # Train with existing vocabulary
    python run_streaming_sequence.py --config configs/models/pytorch_seq_gru_fast_deephit.yaml --vocab checkpoints/sequence_vocab.joblib
"""
import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_processed_data(data_path: str, sample_frac: float = None) -> pd.DataFrame:
    """Load processed parquet data (year-by-year files)."""
    logger.info(f"Loading data from {data_path}")

    # Read using pyarrow for efficiency
    dataset = pq.ParquetDataset(data_path)

    # Get total rows
    total_rows = sum(piece.metadata.num_rows for piece in dataset.fragments)
    logger.info(f"Total rows available: {total_rows:,}")

    if sample_frac and sample_frac < 1.0:
        logger.info(f"Sampling {sample_frac*100:.1f}% of data")
        # Read year-by-year and sample to avoid OOM
        dfs = []
        for piece in dataset.fragments:
            df_year = piece.to_table().to_pandas()
            df_sampled = df_year.sample(frac=sample_frac, random_state=42)
            dfs.append(df_sampled)
            del df_year
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        logger.info(f"Sampled to {len(df):,} rows")
    else:
        # Read all
        df = dataset.read().to_pandas()
        logger.info(f"Loaded {len(df):,} rows")

    return df


def build_vocabulary(df: pd.DataFrame, vocab_path: str):
    """Build and save vocabulary from processed data."""
    from src.sequence.vocabulary import LifeEventVocabulary

    logger.info("Building vocabulary from data")

    vocab = LifeEventVocabulary()
    vocab.build_from_dataframe(
        df,
        year_range=(df['year'].min(), df['year'].max()),
        age_col='age',
        age_group_col='age_group',
        nationality_col='eerste_nationaliteit',
        gender_col='gender',
        coupled_col='coupled',
        hh_pos_col='hh_pos',
        income_quintile_col='income_quintile',
        n_muni_bins=10,  # Will create tokens even if no muni data
    )

    vocab.save(vocab_path)
    logger.info(f"Vocabulary saved: {vocab_path}")
    logger.info(f"  Vocabulary size: {vocab.vocab_size}")

    return vocab


def train_sequence_model(
    df: pd.DataFrame,
    config: dict,
    vocab_path: str,
    checkpoint_dir: str = "checkpoints/sequence",
):
    """Train sequence model on processed data."""
    from src.sequence.vocabulary import LifeEventVocabulary
    from src.sequence.dataset import SequenceDataset
    from src.sequence.estimator import PyTorchSequenceEstimator

    logger.info("Initializing sequence model training")

    # Load vocabulary
    vocab = LifeEventVocabulary.load(vocab_path)
    logger.info(f"Loaded vocabulary: {vocab.vocab_size} tokens")

    # Extract config params
    model_params = config.get("model", {}).get("params", {})
    events = config.get("events", [])
    horizons = config.get("horizons", [1, 3, 5])

    # Create train/val split (time-based)
    years = sorted(df['year'].unique())
    train_cutoff = years[-3]  # Last 2 years for validation

    df_train = df[df['year'] < train_cutoff].copy()
    df_val = df[df['year'] >= train_cutoff].copy()

    logger.info(f"Train: {len(df_train):,} rows (years {df_train['year'].min()}-{df_train['year'].max()})")
    logger.info(f"Val:   {len(df_val):,} rows (years {df_val['year'].min()}-{df_val['year'].max()})")

    # Create datasets
    logger.info("Creating sequence datasets")

    max_seq_len = model_params.get("max_seq_len", 64)
    use_numeric = bool(model_params.get("use_numeric_features", False))

    # Determine ID column name (id for local client, sid for server/synthetic)
    id_col = 'id' if 'id' in df.columns else 'sid'
    logger.info(f"Using ID column: {id_col}")

    train_dataset = SequenceDataset(
        df=df_train,
        vocabulary=vocab,
        events=events,
        horizons=horizons,
        max_seq_len=max_seq_len,
        use_numeric_features=use_numeric,
        id_col=id_col,
    )

    val_dataset = SequenceDataset(
        df=df_val,
        vocabulary=vocab,
        events=events,
        horizons=horizons,
        max_seq_len=max_seq_len,
        use_numeric_features=use_numeric,
        id_col=id_col,
    )

    logger.info(f"Train dataset: {len(train_dataset):,} sequences")
    logger.info(f"Val dataset:   {len(val_dataset):,} sequences")

    # Build estimator
    logger.info("Building sequence estimator")

    # Add events and horizons to config for estimator
    full_config = config.copy()
    full_config['events'] = events
    full_config['horizons'] = horizons

    estimator = PyTorchSequenceEstimator(
        model_config=full_config,
        device_config=config.get('device', {}),
    )

    # Train
    logger.info("Starting training")
    print("\n" + "="*60)
    print("SEQUENCE MODEL TRAINING")
    print("="*60)
    print(f"Model: {model_params.get('encoder_type', 'gru').upper()}")
    print(f"Loss: {model_params.get('loss_type', 'bce').upper()}")
    print(f"Events: {len(events)}")
    print(f"Horizons: {horizons}")
    print(f"Max seq length: {max_seq_len}")
    print(f"Embed dim: {model_params.get('embed_dim', 128)}")
    print(f"Numeric features: {'enabled' if use_numeric else 'disabled'}")
    print("="*60)
    print()

    result = estimator.fit(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        vocabulary=vocab,
    )

    # Save checkpoint
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    estimator.save_checkpoint(checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("Training complete")

    return estimator, result


def main():
    parser = argparse.ArgumentParser(
        description="Train GRU sequence model on streaming pipeline output"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML (e.g., configs/models/pytorch_seq_gru_fast_deephit.yaml)"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed_features_with_municipality",
        help="Path to processed parquet data (default: data/processed_features_with_municipality)"
    )
    parser.add_argument(
        "--vocab",
        default="checkpoints/sequence_vocab.joblib",
        help="Path to vocabulary file (will be created if doesn't exist)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/sequence",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Sample fraction of data (for testing, e.g., 0.01 = 1%%)"
    )
    parser.add_argument(
        "--rebuild-vocab",
        action="store_true",
        help="Force rebuild vocabulary even if it exists"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("GRU SEQUENCE MODEL - STREAMING DATA")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Data: {args.data_path}")
    print(f"Vocabulary: {args.vocab}")
    print("="*60)
    print()

    # Check data path exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data path not found: {args.data_path}")
        logger.error("Run run_municipality_streaming.py first to process features")
        sys.exit(1)

    # Load data
    df = load_processed_data(args.data_path, sample_frac=args.sample_frac)

    # Build or load vocabulary
    if not os.path.exists(args.vocab) or args.rebuild_vocab:
        if args.rebuild_vocab and os.path.exists(args.vocab):
            logger.info(f"Rebuilding vocabulary (--rebuild-vocab)")
        vocab = build_vocabulary(df, args.vocab)
    else:
        logger.info(f"Using existing vocabulary: {args.vocab}")

    # Train model
    estimator, result = train_sequence_model(
        df=df,
        config=config,
        vocab_path=args.vocab,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Vocabulary: {args.vocab}")
    if result and hasattr(result, 'val_metrics'):
        print(f"Best val loss: {result.val_metrics.get('loss', 'N/A')}")
    print("="*60)


if __name__ == "__main__":
    main()
