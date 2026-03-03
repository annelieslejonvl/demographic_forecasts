#!/usr/bin/env python3
"""
Server-side: pretrain on synthetic data and export a package for the client.

Run this on the SERVER (Linux). It pretrains and saves the package locally.
The client (Windows laptop) will SCP it down.

Usage:
    python run_server_pretrain.py \
        --config configs/models/pytorch_seq_gru_fast_deephit.yaml \
        --vocab-path checkpoints/sequence_vocab.joblib \
        --server-data-path /path/to/synthetic_data \
        --exchange-dir /home/annelies/federated_exchange \
        --server-pretrain-epochs 30 \
        --cutoffs 2016,2017,2018 \
        --history-len 5
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import torch
import yaml

from src.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Server: pretrain + export for federated learning")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--vocab-path", default="checkpoints/sequence_vocab.joblib")
    parser.add_argument("--server-data-path", required=True, help="Path to synthetic data (parquet)")
    parser.add_argument("--exchange-dir", required=True,
                        help="Directory on server to store round packages")
    parser.add_argument("--server-pretrain-epochs", type=int, default=30)
    parser.add_argument("--cutoffs", required=True, help="Comma-separated cutoff years")
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between checks for client result (default: 30)")
    parser.add_argument("--timeout", type=int, default=14400,
                        help="Max seconds to wait for client result (default: 4h)")

    args = parser.parse_args()

    log_file = setup_logging(log_file=f"server_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Logging to: {log_file}")
    logger = logging.getLogger(__name__)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cutoff_years = [int(c.strip()) for c in args.cutoffs.split(",")]
    os.makedirs(args.exchange_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FEDERATED SERVER (Pretrain + Wait for Client)")
    print(f"{'='*60}")
    print(f"  Synthetic data: {args.server_data_path}")
    print(f"  Exchange dir:   {args.exchange_dir}")
    print(f"  Pretrain epochs: {args.server_pretrain_epochs}")
    print(f"  Cutoffs: {cutoff_years}")
    print()

    from src.federated import protocol
    from src.sequence.dataset import SequenceDataset
    from src.sequence.estimator import PyTorchSequenceEstimator
    from src.sequence.vocabulary import LifeEventVocabulary
    import copy
    import pandas as pd
    import pyarrow.parquet as pq

    vocab = LifeEventVocabulary.load(args.vocab_path)
    model = None
    round_history = []

    for round_idx, cutoff in enumerate(cutoff_years):
        print(f"\n--- Round {round_idx + 1}/{len(cutoff_years)}: cutoff={cutoff} ---")

        # ---- Phase 1: Server pretrains on synthetic data ----
        print(f"\n  [SERVER] Pretraining for {args.server_pretrain_epochs} epochs...")

        model_params = config.get("model", {}).get("params", {})
        events = config.get("events", [])
        horizons = config.get("horizons", [1, 3, 5])
        max_horizon = max(horizons)
        min_hist_year = cutoff - args.history_len
        max_seq_len = model_params.get("max_seq_len", 64)
        use_numeric = bool(model_params.get("use_numeric_features", False))

        # Load synthetic data
        logger.info("Loading synthetic data...")
        dataset = pq.ParquetDataset(args.server_data_path)
        df = dataset.read().to_pandas()
        df = df[(df['year'] > min_hist_year) & (df['year'] <= cutoff + max_horizon)]
        id_col = 'sid' if 'sid' in df.columns else 'id'

        persons_hist = set(df[df['year'] <= cutoff][id_col].unique())
        persons_future = set(df[df['year'] > cutoff][id_col].unique())
        valid = persons_hist & persons_future
        df_train = df[df[id_col].isin(valid)].copy()
        print(f"  Synthetic data: {len(df_train):,} rows, {len(valid):,} persons")

        train_ds = SequenceDataset(
            df=df_train, vocabulary=vocab, events=events, horizons=horizons,
            max_seq_len=max_seq_len, use_numeric_features=use_numeric, id_col=id_col,
        )
        del df, df_train

        server_config = copy.deepcopy(config)
        server_config['model']['params']['epochs'] = args.server_pretrain_epochs

        initial_state_dict = None
        if model is not None:
            initial_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        estimator = PyTorchSequenceEstimator(
            model_config=server_config.get('model', {}),
            device_config=server_config.get('device', {}),
        )
        result = estimator.fit(
            train_dataset=train_ds, eval_dataset=None,
            vocabulary=vocab, initial_state_dict=initial_state_dict,
        )
        model = estimator.model_
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"  [SERVER] Pretraining done: loss={result.metrics.get('train_loss', 0):.4f}")

        # ---- Phase 2: Export package ----
        round_dir = os.path.join(args.exchange_dir, f"round_{round_idx:03d}_cut{cutoff}")
        package_dir = os.path.join(round_dir, "outgoing")
        result_dir = os.path.join(round_dir, "incoming")

        protocol.export_round_package(
            package_dir=package_dir, vocab_path=args.vocab_path,
            config=config, cutoff_year=cutoff,
            history_len=args.history_len, state_dict=state_dict,
        )

        with open(os.path.join(round_dir, "SERVER_READY"), "w") as f:
            f.write(f"cutoff={cutoff}, timestamp={datetime.now().isoformat()}\n")

        print(f"\n  [SERVER] Package ready at: {package_dir}")
        print(f"  -------------------------------------------------------")
        print(f"  NOW ON YOUR LAPTOP, run these commands:")
        print(f"")
        print(f"    scp -r annelies-w84it:{package_dir} checkpoints/federated/round_{round_idx:03d}_cut{cutoff}/outgoing")
        print(f"")
        print(f"    python run_client_validate.py ^")
        print(f"      --incoming checkpoints/federated/round_{round_idx:03d}_cut{cutoff}/outgoing ^")
        print(f"      --outgoing checkpoints/federated/round_{round_idx:03d}_cut{cutoff}/incoming ^")
        print(f"      --data-path data/processed_features_with_municipality")
        print(f"")
        print(f"    scp -r checkpoints/federated/round_{round_idx:03d}_cut{cutoff}/incoming annelies-w84it:{result_dir}")
        print(f"  -------------------------------------------------------")
        print()

        # ---- Phase 3: Wait for client result ----
        client_done_file = os.path.join(result_dir, "train_metrics.json")
        waited = 0
        while not os.path.exists(client_done_file):
            time.sleep(args.poll_interval)
            waited += args.poll_interval
            if waited % 120 == 0:
                print(f"  [SERVER] Waiting for client result at {result_dir}... ({waited}s)")
            if waited >= args.timeout:
                print(f"  [SERVER] TIMEOUT after {args.timeout}s")
                sys.exit(1)

        print(f"  [SERVER] Client result received! (waited {waited}s)")

        # ---- Phase 4: Import result ----
        client_result = protocol.import_round_result(result_dir)
        metrics = client_result.get("train_metrics", {})

        if metrics.get("epochs_trained", 0) > 0:
            updated_weights = protocol.import_weights(client_result["weights_path"])
            model.load_state_dict(updated_weights)
            print(f"  [SERVER] Loaded fine-tuned weights from client")

        print(f"  Round {round_idx+1}: val_loss={metrics.get('val_loss', '?')}, "
              f"epochs={metrics.get('epochs_trained', 0)}, samples={metrics.get('n_samples', '?')}")
        round_history.append({"round": round_idx + 1, "cutoff": cutoff, **metrics})

    history_path = os.path.join(args.exchange_dir, "round_history.json")
    with open(history_path, "w") as f:
        json.dump(round_history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All {len(cutoff_years)} rounds complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
