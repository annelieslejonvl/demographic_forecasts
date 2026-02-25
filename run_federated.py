#!/usr/bin/env python3
"""
Federated learning runner.

Coordinates training between a server (this machine) and a single client
(remote machine accessible via SSH) where the data lives.

Per cutoff-year the server sends the current model weights to the client,
the client trains locally, and sends updated weights back. The raw data
never leaves the client.

Usage (SSH mode):
    python run_federated.py \
        --config configs/models/pytorch_seq_gru_fast_numeric.yaml \
        --vocab-path checkpoints/sequence_vocab.joblib \
        --ssh-host client-machine \
        --ssh-user annelies \
        --remote-dir /data/federated \
        --cutoffs 2016,2017,2018,2019 \
        --history-len 5

Usage (local test mode — server & client in same process):
    python run_federated.py \
        --config configs/models/pytorch_seq_gru_fast_numeric.yaml \
        --vocab-path checkpoints/sequence_vocab.joblib \
        --data-path data/processed_features_with_municipality \
        --local-mode \
        --cutoffs 2016,2017,2018 \
        --history-len 5
"""
import argparse
import logging
import os
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Federated learning: train a sequence model across server and client via SSH"
    )

    # Config
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model YAML config (e.g., configs/models/pytorch_seq_gru_fast_numeric.yaml)",
    )
    parser.add_argument(
        "--vocab-path",
        default="checkpoints/sequence_vocab.joblib",
        help="Path to the shared vocabulary .joblib file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/federated",
        help="Directory for federated checkpoints",
    )

    # Cutoff configuration
    parser.add_argument(
        "--cutoffs",
        required=True,
        help="Comma-separated cutoff years (e.g., 2016,2017,2018,2019)",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=5,
        help="Number of years of history per window (default: 5)",
    )

    # SSH mode
    parser.add_argument("--ssh-host", help="Remote hostname")
    parser.add_argument("--ssh-user", help="SSH username")
    parser.add_argument("--remote-dir", help="Working directory on remote machine")
    parser.add_argument("--ssh-key", help="Path to SSH private key")

    # Local mode
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run server and client in the same process (for testing)",
    )
    parser.add_argument(
        "--data-path",
        help="Path to parquet data (required for --local-mode)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Validate arguments
    if not args.local_mode:
        if not args.ssh_host:
            parser.error("--ssh-host is required in SSH mode")
        if not args.remote_dir:
            parser.error("--remote-dir is required in SSH mode")
    else:
        if not args.data_path:
            parser.error("--data-path is required in --local-mode")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Parse cutoffs
    cutoff_years = [int(c.strip()) for c in args.cutoffs.split(",")]

    # Check vocab exists
    if not os.path.exists(args.vocab_path):
        print(f"ERROR: Vocabulary file not found: {args.vocab_path}")
        print("Build it first with: python run_with_municipality_sequence.py --reuse --rolling")
        sys.exit(1)

    # Import here to avoid slow imports when just showing --help
    from src.federated.server import FederatedServer

    server = FederatedServer(
        config=config,
        vocab_path=args.vocab_path,
        checkpoint_dir=args.checkpoint_dir,
        ssh_host=args.ssh_host,
        ssh_user=args.ssh_user,
        remote_dir=args.remote_dir,
        ssh_key=args.ssh_key,
        local_mode=args.local_mode,
        data_path=args.data_path,
    )

    server.run(
        cutoff_years=cutoff_years,
        history_len=args.history_len,
    )


if __name__ == "__main__":
    main()
