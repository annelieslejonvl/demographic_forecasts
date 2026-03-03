#!/usr/bin/env python3
"""
Client-side: validate (or fine-tune) pretrained model on real data.

Run this on the CLIENT (your Windows laptop with real data).
It loads a package from --incoming (SCP'd from server), runs validation
on real data, and writes results to --outgoing (SCP back to server).

Usage:
    python run_client_validate.py ^
        --incoming checkpoints/federated/round_000_cut2016/outgoing ^
        --outgoing checkpoints/federated/round_000_cut2016/incoming ^
        --data-path data/processed_features_with_municipality

    Optional: --finetune-epochs 5  (default: 0 = validation only)
"""
import argparse
import logging
import os
import sys
from datetime import datetime

from src.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Client: validate/fine-tune on real data")
    parser.add_argument("--incoming", required=True, help="Directory with server package (weights + config)")
    parser.add_argument("--outgoing", required=True, help="Directory to write results to")
    parser.add_argument("--data-path", required=True, help="Path to real parquet data")
    parser.add_argument("--finetune-epochs", type=int, default=0,
                        help="Fine-tuning epochs (0 = validation only)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Sample fraction for val data (default: 0.05 = 5%%)")

    args = parser.parse_args()

    log_file = setup_logging(log_file=f"client_validate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Logging to: {log_file}")

    print(f"\n{'='*60}")
    print(f"FEDERATED CLIENT (Validate on Real Data)")
    print(f"{'='*60}")
    print(f"  Incoming: {args.incoming}")
    print(f"  Outgoing: {args.outgoing}")
    print(f"  Data: {args.data_path}")
    print(f"  Mode: {'validation only' if args.finetune_epochs == 0 else f'fine-tune {args.finetune_epochs} epochs'}")
    print()

    from src.federated.client import FederatedClient

    client = FederatedClient(
        data_path=args.data_path,
        incoming_dir=args.incoming,
        outgoing_dir=args.outgoing,
        sample_frac=args.sample_frac,
        finetune_epochs=args.finetune_epochs,
    )

    metrics = client.train_round()

    print(f"\n{'='*60}")
    print(f"Client done!")
    print(f"  Val loss: {metrics.get('val_loss', '?')}")
    print(f"  Epochs:   {metrics.get('epochs_trained', 0)}")
    print(f"  Samples:  {metrics.get('n_samples', '?')}")
    print(f"  Results:  {args.outgoing}")
    print(f"{'='*60}")
    print()
    print(f"Now SCP the results back to the server:")
    print(f"  scp -r {args.outgoing} annelies-w84it:<server_exchange_dir>/incoming")


if __name__ == "__main__":
    main()
