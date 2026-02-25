"""
Federated learning client.

Runs on the machine where the data lives. Receives a round package from
the server (weights + vocabulary + config), trains locally on one
cutoff-year window, and exports the updated weights back.

Designed for resource-constrained environments (CPU-only, 32GB RAM).

Usage (called by the server via SSH):
    python -m src.federated.client \
        --incoming /path/to/incoming \
        --outgoing /path/to/outgoing \
        --data-path /path/to/parquet

Or for local testing:
    Used directly by FederatedServer in local_mode.
"""
import argparse
import gc
import logging
import os
import platform
import sys
from typing import Any, Dict, Optional, Set

import numpy as np
import pyarrow.parquet as pq
import torch

from . import protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# CPU / low-RAM defaults — override aggressive server config
# ---------------------------------------------------------------
CPU_OVERRIDES = {
    "use_amp": False,           # AMP is useless without GPU
    "batch_size": 64,           # smaller batches = less RAM
    "lr_find": False,           # LR finder doubles memory usage
    "mc_dropout_samples": 0,    # no MC dropout on CPU
}

# Maximum batch size the client will allow (even if config says higher)
MAX_BATCH_SIZE = 128

# Smaller parquet chunk size to limit peak RAM during cache building
PARQUET_CHUNK_SIZE = 200_000


def _apply_cpu_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply resource-friendly overrides to the server config.

    Modifies config in-place and returns it.
    """
    params = config.get("model", {}).get("params", {})

    for key, val in CPU_OVERRIDES.items():
        old = params.get(key)
        if old != val:
            logger.info("CPU override: %s = %s (was %s)", key, val, old)
            params[key] = val

    # Cap batch size
    if params.get("batch_size", 64) > MAX_BATCH_SIZE:
        logger.info(
            "CPU override: batch_size capped at %d (was %d)",
            MAX_BATCH_SIZE,
            params["batch_size"],
        )
        params["batch_size"] = MAX_BATCH_SIZE

    # Force device to cpu
    config.setdefault("device", {})["type"] = "cpu"

    return config


def _collect_valid_persons_windowed(
    parquet_path: str,
    cutoff_year: int,
    min_history_year: int,
    max_horizon: int,
    chunk_size: int = PARQUET_CHUNK_SIZE,
) -> Set:
    """Find persons with history in (min_history_year, cutoff] AND future in (cutoff, cutoff+max_horizon]."""
    dataset = pq.ParquetDataset(parquet_path)
    history: set = set()
    future: set = set()

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(
            batch_size=chunk_size, columns=["sid", "year"]
        ):
            df = batch.to_pandas()
            hist_mask = (df["year"] > min_history_year) & (
                df["year"] <= cutoff_year
            )
            future_mask = (df["year"] > cutoff_year) & (
                df["year"] <= cutoff_year + max_horizon
            )
            if hist_mask.any():
                history.update(df.loc[hist_mask, "sid"].unique().tolist())
            if future_mask.any():
                future.update(df.loc[future_mask, "sid"].unique().tolist())
            del df

    valid = history & future
    del history, future
    gc.collect()
    return valid


class FederatedClient:
    """Client-side trainer for federated learning.

    Receives a round package, trains on local data, exports results.
    Automatically applies CPU-friendly overrides on resource-constrained
    machines (no GPU, limited RAM).
    """

    def __init__(
        self,
        data_path: str,
        incoming_dir: str,
        outgoing_dir: str,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            data_path: Path to the local parquet dataset.
            incoming_dir: Directory with the round package from the server.
            outgoing_dir: Directory to write the round result to.
            cache_dir: Where to store per-cutoff dataset caches (defaults to
                       a 'cache' subdirectory next to data_path).
        """
        self.data_path = data_path
        self.incoming_dir = incoming_dir
        self.outgoing_dir = outgoing_dir
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(data_path), "federated_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def train_round(self):
        """Run one training round using the package in incoming_dir."""
        from ..sequence.dataset import CachedSequenceDataset
        from ..sequence.estimator import PyTorchSequenceEstimator
        from ..sequence.vocabulary import LifeEventVocabulary

        # Limit PyTorch threads on CPU to avoid over-subscribing the i5
        n_cores = os.cpu_count() or 4
        torch.set_num_threads(max(1, n_cores - 1))
        torch.set_num_interop_threads(1)
        logger.info(
            "PyTorch threads: %d (system cores: %d)",
            torch.get_num_threads(),
            n_cores,
        )

        # 1. Load round package
        package = protocol.import_round_package(self.incoming_dir)
        config = package["config"]
        cutoff_year = package["cutoff_year"]
        history_len = package["history_len"]
        vocab_path = package["vocab_path"]
        weights_path = package["weights_path"]

        # 2. Apply CPU / low-RAM overrides
        config = _apply_cpu_overrides(config)

        model_params = config.get("model", {}).get("params", {})
        events = config.get("events", [])
        horizons = config.get("horizons", [1, 3, 5])
        max_horizon = max(horizons)
        max_seq_len = model_params.get("max_seq_len", 256)
        use_numeric = bool(model_params.get("use_numeric_features", False))

        logger.info(
            "Client training: cutoff=%d, history_len=%d, batch_size=%d",
            cutoff_year,
            history_len,
            model_params.get("batch_size", 64),
        )
        print(f"  Client: training round cutoff={cutoff_year}")
        print(f"  Platform: {platform.system()}, CPU cores: {n_cores}, "
              f"PyTorch threads: {torch.get_num_threads()}")

        # 3. Load vocabulary
        vocabulary = LifeEventVocabulary.load(vocab_path)
        print(f"  Vocabulary loaded: {vocabulary.vocab_size} tokens")

        # 4. Load server weights (if provided)
        initial_state_dict = None
        if weights_path is not None:
            initial_state_dict = protocol.import_weights(weights_path)
            print("  Loaded server weights for warm-start")

        # 5. Build or load dataset cache for this cutoff
        cache_path = os.path.join(
            self.cache_dir, f"cut{cutoff_year}_h{history_len}.pt"
        )
        min_hist_year = cutoff_year - history_len

        cache_exists = os.path.exists(cache_path) or os.path.isdir(
            cache_path.replace(".pt", "_chunks")
        )

        if not cache_exists:
            print(f"  Building dataset cache for cutoff={cutoff_year}...")
            valid_persons = _collect_valid_persons_windowed(
                self.data_path, cutoff_year, min_hist_year, max_horizon
            )
            print(f"  Valid persons: {len(valid_persons):,}")
        else:
            valid_persons = None
            print(f"  Reusing existing cache: {cache_path}")

        train_dataset = CachedSequenceDataset(
            parquet_path=self.data_path,
            vocabulary=vocabulary,
            max_seq_len=max_seq_len,
            events=events,
            horizons=horizons,
            cutoff_year=cutoff_year,
            allowed_sids=valid_persons,
            cache_path=cache_path,
            min_history_year=min_hist_year,
            use_numeric_features=use_numeric,
            chunk_size=PARQUET_CHUNK_SIZE,
        )
        del valid_persons
        gc.collect()
        print(f"  Dataset: {len(train_dataset):,} samples")

        # 6. Build validation dataset (next year after cutoff, if data exists)
        eval_dataset = self._build_validation_dataset(
            config, vocabulary, cutoff_year, history_len, max_horizon,
            max_seq_len, use_numeric,
        )

        # 7. Train
        model_config = config.get("model", {})
        estimator = PyTorchSequenceEstimator(model_config=model_config)

        result = estimator.fit(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            vocabulary=vocabulary,
            initial_state_dict=initial_state_dict,
            device="cpu",
        )

        print(f"  Training complete: {result.metadata.get('epochs_trained', '?')} epochs")

        # 8. Export result
        train_metrics = {
            "cutoff_year": cutoff_year,
            "epochs_trained": result.metadata.get("epochs_trained", 0),
            "train_loss": float(result.metrics.get("train_loss", 0)),
            "val_loss": float(result.metrics.get("val_loss", 0))
            if result.metrics.get("val_loss") is not None
            else None,
            "n_samples": len(train_dataset),
            "n_params": result.metadata.get("n_params", 0),
            "device": "cpu",
            "platform": platform.system(),
        }

        protocol.export_round_result(
            result_dir=self.outgoing_dir,
            state_dict={k: v.cpu() for k, v in estimator.model_.state_dict().items()},
            train_metrics=train_metrics,
        )

        # Free memory
        del estimator, train_dataset, eval_dataset, initial_state_dict
        gc.collect()

        print(f"  Result exported to {self.outgoing_dir}")
        return train_metrics

    def _build_validation_dataset(
        self, config, vocabulary, cutoff_year, history_len, max_horizon,
        max_seq_len, use_numeric,
    ):
        """Build a small validation dataset from the next cutoff year."""
        from ..sequence.dataset import CachedSequenceDataset

        events = config.get("events", [])
        horizons = config.get("horizons", [1, 3, 5])
        val_cutoff = cutoff_year + 1
        val_cache_path = os.path.join(
            self.cache_dir, f"cut{val_cutoff}_h{history_len}_val.pt"
        )
        val_cache_exists = os.path.exists(val_cache_path) or os.path.isdir(
            val_cache_path.replace(".pt", "_chunks")
        )

        try:
            if not val_cache_exists:
                val_persons = _collect_valid_persons_windowed(
                    self.data_path, val_cutoff, val_cutoff - history_len,
                    max_horizon,
                )
                if val_persons:
                    # Subsample to 10% for speed on CPU
                    rng = np.random.RandomState(42)
                    n_val = max(1, int(len(val_persons) * 0.1))
                    val_persons = set(
                        rng.choice(list(val_persons), n_val, replace=False)
                    )
                else:
                    return None
            else:
                val_persons = None

            if val_persons is not None or val_cache_exists:
                eval_dataset = CachedSequenceDataset(
                    parquet_path=self.data_path,
                    vocabulary=vocabulary,
                    max_seq_len=max_seq_len,
                    events=events,
                    horizons=horizons,
                    cutoff_year=val_cutoff,
                    allowed_sids=val_persons,
                    cache_path=val_cache_path,
                    min_history_year=val_cutoff - history_len,
                    use_numeric_features=use_numeric,
                    chunk_size=PARQUET_CHUNK_SIZE,
                )
                del val_persons
                gc.collect()
                print(f"  Val dataset: {len(eval_dataset):,} samples (cutoff={val_cutoff})")
                return eval_dataset
        except Exception as e:
            logger.warning("Could not build validation dataset: %s", e)
            print(f"  No validation data available (cutoff={val_cutoff})")

        return None


def main():
    """CLI entry point for the federated client."""
    parser = argparse.ArgumentParser(
        description="Federated learning client - train one round on local data"
    )
    parser.add_argument(
        "--incoming",
        required=True,
        help="Directory with the round package from the server",
    )
    parser.add_argument(
        "--outgoing",
        required=True,
        help="Directory to write the round result to",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to local parquet data (reads from FEDERATED_DATA_PATH env var if not set)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for dataset caches",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Determine data path
    data_path = args.data_path
    if data_path is None:
        data_path = os.environ.get(
            "FEDERATED_DATA_PATH",
            "data/processed_features_with_municipality",
        )

    print(f"Federated Client starting")
    print(f"  Platform: {platform.system()}")
    print(f"  Incoming: {args.incoming}")
    print(f"  Outgoing: {args.outgoing}")
    print(f"  Data: {data_path}")

    client = FederatedClient(
        data_path=data_path,
        incoming_dir=args.incoming,
        outgoing_dir=args.outgoing,
        cache_dir=args.cache_dir,
    )
    client.train_round()


if __name__ == "__main__":
    main()
