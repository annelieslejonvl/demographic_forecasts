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


def _detect_id_column(parquet_path: str) -> str:
    """Detect whether the dataset uses 'id' or 'sid' column.

    Args:
        parquet_path: Path to parquet dataset

    Returns:
        'id' or 'sid' depending on which column exists
    """
    dataset = pq.ParquetDataset(parquet_path)
    schema = dataset.schema

    # Check schema for id column
    if 'id' in schema.names:
        logger.info("Detected ID column: 'id' (local/client data)")
        return 'id'
    elif 'sid' in schema.names:
        logger.info("Detected ID column: 'sid' (server/synthetic data)")
        return 'sid'
    else:
        raise ValueError(f"No ID column found in dataset. Available columns: {schema.names}")


def _collect_valid_persons_windowed(
    parquet_path: str,
    cutoff_year: int,
    min_history_year: int,
    max_horizon: int,
    chunk_size: int = PARQUET_CHUNK_SIZE,
    id_col: Optional[str] = None,
) -> Set:
    """Find persons with history in (min_history_year, cutoff] AND future in (cutoff, cutoff+max_horizon].

    Args:
        parquet_path: Path to parquet dataset
        cutoff_year: Cutoff year for training
        min_history_year: Minimum year for history
        max_horizon: Maximum horizon for future events
        chunk_size: Batch size for reading parquet
        id_col: ID column name ('id' or 'sid'). Auto-detected if None.
    """
    # Auto-detect ID column if not provided
    if id_col is None:
        id_col = _detect_id_column(parquet_path)

    dataset = pq.ParquetDataset(parquet_path)
    history: set = set()
    future: set = set()

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(
            batch_size=chunk_size, columns=[id_col, "year"]
        ):
            df = batch.to_pandas()
            hist_mask = (df["year"] > min_history_year) & (
                df["year"] <= cutoff_year
            )
            future_mask = (df["year"] > cutoff_year) & (
                df["year"] <= cutoff_year + max_horizon
            )
            if hist_mask.any():
                history.update(df.loc[hist_mask, id_col].unique().tolist())
            if future_mask.any():
                future.update(df.loc[future_mask, id_col].unique().tolist())
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
        sample_frac: Optional[float] = None,
        finetune_epochs: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to the local parquet dataset.
            incoming_dir: Directory with the round package from the server.
            outgoing_dir: Directory to write the round result to.
            cache_dir: Where to store per-cutoff dataset caches (defaults to
                       a 'cache' subdirectory next to data_path).
            sample_frac: Sample fraction for testing (e.g., 0.01 = 1%).
            finetune_epochs: Number of epochs for fine-tuning (None = use config, 0 = validation only).
        """
        self.data_path = data_path
        self.incoming_dir = incoming_dir
        self.outgoing_dir = outgoing_dir
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(data_path), "federated_cache"
        )
        self.sample_frac = sample_frac
        self.finetune_epochs = finetune_epochs
        os.makedirs(self.cache_dir, exist_ok=True)

    def train_round(self):
        """Run one training round using the package in incoming_dir."""
        from ..sequence.dataset import CachedSequenceDataset, SequenceDataset
        from ..sequence.estimator import PyTorchSequenceEstimator
        from ..sequence.vocabulary import LifeEventVocabulary
        import pandas as pd

        # Limit PyTorch threads on CPU to avoid over-subscribing the i5
        # Only set once (PyTorch doesn't allow changing after parallel work starts)
        n_cores = os.cpu_count() or 4
        try:
            torch.set_num_threads(max(1, n_cores - 1))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already set in previous round, ignore
            pass
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

        # 2b. Override epochs if finetune_epochs is specified
        if self.finetune_epochs is not None:
            config["model"]["params"]["epochs"] = self.finetune_epochs
            if self.finetune_epochs == 0:
                logger.info("Client mode: VALIDATION ONLY (no training)")
                print(f"  [CLIENT] Mode: VALIDATION ONLY (0 epochs)")
            else:
                logger.info("Client mode: FINE-TUNING for %d epochs", self.finetune_epochs)
                print(f"  [CLIENT] Mode: FINE-TUNING ({self.finetune_epochs} epochs)")

        model_params = config.get("model", {}).get("params", {})
        events = config.get("events", [])
        horizons = config.get("horizons", [1, 3, 5])
        max_horizon = max(horizons)
        max_seq_len = model_params.get("max_seq_len", 256)
        use_numeric = bool(model_params.get("use_numeric_features", False))

        logger.info(
            "Client training: cutoff=%d, history_len=%d, batch_size=%d, epochs=%d",
            cutoff_year,
            history_len,
            model_params.get("batch_size", 64),
            model_params.get("epochs", 30),
        )
        print(f"  Client: training round cutoff={cutoff_year}")
        print(f"  Platform: {platform.system()}, CPU cores: {n_cores}, "
              f"PyTorch threads: {torch.get_num_threads()}")

        # 3. Detect ID column (id vs sid)
        id_col = _detect_id_column(self.data_path)
        print(f"  Using ID column: {id_col}")

        # 4. Load vocabulary
        vocabulary = LifeEventVocabulary.load(vocab_path)
        print(f"  Vocabulary loaded: {vocabulary.vocab_size} tokens")

        # 5. Load server weights (if provided)
        initial_state_dict = None
        if weights_path is not None:
            initial_state_dict = protocol.import_weights(weights_path)
            print("  Loaded server weights for warm-start")

        # 6. Build dataset
        min_hist_year = cutoff_year - history_len

        # If sampling, use SequenceDataset (on-the-fly tokenization) instead of CachedSequenceDataset
        if self.sample_frac is not None:
            print(f"  Using SequenceDataset with {self.sample_frac*100:.1f}% sampling (no cache)")

            # Load and filter data
            print(f"  Loading data from {self.data_path}...")
            dataset = pq.ParquetDataset(self.data_path)

            # Read year-by-year and filter to window
            dfs = []
            for fragment in dataset.fragments:
                df_year = fragment.to_table().to_pandas()
                # Filter to relevant years: history + horizon
                year_mask = (df_year['year'] > min_hist_year) & (df_year['year'] <= cutoff_year + max_horizon)
                df_filtered = df_year[year_mask].copy()
                if len(df_filtered) > 0:
                    # Sample each year
                    df_sampled = df_filtered.sample(frac=self.sample_frac, random_state=42)
                    dfs.append(df_sampled)
                del df_year, df_filtered

            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            del dfs
            gc.collect()
            print(f"  Loaded {len(df):,} rows (sampled)")

            # Filter to valid persons (have both history and future)
            persons_hist = set(df[df['year'] <= cutoff_year][id_col].unique())
            persons_future = set(df[df['year'] > cutoff_year][id_col].unique())
            valid_persons = persons_hist & persons_future
            print(f"  Valid persons: {len(valid_persons):,}")

            df_train = df[df[id_col].isin(valid_persons) & (df['year'] <= cutoff_year)].copy()
            del df
            gc.collect()
            print(f"  Training data: {len(df_train):,} rows")

            # Build SequenceDataset
            train_dataset = SequenceDataset(
                df=df_train,
                vocabulary=vocabulary,
                events=events,
                horizons=horizons,
                max_seq_len=max_seq_len,
                use_numeric_features=use_numeric,
                id_col=id_col,
            )
            del df_train
            gc.collect()
        else:
            # Use cached dataset for full data
            cache_path = os.path.join(
                self.cache_dir, f"cut{cutoff_year}_h{history_len}.pt"
            )

            cache_exists = os.path.exists(cache_path) or os.path.isdir(
                cache_path.replace(".pt", "_chunks")
            )

            if not cache_exists:
                print(f"  Building dataset cache for cutoff={cutoff_year}...")
                valid_persons = _collect_valid_persons_windowed(
                    self.data_path, cutoff_year, min_hist_year, max_horizon,
                    id_col=id_col
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
                id_col=id_col,
                allowed_sids=valid_persons,
                cache_path=cache_path,
                min_history_year=min_hist_year,
                use_numeric_features=use_numeric,
                chunk_size=PARQUET_CHUNK_SIZE,
            )
            del valid_persons
            gc.collect()

        print(f"  Dataset: {len(train_dataset):,} samples")

        # 7. Build validation dataset (next year after cutoff, if data exists)
        eval_dataset = self._build_validation_dataset(
            config, vocabulary, cutoff_year, history_len, max_horizon,
            max_seq_len, use_numeric, id_col=id_col,
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
        max_seq_len, use_numeric, id_col='sid',
    ):
        """Build a small validation dataset from the next cutoff year.

        Args:
            config: Model configuration
            vocabulary: Token vocabulary
            cutoff_year: Current training cutoff year
            history_len: Years of history per window
            max_horizon: Maximum prediction horizon
            max_seq_len: Maximum sequence length
            use_numeric: Whether to use numeric features
            id_col: ID column name ('id' or 'sid')
        """
        from ..sequence.dataset import CachedSequenceDataset, SequenceDataset
        import pandas as pd

        # Skip validation when sampling (to save time during testing)
        if self.sample_frac is not None:
            print(f"  Skipping validation dataset (sample_frac={self.sample_frac})")
            return None

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
                    max_horizon, id_col=id_col,
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
                    id_col=id_col,
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
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Sample fraction for testing (e.g., 0.01 = 1%%)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=None,
        help="Number of fine-tuning epochs (0 = validation only, None = use config)",
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
        sample_frac=args.sample_frac,
        finetune_epochs=args.finetune_epochs,
    )
    client.train_round()


if __name__ == "__main__":
    main()
