"""
Federated learning server.

Coordinates training rounds with a single remote client via SSH/SCP.
Works on both Windows (OpenSSH) and Linux.

Per cutoff-year:
  1. Export current model weights + vocabulary + config
  2. SCP the package to the client
  3. SSH: run client training on the remote machine
  4. SCP the result (updated weights + metrics) back
  5. Load updated weights, checkpoint, log to MLflow
"""
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from . import protocol

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"


def _quote_remote(path: str) -> str:
    """Quote a remote path for SSH commands. Works on both Windows and Linux."""
    if IS_WINDOWS:
        # Windows OpenSSH passes commands via cmd.exe — use double quotes
        return f'"{path}"'
    else:
        import shlex
        return shlex.quote(path)


class FederatedServer:
    """Server-side coordinator for federated training over SSH."""

    def __init__(
        self,
        config: Dict[str, Any],
        vocab_path: str,
        checkpoint_dir: str = "checkpoints/federated",
        ssh_host: Optional[str] = None,
        ssh_user: Optional[str] = None,
        remote_dir: Optional[str] = None,
        ssh_key: Optional[str] = None,
        local_mode: bool = False,
        data_path: Optional[str] = None,
        sample_frac: Optional[float] = None,
        server_pretrain_epochs: Optional[int] = None,
        server_data_path: Optional[str] = None,
        client_finetune_epochs: int = 0,
    ):
        """
        Args:
            config: Full config dict (same format as YAML configs).
            vocab_path: Path to the shared vocabulary .joblib file.
            checkpoint_dir: Where to store per-round checkpoints.
            ssh_host: Remote hostname (ignored in local_mode).
            ssh_user: SSH username (ignored in local_mode).
            remote_dir: Working directory on the remote machine.
            ssh_key: Optional path to SSH private key.
            local_mode: If True, run client in the same process (for testing).
            data_path: Path to parquet data (used in local_mode).
            sample_frac: Sample fraction for testing (e.g., 0.01 = 1%).
            server_pretrain_epochs: Pretrain on server synthetic data for N epochs.
            server_data_path: Path to server's synthetic data for pretraining.
            client_finetune_epochs: Number of epochs for client fine-tuning (0 = validation only).
        """
        self.config = config
        self.vocab_path = os.path.abspath(vocab_path)
        self.checkpoint_dir = checkpoint_dir
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.remote_dir = remote_dir
        self.ssh_key = ssh_key
        self.local_mode = local_mode
        self.data_path = data_path
        self.sample_frac = sample_frac
        self.server_pretrain_epochs = server_pretrain_epochs
        self.server_data_path = server_data_path
        self.client_finetune_epochs = client_finetune_epochs

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Model will be built on the first round (we need vocab_size, etc.)
        self.model_ = None
        self._round_history: List[Dict[str, Any]] = []

    @property
    def _host_str(self) -> str:
        if self.ssh_user:
            return f"{self.ssh_user}@{self.ssh_host}"
        return self.ssh_host

    def run(
        self,
        cutoff_years: List[int],
        history_len: int = 5,
    ):
        """Run the full federated training loop.

        Args:
            cutoff_years: Ordered list of cutoff years to train on.
            history_len: Years of history per window.
        """
        logger.info(
            "Starting federated training: %d rounds, cutoffs=%s",
            len(cutoff_years),
            cutoff_years,
        )
        print(f"\n{'='*60}")
        print("FEDERATED TRAINING")
        print(f"{'='*60}")
        print(f"  Platform: {platform.system()}")
        print(f"  Cutoff years: {cutoff_years}")
        print(f"  History length: {history_len}")
        print(f"  Mode: {'local' if self.local_mode else 'SSH'}")
        if not self.local_mode:
            print(f"  Remote: {self._host_str}:{self.remote_dir}")
        print()

        mlflow_ok = False
        try:
            import mlflow
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("demographic_forecasts_federated")
            mlflow_ok = True
        except Exception:
            logger.warning("MLflow not available, skipping experiment tracking")

        for round_idx, cutoff in enumerate(cutoff_years):
            print(f"\n--- Round {round_idx + 1}/{len(cutoff_years)}: cutoff={cutoff} ---")
            logger.info("Round %d: cutoff=%d", round_idx + 1, cutoff)

            result = self._run_round(cutoff, history_len, round_idx)

            self._round_history.append({
                "round": round_idx + 1,
                "cutoff_year": cutoff,
                **result.get("train_metrics", {}),
            })

            # Checkpoint after each round
            self._save_checkpoint(cutoff, round_idx)

            # Print summary
            metrics = result.get("train_metrics", {})
            print(f"  Round {round_idx + 1} complete:")
            print(f"    Epochs trained: {metrics.get('epochs_trained', '?')}")
            print(f"    Train loss: {metrics.get('train_loss', '?')}")
            print(f"    Val loss: {metrics.get('val_loss', '?')}")

            if mlflow_ok:
                try:
                    import mlflow
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) and v is not None:
                            mlflow.log_metric(f"client_{k}", v, step=round_idx)
                except Exception:
                    pass

        # Save final history
        history_path = os.path.join(self.checkpoint_dir, "round_history.json")
        with open(history_path, "w") as f:
            json.dump(self._round_history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Federated training complete: {len(cutoff_years)} rounds")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*60}")

    def _run_round(
        self, cutoff_year: int, history_len: int, round_idx: int
    ) -> Dict[str, Any]:
        """Execute one federated round."""
        # PHASE 1: Server-side pretraining (if enabled)
        if self.server_pretrain_epochs is not None and self.server_pretrain_epochs > 0:
            if self.server_data_path is None:
                raise ValueError("server_data_path is required when server_pretrain_epochs > 0")

            logger.info(
                "Server pretraining: %d epochs on synthetic data (cutoff=%d)",
                self.server_pretrain_epochs, cutoff_year
            )
            print(f"\n  [SERVER] Pretraining for {self.server_pretrain_epochs} epochs on synthetic data...")

            state_dict = self._server_pretrain(
                cutoff_year=cutoff_year,
                history_len=history_len,
                n_epochs=self.server_pretrain_epochs
            )
        else:
            # Get current state_dict (None for first round)
            state_dict = None
            if self.model_ is not None:
                state_dict = {
                    k: v.cpu() for k, v in self.model_.state_dict().items()
                }

        # Create package directory
        round_dir = os.path.join(
            self.checkpoint_dir, f"round_{round_idx:03d}_cut{cutoff_year}"
        )
        package_dir = os.path.join(round_dir, "outgoing")
        result_dir = os.path.join(round_dir, "incoming")

        # Export round package
        protocol.export_round_package(
            package_dir=package_dir,
            vocab_path=self.vocab_path,
            config=self.config,
            cutoff_year=cutoff_year,
            history_len=history_len,
            state_dict=state_dict,
        )

        # Run training (local or remote)
        if self.local_mode:
            self._run_local_training(package_dir, result_dir)
        else:
            self._run_remote_training(package_dir, result_dir)

        # Import result
        result = protocol.import_round_result(result_dir)

        # Load updated weights into global model
        updated_weights = protocol.import_weights(result["weights_path"])

        if self.model_ is None:
            self._init_model_from_config(updated_weights)
        else:
            self.model_.load_state_dict(updated_weights)

        logger.info(
            "Round cutoff=%d: loaded updated weights from client",
            cutoff_year,
        )

        return result

    def _init_model_from_config(self, state_dict: Dict[str, torch.Tensor]):
        """Initialize the global model from config + loaded weights."""
        from ..sequence.models import SequenceModel
        from ..sequence.vocabulary import LifeEventVocabulary

        vocab = LifeEventVocabulary.load(self.vocab_path)
        model_params = self.config.get("model", {}).get("params", {})
        events = self.config.get("events", [])
        horizons = self.config.get("horizons", [1, 3, 5])

        use_numeric = bool(model_params.get("use_numeric_features", False))
        if use_numeric:
            from ..sequence.vocabulary import N_NUMERIC_FEATURES
            n_numeric = N_NUMERIC_FEATURES
        else:
            n_numeric = 0

        self.model_ = SequenceModel(
            vocab_size=vocab.vocab_size,
            embed_dim=model_params.get("embed_dim", 128),
            encoder_type=model_params.get("encoder_type", "gru"),
            encoder_config=model_params.get("encoder", {}),
            n_events=len(events),
            n_horizons=len(horizons),
            max_seq_len=model_params.get("max_seq_len", 256),
            head_hidden_dims=model_params.get("head_hidden_dims", [128, 64]),
            dropout=model_params.get("dropout", 0.1),
            loss_type=model_params.get("loss_type", "bce"),
            multi_head=bool(model_params.get("multi_head", False)),
            n_numeric_features=n_numeric,
            numeric_inject=model_params.get("numeric_inject", "add"),
        )
        self.model_.load_state_dict(state_dict)
        logger.info("Initialized global model from client weights")

    def _server_pretrain(
        self, cutoff_year: int, history_len: int, n_epochs: int
    ) -> Dict[str, torch.Tensor]:
        """Pretrain model on server's synthetic data.

        Args:
            cutoff_year: Training cutoff year
            history_len: Years of history per window
            n_epochs: Number of epochs to train

        Returns:
            state_dict: Trained model weights
        """
        from ..sequence.dataset import SequenceDataset
        from ..sequence.estimator import PyTorchSequenceEstimator
        from ..sequence.vocabulary import LifeEventVocabulary
        import pandas as pd
        import pyarrow.parquet as pq

        logger.info("Loading server synthetic data from %s", self.server_data_path)

        # Load server's synthetic data
        dataset = pq.ParquetDataset(self.server_data_path)
        df = dataset.read().to_pandas()
        logger.info(f"Server data: {len(df):,} rows")

        # Filter to cutoff window
        min_hist_year = cutoff_year - history_len
        model_params = self.config.get("model", {}).get("params", {})
        events = self.config.get("events", [])
        horizons = self.config.get("horizons", [1, 3, 5])
        max_horizon = max(horizons)

        df = df[(df['year'] > min_hist_year) & (df['year'] <= cutoff_year + max_horizon)]
        logger.info(f"Filtered to window: {len(df):,} rows")

        # Identify valid persons (have both history and future)
        id_col = 'sid' if 'sid' in df.columns else 'id'
        persons_hist = set(df[df['year'] <= cutoff_year][id_col].unique())
        persons_future = set(df[df['year'] > cutoff_year][id_col].unique())
        valid_persons = persons_hist & persons_future
        logger.info(f"Valid persons for training: {len(valid_persons):,}")

        df_train = df[df[id_col].isin(valid_persons) & (df['year'] <= cutoff_year)].copy()
        logger.info(f"Training data: {len(df_train):,} rows")

        # Build dataset
        vocab = LifeEventVocabulary.load(self.vocab_path)
        max_seq_len = model_params.get("max_seq_len", 64)
        use_numeric = bool(model_params.get("use_numeric_features", False))

        train_dataset = SequenceDataset(
            df=df_train,
            vocabulary=vocab,
            events=events,
            horizons=horizons,
            max_seq_len=max_seq_len,
            use_numeric_features=use_numeric,
            id_col=id_col,
        )
        logger.info(f"Server training dataset: {len(train_dataset):,} sequences")

        # Override epochs in config for server pretraining
        import copy
        server_config = copy.deepcopy(self.config)
        server_config['model']['params']['epochs'] = n_epochs

        # Train
        estimator = PyTorchSequenceEstimator(
            model_config=server_config.get('model', {}),
            device_config=server_config.get('device', {}),
        )

        result = estimator.fit(
            train_dataset=train_dataset,
            eval_dataset=None,  # No validation during server pretraining
            vocabulary=vocab,
        )

        logger.info(
            "Server pretraining complete: %d epochs, final loss: %.4f",
            n_epochs,
            result.metrics.get('train_loss', 0.0)
        )
        print(f"  [SERVER] Pretraining complete: {n_epochs} epochs, loss={result.metrics.get('train_loss', 0.0):.4f}")

        # Store model for future rounds
        self.model_ = estimator.model_

        # Return weights to send to client
        return {k: v.cpu() for k, v in estimator.model_.state_dict().items()}

    # ------------------------------------------------------------------
    # Local mode
    # ------------------------------------------------------------------

    def _run_local_training(self, package_dir: str, result_dir: str):
        """Run client training in the same process (for testing)."""
        from .client import FederatedClient

        if not self.data_path:
            raise ValueError("data_path is required for local_mode")

        client = FederatedClient(
            data_path=self.data_path,
            incoming_dir=package_dir,
            outgoing_dir=result_dir,
            sample_frac=self.sample_frac,
            finetune_epochs=self.client_finetune_epochs,
        )
        client.train_round()

    # ------------------------------------------------------------------
    # SSH mode (Windows OpenSSH + Linux compatible)
    # ------------------------------------------------------------------

    def _run_remote_training(self, package_dir: str, result_dir: str):
        """Run client training on the remote machine via SSH/SCP."""
        remote_incoming = self.remote_dir + "/incoming"
        remote_outgoing = self.remote_dir + "/outgoing"

        # 1. Clean remote dirs and recreate
        self._ssh_run(
            f"rm -rf {_quote_remote(remote_incoming)} {_quote_remote(remote_outgoing)} && "
            f"mkdir -p {_quote_remote(remote_incoming)} {_quote_remote(remote_outgoing)}"
        )

        # 2. Upload package
        print(f"  Sending round package to {self._host_str}...")
        self._scp_to_remote(package_dir, remote_incoming)

        # 3. Run client on remote
        print(f"  Starting remote training...")
        client_cmd = (
            f"cd {_quote_remote(self.remote_dir)} && "
            f"python -m src.federated.client "
            f"--incoming {_quote_remote(remote_incoming)} "
            f"--outgoing {_quote_remote(remote_outgoing)}"
        )
        if self.sample_frac is not None:
            client_cmd += f" --sample-frac {self.sample_frac}"
        if self.client_finetune_epochs > 0:
            client_cmd += f" --finetune-epochs {self.client_finetune_epochs}"
        self._ssh_run(client_cmd)

        # 4. Download result
        print(f"  Receiving results from {self._host_str}...")
        os.makedirs(result_dir, exist_ok=True)
        self._scp_from_remote(remote_outgoing, result_dir)

    def _build_ssh_base(self) -> List[str]:
        """Build the base SSH command with optional key."""
        cmd = ["ssh"]
        if self.ssh_key:
            cmd.extend(["-i", self.ssh_key])
        cmd.append(self._host_str)
        return cmd

    def _build_scp_base(self) -> List[str]:
        """Build the base SCP command with optional key."""
        cmd = ["scp", "-r"]
        if self.ssh_key:
            cmd.extend(["-i", self.ssh_key])
        return cmd

    def _ssh_run(self, command: str):
        """Run a command on the remote machine via SSH.

        Works with Windows OpenSSH and Linux OpenSSH.
        """
        ssh_cmd = self._build_ssh_base()
        ssh_cmd.append(command)

        logger.info("SSH: %s", command)
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            # On Windows, avoid shell=True but let subprocess find ssh in PATH
        )
        if result.returncode != 0:
            logger.error("SSH stderr: %s", result.stderr)
            raise RuntimeError(
                f"Remote command failed (exit {result.returncode}): {result.stderr}"
            )
        if result.stdout.strip():
            logger.info("SSH stdout (truncated): %s", result.stdout.strip()[:500])

    def _scp_to_remote(self, local_dir: str, remote_dir: str):
        """SCP a local directory's contents to the remote machine.

        On Windows, scp -r local_dir/* doesn't expand globs natively,
        so we upload the entire directory and let the remote side handle it.
        """
        scp_cmd = self._build_scp_base()
        # Upload the whole local_dir as a subdirectory, then we already
        # structured the package contents flat in local_dir.
        # scp -r local_dir/* user@host:remote_dir/ works on Linux.
        # On Windows we upload each file individually to avoid glob issues.
        if IS_WINDOWS:
            self._scp_files_windows(local_dir, remote_dir)
        else:
            import glob as _glob
            files = _glob.glob(os.path.join(local_dir, "*"))
            scp_cmd.extend(files)
            scp_cmd.append(f"{self._host_str}:{remote_dir}/")

            logger.info("SCP upload: %d files -> %s:%s", len(files), self._host_str, remote_dir)
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"SCP upload failed: {result.stderr}")

    def _scp_files_windows(self, local_dir: str, remote_dir: str):
        """Upload files one by one on Windows (scp glob expansion is unreliable)."""
        for entry in os.listdir(local_dir):
            local_path = os.path.join(local_dir, entry)
            scp_cmd = self._build_scp_base()
            if os.path.isdir(local_path):
                scp_cmd.extend([local_path, f"{self._host_str}:{remote_dir}/"])
            else:
                scp_cmd.extend([local_path, f"{self._host_str}:{remote_dir}/{entry}"])

            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"SCP upload failed for {entry}: {result.stderr}")
            logger.info("SCP uploaded: %s", entry)

    def _scp_from_remote(self, remote_dir: str, local_dir: str):
        """SCP a remote directory's contents to the local machine."""
        scp_cmd = self._build_scp_base()

        if IS_WINDOWS:
            # On Windows, download the whole remote dir
            scp_cmd.extend([f"{self._host_str}:{remote_dir}/*", local_dir + os.sep])
        else:
            scp_cmd.extend([f"{self._host_str}:{remote_dir}/.", local_dir + "/"])

        logger.info("SCP download: %s:%s -> %s", self._host_str, remote_dir, local_dir)
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            # Fallback: try without /. suffix (some scp versions don't support it)
            scp_cmd2 = self._build_scp_base()
            scp_cmd2.extend([f"{self._host_str}:{remote_dir}", local_dir])
            result2 = subprocess.run(scp_cmd2, capture_output=True, text=True, timeout=600)
            if result2.returncode != 0:
                raise RuntimeError(f"SCP download failed: {result.stderr}\nFallback: {result2.stderr}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, cutoff_year: int, round_idx: int):
        """Save the current global model as a checkpoint."""
        if self.model_ is None:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(
            self.checkpoint_dir, f"global_model_round{round_idx:03d}_cut{cutoff_year}"
        )
        os.makedirs(ckpt_path, exist_ok=True)

        torch.save(
            self.model_.state_dict(),
            os.path.join(ckpt_path, "model_state_dict.pt"),
        )

        metadata = {
            "round_idx": round_idx,
            "cutoff_year": cutoff_year,
            "timestamp": ts,
            "config": self.config,
            "history": self._round_history,
        }
        with open(os.path.join(ckpt_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved checkpoint: %s", ckpt_path)
