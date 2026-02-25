"""
Shared protocol for weight exchange between server and client.

A 'round package' is a directory the server sends to the client containing
everything needed for one training round:

    incoming/
        weights.pt          # model state_dict (or None for first round)
        vocabulary.joblib    # shared LifeEventVocabulary
        round_info.json     # cutoff_year, history_len, config, etc.

A 'round result' is a directory the client sends back:

    outgoing/
        weights.pt          # updated model state_dict after local training
        train_metrics.json  # loss, epochs_trained, n_samples, etc.
"""
import json
import logging
import os
import shutil
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def export_weights(state_dict: Dict[str, torch.Tensor], path: str) -> str:
    """Save a model state_dict to a .pt file.

    Returns the path written to.
    """
    torch.save(state_dict, path)
    n_params = sum(v.numel() for v in state_dict.values())
    logger.info("Exported weights (%d parameters) to %s", n_params, path)
    return path


def import_weights(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a model state_dict from a .pt file."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    n_params = sum(v.numel() for v in state_dict.values())
    logger.info("Imported weights (%d parameters) from %s", n_params, path)
    return state_dict


def export_round_package(
    package_dir: str,
    vocab_path: str,
    config: Dict[str, Any],
    cutoff_year: int,
    history_len: int,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    """Create a round package directory for the client.

    Args:
        package_dir: Directory to write the package into.
        vocab_path: Path to the vocabulary .joblib file.
        config: Full model/training config dict.
        cutoff_year: The cutoff year for this round.
        history_len: Number of years of history per window.
        state_dict: Model weights (None for the very first round).

    Returns:
        The package_dir path.
    """
    os.makedirs(package_dir, exist_ok=True)

    # Weights
    if state_dict is not None:
        export_weights(state_dict, os.path.join(package_dir, "weights.pt"))

    # Vocabulary
    dst_vocab = os.path.join(package_dir, "vocabulary.joblib")
    shutil.copy2(vocab_path, dst_vocab)

    # Round info
    round_info = {
        "cutoff_year": cutoff_year,
        "history_len": history_len,
        "config": config,
    }
    with open(os.path.join(package_dir, "round_info.json"), "w") as f:
        json.dump(round_info, f, indent=2)

    logger.info(
        "Exported round package for cutoff=%d to %s", cutoff_year, package_dir
    )
    return package_dir


def import_round_package(package_dir: str) -> Dict[str, Any]:
    """Read a round package sent by the server.

    Returns:
        Dict with keys: 'weights_path' (str or None), 'vocab_path' (str),
        'config' (dict), 'cutoff_year' (int), 'history_len' (int).
    """
    with open(os.path.join(package_dir, "round_info.json")) as f:
        round_info = json.load(f)

    weights_path = os.path.join(package_dir, "weights.pt")
    if not os.path.exists(weights_path):
        weights_path = None

    return {
        "weights_path": weights_path,
        "vocab_path": os.path.join(package_dir, "vocabulary.joblib"),
        "config": round_info["config"],
        "cutoff_year": round_info["cutoff_year"],
        "history_len": round_info["history_len"],
    }


def export_round_result(
    result_dir: str,
    state_dict: Dict[str, torch.Tensor],
    train_metrics: Dict[str, Any],
) -> str:
    """Create a round result directory for the server.

    Args:
        result_dir: Directory to write the result into.
        state_dict: Updated model weights after local training.
        train_metrics: Training metrics (loss, epochs, n_samples, etc.).

    Returns:
        The result_dir path.
    """
    os.makedirs(result_dir, exist_ok=True)

    export_weights(state_dict, os.path.join(result_dir, "weights.pt"))

    with open(os.path.join(result_dir, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info("Exported round result to %s", result_dir)
    return result_dir


def import_round_result(result_dir: str) -> Dict[str, Any]:
    """Read a round result sent by the client.

    Returns:
        Dict with keys: 'weights_path' (str), 'train_metrics' (dict).
    """
    with open(os.path.join(result_dir, "train_metrics.json")) as f:
        train_metrics = json.load(f)

    return {
        "weights_path": os.path.join(result_dir, "weights.pt"),
        "train_metrics": train_metrics,
    }
