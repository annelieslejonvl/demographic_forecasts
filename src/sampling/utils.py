"""Sampling utility functions."""
from typing import Union, Dict, Any
import numpy as np

from .config import SamplingConfig, SamplingStrategy


def create_sampler(
    backend: str,
    config: Union[SamplingConfig, Dict[str, Any], None] = None,
):
    """Create sampler for specified backend."""
    from .spark import SparkSampler
    from .numpy_sampler import NumpySampler
    
    if isinstance(config, dict):
        config = SamplingConfig.from_dict(config)
    elif config is None:
        config = SamplingConfig()
    
    if backend == "spark":
        return SparkSampler(config)
    else:
        return NumpySampler(config)


def compute_class_weights(
    y: np.ndarray,
    strategy: str = "balanced",
) -> np.ndarray:
    """Compute sample weights for class imbalance."""
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    
    if strategy == "balanced":
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        neg_weight = 1.0
    elif strategy == "sqrt":
        pos_weight = np.sqrt(n_neg / n_pos) if n_pos > 0 else 1.0
        neg_weight = 1.0
    else:
        pos_weight = neg_weight = 1.0
    
    weights = np.where(y == 1, pos_weight, neg_weight)
    return weights.astype(np.float32)
