"""
Sampling Strategies Module

Supports:
- Random, stratified, undersample, oversample
- Hard negative mining (top_prob, margin, entropy)
"""
from .config import SamplingConfig, SamplingStrategy
from .spark import SparkSampler
from .numpy_sampler import NumpySampler, HardNegativeMiner
from .utils import create_sampler, compute_class_weights

__all__ = [
    "SamplingConfig",
    "SamplingStrategy",
    "SparkSampler",
    "NumpySampler",
    "HardNegativeMiner",
    "create_sampler",
    "compute_class_weights",
]
