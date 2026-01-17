"""
Preprocessing Module

Supports:
- Imputation (mean, median, mode, constant)
- Categorical encoding (one-hot, label/string indexer, ordinal)
- Scaling (standard, minmax, robust)
- PCA dimensionality reduction
"""
from .config import (
    ColumnConfig,
    EncodingStrategy,
    ImputationStrategy,
    PreprocessingConfig,
    ScalingStrategy,
)
from .spark import SparkPreprocessor
from .sklearn_preprocessor import SklearnPreprocessor
from .utils import create_preprocessor

__all__ = [
    "ColumnConfig",
    "EncodingStrategy",
    "ImputationStrategy",
    "PreprocessingConfig",
    "ScalingStrategy",
    "SparkPreprocessor",
    "SklearnPreprocessor",
    "create_preprocessor",
]
