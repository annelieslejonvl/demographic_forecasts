"""
Feature Configuration Module

Defines and resolves feature sets for ML pipelines:
- Label column
- Categorical columns (for encoding)
- Numeric columns (for scaling)
- Columns to drop
- Auto-detection of remaining columns
"""
from .config import (
    FeatureConfig,
    FeatureResolver,
    resolve_features,
    load_feature_config,
)

__all__ = [
    "FeatureConfig",
    "FeatureResolver",
    "resolve_features",
    "load_feature_config",
]
