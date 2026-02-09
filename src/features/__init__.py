"""
Feature Configuration Module

Defines and resolves feature sets for ML pipelines:
- Label column
- Categorical columns (for encoding)
- Numeric columns (for scaling)
- Columns to drop
- Auto-detection of remaining columns

Also provides feature engineering functions:
- Socioeconomic features (income, events, age interactions)
- Moving history features
"""
from .config import (
    FeatureConfig,
    FeatureResolver,
    resolve_features,
    load_feature_config,
)

# Socioeconomic feature engineering
from .socioeconomic import (
    create_all_socioeconomic_features,
    create_hh_pos_features,
    create_income_features_spark,
    create_event_interactions_spark,
    create_age_interactions_spark,
    get_socioeconomic_feature_list,
    cast_events_to_bool,
)

__all__ = [
    # Feature configuration
    "FeatureConfig",
    "FeatureResolver",
    "resolve_features",
    "load_feature_config",
    # Socioeconomic features
    "create_all_socioeconomic_features",
    "create_hh_pos_features",
    "create_income_features_spark",
    "create_event_interactions_spark",
    "create_age_interactions_spark",
    "get_socioeconomic_feature_list",
    "cast_events_to_bool",
]
