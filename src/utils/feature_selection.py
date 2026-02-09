"""
Feature selection utilities for loading and applying feature configurations.

This module provides functions to:
- Load feature lists from YAML config files
- Filter features based on config
- Validate that specified features exist in the dataset
"""
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_feature_config(config_path: str) -> Dict[str, Any]:
    """
    Load feature configuration from YAML file.

    Args:
        config_path: Path to feature config YAML (e.g., 'configs/data/features_mixed.yaml')

    Returns:
        Dictionary with feature configuration

    Example config structure:
        features:
          - age_group
          - age_norm
          - ...
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Feature config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if 'features' not in config:
        raise ValueError(f"Config file must contain 'features' key: {config_path}")

    features = config['features']
    print(f"✓ Loaded feature config from {config_path}")
    print(f"  {len(features)} features specified")

    return config


def filter_features_by_config(
    all_features: List[str],
    feature_config: Optional[Dict[str, Any]] = None,
    feature_config_path: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """
    Filter features based on configuration file.

    Args:
        all_features: List of all available features
        feature_config: Pre-loaded feature config dict (optional)
        feature_config_path: Path to feature config YAML (optional)
        verbose: Print filtering information

    Returns:
        Filtered list of features to use

    Usage:
        # Option 1: Pass config dict
        config = load_feature_config('configs/data/features_mixed.yaml')
        features = filter_features_by_config(all_features, feature_config=config)

        # Option 2: Pass config path
        features = filter_features_by_config(all_features,
                                            feature_config_path='configs/data/features_mixed.yaml')

        # Option 3: No config (returns all features)
        features = filter_features_by_config(all_features)
    """
    # If no config provided, return all features
    if feature_config is None and feature_config_path is None:
        if verbose:
            print("ℹ️  No feature config provided, using all available features")
        return all_features

    # Load config if path provided
    if feature_config is None and feature_config_path is not None:
        feature_config = load_feature_config(feature_config_path)

    # Get feature list from config
    specified_features = feature_config.get('features', [])

    # Clean feature names: strip whitespace, remove empty strings
    cleaned_features = []
    for feat in specified_features:
        if isinstance(feat, str):
            feat_cleaned = feat.strip()
            if feat_cleaned:  # Not empty after stripping
                cleaned_features.append(feat_cleaned)
        else:
            print(f"  ⚠️  Skipping non-string feature in config: {feat} (type: {type(feat)})")

    specified_features = cleaned_features
    print(f"✓ Loaded {len(specified_features)} features from config")

    if not specified_features:
        if verbose:
            print("⚠️  Feature config is empty, using all available features")
        return all_features

    # Filter: only keep features that are both specified AND available
    available_set = set(all_features)
    specified_set = set(specified_features)

    # Features that are specified and available
    keep_features = [f for f in specified_features if f in available_set]

    # CRITICAL: Ensure keep_features is a flat list of strings
    # Remove any non-string items or nested structures
    validated_features = []
    for feat in keep_features:
        if isinstance(feat, str):
            validated_features.append(feat)
        else:
            print(f"  ⚠️  Skipping non-string feature: {feat} (type: {type(feat)})")

    keep_features = validated_features

    # CRITICAL: Remove duplicates while preserving order
    # Duplicates cause pandas indexing issues (X[col] returns DataFrame, not Series)
    seen = set()
    unique_features = []
    duplicates = []
    for feat in keep_features:
        if feat not in seen:
            seen.add(feat)
            unique_features.append(feat)
        else:
            duplicates.append(feat)

    if duplicates:
        print(f"  ⚠️  Removed {len(duplicates)} duplicate features: {duplicates[:5]}")

    keep_features = unique_features

    # Report results
    if verbose:
        print("\n" + "=" * 80)
        print("FEATURE SELECTION FROM CONFIG")
        print("=" * 80)
        print(f"  Available features: {len(all_features)}")
        print(f"  Specified in config: {len(specified_features)}")
        print(f"  ✓ Kept (specified & available): {len(keep_features)}")

        # Features specified but not available (warning)
        missing = specified_set - available_set
        if missing:
            print(f"  ⚠️  Specified but NOT available: {len(missing)}")
            if len(missing) <= 10:
                for feat in sorted(missing):
                    print(f"     - {feat}")
            else:
                for feat in sorted(list(missing)[:10]):
                    print(f"     - {feat}")
                print(f"     ... and {len(missing) - 10} more")

        # Features available but not specified (removed)
        removed = available_set - specified_set
        if removed:
            print(f"  ℹ️  Available but NOT specified (removed): {len(removed)}")
            if verbose and len(removed) <= 5:
                for feat in sorted(list(removed)[:5]):
                    print(f"     - {feat}")

        print("=" * 80)

    return keep_features


def apply_feature_selection(
    df,
    feature_cols: List[str],
    feature_config_path: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """
    Apply feature selection to a DataFrame based on config.

    This is a convenience function that combines filtering with validation.

    Args:
        df: DataFrame (pandas or Spark)
        feature_cols: List of candidate feature columns
        feature_config_path: Path to feature config YAML (optional)
        verbose: Print selection information

    Returns:
        Filtered list of features to use for training

    Example:
        feature_cols = [c for c in df.columns if c not in ['sid', 'year', 'label']]
        selected_features = apply_feature_selection(
            df,
            feature_cols,
            feature_config_path='configs/data/features_mixed.yaml'
        )
        X = df[selected_features]
    """
    if feature_config_path is None:
        return feature_cols

    # Filter features
    selected_features = filter_features_by_config(
        feature_cols,
        feature_config_path=feature_config_path,
        verbose=verbose
    )

    return selected_features


def validate_features_exist(
    df,
    required_features: List[str],
    raise_error: bool = True
) -> Dict[str, bool]:
    """
    Validate that required features exist in DataFrame.

    Args:
        df: DataFrame (pandas or Spark)
        required_features: List of features that must exist
        raise_error: Raise ValueError if features missing (default: True)

    Returns:
        Dictionary mapping feature name to exists (bool)

    Raises:
        ValueError: If raise_error=True and features are missing
    """
    # Get column names (works for both pandas and Spark)
    if hasattr(df, 'columns'):
        available_cols = set(df.columns)
    else:
        raise ValueError("DataFrame must have 'columns' attribute")

    # Check each feature
    validation = {feat: feat in available_cols for feat in required_features}

    missing = [feat for feat, exists in validation.items() if not exists]

    if missing:
        msg = f"Missing {len(missing)} required features: {missing[:10]}"
        if raise_error:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return validation


# Convenience function for common use case
def get_features_from_config_or_all(
    all_features: List[str],
    config_path: Optional[str] = None
) -> List[str]:
    """
    Get features from config if provided, otherwise return all features.

    This is the simplest interface for feature selection.

    Args:
        all_features: All available features
        config_path: Optional path to feature config

    Returns:
        List of features to use
    """
    if config_path is None:
        return all_features

    return filter_features_by_config(
        all_features,
        feature_config_path=config_path,
        verbose=True
    )
