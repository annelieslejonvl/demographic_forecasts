"""
Check for temporal data leakage in feature configurations.

This script identifies features that may cause temporal leakage when predicting
outcomes in year t using events from year t.

Usage:
    python check_temporal_leakage.py
    python check_temporal_leakage.py --config configs/data/socioec_features_1.yaml
"""
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Set


# Known problematic feature patterns
LEAKY_PATTERNS = {
    'event': [
        'birth1_event',
        'birth2_event',
        'divorce_event',
        'getalifeother_event',
    ],
    'interaction': [
        'divorce_x_age',
        'birth_x_age',
        'income_drop',
        'income_rise',
        'low_income_birth',
        'high_income_divorce',
        'low_income_divorce',
        'family_break',  # LEAKY: uses current hh_pos which can change due to moving
        'constrained_young_family',  # LEAKY: uses current hh_pos which can change due to moving
    ],
    'state_leaky': [
        # These state variables can change as a result of the outcome (moving)
        # and should always be lagged
        'hh_pos',  # Household position can change when moving - use hh_pos_lag1 instead!
    ],
}

# Safe feature patterns
SAFE_PATTERNS = {
    'lagged': ['_lag1', '_lag2', '_lag3'],
    'censored': ['_censored'],
    'state': ['age', 'MS_ADI', 'coupled', 'eerste_nationaliteit'],  # hh_pos removed - must be lagged!
    'safe_derived': [
        # These are safe versions of potentially leaky features
        'family_break_safe',  # Uses hh_pos_lag1 instead of hh_pos
        'constrained_young_family_safe',  # Uses hh_pos_lag1 instead of hh_pos
        'hh_pos_changed',  # Derived from lagged values
        'income_norm',
        'income_quintile',
        'recent_birth',
        'new_family',
        'family_expansion',
        'recent_mover',
        'frequent_mover',
        'age_x_life_event',
        'mobility_history',
        'total_recent_events',
        'multiple_events',
        'coupled_x_birth',
        'any_life_event_lag1',
        'recent_move_x_life_event',
        'age_norm',
        'young_parent',
        'older_parent',
        'partnership_income',
    ],
}


def load_feature_config(config_path: str) -> Dict:
    """Load feature configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_feature_leakage(feature: str) -> Dict[str, any]:
    """
    Check if a feature may cause temporal leakage.

    Returns:
        dict with keys: is_safe, category, reason
    """
    # Check if it's a safe lagged feature
    for pattern in SAFE_PATTERNS['lagged']:
        if pattern in feature:
            return {
                'is_safe': True,
                'category': 'lagged',
                'reason': f'Contains {pattern} (historical data)'
            }

    # Check if it's a censored (cumulative) feature
    for pattern in SAFE_PATTERNS['censored']:
        if pattern in feature:
            return {
                'is_safe': True,
                'category': 'censored',
                'reason': 'Cumulative history (safe)'
            }

    # Check if it's a safe derived feature (must check before state!)
    if feature in SAFE_PATTERNS['safe_derived']:
        return {
            'is_safe': True,
            'category': 'safe_derived',
            'reason': 'Safe derived feature (uses lagged values)'
        }

    # Check if it's a state variable
    for pattern in SAFE_PATTERNS['state']:
        if pattern in feature:
            return {
                'is_safe': True,
                'category': 'state',
                'reason': 'State variable (not event)'
            }

    # Check for leaky state variables (must be lagged!)
    if feature in LEAKY_PATTERNS['state_leaky']:
        return {
            'is_safe': False,
            'category': 'state_leakage',
            'reason': '⚠️  State variable that changes due to outcome - use lagged version!'
        }

    # Check for known leaky event features
    if feature in LEAKY_PATTERNS['event']:
        return {
            'is_safe': False,
            'category': 'event_leakage',
            'reason': '⚠️  Event in year t (may occur after outcome)'
        }

    # Check for interaction features that may contain leakage
    if feature in LEAKY_PATTERNS['interaction']:
        return {
            'is_safe': False,
            'category': 'interaction_leakage',
            'reason': '⚠️  Interaction with year t event or state'
        }

    # Check if it contains 'event' but not a lag
    if 'event' in feature and not any(lag in feature for lag in ['lag', 'censored']):
        return {
            'is_safe': False,
            'category': 'potential_event_leakage',
            'reason': '⚠️  Contains "event" without lag/censored'
        }

    # Unknown - needs manual review
    return {
        'is_safe': None,
        'category': 'unknown',
        'reason': '❓ Unknown - requires manual review'
    }


def analyze_config(config_path: str) -> Dict:
    """Analyze a feature configuration for temporal leakage."""
    config = load_feature_config(config_path)

    label_col = config.get('label_col', 'unknown')
    cat_cols = config.get('cat_cols', [])
    num_cols = config.get('num_cols', [])

    all_features = cat_cols + num_cols

    results = {
        'config_path': config_path,
        'label_col': label_col,
        'total_features': len(all_features),
        'safe_features': [],
        'leaky_features': [],
        'unknown_features': [],
        'feature_details': {}
    }

    for feature in all_features:
        check = check_feature_leakage(feature)
        results['feature_details'][feature] = check

        if check['is_safe'] is True:
            results['safe_features'].append(feature)
        elif check['is_safe'] is False:
            results['leaky_features'].append(feature)
        else:
            results['unknown_features'].append(feature)

    return results


def print_report(results: Dict) -> None:
    """Print a human-readable report."""
    print("=" * 80)
    print(f"TEMPORAL LEAKAGE ANALYSIS: {Path(results['config_path']).name}")
    print("=" * 80)
    print(f"Label: {results['label_col']}")
    print(f"Total features: {results['total_features']}")
    print()

    # Summary
    n_safe = len(results['safe_features'])
    n_leaky = len(results['leaky_features'])
    n_unknown = len(results['unknown_features'])

    print("SUMMARY:")
    print(f"  ✅ Safe features:     {n_safe:3d} ({n_safe/results['total_features']*100:.1f}%)")
    print(f"  ❌ Leaky features:    {n_leaky:3d} ({n_leaky/results['total_features']*100:.1f}%)")
    print(f"  ❓ Unknown features:  {n_unknown:3d} ({n_unknown/results['total_features']*100:.1f}%)")
    print()

    # Leaky features (detailed)
    if results['leaky_features']:
        print("❌ LEAKY FEATURES (REMOVE THESE):")
        print("-" * 80)
        for feat in results['leaky_features']:
            details = results['feature_details'][feat]
            print(f"  • {feat:30s} - {details['reason']}")
        print()

    # Unknown features (need review)
    if results['unknown_features']:
        print("❓ UNKNOWN FEATURES (MANUAL REVIEW NEEDED):")
        print("-" * 80)
        for feat in results['unknown_features']:
            details = results['feature_details'][feat]
            print(f"  • {feat:30s} - {details['reason']}")
        print()

    # Safe features (summary)
    if results['safe_features']:
        print(f"✅ SAFE FEATURES ({len(results['safe_features'])} total):")
        print("-" * 80)

        # Group by category
        by_category = {}
        for feat in results['safe_features']:
            cat = results['feature_details'][feat]['category']
            by_category.setdefault(cat, []).append(feat)

        for category, features in sorted(by_category.items()):
            print(f"  {category.upper()}: {len(features)} features")
        print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    if n_leaky > 0:
        print("⚠️  WARNING: This config contains features that may cause temporal leakage!")
        print()
        print("Action required:")
        print("  1. Remove the leaky features listed above")
        print("  2. Re-run your experiments with the cleaned config")
        print("  3. Compare model performance (AUC will likely drop by 0.05-0.15)")
        print("  4. Document the difference in your results")
        print()
        print(f"Expected performance impact:")
        print(f"  • AUC may drop by: ~{n_leaky / results['total_features'] * 0.15:.2f}")
        print(f"  • This is REALISTIC, not worse!")
    else:
        print("✅ This config appears safe from temporal leakage!")
        print()
        print("Note: Always verify that:")
        print("  • State variables are measured at the start of year t")
        print("  • Lagged features are truly from previous years")
        print("  • No hidden temporal dependencies exist")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Check feature configs for temporal leakage'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to feature config YAML (default: check all in configs/data/)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export cleaned config to this path (removes leaky features)'
    )

    args = parser.parse_args()

    if args.config:
        # Check single config
        configs = [args.config]
    else:
        # Check all socioec configs
        config_dir = Path('configs/data')
        configs = list(config_dir.glob('socioec_features*.yaml'))

    for config_path in configs:
        results = analyze_config(str(config_path))
        print_report(results)
        print("\n")

        # Export cleaned config if requested
        if args.export and len(configs) == 1:
            export_cleaned_config(config_path, results, args.export)


def export_cleaned_config(config_path: str, results: Dict, export_path: str):
    """Export a cleaned config with leaky features removed."""
    config = load_feature_config(config_path)

    # Remove leaky features
    leaky_features = set(results['leaky_features'])

    if 'cat_cols' in config:
        config['cat_cols'] = [f for f in config['cat_cols'] if f not in leaky_features]

    if 'num_cols' in config:
        config['num_cols'] = [f for f in config['num_cols'] if f not in leaky_features]

    # Add comment about removed features
    if leaky_features:
        config['_removed_features'] = {
            'reason': 'Temporal leakage prevention',
            'removed': sorted(list(leaky_features)),
            'count': len(leaky_features)
        }

    # Write cleaned config
    with open(export_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Cleaned config exported to: {export_path}")
    print(f"   Removed {len(leaky_features)} leaky features")


if __name__ == '__main__':
    main()
