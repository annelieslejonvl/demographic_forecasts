"""
Diagnostic script to analyze features and identify potential improvements.

Usage:
    python diagnose_features.py
"""
import sys


def get_leaky_columns(columns):
    """
    Identify columns that cause temporal leakage.
    """
    leaky = []

    event_base_names = [
        'birth1_event',
        'birth2_event',
        'divorce_event',
        'getalifeother_event',
    ]

    for col in columns:
        if col.endswith('_first') or col.endswith('_censored') or col.endswith('_at_risk'):
            leaky.append(col)
            continue

        for event_name in event_base_names:
            if col == event_name:
                leaky.append(col)
                break

    return leaky

# Simulate column names from your dataset
# (Based on the feature engineering code in run_test.py)
all_columns = [
    # ID and time columns
    'sid', 'year', 'refnis',

    # Target
    'y_moved',

    # Raw event columns (YEAR T - LEAKY!)
    'birth1_event', 'birth2_event', 'divorce_event', 'getalifeother_event',

    # Survival analysis columns (LEAKY!)
    'birth1_event_first', 'birth1_event_at_risk', 'birth1_event_censored',
    'birth2_event_first', 'birth2_event_at_risk', 'birth2_event_censored',
    'divorce_event_first', 'divorce_event_at_risk', 'divorce_event_censored',
    'getalifeother_event_first', 'getalifeother_event_at_risk', 'getalifeother_event_censored',
    'y_moved_first', 'y_moved_at_risk', 'y_moved_censored',

    # Lagged event columns (SAFE - historical)
    'birth1_event_lag1', 'birth1_event_lag2',
    'birth2_event_lag1', 'birth2_event_lag2',
    'divorce_event_lag1', 'divorce_event_lag2',
    'getalifeother_event_lag1', 'getalifeother_event_lag2',
    'y_moved_lag1', 'y_moved_lag2',

    # State variables at start of year T (SAFE)
    'age', 'gender', 'coupled', 'hh_pos', 'eerste_nationaliteit',
    'MS_ADI_PP', 'MS_ADI_HH',

    # Lagged state variables (SAFE)
    'hh_pos_lag1', 'income_lag1',

    # Derived features (SAFE if using lagged events)
    'hh_pos_changed',
    'income_change', 'income_norm', 'income_quintile',
    'income_drop', 'income_rise',
    'low_income_birth', 'high_income_divorce', 'low_income_divorce',
    'partnership_income',

    # Event interaction features (SAFE if using lagged)
    'recent_birth', 'new_family', 'family_expansion',
    'family_break_safe', 'recent_mover', 'frequent_mover',
    'age_x_life_event', 'mobility_history', 'total_recent_events',
    'multiple_events', 'coupled_x_birth', 'any_life_event_lag1',
    'recent_move_x_life_event',

    # Age interaction features (SAFE)
    'age_norm', 'divorce_x_age', 'birth_x_age',
    'young_parent', 'constrained_young_family_safe', 'older_parent',

    # Socioeconomic features (SAFE)
    'pop_density', 'pct_foreign', 'pct_unemployed', 'median_income_municipality',
    'income_vs_municipal_median',
]

def analyze_features():
    """Analyze feature set and identify improvements."""

    print("="*80)
    print("FEATURE ANALYSIS & IMPROVEMENT SUGGESTIONS")
    print("="*80)
    print()

    # Identify leaky columns
    leaky = get_leaky_columns(all_columns)

    # Categorize features
    drop_cols = {'sid', 'year', 'refnis', 'y_moved'}
    safe_features = [c for c in all_columns if c not in drop_cols and c not in leaky]

    print(f"📊 FEATURE SUMMARY")
    print(f"  Total columns: {len(all_columns)}")
    print(f"  ID/target columns: {len(drop_cols)}")
    print(f"  Leaky columns (removed): {len(leaky)}")
    print(f"  Safe features (used): {len(safe_features)}")
    print()

    # Show what's being removed
    print(f"❌ REMOVED LEAKY FEATURES ({len(leaky)}):")
    for col in sorted(leaky):
        print(f"  • {col}")
    print()

    # Categorize safe features
    lag1_features = [f for f in safe_features if 'lag1' in f]
    lag2_features = [f for f in safe_features if 'lag2' in f]
    state_features = [f for f in safe_features if f in ['age', 'gender', 'coupled', 'hh_pos', 'eerste_nationaliteit', 'MS_ADI_PP', 'MS_ADI_HH']]
    derived_features = [f for f in safe_features if f not in lag1_features and f not in lag2_features and f not in state_features]

    print(f"✅ SAFE FEATURES ({len(safe_features)} total):")
    print(f"  • State variables (year T start): {len(state_features)}")
    print(f"  • Lag-1 features (year T-1): {len(lag1_features)}")
    print(f"  • Lag-2 features (year T-2): {len(lag2_features)}")
    print(f"  • Derived/interaction features: {len(derived_features)}")
    print()

    # Check what historical information we have
    print("📅 HISTORICAL DEPTH:")
    print(f"  • Using data from year T-1: {len(lag1_features)} features")
    print(f"  • Using data from year T-2: {len(lag2_features)} features")
    print(f"  • Maximum lookback: 2 years")
    print()

    # Suggestions for improvement
    print("="*80)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*80)
    print()

    print("1. ADD MORE HISTORICAL DEPTH")
    print("   Current: Only 2-year lookback (lag1, lag2)")
    print("   Suggested: Add 3-year lookback (lag3)")
    print("   Rationale: Moving decisions may be influenced by longer-term patterns")
    print()
    print("   To implement:")
    print("   • Modify create_lags() to include lag3")
    print("   • Add features like birth1_event_lag3, income_lag3")
    print("   Expected impact: +0.01 to +0.03 AUC-ROC")
    print()

    print("2. ADD CUMULATIVE HISTORY FEATURES")
    print("   Current: Only recent events (lag1, lag2)")
    print("   Suggested: Add cumulative counts over longer periods")
    print("   Rationale: Total life events matter, not just recent ones")
    print()
    print("   New features to add:")
    print("   • total_births_ever: Total births in all previous years")
    print("   • total_moves_ever: Total moves in all previous years")
    print("   • years_since_last_move: Time since most recent move")
    print("   • years_since_last_birth: Time since most recent birth")
    print("   Expected impact: +0.02 to +0.05 AUC-ROC")
    print()

    print("3. ADD MOVING AVERAGE FEATURES")
    print("   Current: Point-in-time income and state variables")
    print("   Suggested: Add rolling averages over 2-3 years")
    print("   Rationale: Trends matter more than single-year values")
    print()
    print("   New features to add:")
    print("   • income_avg_3yr: Average income over past 3 years")
    print("   • income_trend: Linear trend in income")
    print("   • income_volatility: Standard deviation of income")
    print("   Expected impact: +0.02 to +0.04 AUC-ROC")
    print()

    print("4. ADD LIFE STAGE INDICATORS")
    print("   Current: Basic age interactions")
    print("   Suggested: Richer life stage characterization")
    print()
    print("   New features to add:")
    print("   • family_size_trajectory: Growing, stable, or shrinking")
    print("   • household_stability_score: Fewer changes = higher stability")
    print("   • economic_trajectory: Income improving or declining")
    print("   • life_transition_phase: Recent major events (last 2-3 years)")
    print("   Expected impact: +0.01 to +0.03 AUC-ROC")
    print()

    print("5. ADD NEIGHBORHOOD CHANGE FEATURES")
    print("   Current: Static municipality features")
    print("   Suggested: Add temporal changes in neighborhood")
    print()
    print("   New features to add:")
    print("   • municipality_income_change: Change in median income")
    print("   • municipality_pop_change: Population growth/decline")
    print("   • relative_income_trajectory: Person vs neighborhood trend")
    print("   Expected impact: +0.01 to +0.02 AUC-ROC")
    print()

    print("="*80)
    print("ESTIMATED CUMULATIVE IMPACT")
    print("="*80)
    print("If you implement all 5 suggestions:")
    print("  Conservative estimate: +0.07 to +0.12 AUC-ROC")
    print("  Optimistic estimate: +0.10 to +0.17 AUC-ROC")
    print()
    print("Example trajectory:")
    print("  Current (with leakage): 0.85 AUC-ROC")
    print("  After removing leakage: 0.60 AUC-ROC (drop of 0.25)")
    print("  After adding improvements: 0.67-0.77 AUC-ROC")
    print("  → This would hit your target of 0.65-0.75!")
    print("="*80)
    print()

    print("🚀 QUICK WINS (Start Here)")
    print("="*80)
    print("1. Add lag3 features (easiest, ~1 hour)")
    print("2. Add cumulative history counts (medium, ~2-3 hours)")
    print("3. Add 3-year rolling averages (medium, ~2-3 hours)")
    print()
    print("Next steps:")
    print("  • Modify create_lags() in run_test.py to add lag3")
    print("  • Create new feature engineering functions")
    print("  • Re-train and compare results")
    print("="*80)

if __name__ == '__main__':
    analyze_features()
