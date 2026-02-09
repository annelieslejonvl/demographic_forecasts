#!/usr/bin/env python3
"""
Quick script to check class balance in your processed data.
This helps determine the optimal scale_pos_weight parameter.
"""
import sys

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Error: {e}")
    print("Install missing packages: pip install pandas pyarrow")
    sys.exit(1)

def check_balance(parquet_path):
    """Check class balance in parquet dataset."""
    print("="*80)
    print("CLASS BALANCE ANALYSIS")
    print("="*80)
    print(f"Reading: {parquet_path}")

    try:
        # Try to read with filters for efficiency
        df = pd.read_parquet(parquet_path, columns=['y_moved', 'year'])
        print(f"✓ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Overall balance
    print("\n" + "="*80)
    print("OVERALL CLASS BALANCE")
    print("="*80)
    n_positive = df['y_moved'].sum()
    n_negative = len(df) - n_positive
    ratio = n_negative / n_positive if n_positive > 0 else 0

    print(f"Positive (moved=1):  {n_positive:>12,} ({n_positive/len(df)*100:>5.2f}%)")
    print(f"Negative (moved=0):  {n_negative:>12,} ({n_negative/len(df)*100:>5.2f}%)")
    print(f"Ratio (neg/pos):     {ratio:>12.1f}")
    print()
    print(f"✅ Recommended scale_pos_weight: {ratio:.1f}")

    # By year
    if 'year' in df.columns:
        print("\n" + "="*80)
        print("CLASS BALANCE BY YEAR")
        print("="*80)
        print(f"{'Year':<8} {'Total':>12} {'Moved':>12} {'Rate':>8} {'Ratio':>8}")
        print("-"*80)

        year_stats = []
        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]
            n_pos = year_df['y_moved'].sum()
            n_total = len(year_df)
            rate = n_pos / n_total * 100 if n_total > 0 else 0
            year_ratio = (n_total - n_pos) / n_pos if n_pos > 0 else float('inf')
            year_stats.append({'year': year, 'total': n_total, 'moved': n_pos, 'rate': rate, 'ratio': year_ratio})

            print(f"{year:<8} {n_total:>12,} {n_pos:>12,} {rate:>7.2f}% {year_ratio:>7.1f}")

        # Check for temporal trends
        print("\n" + "="*80)
        print("TEMPORAL ANALYSIS")
        print("="*80)

        import numpy as np
        rates = [s['rate'] for s in year_stats]
        ratios = [s['ratio'] for s in year_stats if s['ratio'] != float('inf')]

        print(f"Move rate over time:")
        print(f"  Min:  {min(rates):.2f}% (year {year_stats[rates.index(min(rates))]['year']})")
        print(f"  Max:  {max(rates):.2f}% (year {year_stats[rates.index(max(rates))]['year']})")
        print(f"  Mean: {np.mean(rates):.2f}%")
        print(f"  Std:  {np.std(rates):.2f}%")

        if len(year_stats) >= 5:
            # Check for trend (simple linear)
            years_num = [s['year'] for s in year_stats]
            trend = np.polyfit(years_num, rates, 1)[0]
            if abs(trend) > 0.01:
                direction = "increasing" if trend > 0 else "decreasing"
                print(f"\n⚠️  TREND DETECTED: Move rate is {direction} over time ({trend:+.3f}% per year)")
                print(f"  → Consider using year-specific weights or sample weights")
            else:
                print(f"\n✓ Stable over time (trend: {trend:+.3f}% per year)")

        # Recommend strategy based on variance
        cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0
        print(f"\nCoefficient of variation: {cv:.3f}")
        if cv < 0.1:
            print("  ✓ Low variance - single scale_pos_weight is OK")
            print(f"    Recommended: {ratio:.1f}")
        elif cv < 0.25:
            print("  ⚠️  Moderate variance - consider:")
            print(f"    1. Use TRAIN set ratio for scale_pos_weight")
            print(f"    2. Or use sample_weight per year")
        else:
            print("  🔴 High variance - strongly recommend:")
            print(f"    1. Use sample_weight (different weight per year)")
            print(f"    2. Or train separate models per time period")

        # Train/test split analysis
        train_years = [s for s in year_stats if s['year'] < 2023]
        test_years = [s for s in year_stats if s['year'] >= 2023]

        if train_years and test_years:
            train_rate = sum(s['moved'] for s in train_years) / sum(s['total'] for s in train_years) * 100
            test_rate = sum(s['moved'] for s in test_years) / sum(s['total'] for s in test_years) * 100
            train_ratio = (sum(s['total'] for s in train_years) - sum(s['moved'] for s in train_years)) / sum(s['moved'] for s in train_years)

            print("\n" + "="*80)
            print("TRAIN/TEST SPLIT ANALYSIS (Years < 2023 vs >= 2023)")
            print("="*80)
            print(f"Train years: {[s['year'] for s in train_years]}")
            print(f"  Move rate: {train_rate:.2f}%")
            print(f"  Ratio: {train_ratio:.1f}")
            print(f"\nTest years: {[s['year'] for s in test_years]}")
            print(f"  Move rate: {test_rate:.2f}%")

            diff = abs(train_rate - test_rate)
            if diff > 0.5:
                print(f"\n⚠️  WARNING: {diff:.2f}% difference between train/test!")
                print(f"  → Use train ratio for scale_pos_weight: {train_ratio:.1f}")
                print(f"  → Performance may differ from validation to test")
            else:
                print(f"\n✓ Similar rates (diff: {diff:.2f}%)")
                print(f"  → Use overall or train ratio: {train_ratio:.1f}")

    # Recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)

    if ratio < 5:
        print("✓ Moderately imbalanced (ratio < 5)")
        print("  → Use scale_pos_weight: 3-5")
        print("  → Standard training should work")
    elif ratio < 15:
        print("⚠️  Significantly imbalanced (ratio 5-15)")
        print(f"  → Use scale_pos_weight: {ratio:.1f}")
        print("  → Consider max_delta_step: 1")
        print("  → Optimize for aucpr, not logloss")
    elif ratio < 50:
        print("🔴 Highly imbalanced (ratio 15-50)")
        print(f"  → Use scale_pos_weight: {ratio:.1f}")
        print("  → Use max_delta_step: 1-2")
        print("  → Consider SMOTE or undersampling")
        print("  → Focus on top-k metrics, not overall F1")
    else:
        print("🔴 EXTREME imbalance (ratio > 50)")
        print("  → Individual prediction may not be feasible")
        print("  → Consider aggregate/group-level modeling")
        print("  → Or multi-class formulation")

    # If temporal variance is high, add sample weight recommendation
    if 'year' in df.columns and len(year_stats) > 1:
        cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0
        if cv > 0.15:
            print("\n💡 TEMPORAL VARIANCE DETECTED")
            print("   Consider using sample weights instead of fixed scale_pos_weight:")
            print("   ")
            print("   # In your training code:")
            print("   sample_weights = df['year'].map({")
            for s in year_stats:
                w = s['ratio'] / ratio  # normalize to overall ratio
                print(f"       {s['year']}: {w:.3f},  # {s['rate']:.2f}% move rate")
            print("   })")
            print("   model.fit(X_train, y_train, sample_weight=sample_weights)")

    print("\n📝 Update your config file:")
    # Use train ratio if available, otherwise overall
    recommended_weight = train_ratio if 'train_ratio' in locals() and train_ratio != float('inf') else ratio
    print(f"   scale_pos_weight: {recommended_weight:.1f}")
    print()

if __name__ == "__main__":
    # Default paths
    paths = [
        "data/processed_features_enhanced",
        "data/processed_features",
    ]

    # Check if user provided path
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]

    # Try each path
    for path in paths:
        import os
        if os.path.exists(path):
            check_balance(path)
            break
    else:
        print(f"Error: No processed data found at:")
        for p in paths:
            print(f"  - {p}")
        print("\nRun feature engineering first:")
        print("  python run_test.py --max-rows=1000000")
