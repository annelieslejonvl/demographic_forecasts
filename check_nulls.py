#!/usr/bin/env python3
"""
Check for NULL values in processed features that cause rows to be dropped during training.
"""
import pandas as pd
import sys

def check_nulls(parquet_path):
    """Check NULL values in processed features."""
    print("="*80)
    print("NULL VALUE ANALYSIS")
    print("="*80)

    try:
        # Load a sample
        df = pd.read_parquet(parquet_path)
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")

        # Check NULL counts per column
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].sort_values(ascending=False)

        if len(null_cols) == 0:
            print("\n✓ No NULL values found!")
            return

        print(f"\n⚠️  Found {len(null_cols)} columns with NULL values:")
        print("-"*80)
        print(f"{'Column':<50} {'NULLs':>10} {'%':>8}")
        print("-"*80)

        for col, count in null_cols.head(30).items():
            pct = count / len(df) * 100
            print(f"{col:<50} {count:>10,} {pct:>7.1f}%")

        # Check rows with ANY NULL
        rows_with_any_null = df.isnull().any(axis=1).sum()
        pct_rows_null = rows_with_any_null / len(df) * 100

        print("\n" + "="*80)
        print("IMPACT ON TRAINING")
        print("="*80)
        print(f"Rows with ANY NULL value: {rows_with_any_null:,} ({pct_rows_null:.1f}%)")
        print(f"Usable rows (no NULLs):   {len(df) - rows_with_any_null:,} ({100-pct_rows_null:.1f}%)")

        if pct_rows_null > 50:
            print("\n🔴 CRITICAL: > 50% of rows have NULL values!")
            print("   → XGBoost will drop these rows")
            print("   → This explains why training is minimal")
        elif pct_rows_null > 20:
            print("\n⚠️  WARNING: > 20% of rows have NULL values")
            print("   → Significant data loss during training")

        # Identify feature groups with most NULLs
        print("\n" + "="*80)
        print("NULL VALUES BY FEATURE GROUP")
        print("="*80)

        groups = {
            'municipality aggregates': [c for c in null_cols.index if c.startswith('muni_') and '_context' not in c],
            'external socioec': [c for c in null_cols.index if '_context' in c],
            'lag features': [c for c in null_cols.index if '_lag' in c and not c.startswith('muni_')],
            'other': [c for c in null_cols.index if not c.startswith('muni_') and '_lag' not in c and '_context' not in c]
        }

        for group_name, cols in groups.items():
            if cols:
                avg_null_pct = null_counts[cols].mean() / len(df) * 100
                print(f"{group_name:30s}: {len(cols):3d} cols, avg {avg_null_pct:5.1f}% NULL")

        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)

        if pct_rows_null > 20:
            print("1. Fill NULL values with medians/means before training")
            print("2. Or: Remove features with >30% NULL values")
            print("3. Or: Check refnis matching (municipality mismatch?)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed_features_with_municipality"
    check_nulls(path)
