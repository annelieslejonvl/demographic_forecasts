#!/usr/bin/env python3
"""
Test spatial imputation with neighbors for socioeconomic data.
"""
import pandas as pd
import numpy as np
from load_socioeconomic_data import load_neighbors, spatial_impute_with_neighbors


def test_spatial_imputation():
    """Test the spatial imputation functionality."""

    print("=" * 80)
    print("TESTING SPATIAL IMPUTATION WITH NEIGHBORS")
    print("=" * 80)

    # Create sample data mimicking Belgian municipality data
    # Using real refnis codes from the neighbors dictionary
    sample_data = pd.DataFrame({
        'refnis': [71071, 72041, 73107, 73001, 11022],
        'municipality_name': ['Muni_A', 'Muni_B', 'Muni_C', 'Muni_D', 'Muni_E'],
        'avg_income': [25000, None, 30000, 28000, None],  # Missing for 72041 and 11022
        'unemployment': [5.0, 6.5, None, 4.2, 7.0],  # Missing for 73107
        'population': [50000, 45000, 60000, None, 40000],  # Missing for 73001
    })

    print("\n📊 BEFORE SPATIAL IMPUTATION:")
    print(sample_data)
    print(f"\nTotal NULLs: {sample_data.isnull().sum().sum()}")

    # Load real neighbors
    neighbors = load_neighbors('neighbors.npy')

    if neighbors is None:
        print("❌ Cannot test without neighbors file")
        return

    # Show neighbors for municipalities with missing data
    print("\n🗺️  Neighbors for municipalities with missing data:")
    for refnis in [72041, 11022, 73107, 73001]:
        if refnis in sample_data['refnis'].values:
            neighbor_codes = neighbors.get(refnis, [])
            print(f"  refnis {refnis}: {len(neighbor_codes)} neighbors - {neighbor_codes[:5]}...")

    # Apply spatial imputation
    feature_cols = ['avg_income', 'unemployment', 'population']
    imputed_data = spatial_impute_with_neighbors(
        sample_data.copy(),
        neighbors,
        feature_cols,
        refnis_col='refnis'
    )

    print("\n📊 AFTER SPATIAL IMPUTATION:")
    print(imputed_data)
    print(f"\nRemaining NULLs: {imputed_data.isnull().sum().sum()}")

    # Verify improvement
    nulls_before = sample_data.isnull().sum().sum()
    nulls_after = imputed_data.isnull().sum().sum()

    if nulls_after < nulls_before:
        print(f"\n✅ SUCCESS: Reduced NULLs from {nulls_before} to {nulls_after}")
        print(f"   Improvement: {((nulls_before - nulls_after) / nulls_before * 100):.1f}%")
    else:
        print(f"\n⚠️  No improvement in NULL count")

    print("\n" + "=" * 80)


def test_full_workflow():
    """Test the full workflow with actual socioeconomic data."""

    print("\n" + "=" * 80)
    print("TESTING FULL WORKFLOW WITH ACTUAL DATA")
    print("=" * 80)

    from load_socioeconomic_data import load_and_prepare_socioec_data

    # Load data with spatial imputation
    df = load_and_prepare_socioec_data('df_socioec.csv', 'neighbors.npy')

    print(f"\n✅ Final data shape: {df.shape}")

    # Check for remaining NULLs
    nulls = df.isnull().sum().sum()
    print(f"   Total NULLs: {nulls}")

    if nulls == 0:
        print("   🎉 All values successfully imputed!")
    else:
        print(f"   ⚠️  {nulls} NULLs remain")

        # Show which columns still have NULLs
        null_cols = df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        if len(null_cols) > 0:
            print("\n   Columns with remaining NULLs:")
            for col, count in null_cols.items():
                print(f"     {col}: {count}")

    print("=" * 80)


if __name__ == "__main__":
    test_spatial_imputation()
    test_full_workflow()
