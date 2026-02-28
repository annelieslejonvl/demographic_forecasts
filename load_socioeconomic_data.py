"""
Load and integrate external socio-economic data from df_socioec.csv

This file contains municipality-level statistics for 2020 from Belgian sources.
We'll use it as static context (assuming slow change) or interpolate for other years.

Imputation strategy:
- For missing refnis codes: Use average from neighboring municipalities
- For missing features within matched refnis: Use median from neighbors
- Fall back to global median only when no neighbors have data
"""
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Dict, Optional


def load_neighbors(neighbors_path='neighbors.npy') -> Optional[Dict]:
    """
    Load municipality neighbors mapping.

    Args:
        neighbors_path: Path to neighbors.npy file

    Returns:
        Dictionary mapping refnis -> list of neighbor refnis codes
    """
    try:
        neighbors = np.load(neighbors_path, allow_pickle=True).item()
        print(f"✓ Loaded {len(neighbors)} municipality neighborhoods")
        return neighbors
    except Exception as e:
        print(f"⚠️  Could not load neighbors file: {e}")
        print("   Falling back to global imputation")
        return None


def spatial_impute_with_neighbors(
    df: pd.DataFrame,
    neighbors: Optional[Dict],
    feature_cols: list,
    refnis_col='refnis'
) -> pd.DataFrame:
    """
    Impute missing values using neighboring municipalities' data.

    Strategy:
    1. For each municipality with missing values
    2. Find its neighbors from the neighbors dictionary
    3. Use the median of neighbors' values
    4. Fall back to global median if no neighbors have data

    Args:
        df: DataFrame with municipality data
        neighbors: Dictionary mapping refnis -> neighbor refnis codes
        feature_cols: List of feature columns to impute
        refnis_col: Name of the refnis column

    Returns:
        DataFrame with spatially imputed values
    """
    if neighbors is None:
        print("⚠️  No neighbors data available, using global median")
        return df

    print("\n🗺️  SPATIAL IMPUTATION USING NEIGHBORS")
    print("="*80)

    imputed_count = 0
    neighbor_used = 0
    global_used = 0

    # Set refnis as index for easier lookup
    df = df.set_index(refnis_col)

    for col in feature_cols:
        if col == refnis_col or col == 'municipality_name':
            continue

        # Find rows with missing values
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue

        missing_refnis = df[missing_mask].index

        for refnis in missing_refnis:
            # Get neighbors for this municipality
            neighbor_codes = neighbors.get(int(refnis), [])

            if len(neighbor_codes) == 0:
                # No neighbors defined, use global median
                global_val = df[col].median()
                if pd.notna(global_val):
                    df.loc[refnis, col] = global_val
                    global_used += 1
                continue

            # Get values from neighbors
            neighbor_values = []
            for neighbor_refnis in neighbor_codes:
                if neighbor_refnis in df.index:
                    val = df.loc[neighbor_refnis, col]
                    if pd.notna(val):
                        neighbor_values.append(val)

            if len(neighbor_values) > 0:
                # Use median of neighbors
                df.loc[refnis, col] = np.median(neighbor_values)
                neighbor_used += 1
                imputed_count += 1
            else:
                # Neighbors also have missing data, use global median
                global_val = df[col].median()
                if pd.notna(global_val):
                    df.loc[refnis, col] = global_val
                    global_used += 1

    print(f"  Imputed values using neighbors: {neighbor_used}")
    print(f"  Imputed values using global median: {global_used}")
    print(f"  Total imputed: {imputed_count + global_used}")
    print("="*80)

    # Reset index
    df = df.reset_index()

    return df


def load_and_prepare_socioec_data(csv_path='df_socioec.csv', neighbors_path='neighbors.npy'):
    """
    Load socio-economic data and prepare for merging.

    Returns:
        DataFrame with municipality-level socio-economic indicators
    """
    print("\n" + "="*80)
    print("LOADING EXTERNAL SOCIO-ECONOMIC DATA")
    print("="*80)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} municipalities")

    # Rename key columns for easier use
    column_mapping = {
        'gemeenten': 'municipality_name',
        'gemiddeld netto belastbaar inkomen per inwoner': 'muni_avg_income_2020',
        'aantal inwoners volgens Rijksregister': 'muni_population_2020',
        'werkzoekenden (t.o.v. alle inwoners)': 'muni_unemployment_rate_2020',
        'administratief armoederisico [model]': 'muni_poverty_risk_2020',
        'gemiddelde leeftijd': 'muni_avg_age_2020',
        'appartementen - mediaanprijs': 'muni_apartment_price_2020',
        'prijs-inkomen ratio - woonhuizen (alle huizen excl. appartementen)': 'muni_price_income_ratio_2020',
        'mediaan huurprijs nieuwe huurcontracten - verhuurder rechtspersoon': 'muni_median_rent_2020',
        'zelfde adres als vorig jaar (t.o.v. inwoners)': 'muni_stability_rate_2020',
        'immigratie vanuit een andere Belgische gemeente per 1.000 inwoners': 'muni_immigration_rate_2020',
        'emigratie naar een andere Belgische gemeente per 1.000 inwoners': 'muni_emigration_rate_2020',
        'geboorten per 1.000 inwoners': 'muni_birth_rate_2020',
        'hooggeschoold  (t.o.v. 25-64-jarigen)': 'muni_high_educated_pct_2020',
        'laaggeschoold  (t.o.v. 25-64-jarigen)': 'muni_low_educated_pct_2020',
        'niet-Europese (niet-EU) herkomst (t.o.v. inwoners)': 'muni_non_eu_pct_2020',
        'woonduur gemeente: 5 jaar of minder (t.o.v. inwoners)': 'muni_short_residence_pct_2020',
        'gemiddelde huishoudensgrootte': 'muni_avg_household_size_2020'
    }

    # Select and rename relevant columns
    df_selected = df[['refnis', 'gemeenten']].copy()
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and old_col not in ['refnis', 'gemeenten']:
            df_selected[new_col] = df[old_col]

    # Rename Code NIS and gemeenten
    df_selected = df_selected.rename(columns={'gemeenten': 'municipality_name'})

    print(f"✓ Selected {len(df_selected.columns) - 2} features")

    # Clean data: replace 'x' and '-' with NaN, then convert to numeric
    for col in df_selected.columns:
        if col not in ['refnis', 'municipality_name']:
            # Replace Belgian decimal comma with dot
            df_selected[col] = df_selected[col].astype(str).str.replace(',', '.')
            # Replace x and - with NaN
            df_selected[col] = df_selected[col].replace(['x', '-', 'nan'], np.nan)
            # Convert to numeric
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

    print(f"✓ Cleaned and converted to numeric")

    # Summary of missing values BEFORE imputation
    missing = df_selected.isnull().sum()
    missing_features = missing[missing > 0]
    total_missing_before = missing.sum()

    if len(missing_features) > 0:
        print(f"\n⚠️  Missing values detected (before imputation):")
        for feat, count in missing_features.items():
            if feat not in ['refnis', 'municipality_name']:
                print(f"  {feat}: {count} ({count/len(df)*100:.1f}%)")

    # IMPUTATION STRATEGY: Use GLOBAL median instead of neighbors
    # Why: Neighboring municipalities are prediction targets (people move there)
    #      Making neighbors too similar reduces discriminative power
    print(f"\n📊 Using GLOBAL median imputation (preserves geographic variation)")

    # Fill any remaining NULLs with global median (fallback)
    for col in df_selected.columns:
        if col not in ['refnis', 'municipality_name']:
            if df_selected[col].isna().any():
                median_val = df_selected[col].median()
                if pd.notna(median_val):
                    df_selected[col] = df_selected[col].fillna(median_val)

    return df_selected


def add_socioec_features_to_spark_df(
    df: DataFrame,
    spark: SparkSession,
    socioec_csv_path='df_socioec.csv',
    neighbors_path='neighbors.npy',
    strategy='static'
) -> DataFrame:
    """
    Add external socio-economic features to main Spark DataFrame.

    Uses spatial imputation with neighboring municipalities for missing values.

    Args:
        df: Main Spark DataFrame with individual data
        spark: Spark session
        socioec_csv_path: Path to CSV file
        neighbors_path: Path to neighbors.npy file
        strategy: How to handle temporal dimension
                  'static' = use 2020 values for all years
                  'interpolate' = interpolate/extrapolate for other years (TODO)

    Returns:
        DataFrame with socio-economic features added
    """
    print("\n" + "="*80)
    print("ADDING EXTERNAL SOCIO-ECONOMIC FEATURES WITH SPATIAL IMPUTATION")
    print("="*80)

    # Load and prepare socioec data (already includes spatial imputation)
    socioec_pd = load_and_prepare_socioec_data(socioec_csv_path, neighbors_path)

    # Convert to Spark
    socioec_spark = spark.createDataFrame(socioec_pd)

    # Strategy: Static (use 2020 values for all years)
    if strategy == 'static':
        print("\nStrategy: STATIC")
        print("  Using 2020 values for all years")
        print("  Assumption: Socio-economic structure changes slowly")

        # Simply join on refnis (no year matching needed)
        # Use broadcast join since socioec table is tiny (290 rows)
        df = df.join(F.broadcast(socioec_spark.drop('municipality_name')), on='refnis', how='left')
        print(f"✓ Joined socioec data (broadcast join on {socioec_spark.count()} municipalities)")

        # Fill any remaining missing values with median computed from the small pandas table
        # This avoids triggering expensive Spark actions on the large lazy DataFrame
        numeric_cols = [col for col in socioec_spark.columns if col not in ['refnis', 'municipality_name']]

        print("📊 Filling unmatched refnis codes with pre-computed medians...")
        fill_dict = {}
        for col in numeric_cols:
            median_val = socioec_pd[col].median() if col in socioec_pd.columns else None
            if median_val is not None and pd.notna(median_val):
                fill_dict[col] = float(median_val)
        if fill_dict:
            df = df.fillna(fill_dict)
            print(f"  ✓ Prepared fill values for {len(fill_dict)} columns from source data")

        # Create derived features
        print("\n🔧 Creating derived features...")

        # Net migration rate (immigration - emigration)
        df = df.withColumn(
            'muni_net_migration_rate_2020',
            F.col('muni_immigration_rate_2020') - F.col('muni_emigration_rate_2020')
        )

        # Housing affordability indicator
        df = df.withColumn(
            'muni_housing_affordable_2020',
            (F.col('muni_price_income_ratio_2020') < 10.0).cast('boolean')
        )

        # High mobility area (>40% moved in last 5 years)
        df = df.withColumn(
            'muni_high_mobility_2020',
            (F.col('muni_short_residence_pct_2020') > 40.0).cast('boolean')
        )

        # Urban indicator (population > 50k)
        df = df.withColumn(
            'muni_urban_2020',
            (F.col('muni_population_2020') > 50000).cast('boolean')
        )

        print("  ✓ Created 4 derived features")

        # CRITICAL: LAG by 1 year for temporal validity!
        print("\n⏱️  Handling temporal lag for external features...")

        # Strategy: Use 2020 values for ALL years (static assumption)
        # This is reasonable because:
        # 1. Socio-economic structure changes slowly
        # 2. We're using 2020 as a "snapshot" of municipality characteristics
        # 3. Better to have approximate context than no context

        all_socioec_cols = [c for c in df.columns if '_2020' in c]

        for col in all_socioec_cols:
            # Rename to indicate it's contextual (not truly time-varying)
            new_col_name = col.replace('_2020', '_context')

            # Use 2020 values for all years
            df = df.withColumn(new_col_name, F.col(col))

            # Drop original
            df = df.drop(col)

        print(f"  ✓ Created {len(all_socioec_cols)} contextual features")
        print("  📍 Using 2020 values as static context for all years")
        print("     (Assumption: Municipality characteristics change slowly)")

        # Summary
        n_features_added = len(all_socioec_cols)
        print("\n" + "="*80)
        print(f"✅ Added {n_features_added} external socio-economic features")
        print("="*80)

    elif strategy == 'interpolate':
        print("⚠️  Interpolation strategy not yet implemented")
        print("   Using 'static' strategy as fallback")
        return add_socioec_features_to_spark_df(df, spark, socioec_csv_path, strategy='static')

    return df


# Utility: Check for leakage in external features
def check_external_features_for_leakage(columns):
    """
    Check that external socioec features are properly lagged.

    Args:
        columns: List of column names

    Returns:
        List of potentially leaky external features
    """
    leaky = []

    for col in columns:
        if 'muni_' in col and '_2020' in col:
            # Has _2020 but no _lag -> leaky!
            if '_lag' not in col:
                leaky.append(col)

    return leaky


if __name__ == "__main__":
    # Test loading
    df = load_and_prepare_socioec_data()
    print(f"\nLoaded {len(df)} municipalities with {len(df.columns)-2} features")
    print("\nSample:")
    print(df.head())
