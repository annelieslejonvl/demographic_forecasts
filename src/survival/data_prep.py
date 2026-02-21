"""
Convert panel data (person x year) to survival analysis format.

For each individual at each observation year, computes:
- duration: years until the NEXT occurrence of the event (or until censoring)
- event_indicator: 1 if event was observed, 0 if right-censored

Handles recurrent events (moving, births) using gap-time approach:
at each observation year, model time to the NEXT occurrence.
"""
import numpy as np
import pandas as pd


def create_survival_dataset(
    df,
    event_col,
    id_col='sid',
    time_col='year',
):
    """
    Compute duration and event indicator for a single event type.

    For each person-year row, looks FORWARD to find the next occurrence
    of the event. If the event occurs, duration = (event_year - current_year)
    and event_indicator = 1. If not, duration = (last_observed_year - current_year)
    and event_indicator = 0 (right-censored).

    Args:
        df: pandas DataFrame in panel format (person x year)
        event_col: Column name for the event indicator (e.g., 'y_moved')
        id_col: Individual ID column
        time_col: Time column (year)

    Returns:
        DataFrame with added columns:
            {event_col}_duration: time to next event or censoring
            {event_col}_event_observed: 1 if event observed, 0 if censored
    """
    duration_col = f"{event_col}_duration"
    observed_col = f"{event_col}_event_observed"

    df = df.sort_values([id_col, time_col]).copy()

    # For each person, find the last observed year (for censoring)
    last_year = df.groupby(id_col)[time_col].transform('max')

    # For each person-year, find the next year where event == 1
    # Strategy: within each person, for rows where event == 1,
    # broadcast the event year forward to all earlier rows.

    # Mark event years
    event_mask = df[event_col].astype(int) == 1
    df['_event_year'] = np.where(event_mask, df[time_col], np.nan)

    # For each person-year, find the NEXT event year (strictly after current year)
    # We do this by sorting and using a reverse cumulative minimum approach.
    # For each row, find the minimum event_year that is > current_year.

    durations = np.full(len(df), np.nan)
    events = np.zeros(len(df), dtype=np.int32)

    # Group by individual for efficiency
    grouped = df.groupby(id_col)
    for _, group in grouped:
        idx = group.index
        years = group[time_col].values
        event_years_raw = group['_event_year'].values
        person_last_year = years[-1]

        # Collect all event years for this person (non-NaN)
        actual_event_years = event_years_raw[~np.isnan(event_years_raw)]

        for i, (row_idx, current_year) in enumerate(zip(idx, years)):
            # Find next event year strictly AFTER current year
            future_events = actual_event_years[actual_event_years > current_year]

            if len(future_events) > 0:
                next_event_year = future_events[0]  # earliest future event
                durations[df.index.get_loc(row_idx)] = next_event_year - current_year
                events[df.index.get_loc(row_idx)] = 1
            else:
                # Right-censored: no future event observed
                durations[df.index.get_loc(row_idx)] = person_last_year - current_year
                events[df.index.get_loc(row_idx)] = 0

    df[duration_col] = durations
    df[observed_col] = events

    # Drop helper column
    df.drop(columns=['_event_year'], inplace=True)

    # Handle duration == 0 for censored observations at last year
    # (person at their last observed year with no event → duration 0, censored)
    # Keep as-is: duration=0, event=0 means "censored immediately" which is valid

    # Minimum duration of 0.5 for AFT models (log(0) is undefined)
    # Only for duration == 0: set to 0.5 (half a year)
    zero_duration = df[duration_col] == 0
    df.loc[zero_duration, duration_col] = 0.5

    n_events = events.sum()
    n_censored = len(events) - n_events
    print(f"  {event_col}: {n_events:,} events, {n_censored:,} censored "
          f"({n_events / len(events):.1%} event rate)")

    return df


def create_survival_dataset_spark(
    df,
    event_col,
    id_col='sid',
    time_col='year',
):
    """
    Spark version of survival dataset creation.

    Computes duration and event indicator using window functions
    for efficient processing of large datasets.

    Args:
        df: Spark DataFrame in panel format
        event_col: Column name for the event indicator
        id_col: Individual ID column
        time_col: Time column

    Returns:
        Spark DataFrame with added duration and event_observed columns
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    duration_col = f"{event_col}_duration"
    observed_col = f"{event_col}_event_observed"

    # Window: for each person, ordered by year, looking forward
    w_forward = Window.partitionBy(id_col).orderBy(time_col) \
        .rowsBetween(1, Window.unboundedFollowing)

    # Window: for each person (unbounded)
    w_all = Window.partitionBy(id_col)

    # Last observed year per person
    df = df.withColumn('_last_year', F.max(F.col(time_col)).over(w_all))

    # Find next event year: minimum year > current year where event == 1
    # Mark event years (NULL if no event)
    df = df.withColumn(
        '_event_year',
        F.when(F.col(event_col).cast('int') == 1, F.col(time_col))
    )

    # Next event year = minimum of _event_year in forward window
    df = df.withColumn(
        '_next_event_year',
        F.min('_event_year').over(w_forward)
    )

    # Duration: if next event exists → next_event_year - current_year
    #           else → last_year - current_year (censored)
    df = df.withColumn(
        duration_col,
        F.when(
            F.col('_next_event_year').isNotNull(),
            F.col('_next_event_year') - F.col(time_col)
        ).otherwise(
            F.col('_last_year') - F.col(time_col)
        ).cast('float')
    )

    # Event observed: 1 if next event exists, 0 otherwise
    df = df.withColumn(
        observed_col,
        F.when(F.col('_next_event_year').isNotNull(), F.lit(1))
        .otherwise(F.lit(0))
        .cast('int')
    )

    # Handle duration == 0 → set to 0.5 for AFT compatibility
    df = df.withColumn(
        duration_col,
        F.when(F.col(duration_col) == 0, F.lit(0.5))
        .otherwise(F.col(duration_col))
    )

    # Clean up helper columns
    df = df.drop('_last_year', '_event_year', '_next_event_year')

    return df


def create_all_survival_datasets(
    df,
    event_cols,
    id_col='sid',
    time_col='year',
    use_spark=False,
):
    """
    Create survival labels for all event types.

    Args:
        df: DataFrame (pandas or Spark)
        event_cols: List of event column names
        id_col: Individual ID column
        time_col: Time column
        use_spark: If True, use Spark implementation

    Returns:
        DataFrame with duration and event_observed columns for each event
    """
    print(f"\nCreating survival datasets for {len(event_cols)} events...")

    create_fn = create_survival_dataset_spark if use_spark else create_survival_dataset

    for event_col in event_cols:
        df = create_fn(df, event_col, id_col=id_col, time_col=time_col)

    return df


def add_duration_and_event_columns(
    df,
    event_col,
    id_col='sid',
    time_col='year',
):
    """
    Convenience wrapper that adds survival columns to an existing DataFrame.
    Same as create_survival_dataset but with a clearer name for the pipeline.
    """
    return create_survival_dataset(df, event_col, id_col=id_col, time_col=time_col)


def create_aft_labels(duration, event_observed):
    """
    Create AFT-compatible labels for XGBoost.

    XGBoost AFT requires:
    - y_lower_bound: lower bound for the time-to-event
    - y_upper_bound: upper bound for the time-to-event

    For right-censored data:
    - Uncensored: y_lower = y_upper = duration
    - Censored: y_lower = duration, y_upper = +inf

    Args:
        duration: array of durations
        event_observed: array of event indicators (1=event, 0=censored)

    Returns:
        y_lower, y_upper arrays
    """
    duration = np.asarray(duration, dtype=np.float32)
    event_observed = np.asarray(event_observed, dtype=np.int32)

    y_lower = duration.copy()
    y_upper = duration.copy()

    # For censored observations, upper bound is infinity
    censored = event_observed == 0
    y_upper[censored] = np.inf

    return y_lower, y_upper


def create_cox_labels(duration, event_observed):
    """
    Create Cox-compatible labels for XGBoost.

    XGBoost Cox expects:
    - Positive value: uncensored, value = duration
    - Negative value: censored, abs(value) = duration

    Args:
        duration: array of durations
        event_observed: array of event indicators (1=event, 0=censored)

    Returns:
        y_cox: array of signed durations
    """
    duration = np.asarray(duration, dtype=np.float32)
    event_observed = np.asarray(event_observed, dtype=np.int32)

    y_cox = duration.copy()
    censored = event_observed == 0
    y_cox[censored] = -y_cox[censored]

    return y_cox


def get_horizon_labels(duration, event_observed, horizon):
    """
    Create binary labels for a specific time horizon.

    Args:
        duration: array of durations
        event_observed: array of event indicators
        horizon: time horizon in years (e.g., 1, 3, 5)

    Returns:
        y_binary: 1 if event occurred within horizon, 0 otherwise
    """
    duration = np.asarray(duration)
    event_observed = np.asarray(event_observed)

    # Event occurred AND within the horizon
    y_binary = ((event_observed == 1) & (duration <= horizon)).astype(np.int32)

    return y_binary


def _compute_survival_labels_vectorized(slim_df, event_col, id_col, time_col):
    """
    Vectorized survival label computation using reverse cummin.

    For each person-year row, finds the next future event year using
    a reverse cumulative minimum of event years within each person.
    Avoids per-person Python loops entirely.

    Operates on numpy arrays and temporary Series to minimize
    memory copies (no DataFrame.copy()).

    Args:
        slim_df: DataFrame sorted by (id_col, time_col) with event_col.
                 Must have a RangeIndex (0..n-1).
        event_col: Event column name
        id_col: Individual ID column
        time_col: Time column

    Returns:
        (duration, event_observed) arrays aligned with slim_df index
    """
    n = len(slim_df)
    sid = slim_df[id_col].values
    year = slim_df[time_col].values.astype(np.float64)
    event = slim_df[event_col].values.astype(np.int32)

    # Mark event years (NaN for non-events)
    evt_yr = np.where(event == 1, year, np.nan)

    # Shift by -1 within each person: exclude current row's event
    # A person boundary is where sid[i] != sid[i-1]
    # shift(-1) means: evt_yr_shifted[i] = evt_yr[i+1] if same person, else NaN
    evt_yr_shifted = np.empty(n, dtype=np.float64)
    evt_yr_shifted[-1] = np.nan  # last row has no next
    evt_yr_shifted[:-1] = evt_yr[1:]
    # NaN at person boundaries
    person_boundary = np.empty(n, dtype=bool)
    person_boundary[0] = False
    person_boundary[1:] = sid[1:] != sid[:-1]
    # The last row of each person should be NaN (no shift into next person)
    # person_boundary[i] is True when row i starts a new person
    # So row i-1 is the last of the previous person
    last_of_person = np.empty(n, dtype=bool)
    last_of_person[:-1] = person_boundary[1:]
    last_of_person[-1] = True
    evt_yr_shifted[last_of_person] = np.nan

    # Reverse cummin within each person using pandas Series (vectorized C code)
    # Create a temporary Series with the shifted event years and use
    # groupby().cummin() on the reversed data
    _s = pd.Series(evt_yr_shifted[::-1].copy())
    _g = pd.Series(sid[::-1].copy())
    next_evt_rev = _s.groupby(_g).cummin()
    next_evt = next_evt_rev.values[::-1].copy()
    del _s, _g, next_evt_rev

    # Last observed year per person
    _yr = pd.Series(year, copy=False)
    _sid = pd.Series(sid, copy=False)
    last_year_arr = _yr.groupby(_sid).transform('max').values

    # Compute duration and event_observed
    has_event = ~np.isnan(next_evt)
    duration = np.where(has_event, next_evt - year, last_year_arr - year)
    event_observed = has_event.astype(np.int32)

    # Duration 0 -> 0.5 for AFT compatibility
    duration = np.where(duration == 0, 0.5, duration).astype(np.float32)

    n_events = event_observed.sum()
    n_total = len(event_observed)
    print(f"    {event_col}: {n_events:,} events, {n_total - n_events:,} censored "
          f"({n_events / n_total:.1%} event rate)")

    return duration, event_observed


def _compute_survival_labels_polars(slim_df, event_col, id_col, time_col):
    """
    Polars-optimized survival label computation (2-3x faster than pandas).

    Uses Polars lazy evaluation and optimized groupby operations for
    faster reverse cummin computation.

    Args:
        slim_df: Polars DataFrame or pandas DataFrame (will convert)
        event_col: Event column name
        id_col: Individual ID column
        time_col: Time column

    Returns:
        (duration, event_observed) numpy arrays
    """
    try:
        import polars as pl
    except ImportError:
        print("    Polars not installed, falling back to pandas implementation")
        return _compute_survival_labels_vectorized(slim_df, event_col, id_col, time_col)

    # Convert to Polars if needed
    if not isinstance(slim_df, pl.DataFrame):
        df = pl.from_pandas(slim_df[[id_col, time_col, event_col]])
    else:
        df = slim_df.select([id_col, time_col, event_col])

    # Mark event years (null for non-events)
    df = df.with_columns([
        pl.when(pl.col(event_col) == 1)
        .then(pl.col(time_col))
        .otherwise(None)
        .alias('event_year')
    ])

    # Shift event_year within each person (exclude current row)
    df = df.with_columns([
        pl.col('event_year').shift(-1).over(id_col).alias('next_event_year_shifted')
    ])

    # Reverse cumulative minimum of future events within each person
    df = df.sort([id_col, pl.col(time_col).reverse()])
    df = df.with_columns([
        pl.col('next_event_year_shifted').cum_min().over(id_col).alias('next_event_year')
    ])
    df = df.sort([id_col, time_col])

    # Last observed year per person
    df = df.with_columns([
        pl.col(time_col).max().over(id_col).alias('last_year')
    ])

    # Compute duration and event_observed
    df = df.with_columns([
        pl.when(pl.col('next_event_year').is_not_null())
        .then(pl.col('next_event_year') - pl.col(time_col))
        .otherwise(pl.col('last_year') - pl.col(time_col))
        .alias('duration'),

        pl.col('next_event_year').is_not_null().cast(pl.Int32).alias('event_observed')
    ])

    # Duration 0 -> 0.5 for AFT compatibility
    df = df.with_columns([
        pl.when(pl.col('duration') == 0)
        .then(0.5)
        .otherwise(pl.col('duration'))
        .cast(pl.Float32)
        .alias('duration')
    ])

    # Extract results
    duration = df['duration'].to_numpy()
    event_observed = df['event_observed'].to_numpy()

    n_events = event_observed.sum()
    n_total = len(event_observed)
    print(f"    {event_col}: {n_events:,} events, {n_total - n_events:,} censored "
          f"({n_events / n_total:.1%} event rate)")

    return duration, event_observed


def create_survival_labels_chunked(
    parquet_path,
    output_path,
    event_cols,
    id_col='sid',
    time_col='year',
    use_polars=True,
):
    """
    Create survival labels in a memory-efficient manner.

    Loads only (sid, year, event_cols) — the slim columns — computes
    survival labels using vectorized operations, then writes per-year
    parquet files by joining labels with original features one year at
    a time.

    Peak memory: ~2 GB for slim table (105M rows × 7 cols) + ~4 GB
    for one year of full features during the write phase.

    Args:
        parquet_path: Path to the source parquet dataset
        output_path: Path to write the survival-labeled parquet files
        event_cols: List of event column names
        id_col: Individual ID column
        time_col: Time column
        use_polars: If True, use Polars for 2-3x speedup (requires polars package)

    Returns:
        output_path
    """
    import pyarrow.parquet as pq_read
    import pyarrow as pa
    import os
    import shutil
    import gc

    if use_polars:
        try:
            import polars as pl
            print(f"\n  Computing survival labels (Polars-accelerated)...")
        except ImportError:
            print(f"\n  Polars not installed, using pandas implementation...")
            use_polars = False

    if not use_polars:
        print(f"\n  Computing survival labels (memory-efficient)...")

    print(f"  Events: {event_cols}")

    # Step 1: Load ONLY the lightweight columns needed
    needed_cols = [id_col, time_col] + event_cols
    print(f"  Loading {len(needed_cols)} columns (out of full feature set)...")

    if use_polars:
        # Use Polars lazy scan for faster parquet reading
        slim_df = pl.scan_parquet(parquet_path).select(needed_cols).collect()
        print(f"    Loaded {len(slim_df):,} rows (slim)")
    else:
        # Pandas implementation
        slim_chunks = []
        total_loaded = 0
        ds = pq_read.ParquetDataset(parquet_path)
        for fragment in ds.fragments:
            for batch in fragment.to_batches(
                batch_size=1_000_000,
                columns=needed_cols,
            ):
                slim_chunks.append(batch.to_pandas())
                total_loaded += len(slim_chunks[-1])
                print(f"    Loaded {total_loaded:,} rows (slim)...",
                      end="\r", flush=True)

        slim_df = pd.concat(slim_chunks, ignore_index=True)
        del slim_chunks
        gc.collect()
        print(f"    Loaded {len(slim_df):,} rows (slim)           ")

    # Step 2: Sort by person and year (required for vectorized computation)
    print("  Sorting by person and year...")
    if use_polars:
        slim_df = slim_df.sort([id_col, time_col])
    else:
        slim_df = slim_df.sort_values([id_col, time_col]).reset_index(drop=True)
    gc.collect()

    # Step 3: Compute labels vectorized per event
    print(f"  Computing survival labels ({'Polars' if use_polars else 'pandas'})...")

    label_compute_fn = _compute_survival_labels_polars if use_polars else _compute_survival_labels_vectorized

    for event_col in event_cols:
        duration, event_observed = label_compute_fn(
            slim_df, event_col, id_col, time_col,
        )
        if use_polars:
            slim_df = slim_df.with_columns([
                pl.Series(f'{event_col}_duration', duration),
                pl.Series(f'{event_col}_event_observed', event_observed),
            ])
        else:
            slim_df[f'{event_col}_duration'] = duration
            slim_df[f'{event_col}_event_observed'] = event_observed
        del duration, event_observed

    gc.collect()

    # Step 4: Keep only label columns for the join phase
    label_cols = [id_col, time_col]
    for event_col in event_cols:
        label_cols += [f'{event_col}_duration', f'{event_col}_event_observed']

    # Drop event columns to save memory before the join
    if use_polars:
        slim_df = slim_df.select(label_cols)
        # Convert to pandas for the merge phase (pandas merge is still needed for pyarrow write)
        slim_df = slim_df.to_pandas()
    else:
        slim_df = slim_df[label_cols].copy()
    gc.collect()

    # Step 5: Write per-year parquet (features + labels)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    print(f"  Joining labels with features and writing parquet...")
    years = sorted(slim_df[time_col].unique())

    for i, year in enumerate(years):
        print(f"    [{i+1}/{len(years)}] Year {year}...", end=" ", flush=True)

        # Read original features for this year only
        year_features = pd.read_parquet(
            parquet_path,
            filters=[(time_col, '==', year)],
        )

        # Get labels for this year
        year_labels = slim_df.loc[slim_df[time_col] == year]

        # Merge
        year_merged = year_features.merge(
            year_labels, on=[id_col, time_col], how='left',
        )

        # Write
        table = pa.Table.from_pandas(year_merged, preserve_index=False)
        pq_read.write_table(
            table,
            os.path.join(output_path, f"year_{year}.parquet"),
        )

        del year_features, year_labels, year_merged, table
        gc.collect()
        print("done")

    del slim_df
    gc.collect()

    # Write success marker
    with open(os.path.join(output_path, "_SUCCESS"), 'w') as f:
        f.write("Success")

    print(f"\n  Survival labels written to {output_path}")
    for event_col in event_cols:
        print(f"    {event_col}: _duration + _event_observed")

    return output_path
