#!/usr/bin/env python3
"""
Pre-tokenize all sequence datasets and cache to disk.

Run this once (overnight if needed) to build all cache files.
Subsequent training runs will load from cache instantly.

Usage:
    # Standard single-cutoff mode
    python build_sequence_cache.py --config configs/models/pytorch_seq_gru.yaml
    python build_sequence_cache.py --sample-fraction 0.1  # For testing

    # Rolling window mode (5yr history, 1yr slide)
    python build_sequence_cache.py --config configs/models/pytorch_seq_gru.yaml --rolling
    python build_sequence_cache.py --config configs/models/pytorch_seq_gru.yaml --rolling --history-len 5
    python build_sequence_cache.py --config configs/models/pytorch_seq_gru.yaml --rolling --rolling-cutoffs 2015,2016,2017,2018,2019,2020
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import pyarrow.parquet as pq

from src.utils.logging_setup import setup_logging, log_stage_start, log_stage_complete
from src.sequence.vocabulary import (
    LifeEventVocabulary, MUNICIPALITY_FEATURE_MAP,
    NUMERIC_FEATURE_COLUMNS, N_NUMERIC_FEATURES,
)
from src.sequence.dataset import CachedSequenceDataset
from run_test import load_model_config
from run_with_municipality_sequence import (
    _build_vocab_from_parquet,
    _collect_valid_persons,
    DEFAULT_EVENTS,
)


def _scan_year_range(parquet_path, chunk_size=500_000):
    """Scan parquet to find min and max year in the dataset."""
    dataset = pq.ParquetDataset(parquet_path)
    min_year = float('inf')
    max_year = float('-inf')
    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=['year']):
            years = batch.to_pandas()['year']
            min_year = min(min_year, int(years.min()))
            max_year = max(max_year, int(years.max()))
    return min_year, max_year


def _collect_valid_persons_windowed(parquet_path, cutoff_year, min_history_year, max_horizon, chunk_size=500_000):
    """Find persons with history in (min_history_year, cutoff] AND future in (cutoff, cutoff+max_horizon]."""
    dataset = pq.ParquetDataset(parquet_path)
    history = set()
    future = set()

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=['sid', 'year']):
            df = batch.to_pandas()
            hist_mask = (df['year'] > min_history_year) & (df['year'] <= cutoff_year)
            future_mask = (df['year'] > cutoff_year) & (df['year'] <= cutoff_year + max_horizon)
            if hist_mask.any():
                history.update(df.loc[hist_mask, 'sid'].unique().tolist())
            if future_mask.any():
                future.update(df.loc[future_mask, 'sid'].unique().tolist())
            del df

    return history & future


def _collect_all_valid_persons(parquet_path, cutoff_years, history_len, max_horizon, chunk_size=500_000):
    """Single-pass: find valid persons for ALL windows at once.

    Returns dict: {cutoff_year: set_of_valid_sids}
    """
    dataset = pq.ParquetDataset(parquet_path)

    # Per cutoff: sets of sids with history / future
    history_sets = {c: set() for c in cutoff_years}
    future_sets = {c: set() for c in cutoff_years}

    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=chunk_size, columns=['sid', 'year']):
            df = batch.to_pandas()
            sids = df['sid'].values
            years = df['year'].values

            for cutoff in cutoff_years:
                min_hist = cutoff - history_len
                hist_mask = (years > min_hist) & (years <= cutoff)
                fut_mask = (years > cutoff) & (years <= cutoff + max_horizon)

                if hist_mask.any():
                    history_sets[cutoff].update(sids[hist_mask].tolist())
                if fut_mask.any():
                    future_sets[cutoff].update(sids[fut_mask].tolist())
            del df

    return {c: history_sets[c] & future_sets[c] for c in cutoff_years}


def _build_rolling_from_existing_cache(
    existing_cache_dir, vocabulary, max_seq_len, events, horizons,
    cutoff_years, history_len, cache_dir, sample_fraction=None,
    use_numeric_features=False,
):
    """Build rolling window caches by reusing an existing single-cutoff cache.

    The existing cache contains per-year shards (shard_XXXX.parquet) with all
    person data already grouped. We skip the expensive parquet-reading and
    sharding steps entirely and just:
      1. Scan shards to find valid persons per window
      2. For each cutoff, load relevant year shards, group by sid, tokenize, save
    """
    import torch
    import pandas as pd

    max_horizon = max(horizons)
    n_horizons = len(horizons)
    n_events = len(events)
    n_outputs = n_events * n_horizons
    sf_tag = f"_sf{sample_fraction}" if sample_fraction else ""
    cache_paths = []

    # Check which caches already exist
    cutoffs_to_build = []
    for cutoff in cutoff_years:
        cache_name = f"rolling_cut{cutoff}_h{history_len}{sf_tag}.pt"
        cache_path = os.path.join(cache_dir, cache_name)
        cache_paths.append(cache_path)
        if os.path.exists(cache_path) or os.path.isdir(cache_path.replace('.pt', '_chunks')):
            print(f"  Cache exists for cutoff={cutoff}, skipping")
        else:
            cutoffs_to_build.append(cutoff)

    if not cutoffs_to_build:
        print("  All caches already exist!")
        return cache_paths

    print(f"  Need to build caches for cutoffs: {cutoffs_to_build}")

    # Index existing shards by year
    log_stage_start("Indexing Existing Cache Shards")
    shard_files = sorted([f for f in os.listdir(existing_cache_dir) if f.endswith('.parquet')])
    print(f"  Found {len(shard_files)} shards in {existing_cache_dir}")

    # Build year -> [shard_paths] mapping
    year_to_shards = {}
    for sf in shard_files:
        path = os.path.join(existing_cache_dir, sf)
        # Read just year column to classify this shard
        mini = pd.read_parquet(path, columns=['year'])
        for yr in mini['year'].unique():
            yr = int(yr)
            if yr not in year_to_shards:
                year_to_shards[yr] = []
            year_to_shards[yr].append(path)
        del mini

    available_years = sorted(year_to_shards.keys())
    print(f"  Available years: {available_years[0]}-{available_years[-1]}")
    for yr in available_years:
        print(f"    {yr}: {len(year_to_shards[yr])} shards")
    log_stage_complete("Indexing Existing Cache Shards")

    # Step 1: Find valid persons per window from the shards
    log_stage_start("Collecting Valid Persons (from existing cache)")
    valid_per_cutoff = {}

    for cutoff in cutoffs_to_build:
        min_hist = cutoff - history_len
        hist_years = [y for y in available_years if y > min_hist and y <= cutoff]
        fut_years = [y for y in available_years if y > cutoff and y <= cutoff + max_horizon]

        # Collect sids with history
        hist_sids = set()
        hist_shard_paths = set()
        for yr in hist_years:
            hist_shard_paths.update(year_to_shards.get(yr, []))
        for path in hist_shard_paths:
            df = pd.read_parquet(path, columns=['sid', 'year'])
            mask = (df['year'] > min_hist) & (df['year'] <= cutoff)
            if mask.any():
                hist_sids.update(df.loc[mask, 'sid'].unique().tolist())
            del df

        # Collect sids with future
        fut_sids = set()
        fut_shard_paths = set()
        for yr in fut_years:
            fut_shard_paths.update(year_to_shards.get(yr, []))
        for path in fut_shard_paths:
            df = pd.read_parquet(path, columns=['sid', 'year'])
            mask = (df['year'] > cutoff) & (df['year'] <= cutoff + max_horizon)
            if mask.any():
                fut_sids.update(df.loc[mask, 'sid'].unique().tolist())
            del df

        valid_per_cutoff[cutoff] = hist_sids & fut_sids
        print(f"    cutoff={cutoff}: {len(valid_per_cutoff[cutoff]):,} valid persons "
              f"(hist years: {hist_years}, fut years: {fut_years})")

    if sample_fraction is not None:
        for cutoff in cutoffs_to_build:
            rng = np.random.RandomState(42 + cutoff)
            valid = valid_per_cutoff[cutoff]
            n_sample = max(1, int(len(valid) * sample_fraction))
            valid_per_cutoff[cutoff] = set(rng.choice(list(valid), n_sample, replace=False))
        print(f"  Subsampled to {sample_fraction:.2%}")
        for cutoff in cutoffs_to_build:
            print(f"    cutoff={cutoff}: {len(valid_per_cutoff[cutoff]):,} persons")

    log_stage_complete("Collecting Valid Persons (from existing cache)")

    # Step 2: Disk-based approach — all windows at once
    # 1. Filter shards -> write temp parquet on disk sorted by sid (low RAM)
    # 2. Read temp parquet per-person, tokenize once, split into all windows
    import tempfile
    import shutil
    import pyarrow as pa
    import pyarrow.compute as pc_local
    import pyarrow.parquet as pq_local

    CHUNK_SIZE = 50_000  # persons per output chunk

    # Union of all valid sids and determine global year range needed
    all_valid = set()
    for cutoff in cutoffs_to_build:
        all_valid |= valid_per_cutoff[cutoff]
    global_min_year = min(c - history_len for c in cutoffs_to_build)
    global_max_year = max(c + max_horizon for c in cutoffs_to_build)
    print(f"  Union of valid persons: {len(all_valid):,}")
    print(f"  Global year range needed: ({global_min_year}, {global_max_year}]")

    # Prepare per-window output chunk dirs and writers
    window_writers = {}  # cutoff -> {chunk_dir, buf_ids, buf_masks, buf_targets, buf_sids, chunk_idx, n_proc}
    for cutoff in cutoffs_to_build:
        cache_name = f"rolling_cut{cutoff}_h{history_len}{sf_tag}"
        cdir = os.path.join(cache_dir, cache_name + "_chunks")
        os.makedirs(cdir, exist_ok=True)
        window_writers[cutoff] = {
            'chunk_dir': cdir,
            'buf_ids': [], 'buf_masks': [], 'buf_targets': [], 'buf_sids': [],
            'buf_numerics': [],
            'chunk_idx': 0, 'n_processed': 0,
        }

    def _flush_window(cutoff):
        w = window_writers[cutoff]
        if not w['buf_ids']:
            return
        chunk_data = {
            'input_ids': torch.from_numpy(np.stack(w['buf_ids'])),
            'attention_masks': torch.from_numpy(np.stack(w['buf_masks'])),
            'targets': torch.from_numpy(np.stack(w['buf_targets'])),
            'sids': np.array(w['buf_sids']),
        }
        if use_numeric_features and w['buf_numerics']:
            chunk_data['numeric_features'] = torch.from_numpy(np.stack(w['buf_numerics']))
        chunk_path = os.path.join(w['chunk_dir'], f"chunk_{w['chunk_idx']:04d}.pt")
        torch.save(chunk_data, chunk_path)
        print(f"    [cut={cutoff}] Wrote chunk {w['chunk_idx']}: "
              f"{len(w['buf_ids']):,} persons")
        w['chunk_idx'] += 1
        w['buf_ids'].clear()
        w['buf_masks'].clear()
        w['buf_targets'].clear()
        w['buf_sids'].clear()
        w['buf_numerics'].clear()
        del chunk_data

    # Pass A: Filter existing shards -> temp parquet on disk, sorted by sid
    log_stage_start("Writing Filtered Temp Parquet")
    temp_dir = tempfile.mkdtemp(prefix='rolling_temp_')
    temp_parquet = os.path.join(temp_dir, 'filtered.parquet')
    print(f"  Temp dir: {temp_dir}")

    relevant_years = [y for y in available_years
                      if y > global_min_year and y <= global_max_year]
    relevant_shard_paths = set()
    for yr in relevant_years:
        relevant_shard_paths.update(year_to_shards.get(yr, []))

    print(f"  Filtering {len(relevant_shard_paths)} shards (years {relevant_years[0]}-{relevant_years[-1]})...")
    writer = None
    n_shards_read = 0
    total_rows_written = 0

    for path in sorted(relevant_shard_paths):
        df = pd.read_parquet(path)
        df = df[(df['year'] > global_min_year) & (df['year'] <= global_max_year)]
        df = df[df['sid'].isin(all_valid)]
        if len(df) > 0:
            # Sort by sid so persons cluster together in the file
            df = df.sort_values('sid')
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq_local.ParquetWriter(temp_parquet, table.schema)
            writer.write_table(table)
            total_rows_written += len(df)
            del table
        del df
        n_shards_read += 1
        if n_shards_read % 50 == 0:
            print(f"    {n_shards_read}/{len(relevant_shard_paths)} shards, "
                  f"{total_rows_written:,} rows written")

    if writer is not None:
        writer.close()
    print(f"  Temp file: {total_rows_written:,} rows "
          f"({os.path.getsize(temp_parquet) / 1024**2:.0f} MB)")
    log_stage_complete("Writing Filtered Temp Parquet")

    if total_rows_written == 0:
        print("  WARNING: No data found, skipping all windows")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return cache_paths

    # Pass A.5: Sort temp parquet globally by (sid, year)
    # This is critical: the shards are per-year, so the same person's data
    # from different years ends up in different row groups. Without global
    # sorting, Pass B would process each person with incomplete data.
    log_stage_start("Sorting Temp Parquet by (sid, year)")
    print(f"  Loading temp parquet as PyArrow Table for sorting...")
    table = pq_local.read_table(temp_parquet)
    print(f"  Loaded {table.num_rows:,} rows, sorting by (sid, year)...")
    sorted_indices = pc_local.sort_indices(table, sort_keys=[('sid', 'ascending'), ('year', 'ascending')])
    table = table.take(sorted_indices)
    del sorted_indices
    sorted_parquet = os.path.join(temp_dir, 'sorted.parquet')
    pq_local.write_table(table, sorted_parquet, row_group_size=500_000)
    del table
    # Replace temp file with sorted version
    os.remove(temp_parquet)
    temp_parquet = sorted_parquet
    print(f"  Sorted file: {os.path.getsize(temp_parquet) / 1024**2:.0f} MB")
    log_stage_complete("Sorting Temp Parquet by (sid, year)")

    # Pass B: Vectorized batch tokenization
    # Instead of per-person pandas operations, we:
    # 1. Pre-tokenize ALL rows in a batch into a token matrix using numpy
    # 2. For each person, slice their row tokens, add BOS/SEP/EOS
    # 3. Build targets via vectorized numpy ops
    log_stage_start("Tokenizing All Windows (vectorized)")
    print(f"  Reading sorted temp parquet and tokenizing in bulk...")

    BATCH_SIZE = 500_000
    total_persons = 0
    t2id = vocabulary._token_to_id
    unk = vocabulary.UNK

    # Pre-build lookup tables for vectorized tokenization
    # Municipality quantile bins
    from src.sequence.vocabulary import MUNICIPALITY_FEATURE_MAP, EVENT_TOKEN_MAP
    muni_bin_info = []  # (col_name, prefix, bin_edges)
    for col, prefix in MUNICIPALITY_FEATURE_MAP.items():
        bin_edges = vocabulary._muni_quantile_bins.get(col)
        if bin_edges is not None:
            n_bins = max(len(bin_edges) - 1, 1)
            # Pre-build token ID array for each quintile
            q_ids = np.array([t2id.get(f'{prefix}_Q{q}', unk)
                              for q in range(1, n_bins + 1)], dtype=np.int64)
            muni_bin_info.append((col, bin_edges, q_ids, n_bins))

    # Event token IDs
    event_info = [(ec, t2id.get(tn, unk)) for ec, tn in EVENT_TOKEN_MAP.items()]
    no_event_id = t2id.get('NO_EVENT', unk)
    bos_id = vocabulary.BOS
    sep_id = vocabulary.SEP
    eos_id = vocabulary.EOS
    pad_id = vocabulary.PAD

    def _tokenize_batch_vectorized(df):
        """Tokenize all rows in a DataFrame at once using numpy vectorization.

        Returns per-row token matrix: (n_rows, tokens_per_row) and a
        year array for windowing.
        """
        n = len(df)
        # Each row produces: year + n_muni + age + gender + coupled + nat + hhpos + income + event = ~13 tokens
        # We'll build columns of token IDs
        token_cols = []

        # Year tokens
        if 'year' in df.columns:
            years = df['year'].values.astype(int)
            year_tokens = np.array([t2id.get(f'YEAR_{y}', unk) for y in years], dtype=np.int64)
            token_cols.append(year_tokens)

        # Municipality tokens (vectorized quantile binning)
        for col, bin_edges, q_ids, n_bins in muni_bin_info:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float64)
                # np.searchsorted on bin edges, clip to valid range
                bin_idx = np.searchsorted(bin_edges, vals, side='right')
                bin_idx = np.clip(bin_idx, 1, n_bins) - 1  # 0-indexed into q_ids
                col_tokens = q_ids[bin_idx]
                # Handle NaN: set to unk
                nan_mask = np.isnan(vals)
                if nan_mask.any():
                    col_tokens[nan_mask] = unk
                token_cols.append(col_tokens)

        # Age group
        if 'age_group' in df.columns:
            ag = df['age_group'].values
            ag_valid = ~pd.isna(ag)
            ag_tokens = np.full(n, unk, dtype=np.int64)
            if ag_valid.any():
                ag_int = ag[ag_valid].astype(int)
                ag_tokens[ag_valid] = np.array(
                    [t2id.get(f'AGE_{a}', unk) for a in ag_int], dtype=np.int64)
            token_cols.append(ag_tokens)
        elif 'age' in df.columns:
            age = df['age'].values
            age_valid = ~pd.isna(age)
            age_tokens = np.full(n, unk, dtype=np.int64)
            if age_valid.any():
                decades = np.minimum(age[age_valid].astype(int) // 10, 9)
                age_tokens[age_valid] = np.array(
                    [t2id.get(f'AGE_{d}', unk) for d in decades], dtype=np.int64)
            token_cols.append(age_tokens)

        # Gender
        if 'gender' in df.columns:
            g = df['gender'].values
            g_valid = ~pd.isna(g)
            g_tokens = np.full(n, unk, dtype=np.int64)
            if g_valid.any():
                g_tokens[g_valid] = np.array(
                    [t2id.get(f'GENDER_{int(v)}', unk) for v in g[g_valid]], dtype=np.int64)
            token_cols.append(g_tokens)

        # Coupled
        if 'coupled' in df.columns:
            c = df['coupled'].values
            c_valid = ~pd.isna(c)
            c_tokens = np.full(n, unk, dtype=np.int64)
            if c_valid.any():
                c_tokens[c_valid] = np.array(
                    [t2id.get(f'COUPLED_{int(v)}', unk) for v in c[c_valid]], dtype=np.int64)
            token_cols.append(c_tokens)

        # Nationality
        if 'eerste_nationaliteit' in df.columns:
            nat = df['eerste_nationaliteit'].values
            nat_valid = ~pd.isna(nat)
            nat_tokens = np.full(n, unk, dtype=np.int64)
            if nat_valid.any():
                nat_tokens[nat_valid] = np.array(
                    [t2id.get(f'NAT_{int(v)}', unk) for v in nat[nat_valid]], dtype=np.int64)
            token_cols.append(nat_tokens)

        # Household position
        if 'hh_pos' in df.columns:
            hh = df['hh_pos'].values
            hh_valid = ~pd.isna(hh)
            hh_tokens = np.full(n, unk, dtype=np.int64)
            if hh_valid.any():
                hh_tokens[hh_valid] = np.array(
                    [t2id.get(f'HHPOS_{int(v)}', unk) for v in hh[hh_valid]], dtype=np.int64)
            token_cols.append(hh_tokens)

        # Income quintile
        if 'income_quintile' in df.columns:
            inc = df['income_quintile'].values
            inc_valid = ~pd.isna(inc)
            inc_tokens = np.full(n, unk, dtype=np.int64)
            if inc_valid.any():
                inc_tokens[inc_valid] = np.array(
                    [t2id.get(f'INCOME_Q{int(v)}', unk) for v in inc[inc_valid]], dtype=np.int64)
            token_cols.append(inc_tokens)

        # Event tokens: one column that is either the event ID or NO_EVENT
        event_token_col = np.full(n, no_event_id, dtype=np.int64)
        for ec, tid in event_info:
            if ec in df.columns:
                ev = df[ec].values
                fired = (~pd.isna(ev)) & (ev.astype(float) == 1.0)
                event_token_col[fired] = tid
        token_cols.append(event_token_col)

        # Stack: (n_rows, n_token_cols)
        row_tokens = np.column_stack(token_cols)

        # Extract normalized numeric features if enabled
        row_numerics = None
        if use_numeric_features:
            row_numerics = np.zeros((n, N_NUMERIC_FEATURES), dtype=np.float32)
            for ni, nc in enumerate(NUMERIC_FEATURE_COLUMNS):
                if nc in df.columns:
                    vals = pd.to_numeric(df[nc], errors='coerce').values.astype(np.float64)
                    mean, std = vocabulary._numeric_stats.get(nc, (0.0, 1.0))
                    normalized = (vals - mean) / max(std, 1e-8)
                    nan_mask = np.isnan(vals)
                    if nan_mask.any():
                        normalized[nan_mask] = 0.0
                    row_numerics[:, ni] = normalized

        return row_tokens, row_numerics

    def _assemble_and_emit(sids, years, row_tokens, event_values, row_start, row_end_per_person, row_numerics=None):
        """For a group of persons (contiguous in the sorted data), assemble
        sequences and targets for all cutoff windows."""
        nonlocal total_persons

        n_persons = len(row_start)
        tokens_per_row = row_tokens.shape[1]

        for pi in range(n_persons):
            rs = row_start[pi]
            re = row_end_per_person[pi]
            sid = sids[pi]
            p_years = years[rs:re]
            p_tokens = row_tokens[rs:re]  # (n_years, tokens_per_row)
            p_events = event_values[rs:re]  # (n_years, n_events)
            p_numerics = row_numerics[rs:re] if row_numerics is not None else None

            for cutoff in cutoffs_to_build:
                if sid not in valid_per_cutoff[cutoff]:
                    continue

                min_hist = cutoff - history_len
                hist_mask = (p_years > min_hist) & (p_years <= cutoff)
                if not hist_mask.any():
                    continue

                # Build token sequence from history rows
                hist_tokens = p_tokens[hist_mask]
                hist_numerics = p_numerics[hist_mask] if p_numerics is not None else None
                n_hist_rows = hist_tokens.shape[0]
                # Flatten with SEP between rows: BOS + row0 + SEP + row1 + ... + EOS
                total_seq_len = 2 + n_hist_rows * tokens_per_row + (n_hist_rows - 1)
                seq = np.empty(total_seq_len, dtype=np.int64)
                seq[0] = bos_id
                pos = 1

                # Build parallel numeric array if enabled
                if hist_numerics is not None:
                    seq_numeric = np.zeros((total_seq_len, N_NUMERIC_FEATURES), dtype=np.float32)
                else:
                    seq_numeric = None

                for ri in range(n_hist_rows):
                    if ri > 0:
                        seq[pos] = sep_id
                        # SEP gets zeros (already initialized)
                        pos += 1
                    seq[pos:pos + tokens_per_row] = hist_tokens[ri]
                    if seq_numeric is not None:
                        # Broadcast this row's numeric values to all token positions
                        seq_numeric[pos:pos + tokens_per_row] = hist_numerics[ri]
                    pos += tokens_per_row
                seq[pos] = eos_id
                # EOS gets zeros (already initialized)
                pos += 1
                seq = seq[:pos]
                if seq_numeric is not None:
                    seq_numeric = seq_numeric[:pos]

                # Pad/truncate
                seq_len = len(seq)
                if seq_len >= max_seq_len:
                    input_ids = seq[-max_seq_len:].copy()
                    att_mask = np.ones(max_seq_len, dtype=np.float32)
                    if seq_numeric is not None:
                        num_feat = seq_numeric[-max_seq_len:].copy()
                else:
                    input_ids = np.full(max_seq_len, pad_id, dtype=np.int64)
                    input_ids[:seq_len] = seq
                    att_mask = np.zeros(max_seq_len, dtype=np.float32)
                    att_mask[:seq_len] = 1.0
                    if seq_numeric is not None:
                        num_feat = np.zeros((max_seq_len, N_NUMERIC_FEATURES), dtype=np.float32)
                        num_feat[:seq_len] = seq_numeric

                # Build targets from future rows
                fut_mask = (p_years > cutoff) & (p_years <= cutoff + max_horizon)
                targets = np.zeros(n_outputs, dtype=np.float32)
                if fut_mask.any():
                    fut_years = p_years[fut_mask]
                    fut_ev = p_events[fut_mask]  # (n_fut, n_events)
                    for ei in range(n_events):
                        for hi, horizon in enumerate(horizons):
                            h_mask = fut_years <= cutoff + horizon
                            if h_mask.any() and fut_ev[h_mask, ei].max() > 0:
                                targets[ei * n_horizons + hi] = 1.0

                w = window_writers[cutoff]
                w['buf_ids'].append(input_ids)
                w['buf_masks'].append(att_mask)
                w['buf_targets'].append(targets)
                w['buf_sids'].append(sid)
                if seq_numeric is not None:
                    w['buf_numerics'].append(num_feat)
                w['n_processed'] += 1

                if len(w['buf_ids']) >= CHUNK_SIZE:
                    _flush_window(cutoff)

            total_persons += 1

        if total_persons % 500_000 == 0:
            print(f"    Processed {total_persons:,} persons...")

    # Read sorted parquet and process in batches
    pf = pq_local.ParquetFile(temp_parquet)
    pending_sid = None
    pending_tokens = None  # numpy arrays instead of DataFrames
    pending_years = None
    pending_events = None

    # Determine event columns present
    sample_batch = next(pf.iter_batches(batch_size=10))
    sample_df = sample_batch.to_pandas()
    event_cols_present = [ec for ec, _ in event_info if ec in sample_df.columns]
    del sample_batch, sample_df

    pf = pq_local.ParquetFile(temp_parquet)  # re-open
    pending_numerics = None
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()

        # Vectorized tokenization of entire batch
        row_tokens, row_numerics = _tokenize_batch_vectorized(df)
        batch_years = df['year'].values.astype(int)
        batch_sids = df['sid'].values

        # Event values matrix for targets
        batch_events = np.zeros((len(df), n_events), dtype=np.float32)
        for ei, (ec, _) in enumerate(event_info):
            if ec in df.columns:
                vals = df[ec].values.astype(float)
                valid = ~np.isnan(vals)
                batch_events[valid, ei] = vals[valid]

        del df

        # Process persons: since data is sorted by (sid, year), find person boundaries
        sid_changes = np.where(batch_sids[1:] != batch_sids[:-1])[0] + 1
        boundaries = np.concatenate([[0], sid_changes, [len(batch_sids)]])

        for bi in range(len(boundaries) - 1):
            start = boundaries[bi]
            end = boundaries[bi + 1]
            sid = batch_sids[start]

            cur_years = batch_years[start:end]
            cur_tokens = row_tokens[start:end]
            cur_events = batch_events[start:end]
            cur_numerics = row_numerics[start:end] if row_numerics is not None else None

            if pending_sid is not None and sid != pending_sid:
                # Emit the pending person
                _assemble_and_emit(
                    [pending_sid], pending_years, pending_tokens, pending_events,
                    np.array([0]), np.array([len(pending_years)]),
                    row_numerics=pending_numerics,
                )
                pending_sid = None
                pending_numerics = None

            if pending_sid == sid:
                # Extend pending person (split across batch boundary)
                pending_years = np.concatenate([pending_years, cur_years])
                pending_tokens = np.vstack([pending_tokens, cur_tokens])
                pending_events = np.vstack([pending_events, cur_events])
                if cur_numerics is not None and pending_numerics is not None:
                    pending_numerics = np.vstack([pending_numerics, cur_numerics])
            else:
                # Check if this is the last person in the batch (might continue in next)
                if bi == len(boundaries) - 2:
                    # Last person in batch — hold as pending
                    pending_sid = sid
                    pending_years = cur_years.copy()
                    pending_tokens = cur_tokens.copy()
                    pending_events = cur_events.copy()
                    pending_numerics = cur_numerics.copy() if cur_numerics is not None else None
                else:
                    # Complete person within this batch — emit directly
                    _assemble_and_emit(
                        [sid], cur_years, cur_tokens, cur_events,
                        np.array([0]), np.array([len(cur_years)]),
                        row_numerics=cur_numerics,
                    )

        del row_tokens, row_numerics, batch_years, batch_sids, batch_events

    # Emit last pending person
    if pending_sid is not None:
        _assemble_and_emit(
            [pending_sid], pending_years, pending_tokens, pending_events,
            np.array([0]), np.array([len(pending_years)]),
            row_numerics=pending_numerics,
        )

    # Flush remaining buffers for all windows
    for cutoff in cutoffs_to_build:
        _flush_window(cutoff)
        w = window_writers[cutoff]
        print(f"  cutoff={cutoff}: {w['n_processed']:,} persons -> "
              f"{w['chunk_idx']} chunks")

    log_stage_complete("Tokenizing All Windows")

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"  Cleaned up temp dir")

    return cache_paths


def _build_rolling_caches(
    data_path, vocabulary, max_seq_len, events, horizons,
    cutoff_years, history_len, cache_dir, sample_fraction=None,
    use_numeric_features=False,
):
    """Build all rolling window caches with a SINGLE pass through the parquet.

    Strategy:
      1. Single parquet scan to find valid persons for all windows
      2. Single Pass 1: group all person data into temp shards (once)
      3. Single Pass 2a: merge shards (once)
      4. Per-window Pass 2b: tokenize from merged data for each cutoff
         (fast: just slicing already-loaded person data, no re-reading parquet)
    """
    import tempfile
    import torch
    import pandas as pd
    import pyarrow.compute as pc

    max_horizon = max(horizons)
    n_horizons = len(horizons)
    n_events = len(events)
    n_outputs = n_events * n_horizons
    sf_tag = f"_sf{sample_fraction}" if sample_fraction else ""
    cache_paths = []

    # Check which caches already exist
    cutoffs_to_build = []
    for cutoff in cutoff_years:
        cache_name = f"rolling_cut{cutoff}_h{history_len}{sf_tag}.pt"
        cache_path = os.path.join(cache_dir, cache_name)
        cache_paths.append(cache_path)
        if os.path.exists(cache_path) or os.path.isdir(cache_path.replace('.pt', '_chunks')):
            print(f"  Cache exists for cutoff={cutoff}, skipping")
        else:
            cutoffs_to_build.append(cutoff)

    if not cutoffs_to_build:
        print("  All caches already exist!")
        return cache_paths

    print(f"  Need to build caches for cutoffs: {cutoffs_to_build}")

    # Step 1: Single-pass valid person collection for ALL windows
    log_stage_start("Collecting Valid Persons (all windows)")
    print(f"  Single parquet scan for {len(cutoffs_to_build)} windows...")
    valid_per_cutoff = _collect_all_valid_persons(
        data_path, cutoffs_to_build, history_len, max_horizon,
    )
    for cutoff in cutoffs_to_build:
        print(f"    cutoff={cutoff}: {len(valid_per_cutoff[cutoff]):,} valid persons")

    if sample_fraction is not None:
        for cutoff in cutoffs_to_build:
            rng = np.random.RandomState(42 + cutoff)
            valid = valid_per_cutoff[cutoff]
            n_sample = max(1, int(len(valid) * sample_fraction))
            valid_per_cutoff[cutoff] = set(rng.choice(list(valid), n_sample, replace=False))
        print(f"  Subsampled to {sample_fraction:.2%}")
        for cutoff in cutoffs_to_build:
            print(f"    cutoff={cutoff}: {len(valid_per_cutoff[cutoff]):,} persons")

    # Union of all valid persons across all windows
    all_valid = set()
    for cutoff in cutoffs_to_build:
        all_valid |= valid_per_cutoff[cutoff]
    print(f"  Union of all valid persons: {len(all_valid):,}")
    log_stage_complete("Collecting Valid Persons (all windows)")

    # Step 2: Single Pass 1 — group parquet by person into merged sorted file
    log_stage_start("Grouping Person Data (single pass)")
    print(f"  Reading parquet and grouping {len(all_valid):,} persons...")

    dataset = pq.ParquetDataset(data_path)
    schema_cols = set(dataset.schema.names)
    required_cols = [
        'sid', 'year', 'age_group', 'age', 'gender', 'coupled',
        'eerste_nationaliteit', 'hh_pos', 'income_quintile',
    ]
    from src.sequence.vocabulary import MUNICIPALITY_FEATURE_MAP
    muni_cols = [c for c in MUNICIPALITY_FEATURE_MAP.keys() if c in schema_cols]
    event_cols = [e for e in events if e in schema_cols]
    columns = [c for c in required_cols if c in schema_cols] + muni_cols + event_cols

    temp_dir = tempfile.mkdtemp(prefix='rolling_cache_')
    print(f"  Temp directory: {temp_dir}")

    # Pass 1: Buffer persons and write shards
    person_buffer = {}
    shard_idx = 0
    max_buffer = 50_000

    if len(all_valid) < 1_000_000:
        filter_expr = pc.field('sid').isin(list(all_valid))
    else:
        filter_expr = None

    def _flush():
        nonlocal shard_idx
        if not person_buffer:
            return
        shard_data = []
        for sid, dfs in person_buffer.items():
            shard_data.append(pd.concat(dfs).sort_values('year'))
        shard_df = pd.concat(shard_data, ignore_index=True)
        shard_path = os.path.join(temp_dir, f"shard_{shard_idx:04d}.parquet")
        shard_df.to_parquet(shard_path, index=False)
        print(f"    Wrote shard {shard_idx}: {len(person_buffer):,} persons")
        shard_idx += 1
        person_buffer.clear()

    n_chunks_read = 0
    for fragment in dataset.fragments:
        for batch in fragment.to_batches(batch_size=500_000, columns=columns, filter=filter_expr):
            chunk = batch.to_pandas()
            if len(chunk) == 0:
                continue
            for sid, rows in chunk.groupby('sid'):
                if sid not in all_valid:
                    continue
                if sid not in person_buffer:
                    person_buffer[sid] = []
                person_buffer[sid].append(rows)
            n_chunks_read += 1
            if n_chunks_read % 20 == 0:
                print(f"    Read {n_chunks_read} chunks, buffered {len(person_buffer):,} persons")
            if len(person_buffer) >= max_buffer:
                _flush()
            del chunk

    if person_buffer:
        _flush()

    # Pass 2a: Merge shards into single sorted file
    shard_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.parquet')])
    print(f"  Merging {len(shard_files)} shards...")
    merged_path = os.path.join(temp_dir, '_merged_sorted.parquet')

    chunk_dfs = []
    for idx, sf in enumerate(shard_files):
        chunk_dfs.append(pd.read_parquet(os.path.join(temp_dir, sf)))
        if len(chunk_dfs) >= 50 or idx == len(shard_files) - 1:
            merged = pd.concat(chunk_dfs, ignore_index=True)
            merged = merged.drop_duplicates(subset=['sid', 'year'])
            merged = merged.sort_values(['sid', 'year'])
            if os.path.exists(merged_path):
                existing = pd.read_parquet(merged_path)
                merged = pd.concat([existing, merged], ignore_index=True)
                merged = merged.drop_duplicates(subset=['sid', 'year'])
                merged = merged.sort_values(['sid', 'year'])
            merged.to_parquet(merged_path, index=False)
            del merged
            chunk_dfs = []

    log_stage_complete("Grouping Person Data (single pass)")

    # Step 3: Per-window tokenization from the merged file
    # This is the fast part — just slicing already-grouped data
    for cutoff in cutoffs_to_build:
        cache_name = f"rolling_cut{cutoff}_h{history_len}{sf_tag}.pt"
        cache_path = os.path.join(cache_dir, cache_name)
        min_hist_year = cutoff - history_len
        valid_sids = valid_per_cutoff[cutoff]

        log_stage_start(f"Tokenizing Window (cutoff={cutoff})")
        print(f"  cutoff={cutoff}: history ({min_hist_year}, {cutoff}], "
              f"targets ({cutoff}, {cutoff + max_horizon}]")

        all_input_ids = []
        all_masks = []
        all_targets = []
        all_sids_list = []
        all_numerics = []
        n_processed = 0

        for chunk in pd.read_parquet(merged_path, chunksize=100_000):
            for sid, person_df in chunk.groupby('sid'):
                if sid not in valid_sids:
                    continue

                person_df = person_df.sort_values('year')
                history = person_df[
                    (person_df['year'] > min_hist_year) &
                    (person_df['year'] <= cutoff)
                ]
                if len(history) == 0:
                    continue

                if use_numeric_features:
                    tokens, numerics = vocabulary.tokenize_person_history_with_numerics(
                        history, time_col='year', max_year=cutoff,
                        min_year=min_hist_year,
                    )
                else:
                    tokens = vocabulary.tokenize_person_history(
                        history, time_col='year', max_year=cutoff,
                        min_year=min_hist_year,
                    )
                    numerics = None

                # Build targets
                future = person_df[
                    (person_df['year'] > cutoff) &
                    (person_df['year'] <= cutoff + max_horizon)
                ]
                targets = np.zeros(n_outputs, dtype=np.float32)
                for ei, event_col in enumerate(events):
                    if event_col not in future.columns:
                        continue
                    for hi, horizon in enumerate(horizons):
                        window = future[future['year'] <= cutoff + horizon]
                        if len(window) > 0 and window[event_col].astype(int).sum() > 0:
                            targets[ei * n_horizons + hi] = 1.0

                # Pad/truncate
                seq_len = len(tokens)
                if seq_len >= max_seq_len:
                    input_ids = np.array(tokens[-max_seq_len:], dtype=np.int64)
                    mask = np.ones(max_seq_len, dtype=np.float32)
                    if numerics is not None:
                        num_feat = numerics[-max_seq_len:].copy()
                else:
                    input_ids = np.full(max_seq_len, vocabulary.PAD, dtype=np.int64)
                    input_ids[:seq_len] = tokens
                    mask = np.zeros(max_seq_len, dtype=np.float32)
                    mask[:seq_len] = 1.0
                    if numerics is not None:
                        num_feat = np.zeros((max_seq_len, N_NUMERIC_FEATURES), dtype=np.float32)
                        num_feat[:seq_len] = numerics

                all_input_ids.append(input_ids)
                all_masks.append(mask)
                all_targets.append(targets)
                all_sids_list.append(sid)
                if numerics is not None:
                    all_numerics.append(num_feat)
                n_processed += 1

                if n_processed % 100_000 == 0:
                    print(f"    Tokenized {n_processed:,} persons...")

            del chunk

        print(f"  cutoff={cutoff}: {n_processed:,} persons tokenized")

        if all_input_ids:
            cache_data = {
                'input_ids': torch.from_numpy(np.stack(all_input_ids)),
                'attention_masks': torch.from_numpy(np.stack(all_masks)),
                'targets': torch.from_numpy(np.stack(all_targets)),
                'sids': np.array(all_sids_list),
            }
            if use_numeric_features and all_numerics:
                cache_data['numeric_features'] = torch.from_numpy(np.stack(all_numerics))
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            torch.save(cache_data, cache_path)
            print(f"  Saved: {cache_path} ({os.path.getsize(cache_path) / 1024**2:.1f} MB)")
            del cache_data

        del all_input_ids, all_masks, all_targets, all_sids_list, all_numerics
        log_stage_complete(f"Tokenizing Window (cutoff={cutoff})")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"  Cleaned up temp directory")

    return cache_paths


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize sequence datasets')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--sample-fraction', type=float,
                        help='Optional: build cache for a subset (for testing)')
    parser.add_argument('--cutoff-year', type=int, default=2022,
                        help='Year cutoff for train/test split')
    parser.add_argument('--data-path', type=str,
                        default='data/processed/features/with_municipality_features.parquet',
                        help='Path to input parquet file')
    # Rolling window arguments
    parser.add_argument('--rolling', action='store_true',
                        help='Build rolling window caches instead of single-cutoff')
    parser.add_argument('--history-len', type=int, default=5,
                        help='History window length in years (default: 5)')
    parser.add_argument('--rolling-cutoffs', type=str,
                        help='Comma-separated cutoff years for rolling windows. '
                             'Auto-detected from data if not specified.')
    parser.add_argument('--existing-cache', type=str,
                        help='Path to existing single-cutoff cache directory with '
                             'per-year shards. Reuses these shards instead of '
                             're-reading the raw parquet (much faster).')
    parser.add_argument('--numeric-features', action='store_true',
                        help='Include normalized numeric features alongside tokens '
                             '(hybrid mode). Requires vocabulary with computed '
                             'numeric stats.')

    args = parser.parse_args()

    log_file = setup_logging(
        log_file=f"build_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        log_dir='.'
    )
    print(f"Logging to: {log_file}\n")

    print("=" * 70)
    print("SEQUENCE DATASET CACHE BUILDER")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Data: {args.data_path}")
    if args.rolling:
        print(f"Mode: ROLLING WINDOW (history_len={args.history_len})")
        if args.existing_cache:
            print(f"  Reusing existing cache: {args.existing_cache}")
    else:
        print(f"Mode: Single cutoff (year={args.cutoff_year})")
    if args.numeric_features:
        print(f"Numeric features: ENABLED (hybrid mode)")
    if args.sample_fraction:
        print(f"Sample fraction: {args.sample_fraction:.2%} (subset mode)")
    else:
        print("Sample fraction: 100% (FULL DATASET)")
    print("=" * 70)
    print()

    # Load config
    config = load_model_config(args.config)
    events = config.get('events', DEFAULT_EVENTS)
    horizons = config.get('horizons', [1, 3, 5])
    max_seq_len = config.get('model', {}).get('params', {}).get('max_seq_len', 256)

    print(f"Events: {', '.join(events)}")
    print(f"Horizons: {horizons}")
    print(f"Max sequence length: {max_seq_len}")
    print()

    # Check data exists (skip for rolling with existing cache)
    if args.rolling and args.existing_cache:
        print(f"Using existing cache, skipping parquet data check")
        print()
    elif not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        sys.exit(1)
    else:
        dataset = pq.ParquetDataset(args.data_path)
        total_rows = sum(fragment.metadata.num_rows for fragment in dataset.fragments)
        print(f"Total rows in dataset: {total_rows:,}")
        print()

    # Build vocabulary
    log_stage_start("Building Vocabulary")

    vocab_path = "checkpoints/sequence_vocab.joblib"
    os.makedirs("checkpoints", exist_ok=True)

    if os.path.exists(vocab_path):
        print(f"Loading existing vocabulary from {vocab_path}...")
        vocabulary = LifeEventVocabulary.load(vocab_path)
        missing_tokens = [
            f"{prefix}_Q1"
            for prefix in MUNICIPALITY_FEATURE_MAP.values()
            if vocabulary.token_to_id(f"{prefix}_Q1") == vocabulary.UNK
        ]
        if missing_tokens:
            print("Vocabulary missing municipality tokens; rebuilding...")
            vocabulary = _build_vocab_from_parquet(args.data_path, vocab_path)
    else:
        print("Building vocabulary from parquet (this may take a few minutes)...")
        vocabulary = _build_vocab_from_parquet(args.data_path, vocab_path)

    print(f"Vocabulary size: {vocabulary.vocab_size} tokens")

    # Compute numeric stats if needed
    if args.numeric_features and not vocabulary._numeric_stats:
        if args.rolling and args.existing_cache:
            # Compute from existing cache shards
            print("Computing numeric normalization stats from existing cache shards...")
            import pandas as pd
            shard_files = sorted([
                os.path.join(args.existing_cache, f)
                for f in os.listdir(args.existing_cache) if f.endswith('.parquet')
            ])
            all_dfs = []
            for sf in shard_files:
                all_dfs.append(pd.read_parquet(sf))
            combined = pd.concat(all_dfs, ignore_index=True)
            vocabulary.compute_numeric_stats(combined)
            del all_dfs, combined
        else:
            print("Computing numeric normalization stats from parquet...")
            vocabulary.compute_numeric_stats_from_parquet(args.data_path)
        # Re-save vocabulary with stats
        vocabulary.save(vocab_path)
        print(f"  Saved vocabulary with numeric stats to {vocab_path}")
    elif args.numeric_features:
        print(f"Vocabulary already has numeric stats for {len(vocabulary._numeric_stats)} features")

    log_stage_complete("Building Vocabulary")
    print()

    cache_dir = "checkpoints/sequence_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # ================================================================
    # Rolling window mode
    # ================================================================
    if args.rolling:
        history_len = args.history_len
        max_horizon = max(horizons)

        # Determine cutoff years
        if args.rolling_cutoffs:
            cutoff_years = sorted([int(c.strip()) for c in args.rolling_cutoffs.split(',')])
        elif args.existing_cache:
            # Detect year range from existing cache shards
            import pandas as pd
            print("Auto-detecting year range from existing cache shards...")
            shard_files = sorted([f for f in os.listdir(args.existing_cache) if f.endswith('.parquet')])
            all_years = set()
            for sf in shard_files:
                mini = pd.read_parquet(os.path.join(args.existing_cache, sf), columns=['year'])
                all_years.update(mini['year'].unique().tolist())
                del mini
            min_year, max_year = int(min(all_years)), int(max(all_years))
            print(f"  Data year range: {min_year}-{max_year}")
            first_cutoff = min_year + history_len
            last_cutoff = max_year - max_horizon
            cutoff_years = list(range(first_cutoff, last_cutoff + 1))
        else:
            print("Auto-detecting year range from data...")
            min_year, max_year = _scan_year_range(args.data_path)
            print(f"  Data year range: {min_year}-{max_year}")
            first_cutoff = min_year + history_len
            last_cutoff = max_year - max_horizon
            cutoff_years = list(range(first_cutoff, last_cutoff + 1))

        print(f"Rolling windows: {len(cutoff_years)} cutoffs")
        print(f"  History: {history_len} years per window")
        print(f"  Cutoffs: {cutoff_years}")
        print(f"  Max horizon: {max_horizon}")
        print()

        for cutoff in cutoff_years:
            min_h = cutoff - history_len
            max_f = cutoff + max_horizon
            print(f"  W cutoff={cutoff}: history ({min_h}, {cutoff}] -> targets ({cutoff}, {max_f}]")
        print()

        if args.existing_cache:
            # Fast path: reuse existing per-year shards
            if not os.path.isdir(args.existing_cache):
                print(f"ERROR: Existing cache directory not found: {args.existing_cache}")
                sys.exit(1)
            print(f"  Using existing cache shards from: {args.existing_cache}")
            print()
            cache_paths = _build_rolling_from_existing_cache(
                existing_cache_dir=args.existing_cache,
                vocabulary=vocabulary,
                max_seq_len=max_seq_len,
                events=events,
                horizons=horizons,
                cutoff_years=cutoff_years,
                history_len=history_len,
                cache_dir=cache_dir,
                sample_fraction=args.sample_fraction,
                use_numeric_features=args.numeric_features,
            )
        else:
            # Slow path: read from raw parquet
            cache_paths = _build_rolling_caches(
                data_path=args.data_path,
                vocabulary=vocabulary,
                max_seq_len=max_seq_len,
                events=events,
                horizons=horizons,
                cutoff_years=cutoff_years,
                history_len=history_len,
                cache_dir=cache_dir,
                sample_fraction=args.sample_fraction,
                use_numeric_features=args.numeric_features,
            )

        print("=" * 70)
        print("ROLLING CACHE BUILDING COMPLETE!")
        print("=" * 70)
        print()
        print("Cache files created:")
        for path in cache_paths:
            chunk_dir = path.replace('.pt', '_chunks')
            if os.path.isdir(chunk_dir):
                n_chunks = len([f for f in os.listdir(chunk_dir) if f.endswith('.pt')])
                print(f"  {chunk_dir}/ ({n_chunks} chunks)")
            elif os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  {path} ({size_mb:.1f} MB)")
            else:
                print(f"  {path} (not built)")
        print()
        print("You can now run training with rolling windows:")
        print(f"  python run_with_municipality_sequence.py --reuse --config {args.config} --rolling --history-len {history_len}")
        if args.sample_fraction:
            print(f"      --sample-fraction {args.sample_fraction}")
        print()
        return

    # ================================================================
    # Standard single-cutoff mode
    # ================================================================
    log_stage_start("Collecting Valid Persons")

    cutoff_year = args.cutoff_year
    val_cutoff = cutoff_year - 2

    print(f"Scanning for persons with history before {cutoff_year} and future after...")
    valid_persons = _collect_valid_persons(args.data_path, cutoff_year)
    print(f"Train persons (cutoff={cutoff_year}): {len(valid_persons):,}")

    print(f"Scanning for persons with history before {val_cutoff} and future after...")
    val_persons = _collect_valid_persons(args.data_path, val_cutoff)
    print(f"Val persons (cutoff={val_cutoff}): {len(val_persons):,}")

    # Apply sampling if requested
    if args.sample_fraction is not None:
        if not (0 < args.sample_fraction <= 1.0):
            print("ERROR: --sample-fraction must be in (0, 1]")
            sys.exit(1)

        rng = np.random.RandomState(42)

        n_train = max(1, int(len(valid_persons) * args.sample_fraction))
        valid_persons = set(rng.choice(list(valid_persons), n_train, replace=False))

        n_val = max(1, int(len(val_persons) * args.sample_fraction))
        val_persons = set(rng.choice(list(val_persons), n_val, replace=False))

        print(f"\nSubsampled to {args.sample_fraction:.2%}:")
        print(f"  Train: {len(valid_persons):,} persons")
        print(f"  Val: {len(val_persons):,} persons")

    log_stage_complete("Collecting Valid Persons")
    print()

    # Build cache paths
    sf_tag = f"_sf{args.sample_fraction}" if args.sample_fraction else ""
    train_cache = os.path.join(cache_dir, f"train_cut{cutoff_year}{sf_tag}.pt")
    val_cache = os.path.join(cache_dir, f"val_cut{val_cutoff}{sf_tag}.pt")
    test_cache = os.path.join(cache_dir, f"test_cut{cutoff_year}{sf_tag}.pt")

    print("Cache files:")
    print(f"  Train: {train_cache}")
    print(f"  Val:   {val_cache}")
    print(f"  Test:  {test_cache}")
    print()

    # Check what already exists
    existing = []
    if os.path.exists(train_cache):
        existing.append("train")
    if os.path.exists(val_cache):
        existing.append("val")
    if os.path.exists(test_cache):
        existing.append("test")

    if existing:
        print(f"WARNING: Found existing cache files: {', '.join(existing)}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
        print()

    # Build train cache
    log_stage_start("Building Train Cache")
    print(f"Tokenizing {len(valid_persons):,} train persons...")
    print("This will take time but only needs to run once!")
    print()

    train_dataset = CachedSequenceDataset(
        parquet_path=args.data_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=cutoff_year,
        allowed_sids=valid_persons,
        cache_path=train_cache,
    )
    print(f"Train cache built: {len(train_dataset):,} persons")
    log_stage_complete("Building Train Cache")
    print()

    # Build val cache
    log_stage_start("Building Val Cache")
    print(f"Tokenizing {len(val_persons):,} val persons...")
    print()

    val_dataset = CachedSequenceDataset(
        parquet_path=args.data_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=val_cutoff,
        allowed_sids=val_persons,
        cache_path=val_cache,
    )
    print(f"Val cache built: {len(val_dataset):,} persons")
    log_stage_complete("Building Val Cache")
    print()

    # Build test cache (same as train for now)
    log_stage_start("Building Test Cache")
    print(f"Tokenizing {len(valid_persons):,} test persons...")
    print()

    test_dataset = CachedSequenceDataset(
        parquet_path=args.data_path,
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        events=events,
        horizons=horizons,
        cutoff_year=cutoff_year,
        allowed_sids=valid_persons,
        cache_path=test_cache,
    )
    print(f"Test cache built: {len(test_dataset):,} persons")
    log_stage_complete("Building Test Cache")
    print()

    print("=" * 70)
    print("CACHE BUILDING COMPLETE!")
    print("=" * 70)
    print()
    print("Cache files created:")
    for path in [train_cache, val_cache, test_cache]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {path} ({size_mb:.1f} MB)")
    print()
    print("You can now run training with instant startup:")
    print(f"  python run_with_municipality_sequence.py --reuse --config {args.config}")
    if args.sample_fraction:
        print(f"                                           --sample-fraction {args.sample_fraction}")
    print()


if __name__ == '__main__':
    main()
