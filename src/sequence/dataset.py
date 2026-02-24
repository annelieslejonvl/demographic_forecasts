"""
PyTorch Dataset classes for sequence-based demographic event prediction.

Converts panel data (person x year) into tokenized sequences with
multi-label targets for event prediction at multiple horizons.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

from .vocabulary import LifeEventVocabulary, EVENT_TOKEN_MAP, MUNICIPALITY_FEATURE_MAP

logger = logging.getLogger(__name__)

DEFAULT_EVENTS = list(EVENT_TOKEN_MAP.keys())
DEFAULT_HORIZONS = [1, 3, 5]


class ChunkShuffledSampler(Sampler):
    """Sampler that shuffles chunk order and items within each chunk.

    This avoids random cross-chunk access which would cause constant
    disk I/O with lazy-loaded chunked datasets. Instead:
    - Chunks are visited in random order each epoch
    - Within each chunk, items are randomly shuffled
    - Each chunk is only loaded once per epoch
    """

    def __init__(self, chunk_offsets, chunk_sizes):
        self.chunk_offsets = chunk_offsets
        self.chunk_sizes = chunk_sizes
        self.n_chunks = len(chunk_sizes)
        self.total = sum(chunk_sizes)

    def __iter__(self):
        chunk_order = torch.randperm(self.n_chunks).tolist()
        for chunk_idx in chunk_order:
            offset = self.chunk_offsets[chunk_idx]
            size = self.chunk_sizes[chunk_idx]
            within_chunk = torch.randperm(size).tolist()
            for local_idx in within_chunk:
                yield offset + local_idx

    def __len__(self):
        return self.total


class BalancedChunkSampler(Sampler):
    """Sampler that oversamples chunks with higher event rates.

    Chunks with more positive 1-year events are sampled more often per epoch.
    Each sample carries an importance weight (Horvitz-Thompson correction)
    so the loss remains unbiased.

    Args:
        chunk_offsets: Start index of each chunk in the global dataset.
        chunk_sizes: Number of samples in each chunk.
        chunk_event_rates: Per-chunk event rate (fraction of samples with any
            positive 1-year target).
        mix_ratio: Blend between event-rate weighting and uniform.
            Final weight = mix_ratio * rate_weight + (1 - mix_ratio) * uniform.
    """

    def __init__(
        self,
        chunk_offsets: List[int],
        chunk_sizes: List[int],
        chunk_event_rates: List[float],
        mix_ratio: float = 0.7,
    ):
        self.chunk_offsets = chunk_offsets
        self.chunk_sizes = chunk_sizes
        self.n_chunks = len(chunk_sizes)
        self.total = sum(chunk_sizes)

        # Compute sampling weights per chunk
        rates = np.array(chunk_event_rates, dtype=np.float64).clip(min=1e-6)
        rate_weights = rates / rates.sum()
        uniform = np.ones(self.n_chunks) / self.n_chunks
        weights = mix_ratio * rate_weights + (1 - mix_ratio) * uniform
        weights = weights / weights.sum()

        # Compute repeats: how many times each chunk appears per epoch
        repeats_float = weights * self.n_chunks
        self.repeats = np.round(repeats_float).astype(int).clip(min=1)

        # Importance weights: IW = P_uniform / P_weighted (normalize so mean=1)
        p_uniform = np.array(chunk_sizes, dtype=np.float64) / self.total
        p_weighted = np.zeros(self.n_chunks, dtype=np.float64)
        for ci in range(self.n_chunks):
            p_weighted[ci] = self.repeats[ci] * chunk_sizes[ci]
        p_weighted = p_weighted / p_weighted.sum()
        self.chunk_importance_weights = (p_uniform / p_weighted.clip(min=1e-10))
        # Normalize to mean=1
        self.chunk_importance_weights /= self.chunk_importance_weights.mean()

        self._effective_len = int(sum(self.repeats[ci] * chunk_sizes[ci] for ci in range(self.n_chunks)))

    def __iter__(self):
        # Build epoch schedule: each chunk appears self.repeats[ci] times
        chunk_schedule = []
        for ci in range(self.n_chunks):
            for _ in range(self.repeats[ci]):
                chunk_schedule.append(ci)

        # Shuffle the chunk order
        rng = np.random.default_rng()
        rng.shuffle(chunk_schedule)

        for chunk_idx in chunk_schedule:
            offset = self.chunk_offsets[chunk_idx]
            size = self.chunk_sizes[chunk_idx]
            within_chunk = torch.randperm(size).tolist()
            for local_idx in within_chunk:
                yield offset + local_idx

    def __len__(self):
        return self._effective_len

    def get_sample_weight(self, global_idx: int) -> float:
        """Get importance weight for a sample given its global index."""
        import bisect
        chunk_idx = bisect.bisect_right(self.chunk_offsets, global_idx) - 1
        chunk_idx = max(0, min(chunk_idx, self.n_chunks - 1))
        return float(self.chunk_importance_weights[chunk_idx])


class SequenceDataset(Dataset):
    """
    Map-style PyTorch Dataset that converts panel data to token sequences.

    For each person:
    - Tokenizes their history up to cutoff_year as input
    - Builds multi-label targets: for each event, whether it occurs
      within each horizon (1, 3, 5 years) after cutoff_year

    Args:
        df: Panel DataFrame with columns sid, year, refnis, events, etc.
        vocabulary: Built LifeEventVocabulary instance.
        max_seq_len: Maximum sequence length (pad/truncate).
        events: Event column names to predict.
        horizons: Prediction horizons in years.
        cutoff_year: Use history up to this year as input;
                     predict events after this year.
        id_col: Person ID column.
        time_col: Time column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocabulary: LifeEventVocabulary,
        max_seq_len: int = 256,
        events: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        cutoff_year: int = 2022,
        id_col: str = 'sid',
        time_col: str = 'year',
        min_history_year: Optional[int] = None,
    ):
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len
        self.events = events or DEFAULT_EVENTS
        self.horizons = horizons or DEFAULT_HORIZONS
        self.cutoff_year = cutoff_year
        self.id_col = id_col
        self.time_col = time_col
        self.min_history_year = min_history_year

        self.n_events = len(self.events)
        self.n_horizons = len(self.horizons)
        self.n_outputs = self.n_events * self.n_horizons

        # Group by person and precompute
        self._prepare(df)

    def _prepare(self, df: pd.DataFrame):
        """Group data by person and precompute sequences + targets."""
        df = df.sort_values([self.id_col, self.time_col])

        grouped = df.groupby(self.id_col)
        self._person_ids = []
        self._sequences = []
        self._targets = []

        n_skipped = 0
        for sid, person_df in grouped:
            # Input: history up to cutoff_year (and after min_history_year if set)
            history = person_df[person_df[self.time_col] <= self.cutoff_year]
            if self.min_history_year is not None:
                history = history[history[self.time_col] > self.min_history_year]
            if len(history) == 0:
                n_skipped += 1
                continue

            # Tokenize history
            tokens = self.vocabulary.tokenize_person_history(
                history,
                time_col=self.time_col,
                max_year=self.cutoff_year,
                min_year=self.min_history_year,
            )

            # Build targets from future observations
            future = person_df[person_df[self.time_col] > self.cutoff_year]
            targets = self._build_targets(future)

            self._person_ids.append(sid)
            self._sequences.append(tokens)
            self._targets.append(targets)

        if n_skipped > 0:
            logger.info(f"Skipped {n_skipped} persons with no history before cutoff")

        logger.info(
            f"SequenceDataset: {len(self._person_ids)} persons, "
            f"cutoff={self.cutoff_year}, "
            f"max_seq_len={self.max_seq_len}"
        )

    def _build_targets(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Build multi-label targets for each event at each horizon.

        For event e and horizon h: target=1 if event e occurs within
        h years after cutoff_year.

        Returns: array of shape (n_events * n_horizons,)
        """
        targets = np.zeros(self.n_outputs, dtype=np.float32)

        for ei, event_col in enumerate(self.events):
            if event_col not in future_df.columns:
                continue
            for hi, horizon in enumerate(self.horizons):
                max_year = self.cutoff_year + horizon
                window = future_df[future_df[self.time_col] <= max_year]
                if len(window) > 0 and window[event_col].astype(int).sum() > 0:
                    targets[ei * self.n_horizons + hi] = 1.0

        return targets

    def _pad_or_truncate(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or truncate token sequence to max_seq_len."""
        seq_len = len(tokens)
        if seq_len >= self.max_seq_len:
            # Truncate: keep last max_seq_len tokens (most recent history)
            input_ids = np.array(tokens[-self.max_seq_len:], dtype=np.int64)
            attention_mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            # Pad with PAD token
            input_ids = np.full(self.max_seq_len, self.vocabulary.PAD, dtype=np.int64)
            input_ids[:seq_len] = tokens
            attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
            attention_mask[:seq_len] = 1.0

        return input_ids, attention_mask

    def __len__(self) -> int:
        return len(self._person_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self._sequences[idx]
        targets = self._targets[idx]

        input_ids, attention_mask = self._pad_or_truncate(tokens)

        return {
            'input_ids': torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'targets': torch.from_numpy(targets),
        }

    def get_pos_weights(self) -> torch.Tensor:
        """Compute positive class weight for each output label (for BCEWithLogitsLoss)."""
        all_targets = np.stack(self._targets)  # (n_persons, n_outputs)
        n_pos = all_targets.sum(axis=0).clip(min=1.0)
        n_neg = (len(all_targets) - n_pos).clip(min=1.0)
        pos_weight = n_neg / n_pos
        return torch.from_numpy(pos_weight.astype(np.float32))


class StreamingSequenceDataset(IterableDataset):
    """
    Streaming (iterable) variant for datasets too large to fit in memory.

    Reads parquet files in chunks, groups by person, tokenizes,
    and yields complete sequences.

    Args:
        parquet_path: Path to parquet directory.
        vocabulary: Built LifeEventVocabulary.
        max_seq_len: Maximum sequence length.
        events: Event column names.
        horizons: Prediction horizons.
        cutoff_year: History cutoff year.
        chunk_size: Rows per chunk when reading parquet.
        id_col: Person ID column.
        time_col: Time column.
    """

    def __init__(
        self,
        parquet_path: str,
        vocabulary: LifeEventVocabulary,
        max_seq_len: int = 256,
        events: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        cutoff_year: int = 2022,
        chunk_size: int = 500_000,
        id_col: str = 'sid',
        time_col: str = 'year',
        allowed_sids: Optional[Set[Any]] = None,
        return_ids: bool = False,
        n_samples: Optional[int] = None,
        min_history_year: Optional[int] = None,
    ):
        self.parquet_path = parquet_path
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len
        self.events = events or DEFAULT_EVENTS
        self.horizons = horizons or DEFAULT_HORIZONS
        self.cutoff_year = cutoff_year
        self.chunk_size = chunk_size
        self.id_col = id_col
        self.time_col = time_col
        self.allowed_sids = allowed_sids
        self.return_ids = return_ids
        self.n_samples = n_samples
        self.min_history_year = min_history_year
        self.n_events = len(self.events)
        self.n_horizons = len(self.horizons)
        self.n_outputs = self.n_events * self.n_horizons
        self._columns = [
            self.id_col,
            self.time_col,
            'age_group',
            'age',
            'gender',
            'coupled',
            'eerste_nationaliteit',
            'hh_pos',
            'income_quintile',
        ] + list(MUNICIPALITY_FEATURE_MAP.keys()) + self.events

    def __len__(self) -> int:
        if self.n_samples is None:
            raise TypeError("Length is not known for this streaming dataset")
        return self.n_samples

    def __iter__(self):
        import pyarrow.parquet as pq
        import torch.utils.data

        dataset = pq.ParquetDataset(self.parquet_path)
        schema_cols = set(dataset.schema.names)
        columns = [c for c in self._columns if c in schema_cols]
        worker_info = torch.utils.data.get_worker_info()
        fragments = dataset.fragments
        if worker_info is not None:
            fragments = fragments[worker_info.id::worker_info.num_workers]

        # Buffer to accumulate rows per person across chunks
        person_buffer: Dict[Any, List[pd.DataFrame]] = {}

        for fragment in fragments:
            for batch in fragment.to_batches(batch_size=self.chunk_size, columns=columns):
                chunk = batch.to_pandas()
                chunk = chunk.sort_values([self.id_col, self.time_col])

                for sid, person_rows in chunk.groupby(self.id_col):
                    if self.allowed_sids is not None and sid not in self.allowed_sids:
                        continue
                    if sid not in person_buffer:
                        person_buffer[sid] = []
                    person_buffer[sid].append(person_rows)

                # Yield complete persons (those whose data is fully loaded)
                # Heuristic: yield persons not seen in this chunk
                chunk_sids = set(chunk[self.id_col].unique())
                complete_sids = [
                    sid for sid in person_buffer
                    if sid not in chunk_sids
                ]

                for sid in complete_sids:
                    person_df = pd.concat(person_buffer.pop(sid)).sort_values(self.time_col)
                    sample = self._process_person(person_df, sid=sid)
                    if sample is not None:
                        yield sample

                del chunk

        # Yield remaining buffered persons
        for sid, dfs in person_buffer.items():
            person_df = pd.concat(dfs).sort_values(self.time_col)
            sample = self._process_person(person_df, sid=sid)
            if sample is not None:
                yield sample

    def _process_person(
        self,
        person_df: pd.DataFrame,
        sid: Optional[Any] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize a person's history and build targets."""
        history = person_df[person_df[self.time_col] <= self.cutoff_year]
        if self.min_history_year is not None:
            history = history[history[self.time_col] > self.min_history_year]
        if len(history) == 0:
            return None

        tokens = self.vocabulary.tokenize_person_history(
            history, time_col=self.time_col, max_year=self.cutoff_year,
            min_year=self.min_history_year,
        )

        future = person_df[person_df[self.time_col] > self.cutoff_year]
        targets = self._build_targets(future)

        input_ids, attention_mask = self._pad_or_truncate(tokens)

        sample = {
            'input_ids': torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'targets': torch.from_numpy(targets),
        }
        if self.return_ids and sid is not None:
            sample['sid'] = sid
        return sample

    def _build_targets(self, future_df: pd.DataFrame) -> np.ndarray:
        targets = np.zeros(self.n_outputs, dtype=np.float32)
        for ei, event_col in enumerate(self.events):
            if event_col not in future_df.columns:
                continue
            for hi, horizon in enumerate(self.horizons):
                max_year = self.cutoff_year + horizon
                window = future_df[future_df[self.time_col] <= max_year]
                if len(window) > 0 and window[event_col].astype(int).sum() > 0:
                    targets[ei * self.n_horizons + hi] = 1.0
        return targets

    def _pad_or_truncate(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = len(tokens)
        if seq_len >= self.max_seq_len:
            input_ids = np.array(tokens[-self.max_seq_len:], dtype=np.int64)
            attention_mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            input_ids = np.full(self.max_seq_len, self.vocabulary.PAD, dtype=np.int64)
            input_ids[:seq_len] = tokens
            attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
            attention_mask[:seq_len] = 1.0
        return input_ids, attention_mask

    def get_pos_weights(self, max_samples: Optional[int] = None) -> torch.Tensor:
        """Compute positive class weights by streaming through the dataset."""
        n_pos = np.zeros(self.n_outputs, dtype=np.float64)
        n_total = 0
        for sample in self:
            targets = sample['targets'].numpy()
            n_pos += targets
            n_total += 1
            if max_samples is not None and n_total >= max_samples:
                break

        n_pos = np.clip(n_pos, 1.0, None)
        n_neg = np.clip(n_total - n_pos, 1.0, None)
        pos_weight = n_neg / n_pos
        return torch.from_numpy(pos_weight.astype(np.float32))


class CachedSequenceDataset(Dataset):
    """
    Pre-tokenizes all persons once from parquet, caches to disk as .pt,
    then serves from memory. Eliminates per-epoch re-tokenization overhead.

    First call: streams parquet, tokenizes, saves cache.
    Subsequent calls (with reuse=True): loads from cache instantly.
    """

    def __init__(
        self,
        parquet_path: str,
        vocabulary: LifeEventVocabulary,
        max_seq_len: int = 256,
        events: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        cutoff_year: int = 2022,
        chunk_size: int = 500_000,
        id_col: str = 'sid',
        time_col: str = 'year',
        allowed_sids: Optional[Set[Any]] = None,
        return_ids: bool = False,
        cache_path: Optional[str] = None,
        min_history_year: Optional[int] = None,
    ):
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len
        self.events = events or DEFAULT_EVENTS
        self.horizons = horizons or DEFAULT_HORIZONS
        self.cutoff_year = cutoff_year
        self.chunk_size = chunk_size
        self.id_col = id_col
        self.time_col = time_col
        self.return_ids = return_ids
        self.min_history_year = min_history_year
        self.n_events = len(self.events)
        self.n_horizons = len(self.horizons)
        self.n_outputs = self.n_events * self.n_horizons

        self._columns = [
            self.id_col,
            self.time_col,
            'age_group', 'age', 'gender', 'coupled',
            'eerste_nationaliteit', 'hh_pos', 'income_quintile',
        ] + list(MUNICIPALITY_FEATURE_MAP.keys()) + self.events

        # Try loading from cache (supports single .pt file or chunk directory)
        chunk_dir = cache_path.replace('.pt', '_chunks') if cache_path else None
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached dataset from {cache_path}")
            cache = torch.load(cache_path, map_location='cpu')
            self._input_ids = cache['input_ids']
            self._attention_masks = cache['attention_masks']
            self._targets = cache['targets']
            self._sids = cache.get('sids')
            logger.info(f"CachedSequenceDataset: {len(self._input_ids)} persons (from cache)")
        elif chunk_dir and os.path.isdir(chunk_dir):
            logger.info(f"Loading cached dataset from chunks in {chunk_dir}")
            self._load_from_chunks(chunk_dir)
            logger.info(f"CachedSequenceDataset: {len(self)} persons (from {len(self._chunk_paths)} chunks)")
        else:
            # Build from parquet
            self._build_from_parquet(parquet_path, allowed_sids)
            if cache_path:
                import os as _os
                _os.makedirs(_os.path.dirname(cache_path) or '.', exist_ok=True)
                torch.save({
                    'input_ids': self._input_ids,
                    'attention_masks': self._attention_masks,
                    'targets': self._targets,
                    'sids': self._sids,
                }, cache_path)
                logger.info(f"Cached dataset saved to {cache_path}")

    def _load_from_chunks(self, chunk_dir: str):
        """Load dataset from multiple chunk .pt files using lazy loading.

        Only stores file paths and sizes at init time. Chunks are loaded
        on demand in __getitem__ and cached with an LRU cache to avoid
        re-reading from disk every access while keeping memory bounded.
        """
        import json as _json

        chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.pt')])
        manifest_path = os.path.join(chunk_dir, 'manifest.json')

        self._chunk_paths = [os.path.join(chunk_dir, f) for f in chunk_files]
        self._chunk_sizes = []
        self._chunk_offsets = []

        # Try to load manifest (avoids touching chunk files at all)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = _json.load(f)
            # Validate manifest matches current chunk files
            if manifest.get('chunk_files') == chunk_files:
                self._chunk_sizes = manifest['chunk_sizes']
                logger.info(f"Loaded chunk manifest ({len(chunk_files)} chunks)")
            else:
                self._chunk_sizes = []  # Stale manifest, rebuild

        # Build sizes by loading first chunk to infer size (all chunks are same size
        # except possibly the last), or load each if needed
        if not self._chunk_sizes:
            logger.info(f"Building chunk manifest for {len(chunk_files)} chunks...")
            for chunk_path in self._chunk_paths:
                chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
                self._chunk_sizes.append(chunk['input_ids'].shape[0])
                del chunk
            # Save manifest for next time
            manifest = {
                'chunk_files': chunk_files,
                'chunk_sizes': self._chunk_sizes,
            }
            with open(manifest_path, 'w') as f:
                _json.dump(manifest, f)
            logger.info(f"Saved chunk manifest to {manifest_path}")

        # Build offsets
        offset = 0
        for size in self._chunk_sizes:
            self._chunk_offsets.append(offset)
            offset += size

        self._total_len = offset
        self._chunk_offsets.append(offset)  # sentinel

        # LRU cache: keep a few chunks in memory (default: 2)
        self._chunk_cache = {}
        self._chunk_cache_order = []
        self._chunk_cache_max = 2

        # Set attributes for compatibility
        self._input_ids = None
        self._attention_masks = None
        self._targets = None
        self._sids = None
        self._using_chunks = True

    def _get_chunk(self, chunk_idx: int):
        """Get a chunk by index, using LRU cache."""
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        # Load from disk
        chunk = torch.load(self._chunk_paths[chunk_idx], map_location='cpu', weights_only=False)

        # Evict oldest if cache is full
        while len(self._chunk_cache) >= self._chunk_cache_max:
            oldest = self._chunk_cache_order.pop(0)
            self._chunk_cache.pop(oldest, None)

        self._chunk_cache[chunk_idx] = chunk
        self._chunk_cache_order.append(chunk_idx)
        return chunk

    def _build_from_parquet(self, parquet_path: str, allowed_sids: Optional[Set] = None):
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import tempfile
        import shutil

        dataset = pq.ParquetDataset(parquet_path)
        schema_cols = set(dataset.schema.names)
        columns = [c for c in self._columns if c in schema_cols]

        # Two-pass strategy to avoid OOM:
        # Pass 1: Group data by person and write to temp files
        # Pass 2: Read temp files and tokenize

        temp_dir = tempfile.mkdtemp(prefix='sequence_cache_')
        logger.info(f"Using temp directory: {temp_dir}")

        try:
            # Pass 1: Group by person and save to parquet shards
            person_buffer: Dict[Any, List[pd.DataFrame]] = {}
            shard_idx = 0
            max_buffer_persons = 50000  # Flush every 50k persons

            target_count = len(allowed_sids) if allowed_sids is not None else None
            sids_seen = set()

            def _flush_buffer_to_shard():
                """Write buffered persons to a parquet shard on disk."""
                nonlocal shard_idx
                if not person_buffer:
                    return

                shard_data = []
                for sid, dfs in person_buffer.items():
                    person_df = pd.concat(dfs).sort_values(self.time_col)
                    shard_data.append(person_df)

                if shard_data:
                    shard_df = pd.concat(shard_data, ignore_index=True)
                    shard_path = os.path.join(temp_dir, f"shard_{shard_idx:04d}.parquet")
                    shard_df.to_parquet(shard_path, index=False)
                    logger.info(f"Wrote shard {shard_idx} with {len(person_buffer):,} persons to disk")
                    shard_idx += 1
                    person_buffer.clear()

            # Build pyarrow filter if we have allowed_sids
            if allowed_sids is not None and len(allowed_sids) < 1_000_000:
                allowed_list = list(allowed_sids)
                filter_expr = pc.field(self.id_col).isin(allowed_list)
                logger.info(f"Using pyarrow filter for {len(allowed_list):,} target persons")
            else:
                filter_expr = None

            n_chunks_read = 0
            logger.info("Pass 1: Grouping data by person and writing to temp shards...")

            for fragment in dataset.fragments:
                for batch in fragment.to_batches(batch_size=self.chunk_size, columns=columns, filter=filter_expr):
                    chunk = batch.to_pandas()

                    if len(chunk) == 0:
                        continue

                    chunk = chunk.sort_values([self.id_col, self.time_col])

                    # Accumulate rows per person
                    for sid, person_rows in chunk.groupby(self.id_col):
                        if allowed_sids is not None and sid not in allowed_sids:
                            continue
                        if sid not in person_buffer:
                            person_buffer[sid] = []
                        person_buffer[sid].append(person_rows)

                    n_chunks_read += 1
                    if n_chunks_read % 50 == 0:
                        logger.info(f"Pass 1: Read {n_chunks_read} chunks, buffered {len(person_buffer):,} persons")

                    # Flush buffer to disk when it gets too large
                    if len(person_buffer) >= max_buffer_persons:
                        _flush_buffer_to_shard()

                    del chunk

            # Flush any remaining persons
            if person_buffer:
                _flush_buffer_to_shard()

            logger.info(f"Pass 1 complete: Created {shard_idx} shards in {temp_dir}")

            # Pass 2: Stream-process shards with minimal memory
            # Strategy: Sort shards by person ID, then stream-merge and tokenize
            logger.info("Pass 2: Preparing shards for efficient streaming...")
            shard_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.parquet')])
            logger.info(f"Pass 2: Found {len(shard_files)} shards")

            # Step 2a: Create a single sorted, deduplicated parquet from all shards
            # This uses disk instead of memory
            merged_path = os.path.join(temp_dir, '_merged_sorted.parquet')
            logger.info("Pass 2a: Merging and sorting shards on disk...")

            chunk_dfs = []
            for idx, shard_file in enumerate(shard_files):
                shard_path = os.path.join(temp_dir, shard_file)
                df = pd.read_parquet(shard_path)
                chunk_dfs.append(df)

                # Write to disk every 50 shards to limit memory
                if len(chunk_dfs) >= 50 or idx == len(shard_files) - 1:
                    if len(chunk_dfs) > 0:
                        chunk = pd.concat(chunk_dfs, ignore_index=True)
                        chunk = chunk.drop_duplicates(subset=[self.id_col, self.time_col])
                        chunk = chunk.sort_values([self.id_col, self.time_col])

                        # Append to merged file
                        if os.path.exists(merged_path):
                            existing = pd.read_parquet(merged_path)
                            chunk = pd.concat([existing, chunk], ignore_index=True)
                            chunk = chunk.drop_duplicates(subset=[self.id_col, self.time_col])
                            chunk = chunk.sort_values([self.id_col, self.time_col])

                        chunk.to_parquet(merged_path, index=False)
                        del chunk
                        chunk_dfs = []

                        if (idx + 1) % 50 == 0:
                            logger.info(f"Pass 2a: Merged {idx + 1}/{len(shard_files)} shards...")

            logger.info("Pass 2a: Shards merged and sorted on disk")

            # Step 2b: Stream through sorted merged file and tokenize
            logger.info("Pass 2b: Streaming tokenization from sorted data...")
            all_input_ids = []
            all_masks = []
            all_targets = []
            all_sids = []
            n_processed = 0

            # Read merged parquet in batches via PyArrow
            import pyarrow.parquet as pq_local
            pf = pq_local.ParquetFile(merged_path)
            for batch in pf.iter_batches(batch_size=100_000):
                chunk = batch.to_pandas()
                for sid, person_df in chunk.groupby(self.id_col):
                    person_df = person_df.sort_values(self.time_col)
                    history = person_df[person_df[self.time_col] <= self.cutoff_year]
                    if self.min_history_year is not None:
                        history = history[history[self.time_col] > self.min_history_year]
                    if len(history) == 0:
                        continue

                    tokens = self.vocabulary.tokenize_person_history(
                        history, time_col=self.time_col, max_year=self.cutoff_year,
                        min_year=self.min_history_year,
                    )
                    future = person_df[person_df[self.time_col] > self.cutoff_year]
                    targets = self._build_targets(future)
                    input_ids, attention_mask = self._pad_or_truncate(tokens)

                    all_input_ids.append(input_ids)
                    all_masks.append(attention_mask)
                    all_targets.append(targets)
                    all_sids.append(sid)

                    n_processed += 1
                    if n_processed % 50000 == 0:
                        logger.info(f"Pass 2b: Tokenized {n_processed:,} persons...")

                del chunk

            logger.info(f"Pass 2 complete: Tokenized {n_processed:,} unique persons")

            if all_input_ids:
                self._input_ids = torch.from_numpy(np.stack(all_input_ids))
                self._attention_masks = torch.from_numpy(np.stack(all_masks))
                self._targets = torch.from_numpy(np.stack(all_targets))
                self._sids = np.array(all_sids)
            else:
                self._input_ids = torch.zeros(0, self.max_seq_len, dtype=torch.long)
                self._attention_masks = torch.zeros(0, self.max_seq_len, dtype=torch.float32)
                self._targets = torch.zeros(0, self.n_outputs, dtype=torch.float32)
                self._sids = np.array([])

            logger.info(f"CachedSequenceDataset: {len(self._input_ids)} persons pre-tokenized")

        finally:
            # TEMPORARILY DISABLED: Keep temp directory for recovery if needed
            # Clean up temp directory
            # if os.path.exists(temp_dir):
            #     shutil.rmtree(temp_dir)
            #     logger.info(f"Cleaned up temp directory: {temp_dir}")
            logger.info(f"Temp directory preserved for recovery: {temp_dir}")

    def _build_targets(self, future_df: pd.DataFrame) -> np.ndarray:
        targets = np.zeros(self.n_outputs, dtype=np.float32)
        for ei, event_col in enumerate(self.events):
            if event_col not in future_df.columns:
                continue
            for hi, horizon in enumerate(self.horizons):
                max_year = self.cutoff_year + horizon
                window = future_df[future_df[self.time_col] <= max_year]
                if len(window) > 0 and window[event_col].astype(int).sum() > 0:
                    targets[ei * self.n_horizons + hi] = 1.0
        return targets

    def _pad_or_truncate(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = len(tokens)
        if seq_len >= self.max_seq_len:
            input_ids = np.array(tokens[-self.max_seq_len:], dtype=np.int64)
            attention_mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            input_ids = np.full(self.max_seq_len, self.vocabulary.PAD, dtype=np.int64)
            input_ids[:seq_len] = tokens
            attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
            attention_mask[:seq_len] = 1.0
        return input_ids, attention_mask

    def _resolve_chunk(self, idx: int):
        """Map global index to (chunk, local_index) for chunked loading."""
        import bisect
        chunk_idx = bisect.bisect_right(self._chunk_offsets, idx) - 1
        local_idx = idx - self._chunk_offsets[chunk_idx]
        return self._get_chunk(chunk_idx), local_idx

    def __len__(self) -> int:
        if getattr(self, '_using_chunks', False):
            return self._total_len
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if getattr(self, '_using_chunks', False):
            chunk, local_idx = self._resolve_chunk(idx)
            input_ids = chunk['input_ids'][local_idx]
            attention_mask = chunk['attention_masks'][local_idx]

            # Truncate if chunk was built with a larger max_seq_len
            chunk_seq_len = input_ids.shape[0]
            if chunk_seq_len > self.max_seq_len:
                # Find actual length, keep most recent tokens
                actual_len = int(attention_mask.sum().item())
                if actual_len > self.max_seq_len:
                    # Take last max_seq_len real tokens
                    start = actual_len - self.max_seq_len
                    input_ids = input_ids[start:actual_len].clone()
                    attention_mask = torch.ones(self.max_seq_len, dtype=attention_mask.dtype)
                else:
                    # Just trim the padding tail
                    input_ids = input_ids[:self.max_seq_len].clone()
                    attention_mask = attention_mask[:self.max_seq_len].clone()

            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'targets': chunk['targets'][local_idx],
            }
            if self.return_ids and 'sids' in chunk:
                sample['sid'] = chunk['sids'][local_idx]
            iw = getattr(self, '_importance_weights', None)
            if iw is not None:
                sample['sample_weight'] = iw[idx]
            return sample

        sample = {
            'input_ids': self._input_ids[idx],
            'attention_mask': self._attention_masks[idx],
            'targets': self._targets[idx],
        }
        if self.return_ids and self._sids is not None and len(self._sids) > 0:
            sample['sid'] = self._sids[idx]
        iw = getattr(self, '_importance_weights', None)
        if iw is not None:
            sample['sample_weight'] = iw[idx]
        return sample

    def get_pos_weights(self, max_samples: Optional[int] = None) -> torch.Tensor:
        if getattr(self, '_using_chunks', False):
            import gc as _gc
            # Stream through chunks one at a time to compute pos weights
            total_pos = None
            total_n = 0
            for i in range(len(self._chunk_paths)):
                chunk = torch.load(self._chunk_paths[i], map_location='cpu', weights_only=False)
                t = chunk['targets'].numpy()
                del chunk
                _gc.collect()
                if max_samples is not None and total_n + len(t) > max_samples:
                    t = t[:max_samples - total_n]
                if total_pos is None:
                    total_pos = t.sum(axis=0)
                else:
                    total_pos += t.sum(axis=0)
                total_n += len(t)
                if max_samples is not None and total_n >= max_samples:
                    break
            n_pos = total_pos.clip(min=1.0)
            n_neg = (total_n - n_pos).clip(min=1.0)
            pos_weight = n_neg / n_pos
            return torch.from_numpy(pos_weight.astype(np.float32))

        targets = self._targets.numpy() if max_samples is None else self._targets[:max_samples].numpy()
        n_pos = targets.sum(axis=0).clip(min=1.0)
        n_neg = (len(targets) - n_pos).clip(min=1.0)
        pos_weight = n_neg / n_pos
        return torch.from_numpy(pos_weight.astype(np.float32))

    def get_chunk_event_rates(
        self,
        n_events: int,
        n_horizons: int,
        target_horizon_idx: int = 0,
    ) -> List[float]:
        """Compute per-chunk event rate for balanced sampling.

        For each chunk, returns the fraction of samples that have at least one
        positive 1-year (first horizon) target across all events.

        Args:
            n_events: Number of events.
            n_horizons: Number of horizons.
            target_horizon_idx: Which horizon to use for rate computation (0 = shortest).

        Returns:
            List of event rates, one per chunk.
        """
        import gc as _gc
        if not getattr(self, '_using_chunks', False):
            # Non-chunked: return single rate for whole dataset
            targets = self._targets.numpy()
            any_pos = np.zeros(len(targets), dtype=bool)
            for ei in range(n_events):
                col = ei * n_horizons + target_horizon_idx
                any_pos |= (targets[:, col] > 0.5)
            return [float(any_pos.mean())]

        rates = []
        for i in range(len(self._chunk_paths)):
            chunk = torch.load(self._chunk_paths[i], map_location='cpu', weights_only=False)
            t = chunk['targets'].numpy()
            del chunk
            _gc.collect()

            any_pos = np.zeros(len(t), dtype=bool)
            for ei in range(n_events):
                col = ei * n_horizons + target_horizon_idx
                any_pos |= (t[:, col] > 0.5)
            rates.append(float(any_pos.mean()))

        logger.info(
            f"Chunk event rates (horizon {target_horizon_idx}): "
            f"min={min(rates):.4f}, max={max(rates):.4f}, "
            f"mean={np.mean(rates):.4f}"
        )
        return rates


def sequence_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate that dynamically pads to the max length within the batch
    rather than the global max_seq_len, saving compute.
    """
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    targets = torch.stack([b['targets'] for b in batch])

    # Trim to max actual length in this batch
    actual_lengths = attention_mask.sum(dim=1).long()
    max_len = int(actual_lengths.max().item())
    if max_len < input_ids.size(1):
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'targets': targets,
    }
    if 'sid' in batch[0]:
        output['sid'] = np.array([b['sid'] for b in batch])
    if 'sample_weight' in batch[0]:
        output['sample_weight'] = torch.stack([b['sample_weight'] for b in batch])
    return output


class RollingWindowDataset(Dataset):
    """
    In-memory dataset that creates multiple samples per person using
    rolling cutoff windows. Each person appears once per valid window
    (has both history and future data for that window).

    Example with history_len=5, horizons=[1,3,5]:
      Window 1: history 2011-2015, cutoff=2015, targets 2016-2020
      Window 2: history 2012-2016, cutoff=2016, targets 2017-2021
      ...

    Args:
        df: Panel DataFrame with columns sid, year, events, etc.
        vocabulary: Built LifeEventVocabulary instance.
        cutoff_years: List of cutoff years for rolling windows.
        history_len: Number of years of history per window.
        max_seq_len: Maximum sequence length (pad/truncate).
        events: Event column names to predict.
        horizons: Prediction horizons in years.
        id_col: Person ID column.
        time_col: Time column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocabulary: LifeEventVocabulary,
        cutoff_years: List[int],
        history_len: int = 5,
        max_seq_len: int = 256,
        events: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        id_col: str = 'sid',
        time_col: str = 'year',
    ):
        self.vocabulary = vocabulary
        self.cutoff_years = sorted(cutoff_years)
        self.history_len = history_len
        self.max_seq_len = max_seq_len
        self.events = events or DEFAULT_EVENTS
        self.horizons = horizons or DEFAULT_HORIZONS
        self.id_col = id_col
        self.time_col = time_col

        self.n_events = len(self.events)
        self.n_horizons = len(self.horizons)
        self.n_outputs = self.n_events * self.n_horizons

        self._prepare_rolling(df)

    def _prepare_rolling(self, df: pd.DataFrame):
        """Create one sample per person per valid rolling window."""
        df = df.sort_values([self.id_col, self.time_col])
        grouped = df.groupby(self.id_col)

        self._person_ids = []
        self._cutoff_labels = []
        self._sequences = []
        self._targets = []

        max_horizon = max(self.horizons)
        n_skipped = 0

        for sid, person_df in grouped:
            for cutoff in self.cutoff_years:
                min_year = cutoff - self.history_len

                # History: years in (min_year, cutoff]
                history = person_df[
                    (person_df[self.time_col] > min_year)
                    & (person_df[self.time_col] <= cutoff)
                ]
                if len(history) == 0:
                    n_skipped += 1
                    continue

                # Future: years in (cutoff, cutoff + max_horizon]
                future = person_df[
                    (person_df[self.time_col] > cutoff)
                    & (person_df[self.time_col] <= cutoff + max_horizon)
                ]
                if len(future) == 0:
                    n_skipped += 1
                    continue

                tokens = self.vocabulary.tokenize_person_history(
                    history,
                    time_col=self.time_col,
                    max_year=cutoff,
                    min_year=min_year,
                )
                targets = self._build_targets(future, cutoff_year=cutoff)

                self._person_ids.append(sid)
                self._cutoff_labels.append(cutoff)
                self._sequences.append(tokens)
                self._targets.append(targets)

        if n_skipped > 0:
            logger.info(f"Skipped {n_skipped} person-window pairs (no history or future)")

        per_window = {}
        for c in self._cutoff_labels:
            per_window[c] = per_window.get(c, 0) + 1
        window_str = ', '.join(f"{y}:{n:,}" for y, n in sorted(per_window.items()))

        logger.info(
            f"RollingWindowDataset: {len(self._person_ids):,} total samples "
            f"from {len(self.cutoff_years)} windows [{window_str}]"
        )

    def _build_targets(
        self, future_df: pd.DataFrame, cutoff_year: int,
    ) -> np.ndarray:
        targets = np.zeros(self.n_outputs, dtype=np.float32)
        for ei, event_col in enumerate(self.events):
            if event_col not in future_df.columns:
                continue
            for hi, horizon in enumerate(self.horizons):
                max_year = cutoff_year + horizon
                window = future_df[future_df[self.time_col] <= max_year]
                if len(window) > 0 and window[event_col].astype(int).sum() > 0:
                    targets[ei * self.n_horizons + hi] = 1.0
        return targets

    def _pad_or_truncate(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = len(tokens)
        if seq_len >= self.max_seq_len:
            input_ids = np.array(tokens[-self.max_seq_len:], dtype=np.int64)
            attention_mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            input_ids = np.full(self.max_seq_len, self.vocabulary.PAD, dtype=np.int64)
            input_ids[:seq_len] = tokens
            attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
            attention_mask[:seq_len] = 1.0
        return input_ids, attention_mask

    def __len__(self) -> int:
        return len(self._person_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self._sequences[idx]
        targets = self._targets[idx]
        input_ids, attention_mask = self._pad_or_truncate(tokens)
        return {
            'input_ids': torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'targets': torch.from_numpy(targets),
        }

    def get_pos_weights(self, max_samples: Optional[int] = None) -> torch.Tensor:
        all_targets = np.stack(self._targets)
        if max_samples is not None:
            all_targets = all_targets[:max_samples]
        n_pos = all_targets.sum(axis=0).clip(min=1.0)
        n_neg = (len(all_targets) - n_pos).clip(min=1.0)
        pos_weight = n_neg / n_pos
        return torch.from_numpy(pos_weight.astype(np.float32))


class CachedRollingWindowDataset(Dataset):
    """
    Loads pre-built per-cutoff cache files and presents them as a single
    dataset. Each cache is a .pt file (or _chunks directory) from one
    rolling window cutoff year.

    Reuses the chunked loading infrastructure from CachedSequenceDataset
    for memory-efficient access to large datasets.

    Args:
        cache_paths: List of .pt cache file paths (one per cutoff).
        cutoff_years: Corresponding cutoff years for each cache.
        max_seq_len: Maximum sequence length.
        events: Event column names.
        horizons: Prediction horizons.
    """

    def __init__(
        self,
        cache_paths: List[str],
        cutoff_years: List[int],
        max_seq_len: int = 256,
        events: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        return_ids: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.events = events or DEFAULT_EVENTS
        self.horizons = horizons or DEFAULT_HORIZONS
        self.return_ids = return_ids
        self.n_events = len(self.events)
        self.n_horizons = len(self.horizons)
        self.n_outputs = self.n_events * self.n_horizons
        self.cutoff_years = cutoff_years

        # Collect all chunk paths from all windows
        self._chunk_paths = []
        self._chunk_sizes = []
        self._chunk_cutoffs = []  # which cutoff each chunk belongs to

        for cache_path, cutoff in zip(cache_paths, cutoff_years):
            chunk_dir = cache_path.replace('.pt', '_chunks')

            if os.path.isdir(chunk_dir):
                # Chunked cache: load manifest or scan chunks
                self._load_window_chunks(chunk_dir, cutoff)
            elif os.path.exists(cache_path):
                # Single .pt file: treat as one chunk
                cache = torch.load(cache_path, map_location='cpu', weights_only=False)
                n = cache['input_ids'].shape[0]
                self._chunk_paths.append(cache_path)
                self._chunk_sizes.append(n)
                self._chunk_cutoffs.append(cutoff)
                del cache
                logger.info(f"  Window cutoff={cutoff}: {n:,} samples (single file)")
            else:
                logger.warning(
                    f"Cache not found for cutoff={cutoff}: {cache_path}. "
                    f"Run build_sequence_cache.py --rolling first."
                )

        # Build offsets for global indexing
        self._chunk_offsets = []
        offset = 0
        for size in self._chunk_sizes:
            self._chunk_offsets.append(offset)
            offset += size
        self._chunk_offsets.append(offset)  # sentinel
        self._total_len = offset

        # LRU cache for chunk data
        self._chunk_cache = {}
        self._chunk_cache_order = []
        self._chunk_cache_max = 3
        self._using_chunks = True

        per_window = {}
        for ci, cutoff in enumerate(self._chunk_cutoffs):
            per_window[cutoff] = per_window.get(cutoff, 0) + self._chunk_sizes[ci]
        window_str = ', '.join(f"{y}:{n:,}" for y, n in sorted(per_window.items()))
        logger.info(
            f"CachedRollingWindowDataset: {self._total_len:,} total samples "
            f"from {len(cutoff_years)} windows [{window_str}]"
        )

    def _load_window_chunks(self, chunk_dir: str, cutoff: int):
        """Load chunk metadata from a single window's chunk directory."""
        import json as _json

        chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.pt')])
        manifest_path = os.path.join(chunk_dir, 'manifest.json')

        sizes = []
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = _json.load(f)
            if manifest.get('chunk_files') == chunk_files:
                sizes = manifest['chunk_sizes']

        if not sizes:
            for chunk_file in chunk_files:
                chunk_path = os.path.join(chunk_dir, chunk_file)
                chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
                sizes.append(chunk['input_ids'].shape[0])
                del chunk

        total = sum(sizes)
        for chunk_file, size in zip(chunk_files, sizes):
            self._chunk_paths.append(os.path.join(chunk_dir, chunk_file))
            self._chunk_sizes.append(size)
            self._chunk_cutoffs.append(cutoff)

        logger.info(f"  Window cutoff={cutoff}: {total:,} samples ({len(chunk_files)} chunks)")

    def _get_chunk(self, chunk_idx: int):
        """Get a chunk by index, using LRU cache."""
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        chunk = torch.load(self._chunk_paths[chunk_idx], map_location='cpu', weights_only=False)

        while len(self._chunk_cache) >= self._chunk_cache_max:
            oldest = self._chunk_cache_order.pop(0)
            self._chunk_cache.pop(oldest, None)

        self._chunk_cache[chunk_idx] = chunk
        self._chunk_cache_order.append(chunk_idx)
        return chunk

    def _resolve_chunk(self, idx: int):
        """Map global index to (chunk_data, local_index)."""
        import bisect
        chunk_idx = bisect.bisect_right(self._chunk_offsets, idx) - 1
        local_idx = idx - self._chunk_offsets[chunk_idx]
        return self._get_chunk(chunk_idx), local_idx

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk, local_idx = self._resolve_chunk(idx)
        input_ids = chunk['input_ids'][local_idx]
        attention_mask = chunk['attention_masks'][local_idx]

        # Truncate if chunk was built with a larger max_seq_len
        chunk_seq_len = input_ids.shape[0]
        if chunk_seq_len > self.max_seq_len:
            actual_len = int(attention_mask.sum().item())
            if actual_len > self.max_seq_len:
                start = actual_len - self.max_seq_len
                input_ids = input_ids[start:actual_len].clone()
                attention_mask = torch.ones(self.max_seq_len, dtype=attention_mask.dtype)
            else:
                input_ids = input_ids[:self.max_seq_len].clone()
                attention_mask = attention_mask[:self.max_seq_len].clone()

        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': chunk['targets'][local_idx],
        }
        if self.return_ids and 'sids' in chunk:
            sample['sid'] = chunk['sids'][local_idx]
        return sample

    def get_pos_weights(self, max_samples: Optional[int] = None) -> torch.Tensor:
        """Compute pos weights by streaming through all chunks."""
        import gc as _gc
        total_pos = None
        total_n = 0
        for i in range(len(self._chunk_paths)):
            chunk = torch.load(self._chunk_paths[i], map_location='cpu', weights_only=False)
            t = chunk['targets'].numpy()
            del chunk
            _gc.collect()
            if max_samples is not None and total_n + len(t) > max_samples:
                t = t[:max_samples - total_n]
            if total_pos is None:
                total_pos = t.sum(axis=0)
            else:
                total_pos += t.sum(axis=0)
            total_n += len(t)
            if max_samples is not None and total_n >= max_samples:
                break
        n_pos = total_pos.clip(min=1.0)
        n_neg = (total_n - n_pos).clip(min=1.0)
        pos_weight = n_neg / n_pos
        return torch.from_numpy(pos_weight.astype(np.float32))

    def get_chunk_event_rates(
        self,
        n_events: int,
        n_horizons: int,
        target_horizon_idx: int = 0,
    ) -> List[float]:
        """Compute per-chunk event rate for balanced sampling."""
        import gc as _gc
        rates = []
        for i in range(len(self._chunk_paths)):
            chunk = torch.load(self._chunk_paths[i], map_location='cpu', weights_only=False)
            t = chunk['targets'].numpy()
            del chunk
            _gc.collect()

            any_pos = np.zeros(len(t), dtype=bool)
            for ei in range(n_events):
                col = ei * n_horizons + target_horizon_idx
                any_pos |= (t[:, col] > 0.5)
            rates.append(float(any_pos.mean()))

        logger.info(
            f"Rolling chunk event rates (horizon {target_horizon_idx}): "
            f"min={min(rates):.4f}, max={max(rates):.4f}, "
            f"mean={np.mean(rates):.4f}"
        )
        return rates
