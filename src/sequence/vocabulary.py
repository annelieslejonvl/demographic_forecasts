"""
Token vocabulary for demographic life-event sequences.

Maps person-year observations (year, municipality context, demographics, events)
to integer token IDs for use with sequence models.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Event column names and their corresponding token names
EVENT_TOKEN_MAP = {
    'y_moved': 'MOVED',
    'birth1_event': 'BIRTH1',
    'birth2_event': 'BIRTH2',
    'divorce_event': 'DIVORCE',
    'getalifeother_event': 'PARTNERSHIP',
}

# Municipality socioeconomic features -> token prefixes
# Each is binned into quantiles (Q1-Q10) during vocabulary building
MUNICIPALITY_FEATURE_MAP = {
    'muni_urban_context': 'MUNI_URBAN',
    'muni_income_trend_3yr': 'MUNI_INCOME_TREND',
    'muni_immigration_rate_context': 'MUNI_IMMIG_RATE',
    'muni_median_age_lag1': 'MUNI_MED_AGE',
    'muni_avg_household_size_context': 'MUNI_HH_SIZE',
    'muni_emigration_rate_context': 'MUNI_EMIG_RATE',
}

# Features to keep as raw numeric values alongside tokens (hybrid mode)
NUMERIC_FEATURE_COLUMNS = [
    'muni_urban_context',
    'muni_income_trend_3yr',
    'muni_immigration_rate_context',
    'muni_median_age_lag1',
    'muni_avg_household_size_context',
    'muni_emigration_rate_context',
    'age',
]
N_NUMERIC_FEATURES = len(NUMERIC_FEATURE_COLUMNS)


class LifeEventVocabulary:
    """
    Manages token vocabulary for demographic life-event sequences.

    Token types:
      - Special: PAD (0), BOS (1), EOS (2), SEP (3), UNK (4)
      - Year tokens: YEAR_2011 .. YEAR_2025
      - Municipality context tokens: MUNI_{feature}_Q{1..10} (quantile bins)
      - Age group tokens: AGE_{0..9}
      - Gender tokens: GENDER_0, GENDER_1
      - Coupled tokens: COUPLED_0, COUPLED_1
      - Nationality tokens: NAT_{code}
      - Household position tokens: HHPOS_{code}
      - Income quintile tokens: INCOME_Q{1..5}
      - Event tokens: MOVED, BIRTH1, BIRTH2, DIVORCE, PARTNERSHIP, NO_EVENT
    """

    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3
    UNK = 4

    SPECIAL_TOKENS = {
        'PAD': 0,
        'BOS': 1,
        'EOS': 2,
        'SEP': 3,
        'UNK': 4,
    }

    def __init__(self):
        self._token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self._id_to_token: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self._next_id: int = len(self.SPECIAL_TOKENS)
        self._built = False
        # Quantile bin edges for municipality features (computed from data)
        self._muni_quantile_bins: Dict[str, np.ndarray] = {}
        # Normalization stats for numeric features: {col: (mean, std)}
        self._numeric_stats: Dict[str, Tuple[float, float]] = {}

    def _add_token(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        tid = self._next_id
        self._token_to_id[token] = tid
        self._id_to_token[tid] = token
        self._next_id += 1
        return tid

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        year_range: tuple = (2011, 2025),
        age_col: str = 'age',
        age_group_col: str = 'age_group',
        nationality_col: str = 'eerste_nationaliteit',
        gender_col: str = 'gender',
        coupled_col: str = 'coupled',
        hh_pos_col: str = 'hh_pos',
        income_quintile_col: str = 'income_quintile',
        n_muni_bins: int = 10,
    ) -> "LifeEventVocabulary":
        """
        Build vocabulary from a DataFrame by scanning unique values.

        Args:
            df: Panel DataFrame with person-year rows.
            year_range: (min_year, max_year) inclusive.
            n_muni_bins: Number of quantile bins for municipality features.
        """
        # Year tokens
        for y in range(year_range[0], year_range[1] + 1):
            self._add_token(f'YEAR_{y}')

        # Municipality socioeconomic tokens (quantile-binned)
        for muni_col, token_prefix in MUNICIPALITY_FEATURE_MAP.items():
            if muni_col in df.columns:
                values = pd.to_numeric(df[muni_col], errors='coerce').astype('float64').dropna()
                if len(values) > 0:
                    # Compute quantile bin edges from the data
                    quantiles = np.linspace(0, 1, n_muni_bins + 1)
                    bin_edges = np.quantile(values, quantiles)
                    # Deduplicate edges (can happen with skewed distributions)
                    bin_edges = np.unique(bin_edges)
                    self._muni_quantile_bins[muni_col] = bin_edges

            # Always add all quantile tokens (Q1..Qn) for consistency
            for q in range(1, n_muni_bins + 1):
                self._add_token(f'{token_prefix}_Q{q}')

        # Age group tokens (decade bins 0-9)
        if age_group_col in df.columns:
            for ag in sorted(df[age_group_col].dropna().unique()):
                self._add_token(f'AGE_{int(ag)}')
        elif age_col in df.columns:
            for decade in range(10):
                self._add_token(f'AGE_{decade}')

        # Gender tokens
        if gender_col in df.columns:
            for g in sorted(df[gender_col].dropna().unique()):
                self._add_token(f'GENDER_{int(g)}')
        else:
            self._add_token('GENDER_0')
            self._add_token('GENDER_1')

        # Coupled tokens
        if coupled_col in df.columns:
            for c in sorted(df[coupled_col].dropna().unique()):
                self._add_token(f'COUPLED_{int(c)}')
        else:
            self._add_token('COUPLED_0')
            self._add_token('COUPLED_1')

        # Nationality tokens
        if nationality_col in df.columns:
            for nat in sorted(df[nationality_col].dropna().unique()):
                self._add_token(f'NAT_{int(nat)}')

        # Household position tokens
        if hh_pos_col in df.columns:
            for hp in sorted(df[hh_pos_col].dropna().unique()):
                self._add_token(f'HHPOS_{int(hp)}')

        # Income quintile tokens
        if income_quintile_col in df.columns:
            for q in sorted(df[income_quintile_col].dropna().unique()):
                self._add_token(f'INCOME_Q{int(q)}')
        else:
            for q in range(1, 6):
                self._add_token(f'INCOME_Q{q}')

        # Event tokens (always added) + NO_EVENT for years without any event
        self._add_token('NO_EVENT')
        for event_col, token_name in EVENT_TOKEN_MAP.items():
            self._add_token(token_name)

        self._built = True
        logger.info(f"Vocabulary built: {self.vocab_size} tokens")
        return self

    def _get_muni_quintile(self, col: str, value: float) -> int:
        """Map a municipality feature value to its quintile bin (1-based)."""
        bin_edges = self._muni_quantile_bins.get(col)
        if bin_edges is None:
            return 3  # Default to middle bin if no edges available
        # np.searchsorted gives the insertion index; clip to valid range
        bin_idx = int(np.searchsorted(bin_edges, value, side='right'))
        # Clamp to [1, n_bins] (n_bins = len(edges) - 1, but we may
        # have fewer edges after dedup)
        n_bins = max(len(bin_edges) - 1, 1)
        return max(1, min(bin_idx, n_bins))

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.UNK)

    def id_to_token(self, tid: int) -> str:
        return self._id_to_token.get(tid, 'UNK')

    def tokenize_year_observation(
        self,
        row: pd.Series,
        year_col: str = 'year',
        age_col: str = 'age',
        age_group_col: str = 'age_group',
        gender_col: str = 'gender',
        coupled_col: str = 'coupled',
        nationality_col: str = 'eerste_nationaliteit',
        hh_pos_col: str = 'hh_pos',
        income_quintile_col: str = 'income_quintile',
        event_cols: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Convert a single person-year row to a list of token IDs.

        Order: year, municipality context (quintile-binned socioeconomic
               features), age, gender, coupled, nationality,
               household position, income quintile, then event tokens
               (only if event occurred).
        """
        tokens = []

        # Year
        if year_col in row.index:
            tokens.append(self.token_to_id(f'YEAR_{int(row[year_col])}'))

        # Municipality socioeconomic features (quintile tokens)
        for muni_col, token_prefix in MUNICIPALITY_FEATURE_MAP.items():
            if muni_col in row.index and pd.notna(row[muni_col]):
                q = self._get_muni_quintile(muni_col, float(row[muni_col]))
                tokens.append(self.token_to_id(f'{token_prefix}_Q{q}'))

        # Age group
        if age_group_col in row.index and pd.notna(row[age_group_col]):
            tokens.append(self.token_to_id(f'AGE_{int(row[age_group_col])}'))
        elif age_col in row.index and pd.notna(row[age_col]):
            decade = min(int(row[age_col]) // 10, 9)
            tokens.append(self.token_to_id(f'AGE_{decade}'))

        # Gender
        if gender_col in row.index and pd.notna(row[gender_col]):
            tokens.append(self.token_to_id(f'GENDER_{int(row[gender_col])}'))

        # Coupled
        if coupled_col in row.index and pd.notna(row[coupled_col]):
            tokens.append(self.token_to_id(f'COUPLED_{int(row[coupled_col])}'))

        # Nationality
        if nationality_col in row.index and pd.notna(row[nationality_col]):
            tokens.append(self.token_to_id(f'NAT_{int(row[nationality_col])}'))

        # Household position
        if hh_pos_col in row.index and pd.notna(row[hh_pos_col]):
            tokens.append(self.token_to_id(f'HHPOS_{int(row[hh_pos_col])}'))

        # Income quintile
        if income_quintile_col in row.index and pd.notna(row[income_quintile_col]):
            tokens.append(self.token_to_id(f'INCOME_Q{int(row[income_quintile_col])}'))

        # Event tokens (only emit if event occurred)
        if event_cols is None:
            event_cols = list(EVENT_TOKEN_MAP.keys())
        event_fired = False
        for event_col in event_cols:
            if event_col in row.index:
                val = row[event_col]
                if pd.notna(val) and int(val) == 1:
                    tokens.append(self.token_to_id(EVENT_TOKEN_MAP[event_col]))
                    event_fired = True

        # Explicit NO_EVENT token when no events occurred this year
        if not event_fired:
            tokens.append(self.token_to_id('NO_EVENT'))

        return tokens

    def tokenize_person_history(
        self,
        person_df: pd.DataFrame,
        time_col: str = 'year',
        max_year: Optional[int] = None,
        min_year: Optional[int] = None,
        **kwargs,
    ) -> List[int]:
        """
        Convert a person's full panel into a token sequence.

        Args:
            person_df: DataFrame rows for a single person, sorted by year.
            max_year: Only include observations up to this year (inclusive).
            min_year: Only include observations after this year (exclusive).

        Returns:
            [BOS, year1_tokens, SEP, year2_tokens, SEP, ..., EOS]
        """
        person_df = person_df.sort_values(time_col)
        if max_year is not None:
            person_df = person_df[person_df[time_col] <= max_year]
        if min_year is not None:
            person_df = person_df[person_df[time_col] > min_year]

        if len(person_df) == 0:
            return [self.BOS, self.EOS]

        return self._tokenize_person_fast(person_df, time_col=time_col)

    def tokenize_person_history_with_numerics(
        self,
        person_df: pd.DataFrame,
        time_col: str = 'year',
        max_year: Optional[int] = None,
        min_year: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Tokenize a person's history and extract aligned numeric features.

        Returns:
            tokens: List[int] - token IDs (same as tokenize_person_history)
            numeric_features: np.ndarray of shape (len(tokens), N_NUMERIC_FEATURES)
                Each token position gets the numeric vector for its year-observation.
                BOS, SEP, EOS positions get zeros.
        """
        person_df = person_df.sort_values(time_col)
        if max_year is not None:
            person_df = person_df[person_df[time_col] <= max_year]
        if min_year is not None:
            person_df = person_df[person_df[time_col] > min_year]

        if len(person_df) == 0:
            tokens = [self.BOS, self.EOS]
            numeric = np.zeros((2, N_NUMERIC_FEATURES), dtype=np.float32)
            return tokens, numeric

        return self._tokenize_person_fast_with_numerics(person_df, time_col=time_col)

    def _tokenize_person_fast(
        self,
        person_df: pd.DataFrame,
        time_col: str = 'year',
    ) -> List[int]:
        """Vectorized tokenization — avoids iterrows() for speed."""
        tokens = [self.BOS]
        cols = person_df.columns
        t2id = self._token_to_id
        unk = self.UNK

        # Pre-check which columns exist
        has_year = time_col in cols
        has_age_group = 'age_group' in cols
        has_age = 'age' in cols
        has_gender = 'gender' in cols
        has_coupled = 'coupled' in cols
        has_nat = 'eerste_nationaliteit' in cols
        has_hhpos = 'hh_pos' in cols
        has_income = 'income_quintile' in cols

        # Municipality columns that exist
        muni_cols_present = [
            (c, prefix)
            for c, prefix in MUNICIPALITY_FEATURE_MAP.items()
            if c in cols
        ]

        # Event columns that exist
        event_cols_present = [
            (ec, t2id.get(tn, unk))
            for ec, tn in EVENT_TOKEN_MAP.items()
            if ec in cols
        ]
        no_event_id = t2id.get('NO_EVENT', unk)

        # Get numpy arrays for fast access
        values = person_df.values
        col_idx = {c: i for i, c in enumerate(cols)}

        year_ci = col_idx.get(time_col)
        ag_ci = col_idx.get('age_group')
        age_ci = col_idx.get('age')
        gender_ci = col_idx.get('gender')
        coupled_ci = col_idx.get('coupled')
        nat_ci = col_idx.get('eerste_nationaliteit')
        hhpos_ci = col_idx.get('hh_pos')
        income_ci = col_idx.get('income_quintile')

        muni_cis = [(col_idx[c], prefix) for c, prefix in muni_cols_present]
        event_cis = [(col_idx[ec], tid) for ec, tid in event_cols_present]

        for row_i in range(len(values)):
            if row_i > 0:
                tokens.append(self.SEP)

            row = values[row_i]

            # Year
            if has_year:
                v = row[year_ci]
                if v == v:  # fast NaN check
                    tokens.append(t2id.get(f'YEAR_{int(v)}', unk))

            # Municipality features
            for ci, prefix in muni_cis:
                v = row[ci]
                if v == v:  # not NaN
                    q = self._get_muni_quintile_raw(ci, float(v), prefix)
                    tokens.append(q)

            # Age group
            if has_age_group:
                v = row[ag_ci]
                if v == v:
                    tokens.append(t2id.get(f'AGE_{int(v)}', unk))
            elif has_age:
                v = row[age_ci]
                if v == v:
                    tokens.append(t2id.get(f'AGE_{min(int(v) // 10, 9)}', unk))

            # Gender
            if has_gender:
                v = row[gender_ci]
                if v == v:
                    tokens.append(t2id.get(f'GENDER_{int(v)}', unk))

            # Coupled
            if has_coupled:
                v = row[coupled_ci]
                if v == v:
                    tokens.append(t2id.get(f'COUPLED_{int(v)}', unk))

            # Nationality
            if has_nat:
                v = row[nat_ci]
                if v == v:
                    tokens.append(t2id.get(f'NAT_{int(v)}', unk))

            # Household position
            if has_hhpos:
                v = row[hhpos_ci]
                if v == v:
                    tokens.append(t2id.get(f'HHPOS_{int(v)}', unk))

            # Income quintile
            if has_income:
                v = row[income_ci]
                if v == v:
                    tokens.append(t2id.get(f'INCOME_Q{int(v)}', unk))

            # Events
            event_fired = False
            for ci, tid in event_cis:
                v = row[ci]
                if v == v and int(v) == 1:
                    tokens.append(tid)
                    event_fired = True
            if not event_fired:
                tokens.append(no_event_id)

        tokens.append(self.EOS)
        return tokens

    def _tokenize_person_fast_with_numerics(
        self,
        person_df: pd.DataFrame,
        time_col: str = 'year',
    ) -> Tuple[List[int], np.ndarray]:
        """Tokenize with parallel numeric feature extraction."""
        tokens = [self.BOS]
        numeric_rows = [np.zeros(N_NUMERIC_FEATURES, dtype=np.float32)]  # BOS = zeros

        cols = person_df.columns
        t2id = self._token_to_id
        unk = self.UNK

        has_year = time_col in cols
        has_age_group = 'age_group' in cols
        has_age = 'age' in cols
        has_gender = 'gender' in cols
        has_coupled = 'coupled' in cols
        has_nat = 'eerste_nationaliteit' in cols
        has_hhpos = 'hh_pos' in cols
        has_income = 'income_quintile' in cols

        muni_cols_present = [
            (c, prefix) for c, prefix in MUNICIPALITY_FEATURE_MAP.items() if c in cols
        ]
        event_cols_present = [
            (ec, t2id.get(tn, unk)) for ec, tn in EVENT_TOKEN_MAP.items() if ec in cols
        ]
        no_event_id = t2id.get('NO_EVENT', unk)

        values = person_df.values
        col_idx = {c: i for i, c in enumerate(cols)}

        year_ci = col_idx.get(time_col)
        ag_ci = col_idx.get('age_group')
        age_ci = col_idx.get('age')
        gender_ci = col_idx.get('gender')
        coupled_ci = col_idx.get('coupled')
        nat_ci = col_idx.get('eerste_nationaliteit')
        hhpos_ci = col_idx.get('hh_pos')
        income_ci = col_idx.get('income_quintile')

        muni_cis = [(col_idx[c], prefix) for c, prefix in muni_cols_present]
        event_cis = [(col_idx[ec], tid) for ec, tid in event_cols_present]

        # Pre-compute numeric column indices
        numeric_col_indices = []
        for nc in NUMERIC_FEATURE_COLUMNS:
            numeric_col_indices.append(col_idx.get(nc))

        for row_i in range(len(values)):
            if row_i > 0:
                tokens.append(self.SEP)
                numeric_rows.append(np.zeros(N_NUMERIC_FEATURES, dtype=np.float32))  # SEP = zeros

            row = values[row_i]

            # Extract normalized numeric features for this row
            row_numeric = np.zeros(N_NUMERIC_FEATURES, dtype=np.float32)
            for ni, nc in enumerate(NUMERIC_FEATURE_COLUMNS):
                ci = numeric_col_indices[ni]
                if ci is not None:
                    v = row[ci]
                    if v == v:  # not NaN
                        row_numeric[ni] = self.normalize_numeric(nc, float(v))

            # Count tokens before this row to know how many to broadcast
            tokens_before = len(tokens)

            # --- Standard tokenization (same as _tokenize_person_fast) ---
            if has_year:
                v = row[year_ci]
                if v == v:
                    tokens.append(t2id.get(f'YEAR_{int(v)}', unk))

            for ci, prefix in muni_cis:
                v = row[ci]
                if v == v:
                    q = self._get_muni_quintile_raw(ci, float(v), prefix)
                    tokens.append(q)

            if has_age_group:
                v = row[ag_ci]
                if v == v:
                    tokens.append(t2id.get(f'AGE_{int(v)}', unk))
            elif has_age:
                v = row[age_ci]
                if v == v:
                    tokens.append(t2id.get(f'AGE_{min(int(v) // 10, 9)}', unk))

            if has_gender:
                v = row[gender_ci]
                if v == v:
                    tokens.append(t2id.get(f'GENDER_{int(v)}', unk))

            if has_coupled:
                v = row[coupled_ci]
                if v == v:
                    tokens.append(t2id.get(f'COUPLED_{int(v)}', unk))

            if has_nat:
                v = row[nat_ci]
                if v == v:
                    tokens.append(t2id.get(f'NAT_{int(v)}', unk))

            if has_hhpos:
                v = row[hhpos_ci]
                if v == v:
                    tokens.append(t2id.get(f'HHPOS_{int(v)}', unk))

            if has_income:
                v = row[income_ci]
                if v == v:
                    tokens.append(t2id.get(f'INCOME_Q{int(v)}', unk))

            event_fired = False
            for ci, tid in event_cis:
                v = row[ci]
                if v == v and int(v) == 1:
                    tokens.append(tid)
                    event_fired = True
            if not event_fired:
                tokens.append(no_event_id)

            # Broadcast numeric vector to all data tokens from this row
            n_data_tokens = len(tokens) - tokens_before
            for _ in range(n_data_tokens):
                numeric_rows.append(row_numeric)

        tokens.append(self.EOS)
        numeric_rows.append(np.zeros(N_NUMERIC_FEATURES, dtype=np.float32))  # EOS = zeros

        numeric_features = np.stack(numeric_rows, axis=0)
        return tokens, numeric_features

    def _get_muni_quintile_raw(self, col_idx: int, value: float, prefix: str) -> int:
        """Fast quintile lookup returning token ID directly.

        Uses prefix to find the column name in _muni_quantile_bins via the
        reverse mapping cached on first call.
        """
        if not hasattr(self, '_prefix_to_col'):
            self._prefix_to_col = {
                v: k for k, v in MUNICIPALITY_FEATURE_MAP.items()
            }
        col = self._prefix_to_col.get(prefix)
        if col is None:
            return self._token_to_id.get(f'{prefix}_Q5', self.UNK)
        bin_edges = self._muni_quantile_bins.get(col)
        if bin_edges is None:
            return self._token_to_id.get(f'{prefix}_Q5', self.UNK)
        bin_idx = int(np.searchsorted(bin_edges, value, side='right'))
        n_bins = max(len(bin_edges) - 1, 1)
        q = max(1, min(bin_idx, n_bins))
        return self._token_to_id.get(f'{prefix}_Q{q}', self.UNK)

    def compute_numeric_stats(self, df: pd.DataFrame) -> None:
        """Compute mean/std for numeric features from a DataFrame."""
        self._numeric_stats = {}
        for col in NUMERIC_FEATURE_COLUMNS:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    self._numeric_stats[col] = (float(values.mean()), float(values.std()))
                else:
                    self._numeric_stats[col] = (0.0, 1.0)
        logger.info(f"Computed numeric stats for {len(self._numeric_stats)} features")

    def compute_numeric_stats_from_parquet(
        self, parquet_path: str, chunk_size: int = 500_000,
    ) -> None:
        """Compute mean/std for numeric features by streaming parquet (Welford)."""
        import pyarrow.parquet as pq

        dataset = pq.ParquetDataset(parquet_path)
        available = [
            c for c in NUMERIC_FEATURE_COLUMNS
            if c in set(dataset.schema.names)
        ]

        counts: Dict[str, int] = {c: 0 for c in available}
        means: Dict[str, float] = {c: 0.0 for c in available}
        m2s: Dict[str, float] = {c: 0.0 for c in available}

        for fragment in dataset.fragments:
            for batch in fragment.to_batches(batch_size=chunk_size, columns=available):
                df = batch.to_pandas()
                for col in available:
                    vals = pd.to_numeric(df[col], errors='coerce').dropna().values.astype(np.float64)
                    n_new = len(vals)
                    if n_new == 0:
                        continue
                    batch_mean = float(vals.mean())
                    batch_var = float(vals.var()) if n_new > 1 else 0.0
                    n_old = counts[col]
                    n_total = n_old + n_new
                    delta = batch_mean - means[col]
                    means[col] = (n_old * means[col] + n_new * batch_mean) / n_total
                    m2s[col] += batch_var * (n_new - 1) + delta ** 2 * n_old * n_new / n_total
                    counts[col] = n_total
                del df

        self._numeric_stats = {}
        for col in available:
            if counts[col] > 1:
                std = (m2s[col] / (counts[col] - 1)) ** 0.5
                self._numeric_stats[col] = (means[col], max(std, 1e-8))
            else:
                self._numeric_stats[col] = (0.0, 1.0)

        logger.info(f"Computed numeric stats for {len(self._numeric_stats)} features (from parquet)")
        for col, (m, s) in self._numeric_stats.items():
            logger.info(f"  {col}: mean={m:.4f}, std={s:.4f}")

    def normalize_numeric(self, col: str, value: float) -> float:
        """Normalize a single numeric value using stored stats."""
        mean, std = self._numeric_stats.get(col, (0.0, 1.0))
        return (value - mean) / max(std, 1e-8)

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def save(self, path: str) -> None:
        joblib.dump({
            'token_to_id': self._token_to_id,
            'id_to_token': self._id_to_token,
            'next_id': self._next_id,
            'built': self._built,
            'muni_quantile_bins': self._muni_quantile_bins,
            'numeric_stats': self._numeric_stats,
        }, path)
        logger.info(f"Vocabulary saved to {path} ({self.vocab_size} tokens)")

    @classmethod
    def load(cls, path: str) -> "LifeEventVocabulary":
        data = joblib.load(path)
        vocab = cls()
        vocab._token_to_id = data['token_to_id']
        vocab._id_to_token = data['id_to_token']
        vocab._next_id = data['next_id']
        vocab._built = data['built']
        vocab._muni_quantile_bins = data.get('muni_quantile_bins', {})
        vocab._numeric_stats = data.get('numeric_stats', {})
        logger.info(f"Vocabulary loaded from {path} ({vocab.vocab_size} tokens)")
        return vocab

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"LifeEventVocabulary(size={self.vocab_size}, built={self._built})"
