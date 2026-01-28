"""
Dataset building utilities.

DatasetSpec: Configuration for train/valid/test splits and sampling
DatasetBuilder: Creates time-based splits and applies sampling strategies
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .sampling import SamplingConfig, SamplingStrategy, SparkSampler


@dataclass
class DatasetSpec:
    """
    Dataset specification - defines splits and sampling strategy.
    
    Maps to configs/datasets/default.yaml structure.
    """
    
    key_cols: List[str] = field(default_factory=list)
    
    # Split configuration
    split: Dict[str, Any] = field(default_factory=lambda: {
        "year_col": "year",
        "train_range": [2018, 2022],
        "valid_range": None,
        "test_range": [2023, 2025],
    })
    
    # Mode-specific configurations
    tune: Dict[str, Any] = field(default_factory=lambda: {
        "sampling": {"type": "stratified", "fraction": 0.1},
        "hard_negatives": {"enabled": False},
    })
    
    final: Dict[str, Any] = field(default_factory=lambda: {
        "sampling": {"type": "none"},
        "hard_negatives": {"enabled": False},
    })
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetSpec":
        """Create from dictionary (YAML config)."""
        return cls(
            key_cols=d.get("key_cols", []),
            split=d.get("split", {}),
            tune=d.get("tune", {}),
            final=d.get("final", {}),
        )
    
    def get_sampling_config(self, mode: str) -> SamplingConfig:
        """Get sampling config for mode (tune or final)."""
        mode_cfg = self.tune if mode == "tune" else self.final
        sampling_dict = mode_cfg.get("sampling", {"type": "none"})
        return SamplingConfig.from_dict(sampling_dict)


class DatasetBuilder:
    """
    Builds train/valid/test datasets with time-based splits and sampling.
    
    Usage:
        builder = DatasetBuilder(label_col="y", key_cols=["id"], year_col="year")
        train, valid, test = builder.time_split(df, (2018, 2022), (2022, 2023), (2023, 2025))
        train_sampled, info = builder.sample_train(train, {"type": "undersample", "target_ratio": 0.5})
    """
    
    def __init__(
        self,
        label_col: str,
        key_cols: Optional[List[str]] = None,
        year_col: str = "year",
    ):
        self.label_col = label_col
        self.key_cols = key_cols or []
        self.year_col = year_col
    
    def time_split(
        self,
        df,
        train_range: Tuple[int, int],
        valid_range: Optional[Tuple[int, int]],
        test_range: Tuple[int, int],
    ) -> Tuple[Any, Optional[Any], Any]:
        """
        Split DataFrame by time periods.
        
        Args:
            df: Spark DataFrame with year_col
            train_range: (start_year, end_year) inclusive for training
            valid_range: (start_year, end_year) for validation, or None
            test_range: (start_year, end_year) for testing
        
        Returns:
            (train_df, valid_df, test_df) - valid_df is None if valid_range is None
        """
        from pyspark.sql import functions as F
        
        year_col = self.year_col
        
        # Training data
        train_df = df.filter(
            (F.col(year_col) >= train_range[0]) & 
            (F.col(year_col) < train_range[1])
        )
        
        # Validation data (optional)
        valid_df = None
        if valid_range:
            valid_df = df.filter(
                (F.col(year_col) >= valid_range[0]) & 
                (F.col(year_col) < valid_range[1])
            )
        
        # Test data
        test_df = df.filter(
            (F.col(year_col) >= test_range[0]) & 
            (F.col(year_col) < test_range[1])
        )
        
        return train_df, valid_df, test_df
    
    def sample_train(
        self,
        train_df,
        sampling_cfg: Dict[str, Any],
        hard_neg_cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply sampling strategy to training data.

        Args:
            train_df: Training Spark DataFrame
            sampling_cfg: Sampling configuration dict
            hard_neg_cfg: Hard negative mining config (optional)

        Returns:
            (sampled_df, info_dict)
        """
        config = SamplingConfig.from_dict(sampling_cfg)

        if config.strategy == SamplingStrategy.NONE:
            n_total = train_df.count()
            return train_df, {"strategy": "none", "n_total": n_total}

        sampler = SparkSampler(config)

        # Hard negative mining requires baseline model
        if config.strategy == SamplingStrategy.HARD_NEGATIVE:
            return self._sample_with_hard_negatives(
                train_df, sampler, hard_neg_cfg or {}
            )

        sampled_df, info = sampler.sample(train_df, self.label_col, self.key_cols)
    
        return sampled_df, info

    def _sample_with_hard_negatives(
        self,
        train_df,
        sampler: SparkSampler,
        hard_neg_cfg: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Sample using hard negative mining with a baseline model.

        Trains a simple logistic regression to score negatives,
        then selects the hardest (highest probability) negatives.
        """
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import functions as F

        print("🔄 Hard negative mining: training baseline model...")

        # Get baseline config
        baseline_cfg = hard_neg_cfg.get("baseline", {})
        sample_frac = baseline_cfg.get("sample_fraction", 0.1)
        max_iter = baseline_cfg.get("params", {}).get("maxIter", 100)
        reg_param = baseline_cfg.get("params", {}).get("regParam", 0.01)

        # Get numeric columns for baseline (exclude label and keys)
        exclude_cols = set([self.label_col] + self.key_cols + [self.year_col])
        feature_cols = [
            c for c in train_df.columns
            if c not in exclude_cols
            and train_df.schema[c].dataType.simpleString() in ('double', 'float', 'int', 'bigint')
        ]

        if not feature_cols:
            raise ValueError("No numeric features found for baseline model")

        print(f"   Using {len(feature_cols)} numeric features for baseline")

        # Sample for baseline training
        baseline_train = train_df.sample(fraction=sample_frac, seed=42)

        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="_baseline_features",
            handleInvalid="skip"
        )
        baseline_train = assembler.transform(baseline_train)

        # Train baseline logistic regression
        lr = LogisticRegression(
            featuresCol="_baseline_features",
            labelCol=self.label_col,
            maxIter=max_iter,
            regParam=reg_param,
        )
        lr_model = lr.fit(baseline_train)
        print(f"   ✓ Baseline model trained")

        # Score all training data
        print("   Scoring negatives...")
        train_assembled = assembler.transform(train_df)
        scored = lr_model.transform(train_assembled)

        # Extract probability of positive class
        from pyspark.ml.functions import vector_to_array
        scores_df = scored.select(
            *self.key_cols,
            vector_to_array(F.col("probability"))[1].alias("score")
        )

        # Apply hard negative sampling
        sampled_df, info = sampler.sample_with_hard_negatives(
            train_df,
            self.label_col,
            scores_df,
            self.key_cols,
            score_col="score",
        )

        print(f"   ✓ Hard negative sampling complete: {info.get('n_hard_negatives', 0)} hard + {info.get('n_random_negatives', 0)} random negatives")

        return sampled_df, info
    
    def deduplicate(
        self,
        df,
        key_cols: Optional[List[str]] = None,
        keep: str = "first",
    ):
        """
        Remove duplicate rows based on key columns.
        
        Args:
            df: Spark DataFrame
            key_cols: Columns to deduplicate on (uses self.key_cols if None)
            keep: "first" or "last"
        
        Returns:
            Deduplicated DataFrame
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        key_cols = key_cols or self.key_cols
        
        if not key_cols:
            return df.dropDuplicates()
        
        if keep == "first":
            window = Window.partitionBy(*key_cols).orderBy(F.monotonically_increasing_id())
        else:
            window = Window.partitionBy(*key_cols).orderBy(F.monotonically_increasing_id().desc())
        
        df_ranked = df.withColumn("_rank", F.row_number().over(window))
        df_dedup = df_ranked.filter(F.col("_rank") == 1).drop("_rank")
        
        return df_dedup
    
    def get_class_distribution(self, df) -> Dict[str, Any]:
        """Get class distribution statistics."""
        from pyspark.sql import functions as F
        
        stats = df.groupBy(self.label_col).count().collect()
        
        distribution = {int(row[self.label_col]): row["count"] for row in stats}
        total = sum(distribution.values())
        
        return {
            "distribution": distribution,
            "total": total,
            "positive_rate": distribution.get(1, 0) / total if total > 0 else 0,
        }
    
    def print_split_summary(
        self,
        train_df,
        valid_df,
        test_df,
    ) -> None:
        """Print summary of data splits."""
        print("=" * 60)
        print("DATASET SPLIT SUMMARY")
        print("=" * 60)
        
        train_stats = self.get_class_distribution(train_df)
        print(f"Train: {train_stats['total']:,} rows, {train_stats['positive_rate']*100:.2f}% positive")
        
        if valid_df is not None:
            valid_stats = self.get_class_distribution(valid_df)
            print(f"Valid: {valid_stats['total']:,} rows, {valid_stats['positive_rate']*100:.2f}% positive")
        
        test_stats = self.get_class_distribution(test_df)
        print(f"Test:  {test_stats['total']:,} rows, {test_stats['positive_rate']*100:.2f}% positive")
        
        print("=" * 60)


def create_dataset_builder(
    spec: DatasetSpec,
    label_col: str,
) -> DatasetBuilder:
    """Factory function to create DatasetBuilder from spec."""
    return DatasetBuilder(
        label_col=label_col,
        key_cols=spec.key_cols,
        year_col=spec.split.get("year_col", "year"),
    )
