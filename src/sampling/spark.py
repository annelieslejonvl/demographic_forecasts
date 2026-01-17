"""Spark DataFrame sampling strategies."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import SamplingConfig, SamplingStrategy

logger = logging.getLogger(__name__)


class SparkSampler:
    """Sampling strategies for Spark DataFrames."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
    
    def sample(
        self,
        df,
        label_col: str,
        key_cols: Optional[List[str]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply sampling strategy to DataFrame."""
        strategy = self.config.strategy
        
        if strategy == SamplingStrategy.NONE:
            return df, {"strategy": "none", "n_original": df.count()}
        elif strategy == SamplingStrategy.RANDOM:
            return self._random_sample(df)
        elif strategy == SamplingStrategy.STRATIFIED:
            return self._stratified_sample(df, label_col)
        elif strategy == SamplingStrategy.UNDERSAMPLE:
            return self._undersample(df, label_col)
        elif strategy == SamplingStrategy.OVERSAMPLE:
            return self._oversample(df, label_col)
        else:
            raise ValueError(
                f"Strategy {strategy} requires external scorer. "
                "Use sample_with_hard_negatives() instead."
            )
    
    def _random_sample(self, df) -> Tuple[Any, Dict[str, Any]]:
        """Simple random sampling."""
        n_original = df.count()
        sampled = df.sample(fraction=self.config.fraction, seed=self.config.seed)
        n_sampled = sampled.count()
        
        return sampled, {
            "strategy": "random",
            "fraction": self.config.fraction,
            "n_original": n_original,
            "n_sampled": n_sampled,
        }
    
    def _stratified_sample(self, df, label_col: str) -> Tuple[Any, Dict[str, Any]]:
        """Stratified sampling preserving class distribution."""
        from pyspark.sql import functions as F
        
        n_original = df.count()
        sampled = df.sampleBy(
            label_col,
            fractions={0: self.config.fraction, 1: self.config.fraction},
            seed=self.config.seed,
        )
        n_sampled = sampled.count()
        
        dist = sampled.groupBy(label_col).count().collect()
        class_counts = {row[label_col]: row["count"] for row in dist}
        
        return sampled, {
            "strategy": "stratified",
            "fraction": self.config.fraction,
            "n_original": n_original,
            "n_sampled": n_sampled,
            "class_counts": class_counts,
        }
    
    def _undersample(self, df, label_col: str) -> Tuple[Any, Dict[str, Any]]:
        """Undersample majority class."""
        from pyspark.sql import functions as F
        
        counts = df.groupBy(label_col).count().collect()
        class_counts = {row[label_col]: row["count"] for row in counts}
        
        n_pos = class_counts.get(1, 0)
        n_neg = class_counts.get(0, 0)
        
        if n_pos == 0 or n_neg == 0:
            return df, {"strategy": "undersample", "error": "Missing class"}
        
        if self.config.target_ratio:
            target_neg = int(n_pos / self.config.target_ratio)
        else:
            target_neg = n_pos
        
        target_neg = max(target_neg, self.config.min_samples)
        neg_fraction = min(target_neg / n_neg, 1.0)
        
        sampled = df.sampleBy(
            label_col,
            fractions={0: neg_fraction, 1: 1.0},
            seed=self.config.seed,
        )
        
        final_counts = sampled.groupBy(label_col).count().collect()
        final_class_counts = {row[label_col]: row["count"] for row in final_counts}
        
        return sampled, {
            "strategy": "undersample",
            "n_pos_original": n_pos,
            "n_neg_original": n_neg,
            "n_pos_sampled": final_class_counts.get(1, 0),
            "n_neg_sampled": final_class_counts.get(0, 0),
            "neg_fraction": neg_fraction,
        }
    
    def _oversample(self, df, label_col: str) -> Tuple[Any, Dict[str, Any]]:
        """Oversample minority class."""
        from pyspark.sql import functions as F
        
        counts = df.groupBy(label_col).count().collect()
        class_counts = {row[label_col]: row["count"] for row in counts}
        
        n_pos = class_counts.get(1, 0)
        n_neg = class_counts.get(0, 0)
        
        if n_pos == 0:
            return df, {"strategy": "oversample", "error": "No positives"}
        
        if self.config.target_ratio:
            target_pos = int(n_neg * self.config.target_ratio)
        else:
            target_pos = n_neg
        
        pos_fraction = target_pos / n_pos
        
        positives = df.filter(F.col(label_col) == 1)
        negatives = df.filter(F.col(label_col) == 0)
        
        if pos_fraction > 1:
            oversampled_pos = positives.sample(
                withReplacement=True,
                fraction=pos_fraction,
                seed=self.config.seed,
            )
        else:
            oversampled_pos = positives
        
        sampled = negatives.union(oversampled_pos)
        
        return sampled, {
            "strategy": "oversample",
            "n_pos_original": n_pos,
            "n_neg_original": n_neg,
            "pos_oversample_ratio": pos_fraction,
            "n_total_sampled": sampled.count(),
        }
    
    def sample_with_hard_negatives(
        self,
        df,
        label_col: str,
        scores_df,
        key_cols: List[str],
        score_col: str = "score",
    ) -> Tuple[Any, Dict[str, Any]]:
        """Sample using hard negative mining."""
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        positives = df.filter(F.col(label_col) == 1)
        n_pos = positives.count()
        
        negatives = df.filter(F.col(label_col) == 0)
        n_neg = negatives.count()
        
        negatives_scored = negatives.join(
            scores_df.select(*key_cols, F.col(score_col).alias("_neg_score")),
            on=key_cols,
            how="left",
        ).fillna({"_neg_score": 0.0})
        
        if self.config.target_ratio:
            target_neg = int(n_pos / self.config.target_ratio)
        else:
            target_neg = n_pos
        
        target_neg = max(target_neg, self.config.min_samples)
        
        n_hard = int(target_neg * self.config.hard_negative_fraction)
        n_random = target_neg - n_hard
        
        # Select hard negatives based on method
        if self.config.hard_negative_method == "top_prob":
            window = Window.orderBy(F.desc("_neg_score"))
            negatives_ranked = negatives_scored.withColumn("_rank", F.row_number().over(window))
            hard_negatives = negatives_ranked.filter(F.col("_rank") <= n_hard)
        elif self.config.hard_negative_method == "margin":
            negatives_scored = negatives_scored.withColumn(
                "_margin", F.abs(F.col("_neg_score") - 0.5)
            )
            window = Window.orderBy(F.asc("_margin"))
            negatives_ranked = negatives_scored.withColumn("_rank", F.row_number().over(window))
            hard_negatives = negatives_ranked.filter(F.col("_rank") <= n_hard)
        elif self.config.hard_negative_method == "entropy":
            negatives_scored = negatives_scored.withColumn(
                "_entropy",
                -F.col("_neg_score") * F.log(F.col("_neg_score") + 1e-10)
                - (1 - F.col("_neg_score")) * F.log(1 - F.col("_neg_score") + 1e-10)
            )
            window = Window.orderBy(F.desc("_entropy"))
            negatives_ranked = negatives_scored.withColumn("_rank", F.row_number().over(window))
            hard_negatives = negatives_ranked.filter(F.col("_rank") <= n_hard)
        else:
            raise ValueError(f"Unknown method: {self.config.hard_negative_method}")
        
        remaining_negatives = negatives_ranked.filter(F.col("_rank") > n_hard)
        random_frac = min(n_random / remaining_negatives.count(), 1.0) if remaining_negatives.count() > 0 else 0
        random_negatives = remaining_negatives.sample(fraction=random_frac, seed=self.config.seed)
        
        drop_cols = ["_neg_score", "_rank", "_margin", "_entropy"]
        hard_negatives = hard_negatives.drop(*[c for c in drop_cols if c in hard_negatives.columns])
        random_negatives = random_negatives.drop(*[c for c in drop_cols if c in random_negatives.columns])
        
        sampled = positives.union(hard_negatives).union(random_negatives)
        
        return sampled, {
            "strategy": "hard_negative",
            "method": self.config.hard_negative_method,
            "n_pos": n_pos,
            "n_neg_original": n_neg,
            "n_hard_negatives": n_hard,
            "n_random_negatives": n_random,
            "hard_negative_fraction": self.config.hard_negative_fraction,
            "n_total_sampled": sampled.count(),
        }
