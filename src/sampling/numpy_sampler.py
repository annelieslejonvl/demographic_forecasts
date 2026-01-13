"""Numpy array sampling strategies."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import SamplingConfig, SamplingStrategy

logger = logging.getLogger(__name__)


class NumpySampler:
    """Sampling strategies for numpy arrays."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply sampling strategy."""
        strategy = self.config.strategy
        
        if strategy == SamplingStrategy.NONE:
            return X, y, sample_weight, {"strategy": "none"}
        elif strategy == SamplingStrategy.RANDOM:
            return self._random_sample(X, y, sample_weight)
        elif strategy == SamplingStrategy.STRATIFIED:
            return self._stratified_sample(X, y, sample_weight)
        elif strategy == SamplingStrategy.UNDERSAMPLE:
            return self._undersample(X, y, sample_weight)
        elif strategy == SamplingStrategy.OVERSAMPLE:
            return self._oversample(X, y, sample_weight)
        else:
            raise ValueError(f"Strategy {strategy} requires external scorer")
    
    def _random_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Random sampling."""
        n = len(y)
        n_sample = int(n * self.config.fraction)
        
        indices = self.rng.choice(n, size=n_sample, replace=False)
        
        X_s = X[indices]
        y_s = y[indices]
        w_s = sample_weight[indices] if sample_weight is not None else None
        
        return X_s, y_s, w_s, {
            "strategy": "random",
            "n_original": n,
            "n_sampled": n_sample,
        }
    
    def _stratified_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Stratified sampling."""
        from sklearn.model_selection import train_test_split
        
        test_size = 1 - self.config.fraction
        
        if test_size > 0:
            X_s, _, y_s, _, idx_s, _ = train_test_split(
                X, y, np.arange(len(y)),
                test_size=test_size,
                stratify=y,
                random_state=self.config.seed,
            )
            w_s = sample_weight[idx_s] if sample_weight is not None else None
        else:
            X_s, y_s, w_s = X, y, sample_weight
        
        return X_s, y_s, w_s, {
            "strategy": "stratified",
            "n_original": len(y),
            "n_sampled": len(y_s),
        }
    
    def _undersample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Undersample majority class."""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        if self.config.target_ratio:
            target_neg = int(n_pos / self.config.target_ratio)
        else:
            target_neg = n_pos
        
        target_neg = max(target_neg, self.config.min_samples)
        target_neg = min(target_neg, n_neg)
        
        neg_sample_idx = self.rng.choice(neg_idx, size=target_neg, replace=False)
        
        all_idx = np.concatenate([pos_idx, neg_sample_idx])
        self.rng.shuffle(all_idx)
        
        X_s = X[all_idx]
        y_s = y[all_idx]
        w_s = sample_weight[all_idx] if sample_weight is not None else None
        
        return X_s, y_s, w_s, {
            "strategy": "undersample",
            "n_pos": n_pos,
            "n_neg_original": n_neg,
            "n_neg_sampled": target_neg,
        }
    
    def _oversample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Oversample minority class."""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        if self.config.target_ratio:
            target_pos = int(n_neg * self.config.target_ratio)
        else:
            target_pos = n_neg
        
        if target_pos > n_pos:
            pos_sample_idx = self.rng.choice(pos_idx, size=target_pos, replace=True)
        else:
            pos_sample_idx = pos_idx
        
        all_idx = np.concatenate([neg_idx, pos_sample_idx])
        self.rng.shuffle(all_idx)
        
        X_s = X[all_idx]
        y_s = y[all_idx]
        w_s = sample_weight[all_idx] if sample_weight is not None else None
        
        return X_s, y_s, w_s, {
            "strategy": "oversample",
            "n_pos_original": n_pos,
            "n_pos_sampled": len(pos_sample_idx),
            "n_neg": n_neg,
        }
    
    def sample_with_hard_negatives(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scores: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Sample with hard negative mining."""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        if self.config.target_ratio:
            target_neg = int(n_pos / self.config.target_ratio)
        else:
            target_neg = n_pos
        
        target_neg = max(target_neg, self.config.min_samples)
        target_neg = min(target_neg, n_neg)
        
        n_hard = int(target_neg * self.config.hard_negative_fraction)
        n_random = target_neg - n_hard
        
        neg_scores = scores[neg_idx]
        
        if self.config.hard_negative_method == "top_prob":
            hard_order = np.argsort(neg_scores)[::-1]
        elif self.config.hard_negative_method == "margin":
            margins = np.abs(neg_scores - 0.5)
            hard_order = np.argsort(margins)
        elif self.config.hard_negative_method == "entropy":
            eps = 1e-10
            entropy = -neg_scores * np.log(neg_scores + eps) - (1 - neg_scores) * np.log(1 - neg_scores + eps)
            hard_order = np.argsort(entropy)[::-1]
        else:
            raise ValueError(f"Unknown method: {self.config.hard_negative_method}")
        
        hard_neg_local_idx = hard_order[:n_hard]
        hard_neg_idx = neg_idx[hard_neg_local_idx]
        
        remaining_local_idx = hard_order[n_hard:]
        if len(remaining_local_idx) > 0 and n_random > 0:
            random_local_idx = self.rng.choice(
                remaining_local_idx,
                size=min(n_random, len(remaining_local_idx)),
                replace=False,
            )
            random_neg_idx = neg_idx[random_local_idx]
        else:
            random_neg_idx = np.array([], dtype=int)
        
        all_idx = np.concatenate([pos_idx, hard_neg_idx, random_neg_idx])
        self.rng.shuffle(all_idx)
        
        X_s = X[all_idx]
        y_s = y[all_idx]
        w_s = sample_weight[all_idx] if sample_weight is not None else None
        
        hard_neg_scores = neg_scores[hard_neg_local_idx]
        
        return X_s, y_s, w_s, {
            "strategy": "hard_negative",
            "method": self.config.hard_negative_method,
            "n_pos": n_pos,
            "n_neg_original": n_neg,
            "n_hard_negatives": len(hard_neg_idx),
            "n_random_negatives": len(random_neg_idx),
            "hard_neg_mean_score": float(np.mean(hard_neg_scores)) if len(hard_neg_scores) > 0 else 0,
            "hard_neg_min_score": float(np.min(hard_neg_scores)) if len(hard_neg_scores) > 0 else 0,
            "hard_neg_max_score": float(np.max(hard_neg_scores)) if len(hard_neg_scores) > 0 else 0,
        }


class HardNegativeMiner:
    """Utility class for hard negative mining workflow."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.baseline_model_ = None
        self.negative_scores_: Optional[np.ndarray] = None
    
    def fit_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        sample_fraction: float = 0.1,
    ) -> Any:
        """Fit baseline model on sampled data."""
        baseline_config = SamplingConfig(
            strategy=SamplingStrategy.UNDERSAMPLE,
            target_ratio=0.5,
            seed=self.config.seed,
        )
        
        sampler = NumpySampler(baseline_config)
        X_base, y_base, _, _ = sampler._undersample(X, y, None)
        
        if sample_fraction < 1.0:
            n_sample = int(len(y_base) * sample_fraction)
            idx = np.random.RandomState(self.config.seed).choice(
                len(y_base), size=n_sample, replace=False
            )
            X_base = X_base[idx]
            y_base = y_base[idx]
        
        self.baseline_model_ = model_factory()
        self.baseline_model_.fit(X_base, y_base)
        
        logger.info(f"Baseline model trained on {len(y_base)} samples")
        
        return self.baseline_model_
    
    def score_negatives(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Score all samples with baseline model."""
        if self.baseline_model_ is None:
            raise RuntimeError("Must fit baseline model first")
        
        if hasattr(self.baseline_model_, "predict_proba"):
            scores = self.baseline_model_.predict_proba(X)[:, 1]
        else:
            scores = self.baseline_model_.decision_function(X)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        self.negative_scores_ = scores
        return scores
    
    def mine_hard_negatives(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Full hard negative mining pipeline."""
        if self.negative_scores_ is None:
            raise RuntimeError("Must score negatives first")
        
        sampler = NumpySampler(self.config)
        return sampler.sample_with_hard_negatives(
            X, y, self.negative_scores_, sample_weight
        )
