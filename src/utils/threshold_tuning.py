"""
Threshold tuning utilities for binary classification.

This module provides utilities to optimize classification thresholds beyond the
default 0.5 value, which can significantly improve model performance for imbalanced
datasets or when specific precision/recall requirements exist.
"""
import numpy as np
from typing import Tuple, Optional, Dict, Callable
from sklearn.metrics import fbeta_score, precision_recall_curve, roc_curve
import logging

logger = logging.getLogger(__name__)


class ThresholdTuner:
    """
    Threshold optimization for binary classification.

    Supports multiple strategies:
    - 'f1': Maximize F-beta score (default beta=1.0)
    - 'youden': Maximize Youden's J statistic (sensitivity + specificity - 1)
    - 'precision_recall': Find threshold meeting precision/recall constraints
    - 'custom': Use custom metric function
    """

    SUPPORTED_STRATEGIES = ['f1', 'youden', 'precision_recall', 'custom']

    def __init__(
        self,
        strategy: str = 'f1',
        beta: float = 1.0,
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
        custom_metric_fn: Optional[Callable] = None
    ):
        """
        Initialize threshold tuner.

        Args:
            strategy: Optimization strategy ('f1', 'youden', 'precision_recall', 'custom')
            beta: Beta parameter for F-beta score (default 1.0 for F1)
            min_precision: Minimum precision constraint (for precision_recall strategy)
            min_recall: Minimum recall constraint (for precision_recall strategy)
            custom_metric_fn: Custom metric function(y_true, y_pred) -> float to maximize
        """
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Strategy '{strategy}' not supported. "
                f"Choose from: {self.SUPPORTED_STRATEGIES}"
            )

        self.strategy = strategy
        self.beta = beta
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.custom_metric_fn = custom_metric_fn

        if strategy == 'custom' and custom_metric_fn is None:
            raise ValueError("custom_metric_fn must be provided for 'custom' strategy")

        if strategy == 'precision_recall':
            if min_precision is None and min_recall is None:
                raise ValueError(
                    "At least one of min_precision or min_recall must be specified "
                    "for 'precision_recall' strategy"
                )

    def tune(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_thresholds: int = 100
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold.

        Args:
            y_true: True binary labels (0/1)
            y_proba: Predicted probabilities for positive class
            n_thresholds: Number of threshold candidates to evaluate

        Returns:
            optimal_threshold: Best threshold value
            stats: Dictionary with tuning statistics
        """
        # Validate inputs
        y_true = np.asarray(y_true).ravel()
        y_proba = np.asarray(y_proba).ravel()

        if len(y_true) != len(y_proba):
            raise ValueError("y_true and y_proba must have same length")

        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only binary values (0/1)")

        if np.any((y_proba < 0) | (y_proba > 1)):
            raise ValueError("y_proba must be in range [0, 1]")

        # Generate threshold candidates
        thresholds = np.linspace(0, 1, n_thresholds)

        # Evaluate each threshold
        if self.strategy == 'f1':
            optimal_threshold, stats = self._tune_f1(y_true, y_proba, thresholds)
        elif self.strategy == 'youden':
            optimal_threshold, stats = self._tune_youden(y_true, y_proba, thresholds)
        elif self.strategy == 'precision_recall':
            optimal_threshold, stats = self._tune_precision_recall(y_true, y_proba, thresholds)
        else:  # custom
            optimal_threshold, stats = self._tune_custom(y_true, y_proba, thresholds)

        return optimal_threshold, stats

    def _tune_f1(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[float, Dict]:
        """Tune threshold to maximize F-beta score."""
        best_score = -1
        best_threshold = 0.5

        scores = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = fbeta_score(y_true, y_pred, beta=self.beta, zero_division=0)
            scores.append(score)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        stats = {
            'strategy': 'f1',
            'beta': self.beta,
            'best_score': best_score,
            'all_scores': scores,
            'all_thresholds': thresholds.tolist()
        }

        return best_threshold, stats

    def _tune_youden(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[float, Dict]:
        """Tune threshold to maximize Youden's J statistic."""
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

        # Youden's J = sensitivity + specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = roc_thresholds[best_idx]
        best_score = j_scores[best_idx]

        stats = {
            'strategy': 'youden',
            'best_score': float(best_score),
            'best_tpr': float(tpr[best_idx]),
            'best_fpr': float(fpr[best_idx]),
            'all_thresholds': roc_thresholds.tolist(),
            'all_j_scores': j_scores.tolist()
        }

        return float(best_threshold), stats

    def _tune_precision_recall(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[float, Dict]:
        """Tune threshold to meet precision/recall constraints."""
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

        # precision_recall_curve returns n_thresholds + 1 values
        # Last precision/recall are for threshold=1.0
        # We need to align arrays

        valid_indices = []
        for i in range(len(pr_thresholds)):
            if self.min_precision is not None and precision[i] < self.min_precision:
                continue
            if self.min_recall is not None and recall[i] < self.min_recall:
                continue
            valid_indices.append(i)

        if not valid_indices:
            logger.warning(
                "No threshold satisfies constraints. Using default 0.5"
            )
            best_threshold = 0.5
            best_idx = 0
        else:
            # Among valid thresholds, maximize F1 score
            best_f1 = -1
            best_idx = valid_indices[0]

            for idx in valid_indices:
                p, r = precision[idx], recall[idx]
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_idx = idx

            best_threshold = pr_thresholds[best_idx]

        stats = {
            'strategy': 'precision_recall',
            'min_precision': self.min_precision,
            'min_recall': self.min_recall,
            'best_precision': float(precision[best_idx]),
            'best_recall': float(recall[best_idx]),
            'all_thresholds': pr_thresholds.tolist(),
            'all_precision': precision.tolist(),
            'all_recall': recall.tolist()
        }

        return float(best_threshold), stats

    def _tune_custom(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[float, Dict]:
        """Tune threshold using custom metric function."""
        best_score = -np.inf
        best_threshold = 0.5

        scores = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = self.custom_metric_fn(y_true, y_pred)
            scores.append(score)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        stats = {
            'strategy': 'custom',
            'best_score': float(best_score),
            'all_scores': scores,
            'all_thresholds': thresholds.tolist()
        }

        return best_threshold, stats


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = 'f1',
    **kwargs
) -> Tuple[float, Dict]:
    """
    Convenience function for threshold tuning.

    Args:
        y_true: True binary labels (0/1)
        y_proba: Predicted probabilities for positive class
        strategy: Optimization strategy ('f1', 'youden', 'precision_recall', 'custom')
        **kwargs: Additional arguments passed to ThresholdTuner

    Returns:
        optimal_threshold: Best threshold value
        stats: Dictionary with tuning statistics

    Examples:
        >>> # Maximize F1 score
        >>> threshold, stats = tune_threshold(y_true, y_proba, strategy='f1')

        >>> # Maximize F2 score (emphasize recall)
        >>> threshold, stats = tune_threshold(y_true, y_proba, strategy='f1', beta=2.0)

        >>> # Maximize Youden's J statistic
        >>> threshold, stats = tune_threshold(y_true, y_proba, strategy='youden')

        >>> # Find threshold with min precision=0.8
        >>> threshold, stats = tune_threshold(
        ...     y_true, y_proba,
        ...     strategy='precision_recall',
        ...     min_precision=0.8
        ... )

        >>> # Use custom metric
        >>> def custom_metric(y_true, y_pred):
        ...     return balanced_accuracy_score(y_true, y_pred)
        >>> threshold, stats = tune_threshold(
        ...     y_true, y_proba,
        ...     strategy='custom',
        ...     custom_metric_fn=custom_metric
        ... )
    """
    tuner = ThresholdTuner(strategy=strategy, **kwargs)
    return tuner.tune(y_true, y_proba)
