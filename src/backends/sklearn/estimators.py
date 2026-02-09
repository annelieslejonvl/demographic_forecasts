"""
Sklearn/cuML backend implementation.
Automatically uses cuML (GPU) if available, otherwise sklearn (CPU).
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..base import (
    BackendFactory,
    BackendType,
    BaseDataLoader,
    BaseEstimator,
    BasePreprocessor,
    DeviceConfig,
    DeviceType,
    PredictResult,
    TrainResult,
)
from ...utils.device import get_device_manager

logger = logging.getLogger(__name__)


# Check cuML availability
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
    CUML_AVAILABLE = True
    logger.info("cuML (GPU) available - will use GPU for logistic regression")
except (ImportError, OSError) as e:
    # OSError catches "libcuml++.so not found" errors
    CUML_AVAILABLE = False
    logger.info(f"cuML not available - will use sklearn (CPU). Reason: {type(e).__name__}")


# =============================================================================
# Data Loading
# =============================================================================

class SklearnDataLoader(BaseDataLoader):
    """Data loader for Sklearn/cuML backend."""

    def from_spark(
        self,
        spark_df: Any,
        label_col: str,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Spark DataFrame to numpy arrays."""
        pdf = spark_df.select(feature_cols + [label_col]).toPandas()
        X = pdf[feature_cols].values.astype(np.float32)
        y = pdf[label_col].values.astype(np.float32)
        return X, y

    def to_spark(self, X: np.ndarray, y: np.ndarray, spark_session: Any) -> Any:
        """Convert numpy arrays to Spark DataFrame."""
        import pandas as pd
        feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df["label"] = y
        return spark_session.createDataFrame(df)

    def create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return data as-is (sklearn doesn't use dataloaders)."""
        return X, y


# =============================================================================
# Preprocessor (reuse PyTorch preprocessor)
# =============================================================================

from ..pytorch.estimators import PyTorchPreprocessor as SklearnPreprocessor


# =============================================================================
# Logistic Regression Classifier
# =============================================================================

class SklearnLogisticRegression(BaseEstimator):
    """
    Sklearn/cuML Logistic Regression.
    Automatically uses cuML (GPU) if available, otherwise sklearn (CPU).
    """

    DEFAULT_PARAMS = {
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 100,
        "tol": 1e-4,
        "solver": "qn",  # cuML default (quasi-newton)
        "random_state": 42,
    }

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ):
        super().__init__(model_config, device_config)

        params = model_config.get("params", {})
        self.params = {**self.DEFAULT_PARAMS, **params}

        self.model_ = None
        self.feature_names_: Optional[List[str]] = None
        self.use_gpu_ = False

    def _get_gpu_device(self) -> str:
        """Check if GPU is available for cuML."""
        if CUML_AVAILABLE:
            dm = get_device_manager()
            if dm.check_cuda_available():
                return "gpu"
        return "cpu"

    def _auto_detect_device(self) -> str:
        """Auto-detect GPU availability."""
        return self._get_gpu_device()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols=None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainResult:
        """Fit the Logistic Regression model."""
        device = self.get_device()
        self.use_gpu_ = (device == "gpu" and CUML_AVAILABLE)

        if self.use_gpu_:
            logger.info(f"Training cuML Logistic Regression on GPU")
            from cuml.linear_model import LogisticRegression

            # cuML parameters
            params = self.params.copy()
            # Map sklearn params to cuML
            if "n_jobs" in params:
                del params["n_jobs"]  # cuML doesn't have n_jobs

            self.model_ = LogisticRegression(**params)
        else:
            logger.info(f"Training sklearn Logistic Regression on CPU")
            from sklearn.linear_model import LogisticRegression

            params = self.params.copy()
            # Add sklearn-specific params
            if "solver" not in params or params["solver"] == "qn":
                params["solver"] = "lbfgs"  # sklearn equivalent
            params["n_jobs"] = -1

            self.model_ = LogisticRegression(**params)

        self.feature_names_ = feature_names

        # Fit model
        self.model_.fit(X, y, sample_weight=sample_weight)
        self._is_fitted = True

        # Compute metrics
        from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

        train_proba = self.model_.predict_proba(X)[:, 1]
        metrics = {
            "train_auc": float(roc_auc_score(y, train_proba)),
            "train_aucpr": float(average_precision_score(y, train_proba)),
            "train_logloss": float(log_loss(y, train_proba)),
        }

        # Eval metrics
        if eval_set:
            X_eval, y_eval = eval_set[0]
            eval_proba = self.model_.predict_proba(X_eval)[:, 1]
            metrics["eval_auc"] = float(roc_auc_score(y_eval, eval_proba))
            metrics["eval_aucpr"] = float(average_precision_score(y_eval, eval_proba))
            metrics["eval_logloss"] = float(log_loss(y_eval, eval_proba))

        return TrainResult(
            model=self.model_,
            metrics=metrics,
            metadata={
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "backend": "cuML (GPU)" if self.use_gpu_ else "sklearn (CPU)",
            },
        )

    def fit_incremental(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
        reset: bool = False,
    ) -> TrainResult:
        """
        Incremental training using partial_fit.
        Not all sklearn models support this, but SGDClassifier does.
        """
        if reset or self.model_ is None:
            # Use SGDClassifier for incremental learning
            if self.use_gpu_:
                # cuML doesn't have partial_fit, fall back to sklearn
                logger.warning("cuML doesn't support incremental learning, using sklearn")
                from sklearn.linear_model import SGDClassifier
                self.model_ = SGDClassifier(
                    loss="log_loss",
                    penalty=self.params.get("penalty", "l2"),
                    alpha=1.0 / self.params.get("C", 1.0),
                    max_iter=1,
                    random_state=self.params.get("random_state", 42),
                    n_jobs=-1,
                )
            else:
                from sklearn.linear_model import SGDClassifier
                self.model_ = SGDClassifier(
                    loss="log_loss",
                    penalty=self.params.get("penalty", "l2"),
                    alpha=1.0 / self.params.get("C", 1.0),
                    max_iter=1,
                    random_state=self.params.get("random_state", 42),
                    n_jobs=-1,
                )

            self.feature_names_ = feature_names

        # Partial fit on this batch
        classes = np.array([0, 1])
        self.model_.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        self._is_fitted = True

        # Compute metrics on this batch
        from sklearn.metrics import roc_auc_score, log_loss

        train_proba = self.model_.predict_proba(X)[:, 1]
        metrics = {
            "batch_auc": float(roc_auc_score(y, train_proba)),
            "batch_logloss": float(log_loss(y, train_proba)),
        }

        return TrainResult(
            model=self.model_,
            metrics=metrics,
            metadata={"batch_size": len(X)},
        )

    def predict(self, X: np.ndarray) -> PredictResult:
        """Generate class predictions using the current threshold."""
        proba = self.predict_proba(X)
        predictions = (proba.probabilities[:, 1] >= self._threshold).astype(int)
        return PredictResult(
            predictions=predictions,
            metadata={'threshold': self._threshold}
        )

    def predict_proba(self, X: np.ndarray) -> PredictResult:
        """Generate probability predictions."""
        probas = self.model_.predict_proba(X)
        return PredictResult(
            predictions=None,
            probabilities=probas,
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature coefficients (weights)."""
        if self.feature_names_ is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model_.coef_[0]))]
        else:
            feature_names = self.feature_names_

        # Return absolute values of coefficients
        return {
            name: float(abs(coef))
            for name, coef in zip(feature_names, self.model_.coef_[0])
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        joblib.dump({
            "model": self.model_,
            "model_config": self.model_config,
            "feature_names": self.feature_names_,
            "params": self.params,
            "use_gpu": self.use_gpu_,
            "threshold": self._threshold,
            "threshold_tuning_stats": self._threshold_tuning_stats,
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        device_config: Optional[DeviceConfig] = None,
    ) -> "SklearnLogisticRegression":
        """Load model from disk."""
        import joblib

        data = joblib.load(path)

        estimator = cls(data["model_config"], device_config)
        estimator.model_ = data["model"]
        estimator.feature_names_ = data["feature_names"]
        estimator.params = data["params"]
        estimator.use_gpu_ = data.get("use_gpu", False)
        estimator._threshold = data.get("threshold", 0.5)
        estimator._threshold_tuning_stats = data.get("threshold_tuning_stats", None)
        estimator._is_fitted = True

        return estimator


# =============================================================================
# Registration
# =============================================================================

def register_sklearn_backend():
    """Register Sklearn/cuML backend components with the factory."""
    BackendFactory.register_preprocessor(BackendType.SKLEARN, SklearnPreprocessor)
    BackendFactory.register_estimator(BackendType.SKLEARN, "logistic_regression", SklearnLogisticRegression)
    BackendFactory.register_data_loader(BackendType.SKLEARN, SklearnDataLoader)

    backend_info = "cuML (GPU)" if CUML_AVAILABLE else "sklearn (CPU)"
    logger.info(f"Sklearn backend registered ({backend_info})")
