"""
Base classes for ML backends.
All backends (Spark, PyTorch, XGBoost) implement these interfaces.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"  # Auto-detect best available


class BackendType(Enum):
    SPARK = "spark"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"


@dataclass
class DeviceConfig:
    """Configuration for compute device selection."""
    device_type: DeviceType = DeviceType.AUTO
    gpu_id: Optional[int] = None  # Specific GPU index, None = auto
    num_workers: int = 4  # For data loading
    memory_fraction: float = 0.9  # GPU memory fraction to use
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeviceConfig":
        return cls(
            device_type=DeviceType(d.get("device", "auto")),
            gpu_id=d.get("gpu_id"),
            num_workers=d.get("num_workers", 4),
            memory_fraction=d.get("memory_fraction", 0.9),
        )


@dataclass
class TrainResult:
    """Standardized training result across backends."""
    model: Any
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass  
class PredictResult:
    """Standardized prediction result across backends."""
    predictions: Any  # Backend-specific: Spark DF, numpy array, torch tensor
    probabilities: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePreprocessor(ABC):
    """Abstract preprocessor interface."""
    
    @abstractmethod
    def fit(self, data: Any, label_col: str, feature_cols: List[str]) -> "BasePreprocessor":
        """Fit preprocessing pipeline on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform data using fitted preprocessor."""
        pass
    
    @abstractmethod
    def fit_transform(self, data: Any, label_col: str, feature_cols: List[str]) -> Any:
        """Fit and transform in one step."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save fitted preprocessor to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BasePreprocessor":
        """Load preprocessor from disk."""
        pass


class BaseEstimator(ABC):
    """Abstract estimator interface for all backends."""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ):
        self.model_config = model_config
        self.device_config = device_config or DeviceConfig()
        self.model_ = None
        self._is_fitted = False
        self._threshold = 0.5  # Default threshold for binary classification
        self._threshold_tuning_stats = None
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @abstractmethod
    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eval_set: Optional[List[Tuple[Any, Any]]] = None,
    ) -> TrainResult:
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> PredictResult:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> PredictResult:
        """Generate probability predictions."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str, device_config: Optional[DeviceConfig] = None) -> "BaseEstimator":
        """Load model from disk."""
        pass
    
    def get_device(self) -> str:
        """Resolve actual device to use based on config and availability."""
        if self.device_config.device_type == DeviceType.CPU:
            return "cpu"
        elif self.device_config.device_type == DeviceType.GPU:
            return self._get_gpu_device()
        else:  # AUTO
            return self._auto_detect_device()
    
    def _get_gpu_device(self) -> str:
        """Get GPU device string. Override per backend."""
        raise NotImplementedError
    
    def _auto_detect_device(self) -> str:
        """Auto-detect best available device. Override per backend."""
        raise NotImplementedError

    def tune_threshold(
        self,
        X_val: Any,
        y_val: Any,
        strategy: str = 'f1',
        **kwargs
    ) -> float:
        """
        Tune classification threshold using validation data.

        Args:
            X_val: Validation features
            y_val: Validation labels
            strategy: Tuning strategy ('f1', 'youden', 'precision_recall', 'custom')
            **kwargs: Additional arguments for threshold tuning

        Returns:
            optimal_threshold: The tuned threshold value
        """
        from ..utils.threshold_tuning import tune_threshold

        if not self.is_fitted:
            raise ValueError("Model must be fitted before tuning threshold")

        proba_result = self.predict_proba(X_val)
        y_proba = self._extract_probabilities(proba_result)
        y_val_array = self._extract_labels(y_val)

        optimal_threshold, stats = tune_threshold(
            y_val_array,
            y_proba,
            strategy=strategy,
            **kwargs
        )

        self._threshold = optimal_threshold
        self._threshold_tuning_stats = stats

        logger.info(
            f"Threshold tuned using {strategy} strategy: "
            f"{optimal_threshold:.4f} (was 0.5)"
        )

        return optimal_threshold

    def _extract_probabilities(self, proba_result: PredictResult) -> np.ndarray:
        """
        Extract probability array from PredictResult.
        Override in subclasses if needed for backend-specific handling.
        """
        probs = proba_result.probabilities
        if probs is None:
            probs = proba_result.predictions

        if hasattr(probs, 'numpy'):
            probs = probs.numpy()

        probs = np.asarray(probs)
        if len(probs.shape) > 1:
            probs = probs[:, 1] if probs.shape[1] == 2 else probs.ravel()

        return probs

    def _extract_labels(self, y: Any) -> np.ndarray:
        """
        Extract label array from various formats.
        Override in subclasses if needed for backend-specific handling.
        """
        if hasattr(y, 'numpy'):
            y = y.numpy()

        y = np.asarray(y)
        if len(y.shape) > 1:
            y = y.ravel()

        return y

    def get_threshold(self) -> float:
        """Get the current classification threshold."""
        return self._threshold

    def set_threshold(self, threshold: float) -> None:
        """
        Manually set the classification threshold.

        Args:
            threshold: Threshold value between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self._threshold = threshold
        logger.info(f"Threshold manually set to: {threshold:.4f}")

    def get_threshold_tuning_stats(self) -> Optional[Dict]:
        """Get statistics from threshold tuning process."""
        return self._threshold_tuning_stats


class BaseDataLoader(ABC):
    """Abstract data loader for converting between formats."""
    
    @abstractmethod
    def from_spark(self, spark_df: Any, label_col: str, feature_cols: List[str]) -> Tuple[Any, Any]:
        """Convert Spark DataFrame to backend-native format."""
        pass
    
    @abstractmethod
    def to_spark(self, X: Any, y: Any, spark_session: Any) -> Any:
        """Convert backend-native format to Spark DataFrame."""
        pass
    
    @abstractmethod
    def create_dataloader(
        self,
        X: Any,
        y: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> Any:
        """Create a batched data loader (for PyTorch)."""
        pass


class BackendFactory:
    """Factory for creating backend-specific components."""
    
    _preprocessors: Dict[BackendType, type] = {}
    _estimators: Dict[BackendType, Dict[str, type]] = {}
    _data_loaders: Dict[BackendType, type] = {}
    
    @classmethod
    def register_preprocessor(cls, backend: BackendType, preprocessor_cls: type):
        cls._preprocessors[backend] = preprocessor_cls
    
    @classmethod
    def register_estimator(cls, backend: BackendType, model_type: str, estimator_cls: type):
        if backend not in cls._estimators:
            cls._estimators[backend] = {}
        cls._estimators[backend][model_type] = estimator_cls
    
    @classmethod
    def register_data_loader(cls, backend: BackendType, loader_cls: type):
        cls._data_loaders[backend] = loader_cls
    
    @classmethod
    def get_preprocessor(cls, backend: BackendType, **kwargs) -> BasePreprocessor:
        if backend not in cls._preprocessors:
            raise ValueError(f"No preprocessor registered for backend: {backend}")
        return cls._preprocessors[backend](**kwargs)
    
    @classmethod
    def get_estimator(
        cls,
        backend: BackendType,
        model_type: str,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ) -> BaseEstimator:
        if backend not in cls._estimators:
            raise ValueError(f"No estimators registered for backend: {backend}")
        if model_type not in cls._estimators[backend]:
            raise ValueError(f"Model type '{model_type}' not registered for backend: {backend}")
        return cls._estimators[backend][model_type](model_config, device_config)
    
    @classmethod
    def get_data_loader(cls, backend: BackendType, **kwargs) -> BaseDataLoader:
        if backend not in cls._data_loaders:
            raise ValueError(f"No data loader registered for backend: {backend}")
        return cls._data_loaders[backend](**kwargs)
    
    @classmethod
    def list_backends(cls) -> List[BackendType]:
        return list(cls._estimators.keys())
    
    @classmethod
    def list_models(cls, backend: BackendType) -> List[str]:
        return list(cls._estimators.get(backend, {}).keys())
