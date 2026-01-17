"""
Multi-backend ML framework.
Supports PyTorch, XGBoost, and Spark with CPU/GPU acceleration.
"""
from .base import (
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
from .pipeline import PipelineConfig, UnifiedPipeline, create_pipeline


def initialize_backends():
    """Initialize and register all available backends."""
    # PyTorch
    try:
        from .pytorch.estimators import register_pytorch_backend
        register_pytorch_backend()
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"PyTorch backend not available: {e}")
    
    # XGBoost
    try:
        from .xgboost.estimators import register_xgboost_backend
        register_xgboost_backend()
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"XGBoost backend not available: {e}")
    
    # Spark
    try:
        from .spark.estimators import register_spark_backend
        register_spark_backend()
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"Spark backend not available: {e}")


# Auto-initialize on import
initialize_backends()


__all__ = [
    # Base classes
    "BackendFactory",
    "BackendType",
    "BaseDataLoader",
    "BaseEstimator",
    "BasePreprocessor",
    "DeviceConfig",
    "DeviceType",
    "PredictResult",
    "TrainResult",
    # Pipeline
    "PipelineConfig",
    "UnifiedPipeline",
    "create_pipeline",
    # Initialization
    "initialize_backends",
]
