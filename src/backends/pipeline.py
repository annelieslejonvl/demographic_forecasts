"""
Unified Pipeline Builder for multi-backend ML workflows.
Orchestrates PyTorch, XGBoost, and Spark backends with consistent API.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import (
    BackendFactory,
    BackendType,
    BaseEstimator,
    BasePreprocessor,
    DeviceConfig,
    DeviceType,
    PredictResult,
    TrainResult,
)
from ..utils.device import get_device_manager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""
    
    # Backend selection
    backend: BackendType = BackendType.SPARK
    model_type: str = "logistic_regression"
    
    # Device configuration
    device: DeviceType = DeviceType.AUTO
    gpu_id: Optional[int] = None
    
    # Preprocessing
    scaling: bool = False
    scaling_type: str = "standard"
    dim_reduction: Optional[Dict[str, Any]] = None
    
    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Column names
    label_col: str = "label"
    features_col: str = "features"
    weight_col: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary (YAML-compatible)."""
        return cls(
            backend=BackendType(d.get("backend", "spark")),
            model_type=d.get("model", {}).get("type", "logistic_regression"),
            device=DeviceType(d.get("device", {}).get("type", "auto")),
            gpu_id=d.get("device", {}).get("gpu_id"),
            scaling=d.get("scaling", {}).get("enabled", False),
            scaling_type=d.get("scaling", {}).get("type", "standard"),
            dim_reduction=d.get("dim_reduction"),
            model_params=d.get("model", {}).get("params", {}),
            label_col=d.get("label_col", "label"),
            features_col=d.get("features_col", "features"),
            weight_col=d.get("weight_col"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backend": self.backend.value,
            "model": {
                "type": self.model_type,
                "params": self.model_params,
            },
            "device": {
                "type": self.device.value,
                "gpu_id": self.gpu_id,
            },
            "scaling": {
                "enabled": self.scaling,
                "type": self.scaling_type,
            },
            "dim_reduction": self.dim_reduction,
            "label_col": self.label_col,
            "features_col": self.features_col,
            "weight_col": self.weight_col,
        }


class UnifiedPipeline:
    """
    Unified ML pipeline that works across backends.
    
    Example usage:
        # From config
        config = PipelineConfig.from_dict(yaml_config)
        pipeline = UnifiedPipeline(config)
        
        # Fit and predict
        result = pipeline.fit(train_df, feature_cols)
        predictions = pipeline.predict(test_df)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device_config = DeviceConfig(
            device_type=config.device,
            gpu_id=config.gpu_id,
        )
        
        self.preprocessor_: Optional[BasePreprocessor] = None
        self.estimator_: Optional[BaseEstimator] = None
        self.feature_cols_: Optional[List[str]] = None
        self._is_fitted = False
    
    @property
    def backend(self) -> BackendType:
        return self.config.backend
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def _create_preprocessor(self) -> BasePreprocessor:
        """Create backend-specific preprocessor."""
        if self.config.backend == BackendType.SPARK:
            return BackendFactory.get_preprocessor(
                BackendType.SPARK,
                scaling=self.config.scaling,
                scaling_type=self.config.scaling_type,
                dim_reduction=self.config.dim_reduction,
                features_col=self.config.features_col,
                label_col=self.config.label_col,
            )
        elif self.config.backend == BackendType.PYTORCH:
            return BackendFactory.get_preprocessor(
                BackendType.PYTORCH,
                scaling=self.config.scaling_type if self.config.scaling else "none",
                dim_reduction=self.config.dim_reduction,
            )
        else:  # XGBoost
            return BackendFactory.get_preprocessor(
                BackendType.XGBOOST,
                scaling="none",  # XGBoost typically doesn't need scaling
                handle_missing="keep",
            )
    
    def _create_estimator(self) -> BaseEstimator:
        """Create backend-specific estimator."""
        model_config = {
            "type": self.config.model_type,
            "params": self.config.model_params,
        }
        return BackendFactory.get_estimator(
            self.config.backend,
            self.config.model_type,
            model_config,
            self.device_config,
        )
    
    def _prepare_data(
        self,
        data: Any,
        feature_cols: List[str],
        fit: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Prepare data for the specific backend.
        
        Returns:
            (X, y) tuple appropriate for the backend
        """
        label_col = self.config.label_col
        
        if self.config.backend == BackendType.SPARK:
            # Spark keeps data as DataFrame
            if fit:
                self.preprocessor_ = self._create_preprocessor()
                transformed = self.preprocessor_.fit_transform(
                    data, label_col, feature_cols
                )
            else:
                transformed = self.preprocessor_.transform(data)
            
            features_col = self.preprocessor_.get_output_col()
            return transformed, None
        
        else:
            # PyTorch/XGBoost need numpy arrays
            loader = BackendFactory.get_data_loader(self.config.backend)
            
            if hasattr(data, "toPandas"):
                # Convert Spark DF to pandas
                pdf = data.select(feature_cols + [label_col]).toPandas()
                X = pdf[feature_cols].values
                y = pdf[label_col].values
            else:
                X, y = data, None
            
            if fit:
                self.preprocessor_ = self._create_preprocessor()
                X = self.preprocessor_.fit_transform(X, label_col, feature_cols)
            else:
                X = self.preprocessor_.transform(X)
            
            return X, y
    
    def fit(
        self,
        train_data: Any,
        feature_cols: List[str],
        eval_data: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> TrainResult:
        """
        Fit the pipeline on training data.
        
        Args:
            train_data: Training data (Spark DF, pandas DF, or numpy array)
            feature_cols: List of feature column names
            eval_data: Optional validation data
            sample_weight: Sample weights (column name for Spark, array otherwise)
        
        Returns:
            TrainResult with fitted model and metrics
        """
        logger.info(f"Fitting pipeline with backend={self.config.backend.value}, "
                   f"model={self.config.model_type}")
        
        self.feature_cols_ = feature_cols
        
        # Prepare training data
        X_train, y_train = self._prepare_data(train_data, feature_cols, fit=True)
        
        # Prepare eval data if provided
        eval_set = None
        if eval_data is not None:
            X_eval, y_eval = self._prepare_data(eval_data, feature_cols, fit=False)
            if self.config.backend != BackendType.SPARK:
                eval_set = [(X_eval, y_eval)]
        
        # Create and fit estimator
        self.estimator_ = self._create_estimator()
        
        if self.config.backend == BackendType.SPARK:
            features_col = self.preprocessor_.get_output_col()
            result = self.estimator_.fit(
                X_train,
                features_col=features_col,
                label_col=self.config.label_col,
                sample_weight=self.config.weight_col,
                eval_set=eval_set,
            )
        else:
            result = self.estimator_.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                feature_names=feature_cols,
            )
        
        self._is_fitted = True
        return result
    
    def predict(self, data: Any) -> PredictResult:
        """Generate predictions on new data."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before predict")
        
        X, _ = self._prepare_data(data, self.feature_cols_, fit=False)
        
        if self.config.backend == BackendType.SPARK:
            features_col = self.preprocessor_.get_output_col()
            # Select only needed columns for prediction
            X = X.select(features_col, self.config.label_col)
        
        return self.estimator_.predict(X)
    
    def predict_proba(self, data: Any) -> PredictResult:
        """Generate probability predictions."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before predict_proba")
        
        X, _ = self._prepare_data(data, self.feature_cols_, fit=False)
        
        if self.config.backend == BackendType.SPARK:
            features_col = self.preprocessor_.get_output_col()
            X = X.select(features_col, self.config.label_col)
        
        return self.estimator_.predict_proba(X)
    
    def save(self, path: str) -> None:
        """Save the entire pipeline (preprocessor + model)."""
        import os
        import json
        
        os.makedirs(path, exist_ok=True)
        
        # Save config
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            config_dict = self.config.to_dict()
            config_dict["feature_cols"] = self.feature_cols_
            json.dump(config_dict, f, indent=2)
        
        # Save preprocessor
        preproc_path = os.path.join(path, "preprocessor")
        self.preprocessor_.save(preproc_path)
        
        # Save estimator
        model_path = os.path.join(path, "model")
        self.estimator_.save(model_path)
        
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "UnifiedPipeline":
        """Load a saved pipeline."""
        import json
        
        # Load config
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        feature_cols = config_dict.pop("feature_cols", [])
        config = PipelineConfig.from_dict(config_dict)
        
        pipeline = cls(config)
        pipeline.feature_cols_ = feature_cols
        
        # Load preprocessor
        preproc_path = os.path.join(path, "preprocessor")
        preprocessor_cls = BackendFactory._preprocessors[config.backend]
        pipeline.preprocessor_ = preprocessor_cls.load(preproc_path)
        
        # Load estimator
        model_path = os.path.join(path, "model")
        pipeline.estimator_ = BackendFactory.get_estimator(
            config.backend,
            config.model_type,
            {"type": config.model_type, "params": config.model_params},
            pipeline.device_config,
        ).load(model_path, pipeline.device_config)
        
        pipeline._is_fitted = True
        logger.info(f"Pipeline loaded from {path}")
        
        return pipeline


def create_pipeline(
    backend: str = "spark",
    model_type: str = "logistic_regression",
    device: str = "auto",
    **kwargs,
) -> UnifiedPipeline:
    """
    Convenience function to create a pipeline.
    
    Args:
        backend: "spark", "pytorch", or "xgboost"
        model_type: Model type specific to backend
        device: "cpu", "gpu", or "auto"
        **kwargs: Additional config options
    
    Returns:
        Configured UnifiedPipeline
    """
    config = PipelineConfig(
        backend=BackendType(backend),
        model_type=model_type,
        device=DeviceType(device),
        **kwargs,
    )
    return UnifiedPipeline(config)
