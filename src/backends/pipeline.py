"""
Unified Pipeline Builder for multi-backend ML workflows.
Orchestrates PyTorch, XGBoost, and Spark backends with consistent API.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
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
    categorical_encoding: str = "onehot"  # onehot, label, ordinal, native
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
            categorical_encoding=d.get("categorical", {}).get("encoding", "onehot"),
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
            "categorical": {
                "encoding": self.categorical_encoding,
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
                categorical_encoding=self.config.categorical_encoding,
                dim_reduction=self.config.dim_reduction,
            )
        else:  # XGBoost
            return BackendFactory.get_preprocessor(
                BackendType.XGBOOST,
                scaling="none",  # XGBoost typically doesn't need scaling
                handle_missing="keep",
                categorical_encoding=self.config.categorical_encoding,
            )
    
    def _create_estimator(self) -> BaseEstimator:
        """Create backend-specific estimator."""
        model_config = {
            "type": self.config.model_type,
            "params": self.config.model_params,
        }
        if self.config.backend == BackendType.XGBOOST and self.config.categorical_encoding == "native":
            model_config["params"] = {
                **model_config["params"],
                "enable_categorical": model_config["params"].get("enable_categorical", True),
            }
        return BackendFactory.get_estimator(
            self.config.backend,
            self.config.model_type,
            model_config,
            self.device_config,
        )
    
    def _prepare_data(
        self,
        data: Tuple[Any, Any],
        feature_cols: List[str],
        fit: bool = False,
        max_samples: Optional[int] = None,  # 👈 Nieuw parameter
    ) -> Tuple[Any, Any]:
        """
        Prepare data for the specific backend.
        
        Args:
            max_samples: For fitting, limit rows to avoid memory issues
            
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
                # CRITICAL: Sample in Spark BEFORE converting to pandas
                if fit and max_samples is not None:
                    n_rows = data.count()
                    if n_rows > max_samples:
                        sample_fraction = max_samples / n_rows
                        print(f"⚠️  Sampling {sample_fraction:.2%} ({max_samples:,}/{n_rows:,} rows) for fitting")
                        data = data.sample(fraction=sample_fraction, seed=42)
                
                # Now safe to convert
                cols_to_select = feature_cols + [label_col]
                pdf = data.select(cols_to_select).toPandas()
                
                print(f"Loaded {len(pdf):,} rows, {len(cols_to_select)} cols, "
                    f"~{pdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")
                
                if self.config.backend == BackendType.XGBOOST and self.config.categorical_encoding == "native":
                    X = pdf[feature_cols]
                else:
                    X = pdf[feature_cols].values
                y = pdf[label_col].values
            else:
                X, y = data
            
            if fit:
                print('Creating preprocessor')
                self.preprocessor_ = self._create_preprocessor()
                print('Fitting and transforming data')
                X = self.preprocessor_.fit_transform(X, label_col, feature_cols)
            else:
                X = self.preprocessor_.transform(X)
            
            return X, y
        
    def fit(
        self,
        train_data,
        eval_data=None,
        feature_cols: List[str] = None,
        sample_weight=None,
        use_batches: bool = None,  # Auto-detect
        batch_size: int = 100_000,
    ):
        """Fit the model."""
        
        # Auto-detect of we batches moeten gebruiken
        if use_batches is None and hasattr(train_data, "count"):
            n_rows = train_data.count()
            use_batches = n_rows > 1_000_000  # Gebruik batches voor >1M rijen
            print(f"Dataset: {n_rows:,} rows → using {'batches' if use_batches else 'full load'}")
        
        if use_batches and self.config.backend != BackendType.SPARK:
            return self._fit_batched(train_data, eval_data, feature_cols, batch_size)
        else:
            return self._fit_full(train_data, eval_data, feature_cols, sample_weight)

    def _fit_full(
        self,
        train_data,
        eval_data,
        feature_cols: List[str],
        sample_weight=None,
    ):
        """Fit model using full dataset (no batching)."""
        self.feature_cols_ = feature_cols

        # Prepare training data
        X_train, y_train = self._prepare_data(
            train_data, feature_cols, fit=True
        )

        # Prepare eval set if provided
        eval_set = None
        if eval_data is not None:
            X_eval, y_eval = self._prepare_data(eval_data, feature_cols, fit=False)
            eval_set = [(X_eval, y_eval)]

        # Create and train estimator
        self.estimator_ = self._create_estimator()

        if self.config.backend == BackendType.SPARK:
            # Spark uses DataFrame directly
            features_col = self.preprocessor_.get_output_col()
            self.estimator_.fit(X_train, features_col=features_col)
        else:
            # PyTorch/XGBoost use numpy arrays
            self.estimator_.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                feature_names=feature_cols,
            )

        self._is_fitted = True
        return self

    def _fit_batched(
        self,
        train_data,
        eval_data,
        feature_cols: List[str],
        batch_size: int,
        label_col: str = 'y_moved',
    ):
        """
        Fit using batch iterator for large datasets.

        For XGBoost: Uses incremental training (each batch updates the model).
        For PyTorch: Collects batches and uses DataLoader for training.
        """
        from ..data.utils import create_batch_iterator_simple as create_batch_iterator

        self.feature_cols_ = feature_cols

        # Step 1: Fit preprocessor on first batch
        print("Step 1: Fitting preprocessor on sample batch...")
        sample_iter = create_batch_iterator(
            train_data,
            batch_size=min(batch_size, 50_000),
            feature_cols=feature_cols,
            label_col=label_col,
        )
        first_batch_X, first_batch_y = next(sample_iter)

        self.preprocessor_ = self._create_preprocessor()
        if self.config.backend == BackendType.XGBOOST and self.config.categorical_encoding == "native":
            X_sample = first_batch_X[feature_cols]
        else:
            X_sample = first_batch_X[feature_cols].values
        self.preprocessor_.fit(X_sample, self.config.label_col, feature_cols)

        # Prepare eval set if provided
        eval_set = None
        if eval_data is not None:
            X_eval, y_eval = self._prepare_data(eval_data, feature_cols, fit=False)
            eval_set = [(X_eval, y_eval)]

        # Step 2: Train based on backend type
        self.estimator_ = self._create_estimator()

        if self.config.backend == BackendType.XGBOOST:
            self._fit_xgboost_incremental(
                train_data, feature_cols, label_col, batch_size, eval_set
            )
        else:
            # PyTorch: collect batches and train with DataLoader
            self._fit_pytorch_batched(
                train_data, feature_cols, label_col, batch_size, eval_set
            )

        self._is_fitted = True
        return self

    def _fit_xgboost_incremental(
        self,
        train_data,
        feature_cols: List[str],
        label_col: str,
        batch_size: int,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """Incrementeel trainen van XGBoost per batch."""
        from ..data.utils import create_batch_iterator_simple as create_batch_iterator

        print(f"Step 2: Training XGBoost incrementally (batch_size={batch_size:,})...")

        batch_iter = create_batch_iterator(
            train_data,
            batch_size=batch_size,
            feature_cols=feature_cols,
            label_col=label_col,
        )

        total_samples = 0
        for i, (batch_X, batch_y) in enumerate(batch_iter):
            # Transform batch
            if self.config.categorical_encoding == "native":
                X_batch = self.preprocessor_.transform(batch_X[feature_cols])
            else:
                X_batch = self.preprocessor_.transform(batch_X[feature_cols].values)
            y_batch = batch_y.values
            total_samples += len(y_batch)

            # Train incrementally
            is_first_batch = (i == 0)
            self.estimator_.fit_incremental(
                X_batch,
                y_batch,
                eval_set=eval_set if is_first_batch else None,  # Eval only on first
                feature_names=feature_cols,
                reset=is_first_batch,
            )

            print(f"  Batch {i+1}: trained on {len(y_batch):,} samples "
                  f"(total: {total_samples:,})")

        print(f"Step 3: Training complete. Total samples: {total_samples:,}")

    def _fit_pytorch_batched(
        self,
        train_data,
        feature_cols: List[str],
        label_col: str,
        batch_size: int,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """Batch training for PyTorch - collect and train with DataLoader."""
        from ..data.utils import create_batch_iterator_simple as create_batch_iterator

        print(f"Step 2: Collecting batches for PyTorch training...")

        X_batches = []
        y_batches = []

        batch_iter = create_batch_iterator(
            train_data,
            batch_size=batch_size,
            feature_cols=feature_cols,
            label_col=label_col,
        )

        for i, (batch_X, batch_y) in enumerate(batch_iter):
            X_processed = self.preprocessor_.transform(batch_X[feature_cols].values)
            X_batches.append(X_processed)
            y_batches.append(batch_y.values)

            if (i + 1) % 10 == 0:
                mem = sum(x.nbytes for x in X_batches) / 1e9
                print(f"  Collected {(i+1)*batch_size:,} rows (~{mem:.2f} GB)")

        X_train = np.vstack(X_batches)
        y_train = np.concatenate(y_batches)

        print(f"Step 3: Training PyTorch model on {len(X_train):,} samples...")
        self.estimator_.fit(X_train, y_train, eval_set=eval_set)

    
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
