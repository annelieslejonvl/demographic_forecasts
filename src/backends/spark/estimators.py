"""
Spark ML backend implementation.
Supports CPU and RAPIDS GPU acceleration.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

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


# =============================================================================
# Data Loading
# =============================================================================

class SparkDataLoader(BaseDataLoader):
    """Data loader for Spark backend (passthrough since data is already Spark)."""
    
    def from_spark(
        self,
        spark_df: Any,
        label_col: str,
        feature_cols: List[str],
    ) -> Tuple[Any, str]:
        """Return Spark DataFrame as-is with column info."""
        return spark_df, label_col
    
    def to_spark(self, X: Any, y: Any, spark_session: Any) -> Any:
        """X is already a Spark DataFrame."""
        return X
    
    def create_dataloader(
        self,
        X: Any,
        y: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        sample_weight: Optional[Any] = None,
    ) -> Any:
        """Spark doesn't use batched data loaders in the same way."""
        return X


# =============================================================================
# Preprocessor
# =============================================================================

class SparkPreprocessor(BasePreprocessor):
    """Preprocessor for Spark backend using Spark ML transformers."""
    
    def __init__(
        self,
        scaling: bool = False,
        scaling_type: str = "standard",
        dim_reduction: Optional[Dict[str, Any]] = None,
        features_col: str = "features",
        label_col: str = "label",
    ):
        self.scaling = scaling
        self.scaling_type = scaling_type
        self.dim_reduction = dim_reduction or {}
        self.features_col = features_col
        self.label_col = label_col
        
        self.pipeline_ = None
        self.pipeline_model_ = None
        self.feature_cols_: Optional[List[str]] = None
        self.output_col_: str = features_col
    
    def fit(
        self,
        data: Any,
        label_col: str,
        feature_cols: List[str],
    ) -> "SparkPreprocessor":
        """Fit Spark ML preprocessing pipeline."""
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            VectorAssembler,
            StandardScaler,
            MinMaxScaler,
            PCA,
        )
        
        self.feature_cols_ = feature_cols
        self.label_col = label_col
        
        stages = []
        current_col = "assembled_features"
        
        # Vector assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=current_col,
            handleInvalid="keep",
        )
        stages.append(assembler)
        
        # Scaling
        if self.scaling:
            scaled_col = "scaled_features"
            if self.scaling_type == "standard":
                scaler = StandardScaler(
                    inputCol=current_col,
                    outputCol=scaled_col,
                    withMean=True,
                    withStd=True,
                )
            else:
                scaler = MinMaxScaler(
                    inputCol=current_col,
                    outputCol=scaled_col,
                )
            stages.append(scaler)
            current_col = scaled_col
        
        # Dimensionality reduction
        if self.dim_reduction.get("enabled", False):
            pca_col = "pca_features"
            k = self.dim_reduction.get("k", 50)
            pca = PCA(inputCol=current_col, outputCol=pca_col, k=k)
            stages.append(pca)
            current_col = pca_col
        
        self.output_col_ = current_col
        
        # Build and fit pipeline
        self.pipeline_ = Pipeline(stages=stages)
        self.pipeline_model_ = self.pipeline_.fit(data)
        
        return self
    
    def transform(self, data: Any) -> Any:
        """Transform data using fitted pipeline."""
        return self.pipeline_model_.transform(data)
    
    def fit_transform(
        self,
        data: Any,
        label_col: str,
        feature_cols: List[str],
    ) -> Any:
        """Fit and transform in one step."""
        self.fit(data, label_col, feature_cols)
        return self.transform(data)
    
    def get_output_col(self) -> str:
        """Get the name of the output features column."""
        return self.output_col_
    
    def save(self, path: str) -> None:
        """Save fitted pipeline to disk."""
        self.pipeline_model_.write().overwrite().save(path)
        
        # Save metadata
        import json
        meta = {
            "feature_cols": self.feature_cols_,
            "output_col": self.output_col_,
            "label_col": self.label_col,
            "scaling": self.scaling,
            "scaling_type": self.scaling_type,
            "dim_reduction": self.dim_reduction,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f)
    
    @classmethod
    def load(cls, path: str) -> "SparkPreprocessor":
        """Load preprocessor from disk."""
        from pyspark.ml import PipelineModel
        import json
        
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)
        
        preprocessor = cls(
            scaling=meta["scaling"],
            scaling_type=meta["scaling_type"],
            dim_reduction=meta["dim_reduction"],
            features_col=meta["output_col"],
            label_col=meta["label_col"],
        )
        preprocessor.pipeline_model_ = PipelineModel.load(path)
        preprocessor.feature_cols_ = meta["feature_cols"]
        preprocessor.output_col_ = meta["output_col"]
        
        return preprocessor


# =============================================================================
# Spark ML Estimators
# =============================================================================

class SparkLogisticRegression(BaseEstimator):
    """Spark ML Logistic Regression estimator."""

    DEFAULT_PARAMS = {
        "maxIter": 100,
        "regParam": 0.01,
        "elasticNetParam": 0.0,
        "tol": 1e-6,
        "threshold": 0.5,
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
        self.features_col_: str = "features"
        self.label_col_: str = "label"

    def _extract_probabilities(self, proba_result: PredictResult):
        """Extract probabilities from Spark DataFrame."""
        import numpy as np
        spark_df = proba_result.predictions
        probs = spark_df.select("probability").rdd.map(lambda row: float(row[0][1])).collect()
        return np.array(probs)

    def _extract_labels(self, y):
        """Extract labels from Spark DataFrame or column."""
        import numpy as np
        if hasattr(y, 'select'):
            labels = y.select(self.label_col_).rdd.map(lambda row: float(row[0])).collect()
            return np.array(labels)
        return np.asarray(y).ravel()

    def _get_gpu_device(self) -> str:
        return "gpu"
    
    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        return "gpu" if dm.check_rapids_available() else "cpu"
    
    def fit(
        self,
        X: Any,  # Spark DataFrame with features column
        y: Any = None,  # Not used; label is in DataFrame
        sample_weight: Optional[str] = None,  # Weight column name
        eval_set: Optional[List[Any]] = None,
        features_col: str = "features",
        label_col: str = "label",
    ) -> TrainResult:
        """Fit the logistic regression model."""
        from pyspark.ml.classification import LogisticRegression
        
        device = self.get_device()
        logger.info(f"Training Spark LogisticRegression on device: {device}")
        
        self.features_col_ = features_col
        self.label_col_ = label_col
        
        # Build estimator
        lr = LogisticRegression(
            featuresCol=features_col,
            labelCol=label_col,
            weightCol=sample_weight if sample_weight else None,
            **self.params,
        )
        
        # Fit
        self.model_ = lr.fit(X)
        self._is_fitted = True

        # Update model threshold to match self._threshold
        self.model_.setThreshold(self._threshold)

        # Training metrics
        summary = self.model_.summary
        metrics = {
            "auc_roc": summary.areaUnderROC,
            "accuracy": summary.accuracy,
        }
        
        return TrainResult(
            model=self.model_,
            metrics=metrics,
            metadata={
                "coefficients": self.model_.coefficients.toArray().tolist(),
                "intercept": self.model_.intercept,
                "num_iterations": summary.totalIterations,
            },
        )
    
    def predict(self, X: Any) -> PredictResult:
        """Generate predictions using the current threshold."""
        predictions = self.model_.transform(X)
        return PredictResult(
            predictions=predictions,
            metadata={'threshold': self._threshold}
        )

    def predict_proba(self, X: Any) -> PredictResult:
        """Generate probability predictions (same as predict for Spark)."""
        return self.predict(X)

    def tune_threshold(self, X_val: Any, y_val: Any = None, strategy: str = 'f1', **kwargs) -> float:
        """Tune threshold and update Spark model."""
        optimal_threshold = super().tune_threshold(X_val, y_val, strategy, **kwargs)
        self.model_.setThreshold(optimal_threshold)
        return optimal_threshold

    def set_threshold(self, threshold: float) -> None:
        """Set threshold on both base class and Spark model."""
        super().set_threshold(threshold)
        if self.model_ is not None:
            self.model_.setThreshold(threshold)

    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model_.write().overwrite().save(path)

        # Save threshold metadata
        import json
        meta_path = os.path.join(path, "threshold_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({
                "threshold": self._threshold,
                "threshold_tuning_stats": self._threshold_tuning_stats,
            }, f)

    @classmethod
    def load(
        cls,
        path: str,
        device_config: Optional[DeviceConfig] = None,
    ) -> "SparkLogisticRegression":
        """Load model from disk."""
        from pyspark.ml.classification import LogisticRegressionModel
        import json

        estimator = cls({}, device_config)
        estimator.model_ = LogisticRegressionModel.load(path)
        estimator._is_fitted = True

        # Load threshold metadata
        meta_path = os.path.join(path, "threshold_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                estimator._threshold = meta.get("threshold", 0.5)
                estimator._threshold_tuning_stats = meta.get("threshold_tuning_stats", None)
                estimator.model_.setThreshold(estimator._threshold)

        return estimator


class SparkRandomForest(BaseEstimator):
    """Spark ML Random Forest classifier."""
    
    DEFAULT_PARAMS = {
        "numTrees": 100,
        "maxDepth": 10,
        "minInstancesPerNode": 1,
        "subsamplingRate": 0.8,
        "featureSubsetStrategy": "sqrt",
        "seed": 42,
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
        self.features_col_: str = "features"
        self.label_col_: str = "label"
    
    def _get_gpu_device(self) -> str:
        return "gpu"
    
    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        return "gpu" if dm.check_rapids_available() else "cpu"
    
    def fit(
        self,
        X: Any,
        y: Any = None,
        sample_weight: Optional[str] = None,
        eval_set: Optional[List[Any]] = None,
        features_col: str = "features",
        label_col: str = "label",
    ) -> TrainResult:
        """Fit the random forest model."""
        from pyspark.ml.classification import RandomForestClassifier
        
        device = self.get_device()
        logger.info(f"Training Spark RandomForest on device: {device}")
        
        # For GPU, use RAPIDS accelerated version if available
        if device == "gpu":
            try:
                from spark_rapids_ml.classification import RandomForestClassifier as RapidsRF
                logger.info("Using RAPIDS-accelerated RandomForest")
                rf = RapidsRF(
                    featuresCol=features_col,
                    labelCol=label_col,
                    **self.params,
                )
            except ImportError:
                logger.warning("RAPIDS ML not available, falling back to CPU")
                rf = RandomForestClassifier(
                    featuresCol=features_col,
                    labelCol=label_col,
                    **self.params,
                )
        else:
            rf = RandomForestClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                **self.params,
            )
        
        self.features_col_ = features_col
        self.label_col_ = label_col
        
        self.model_ = rf.fit(X)
        self._is_fitted = True
        
        return TrainResult(
            model=self.model_,
            metrics={},
            metadata={
                "feature_importances": self.model_.featureImportances.toArray().tolist(),
                "num_trees": self.model_.getNumTrees,
            },
        )
    
    def predict(self, X: Any) -> PredictResult:
        """Generate predictions."""
        predictions = self.model_.transform(X)
        return PredictResult(predictions=predictions)
    
    def predict_proba(self, X: Any) -> PredictResult:
        """Generate probability predictions."""
        return self.predict(X)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model_.write().overwrite().save(path)
    
    @classmethod
    def load(
        cls,
        path: str,
        device_config: Optional[DeviceConfig] = None,
    ) -> "SparkRandomForest":
        """Load model from disk."""
        from pyspark.ml.classification import RandomForestClassificationModel
        
        estimator = cls({}, device_config)
        estimator.model_ = RandomForestClassificationModel.load(path)
        estimator._is_fitted = True
        return estimator


class SparkGBTClassifier(BaseEstimator):
    """Spark ML Gradient Boosted Trees classifier."""
    
    DEFAULT_PARAMS = {
        "maxIter": 100,
        "maxDepth": 5,
        "stepSize": 0.1,
        "subsamplingRate": 0.8,
        "minInstancesPerNode": 1,
        "seed": 42,
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
    
    def _get_gpu_device(self) -> str:
        return "gpu"
    
    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        return "gpu" if dm.check_rapids_available() else "cpu"
    
    def fit(
        self,
        X: Any,
        y: Any = None,
        sample_weight: Optional[str] = None,
        eval_set: Optional[List[Any]] = None,
        features_col: str = "features",
        label_col: str = "label",
    ) -> TrainResult:
        """Fit the GBT model."""
        from pyspark.ml.classification import GBTClassifier
        
        device = self.get_device()
        logger.info(f"Training Spark GBT on device: {device}")
        
        gbt = GBTClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            **self.params,
        )
        
        self.model_ = gbt.fit(X)
        self._is_fitted = True
        
        return TrainResult(
            model=self.model_,
            metrics={},
            metadata={
                "feature_importances": self.model_.featureImportances.toArray().tolist(),
            },
        )
    
    def predict(self, X: Any) -> PredictResult:
        predictions = self.model_.transform(X)
        return PredictResult(predictions=predictions)
    
    def predict_proba(self, X: Any) -> PredictResult:
        return self.predict(X)
    
    def save(self, path: str) -> None:
        self.model_.write().overwrite().save(path)
    
    @classmethod
    def load(cls, path: str, device_config: Optional[DeviceConfig] = None) -> "SparkGBTClassifier":
        from pyspark.ml.classification import GBTClassificationModel
        estimator = cls({}, device_config)
        estimator.model_ = GBTClassificationModel.load(path)
        estimator._is_fitted = True
        return estimator


# =============================================================================
# Registration
# =============================================================================

def register_spark_backend():
    """Register Spark backend components with the factory."""
    BackendFactory.register_preprocessor(BackendType.SPARK, SparkPreprocessor)
    BackendFactory.register_estimator(BackendType.SPARK, "logistic_regression", SparkLogisticRegression)
    BackendFactory.register_estimator(BackendType.SPARK, "random_forest", SparkRandomForest)
    BackendFactory.register_estimator(BackendType.SPARK, "gbt", SparkGBTClassifier)
    BackendFactory.register_data_loader(BackendType.SPARK, SparkDataLoader)
    logger.info("Spark backend registered")
