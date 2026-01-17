"""
XGBoost backend implementation.
Supports CPU and CUDA GPU training.
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


# =============================================================================
# Data Loading
# =============================================================================

class XGBoostDataLoader(BaseDataLoader):
    """Data loader for XGBoost backend."""
    
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
    ) -> Any:
        """Create XGBoost DMatrix."""
        import xgboost as xgb
        return xgb.DMatrix(X, label=y, weight=sample_weight)
    
    def create_dmatrix(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        """Create XGBoost DMatrix directly."""
        import xgboost as xgb
        return xgb.DMatrix(X, label=y, weight=sample_weight)


# =============================================================================
# Preprocessor (reuses sklearn for consistency)
# =============================================================================

class XGBoostPreprocessor(BasePreprocessor):
    """Preprocessor for XGBoost backend.
    
    Note: XGBoost handles missing values internally, so preprocessing
    is often lighter than for neural networks.
    """
    
    def __init__(
        self,
        scaling: str = "none",  # XGBoost doesn't need scaling typically
        handle_missing: str = "keep",  # "keep", "mean", "median"
    ):
        self.scaling = scaling
        self.handle_missing = handle_missing
        self.scaler_ = None
        self.imputer_ = None
        self.feature_cols_: Optional[List[str]] = None
    
    def fit(
        self,
        data: Union[np.ndarray, Any],
        label_col: str,
        feature_cols: List[str],
    ) -> "XGBoostPreprocessor":
        """Fit preprocessing pipeline."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.impute import SimpleImputer
        
        if hasattr(data, "toPandas"):
            X = data.select(feature_cols).toPandas().values
        else:
            X = data
        
        self.feature_cols_ = feature_cols
        
        # Imputation
        if self.handle_missing in ("mean", "median"):
            self.imputer_ = SimpleImputer(strategy=self.handle_missing)
            X = self.imputer_.fit_transform(X)
        
        # Scaling (usually not needed for tree models)
        if self.scaling == "standard":
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
        elif self.scaling == "minmax":
            self.scaler_ = MinMaxScaler()
            self.scaler_.fit(X)
        
        return self
    
    def transform(self, data: Union[np.ndarray, Any]) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if hasattr(data, "toPandas"):
            X = data.select(self.feature_cols_).toPandas().values
        else:
            X = data
        
        X = X.astype(np.float32)
        
        if self.imputer_ is not None:
            X = self.imputer_.transform(X)
        
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return X.astype(np.float32)
    
    def fit_transform(
        self,
        data: Union[np.ndarray, Any],
        label_col: str,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data, label_col, feature_cols)
        return self.transform(data)
    
    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler_,
            "imputer": self.imputer_,
            "feature_cols": self.feature_cols_,
            "scaling": self.scaling,
            "handle_missing": self.handle_missing,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "XGBoostPreprocessor":
        """Load preprocessor from disk."""
        import joblib
        data = joblib.load(path)
        preprocessor = cls(
            scaling=data["scaling"],
            handle_missing=data["handle_missing"],
        )
        preprocessor.scaler_ = data["scaler"]
        preprocessor.imputer_ = data["imputer"]
        preprocessor.feature_cols_ = data["feature_cols"]
        return preprocessor


# =============================================================================
# XGBoost Classifier
# =============================================================================

class XGBoostClassifier(BaseEstimator):
    """XGBoost classifier with CPU/GPU support."""
    
    # Default parameters for binary classification
    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "scale_pos_weight": 1,
        "seed": 42,
    }
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ):
        super().__init__(model_config, device_config)
        
        # Merge default params with user config
        params = model_config.get("params", {})
        self.params = {**self.DEFAULT_PARAMS, **params}
        
        # Training params
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 20)
        self.n_estimators = self.params.pop("n_estimators", 100)
        self.verbose_eval = self.params.pop("verbose_eval", 10)
        
        self.model_ = None
        self.best_iteration_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
    
    def _get_gpu_device(self) -> str:
        dm = get_device_manager()
        gpu_id = self.device_config.gpu_id
        if gpu_id is not None:
            return f"cuda:{gpu_id}"
        best = dm.get_best_gpu()
        return f"cuda:{best}" if best is not None else "cuda:0"
    
    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        if dm.check_cuda_available():
            return self._get_gpu_device()
        return "cpu"
    
    def _get_xgb_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters with device configuration."""
        device = self.get_device()
        params = self.params.copy()
        
        if device.startswith("cuda"):
            params["device"] = device
            params["tree_method"] = "hist"
        else:
            params["device"] = "cpu"
            params["tree_method"] = "hist"
        
        return params
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainResult:
        """Fit the XGBoost model."""
        import xgboost as xgb
        
        device = self.get_device()
        logger.info(f"Training XGBoost on device: {device}")
        
        self.feature_names_ = feature_names
        
        # Create DMatrix
        dtrain = xgb.DMatrix(
            X, label=y, weight=sample_weight,
            feature_names=feature_names,
        )
        
        # Evaluation sets
        evals = [(dtrain, "train")]
        if eval_set:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_names)
                evals.append((deval, f"eval_{i}"))
        
        # Get params with device config
        params = self._get_xgb_params()
        
        # Train
        evals_result = {}
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            evals_result=evals_result,
            verbose_eval=self.verbose_eval,
        )
        
        self.best_iteration_ = self.model_.best_iteration if eval_set else self.n_estimators
        self._is_fitted = True
        
        # Extract final metrics
        metrics = {}
        for ds_name, ds_metrics in evals_result.items():
            for metric_name, values in ds_metrics.items():
                metrics[f"{ds_name}_{metric_name}"] = values[-1]
        
        return TrainResult(
            model=self.model_,
            metrics=metrics,
            metadata={
                "best_iteration": self.best_iteration_,
                "feature_importance": self.model_.get_score(importance_type="gain"),
                "evals_result": evals_result,
            },
        )
    
    def predict(self, X: np.ndarray) -> PredictResult:
        """Generate class predictions."""
        proba = self.predict_proba(X)
        predictions = (proba.probabilities[:, 1] >= 0.5).astype(int)
        return PredictResult(predictions=predictions)
    
    def predict_proba(self, X: np.ndarray) -> PredictResult:
        """Generate probability predictions."""
        import xgboost as xgb
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names_)
        
        # XGBoost binary classification returns P(y=1)
        p1 = self.model_.predict(
            dmatrix,
            iteration_range=(0, self.best_iteration_),
        )
        
        probas_2d = np.column_stack([1 - p1, p1])
        
        return PredictResult(
            predictions=None,
            probabilities=probas_2d,
        )
    
    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.model_.get_score(importance_type=importance_type)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model_.save_model(path)
        
        # Save metadata separately
        import joblib
        meta_path = path + ".meta"
        joblib.dump({
            "model_config": self.model_config,
            "best_iteration": self.best_iteration_,
            "feature_names": self.feature_names_,
            "params": self.params,
        }, meta_path)
    
    @classmethod
    def load(
        cls,
        path: str,
        device_config: Optional[DeviceConfig] = None,
    ) -> "XGBoostClassifier":
        """Load model from disk."""
        import xgboost as xgb
        import joblib
        
        # Load metadata
        meta_path = path + ".meta"
        meta = joblib.load(meta_path)
        
        estimator = cls(meta["model_config"], device_config)
        estimator.model_ = xgb.Booster()
        estimator.model_.load_model(path)
        estimator.best_iteration_ = meta["best_iteration"]
        estimator.feature_names_ = meta["feature_names"]
        estimator.params = meta["params"]
        estimator._is_fitted = True
        
        return estimator


class XGBoostRanker(BaseEstimator):
    """XGBoost ranker for learning-to-rank tasks."""
    
    DEFAULT_PARAMS = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
    }
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ):
        super().__init__(model_config, device_config)
        params = model_config.get("params", {})
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.n_estimators = self.params.pop("n_estimators", 100)
        self.model_ = None
    
    def _get_gpu_device(self) -> str:
        dm = get_device_manager()
        gpu_id = self.device_config.gpu_id
        return dm.get_torch_device(gpu_id).replace("cuda", "gpu")
    
    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        return "gpu:0" if dm.check_cuda_available() else "cpu"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ) -> TrainResult:
        """Fit the ranker."""
        import xgboost as xgb
        
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        dtrain.set_group(group)
        
        params = self.params.copy()
        device = self.get_device()
        if device.startswith("gpu"):
            params["device"] = device.replace("gpu", "cuda")
            params["tree_method"] = "hist"
        
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )
        self._is_fitted = True
        
        return TrainResult(model=self.model_, metrics={})
    
    def predict(self, X: np.ndarray) -> PredictResult:
        """Generate ranking scores."""
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        scores = self.model_.predict(dmatrix)
        return PredictResult(predictions=scores)
    
    def predict_proba(self, X: np.ndarray) -> PredictResult:
        """Not applicable for ranking."""
        return self.predict(X)
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model_.save_model(path)
    
    @classmethod
    def load(cls, path: str, device_config: Optional[DeviceConfig] = None) -> "XGBoostRanker":
        import xgboost as xgb
        estimator = cls({}, device_config)
        estimator.model_ = xgb.Booster()
        estimator.model_.load_model(path)
        estimator._is_fitted = True
        return estimator


# =============================================================================
# Registration
# =============================================================================

def register_xgboost_backend():
    """Register XGBoost backend components with the factory."""
    BackendFactory.register_preprocessor(BackendType.XGBOOST, XGBoostPreprocessor)
    BackendFactory.register_estimator(BackendType.XGBOOST, "classifier", XGBoostClassifier)
    BackendFactory.register_estimator(BackendType.XGBOOST, "ranker", XGBoostRanker)
    BackendFactory.register_data_loader(BackendType.XGBOOST, XGBoostDataLoader)
    logger.info("XGBoost backend registered")
