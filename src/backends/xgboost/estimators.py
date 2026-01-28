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
    is often lighter than for neural networks. Supports native categorical
    handling (no one-hot) for tree-friendly splits.
    """

    def __init__(
        self,
        scaling: str = "none",  # XGBoost doesn't need scaling typically
        handle_missing: str = "keep",  # "keep", "mean", "median"
        categorical_encoding: str = "label",  # "label", "ordinal", "onehot", "native"
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ):
        self.scaling = scaling
        self.handle_missing = handle_missing
        self.categorical_encoding = categorical_encoding
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.scaler_ = None
        self.imputer_ = None
        self.encoders_: Dict[str, Any] = {}  # Per-column encoders
        self.feature_cols_: Optional[List[str]] = None
        self.categorical_categories_: Dict[str, List[str]] = {}
        self.native_categorical_ = False
    
    def fit(
        self,
        data: Union[np.ndarray, Any],
        label_col: str,
        feature_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> "XGBoostPreprocessor":
        """Fit preprocessing pipeline."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
        from sklearn.impute import SimpleImputer

        # Convert to DataFrame if needed
        if hasattr(data, "toPandas"):
            df = data.select(feature_cols).toPandas()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=feature_cols)
        else:
            df = data[feature_cols].copy()

        self.feature_cols_ = feature_cols
        self.categorical_cols = categorical_cols or self.categorical_cols
        self.numeric_cols = numeric_cols or self.numeric_cols
        self.encoders_ = {}
        self.categorical_categories_ = {}
        self.native_categorical_ = False

        # Auto-detect column types if not specified
        if not self.categorical_cols and not self.numeric_cols:
            for col in feature_cols:
                if df[col].dtype == object or df[col].dtype.name == 'category':
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)

        # Fit categorical encoders or capture categories for native handling
        if self.categorical_encoding == "native":
            self.native_categorical_ = True
            for col in self.categorical_cols:
                if col not in df.columns:
                    continue
                col_data = df[col].fillna("__missing__").astype(str)
                categories = list(pd.Series(col_data).unique())
                if "__missing__" not in categories:
                    categories.append("__missing__")
                self.categorical_categories_[col] = categories
        else:
            for col in self.categorical_cols:
                if col not in df.columns:
                    continue
                if self.categorical_encoding in ("label", "ordinal"):
                    encoder = LabelEncoder()
                    # Handle missing values before encoding
                    col_data = df[col].fillna("__missing__").astype(str)
                    encoder.fit(col_data)
                    self.encoders_[col] = encoder
                elif self.categorical_encoding == "onehot":
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    col_data = df[[col]].fillna("__missing__").astype(str)
                    encoder.fit(col_data)
                    self.encoders_[col] = encoder

        # Imputation for numeric columns
        if self.handle_missing in ("mean", "median") and self.numeric_cols:
            numeric_data = df[self.numeric_cols].values
            self.imputer_ = SimpleImputer(strategy=self.handle_missing)
            self.imputer_.fit(numeric_data)

        # Scaling (usually not needed for tree models)
        if self.scaling in ("standard", "minmax") and self.numeric_cols:
            numeric_data = df[self.numeric_cols].values
            if self.imputer_ is not None:
                numeric_data = self.imputer_.transform(numeric_data)
            if self.scaling == "standard":
                self.scaler_ = StandardScaler()
            else:
                self.scaler_ = MinMaxScaler()
            self.scaler_.fit(numeric_data)

        return self
    
    def transform(self, data: Union[np.ndarray, Any]) -> Any:
        """Transform data using fitted preprocessor."""
        import pandas as pd

        # Convert to DataFrame if needed
        if hasattr(data, "toPandas"):
            df = data.select(self.feature_cols_).toPandas()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=self.feature_cols_)
        else:
            df = data[self.feature_cols_].copy()

        if self.categorical_encoding == "native":
            result_df = df.copy()

            if self.numeric_cols:
                numeric_data = result_df[self.numeric_cols].values.astype(np.float32)
                if self.imputer_ is not None:
                    numeric_data = self.imputer_.transform(numeric_data)
                if self.scaler_ is not None:
                    numeric_data = self.scaler_.transform(numeric_data)
                result_df.loc[:, self.numeric_cols] = numeric_data.astype(np.float32)

            for col in self.categorical_cols:
                if col not in result_df.columns:
                    continue
                col_data = result_df[col].fillna("__missing__").astype(str)
                categories = self.categorical_categories_.get(col, ["__missing__"])
                cat_dtype = pd.CategoricalDtype(categories=categories)
                col_cat = pd.Series(pd.Categorical(col_data, dtype=cat_dtype))
                if "__missing__" in categories:
                    col_cat = col_cat.fillna("__missing__")
                result_df[col] = col_cat

            return result_df

        result_arrays = []

        # Process categorical columns
        for col in self.categorical_cols:
            if col not in df.columns or col not in self.encoders_:
                continue
            encoder = self.encoders_[col]
            col_data = df[col].fillna("__missing__").astype(str)

            if self.categorical_encoding in ("label", "ordinal"):
                # Handle unseen labels
                known_classes = set(encoder.classes_)
                col_data = col_data.apply(lambda x: x if x in known_classes else "__missing__")
                encoded = encoder.transform(col_data).reshape(-1, 1)
            else:  # onehot
                encoded = encoder.transform(df[[col]].fillna("__missing__").astype(str))

            result_arrays.append(encoded.astype(np.float32))

        # Process numeric columns
        if self.numeric_cols:
            numeric_data = df[self.numeric_cols].values.astype(np.float32)

            if self.imputer_ is not None:
                numeric_data = self.imputer_.transform(numeric_data)

            if self.scaler_ is not None:
                numeric_data = self.scaler_.transform(numeric_data)

            result_arrays.append(numeric_data.astype(np.float32))

        if not result_arrays:
            return df.values.astype(np.float32)

        return np.hstack(result_arrays).astype(np.float32)
    
    def fit_transform(
        self,
        data: Union[np.ndarray, Any],
        label_col: str,
        feature_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> Any:
        """Fit and transform in one step."""
        self.fit(data, label_col, feature_cols, categorical_cols, numeric_cols)
        return self.transform(data)
    
    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        import joblib
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler_,
            "imputer": self.imputer_,
            "encoders": self.encoders_,
            "feature_cols": self.feature_cols_,
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
            "scaling": self.scaling,
            "handle_missing": self.handle_missing,
            "categorical_encoding": self.categorical_encoding,
            "categorical_categories": self.categorical_categories_,
            "native_categorical": self.native_categorical_,
        }, os.path.join(path, "preprocessor.joblib"))

    @classmethod
    def load(cls, path: str) -> "XGBoostPreprocessor":
        """Load preprocessor from disk."""
        import joblib
        data = joblib.load(os.path.join(path, "preprocessor.joblib"))
        preprocessor = cls(
            scaling=data["scaling"],
            handle_missing=data["handle_missing"],
            categorical_encoding=data.get("categorical_encoding", "label"),
            categorical_cols=data.get("categorical_cols", []),
            numeric_cols=data.get("numeric_cols", []),
        )
        preprocessor.scaler_ = data["scaler"]
        preprocessor.imputer_ = data["imputer"]
        preprocessor.encoders_ = data.get("encoders", {})
        preprocessor.feature_cols_ = data["feature_cols"]
        preprocessor.categorical_categories_ = data.get("categorical_categories", {})
        preprocessor.native_categorical_ = data.get("native_categorical", False)
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
        # GPU-friendly defaults
        "grow_policy": "depthwise",  # Better for GPU
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

        # Extract enable_categorical for DMatrix (not a training param)
        self.enable_categorical = self.params.pop("enable_categorical", False)

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
        import xgboost as xgb

        device = self.get_device()
        params = self.params.copy()

        # Ensure base_score is valid for logistic loss (must be in (0,1))
        # Default to 0.5 if not set or invalid
        if "base_score" not in params or params.get("base_score") is None:
            params["base_score"] = 0.5

        if device.startswith("cuda"):
            # Extract GPU ID from device string (e.g., "cuda:0" -> 0)
            gpu_id = int(device.split(":")[-1]) if ":" in device else 0

            # XGBoost 2.0+ unified API: use device parameter instead of gpu_hist
            params["device"] = f"cuda:{gpu_id}"
            params["tree_method"] = "hist"  # Works for both CPU and GPU

            # GPU-specific optimizations
            if "max_bin" not in params:
                params["max_bin"] = 256  # Higher bins = better quality on GPU

            # XGBoost 3.0+ supports these GPU optimizations
            if xgb.__version__ >= '3.0.0':
                params["sampling_method"] = params.get("sampling_method", "gradient_based")
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
            enable_categorical=self.enable_categorical,
        )
        
        # Evaluation sets
        evals = [(dtrain, "train")]
        if eval_set:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                deval = xgb.DMatrix(
                    X_eval, label=y_eval,
                    feature_names=feature_names,
                    enable_categorical=self.enable_categorical,
                )
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
        Incrementally train the XGBoost model on a batch of data.

        Args:
            X: Feature matrix for this batch
            y: Labels for this batch
            sample_weight: Optional sample weights
            eval_set: Optional evaluation set (used only on first batch typically)
            feature_names: Feature names
            reset: If True, start fresh. If False, continue training existing model.

        Returns:
            TrainResult with updated model
        """
        import xgboost as xgb

        device = self.get_device()

        if reset or self.model_ is None:
            logger.info(f"Starting incremental XGBoost training on device: {device}")
            self.feature_names_ = feature_names
            self.model_ = None  # Will be created on first batch

        # Create DMatrix for this batch
        dtrain = xgb.DMatrix(
            X, label=y, weight=sample_weight,
            feature_names=feature_names or self.feature_names_,
            enable_categorical=self.enable_categorical,
        )

        # Evaluation sets
        evals = [(dtrain, "train")]
        if eval_set:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                deval = xgb.DMatrix(
                    X_eval, label=y_eval,
                    feature_names=feature_names or self.feature_names_,
                    enable_categorical=self.enable_categorical,
                )
                evals.append((deval, f"eval_{i}"))

        # Get params with device config
        params = self._get_xgb_params()

        # IMPORTANT: Remove base_score when continuing training
        # XGBoost stores base_score in the model, and passing it again causes conflicts
        if self.model_ is not None:
            params.pop("base_score", None)

        # Number of rounds per batch (increase for better GPU utilization)
        # GPU trains much faster, so we can afford more rounds per batch
        device = self.get_device()
        if device.startswith("cuda"):
            # GPU: Use more trees per batch to saturate GPU compute
            batch_rounds = max(200, self.n_estimators // 2)  # More trees on GPU
        else:
            # CPU: Keep lower to avoid long batch times
            batch_rounds = max(100, self.n_estimators // 3)
        # Train incrementally: pass existing model to continue training
        evals_result = {}
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=batch_rounds,
            evals=evals,
            xgb_model=self.model_,  # Continue from existing model
            evals_result=evals_result,
            verbose_eval=False,  # Less verbose for batch training
        )

        self.best_iteration_ = self.model_.num_boosted_rounds()
        self._is_fitted = True

        # Extract metrics
        metrics = {}
        for ds_name, ds_metrics in evals_result.items():
            for metric_name, values in ds_metrics.items():
                metrics[f"{ds_name}_{metric_name}"] = values[-1]

        return TrainResult(
            model=self.model_,
            metrics=metrics,
            metadata={"n_trees": self.best_iteration_},
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
        import xgboost as xgb

        # Optimization: DMatrix creation can be expensive for large datasets
        # Consider using DMatrix.quantile_cut for very large datasets
        dmatrix = xgb.DMatrix(
            X,
            feature_names=self.feature_names_,
            enable_categorical=self.enable_categorical,
        )

        # XGBoost binary classification returns P(y=1)
        # Use all trees if best_iteration_ not set (incremental training)
        if self.best_iteration_ is not None:
            p1 = self.model_.predict(
                dmatrix,
                iteration_range=(0, self.best_iteration_),
            )
        else:
            p1 = self.model_.predict(dmatrix)

        probas_2d = np.column_stack([1 - p1, p1])

        return PredictResult(
            predictions=None,
            probabilities=probas_2d,
        )

    def predict_proba_batched(
        self,
        X: np.ndarray,
        batch_size: int = 100_000,
    ) -> PredictResult:
        """
        Generate probability predictions in batches for large datasets.

        This method is more memory-efficient than predict_proba for large datasets,
        as it processes data in batches to avoid creating one massive DMatrix.

        Args:
            X: Feature matrix (can be numpy array or pandas DataFrame)
            batch_size: Number of samples per batch

        Returns:
            PredictResult with probabilities for all samples
        """
        import xgboost as xgb

        n_samples = len(X)

        # If dataset is small, use regular prediction
        if n_samples <= batch_size:
            return self.predict_proba(X)

        # Process in batches
        probabilities_list = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]

            # Create DMatrix for this batch only
            dmatrix = xgb.DMatrix(
                X_batch,
                feature_names=self.feature_names_,
                enable_categorical=self.enable_categorical,
            )

            # Predict
            if self.best_iteration_ is not None:
                p1 = self.model_.predict(
                    dmatrix,
                    iteration_range=(0, self.best_iteration_),
                )
            else:
                p1 = self.model_.predict(dmatrix)

            probas_batch = np.column_stack([1 - p1, p1])
            probabilities_list.append(probas_batch)

        # Concatenate all batch predictions
        all_probabilities = np.vstack(probabilities_list)

        return PredictResult(
            predictions=None,
            probabilities=all_probabilities,
        )
    

    def get_booster(self):
        """Return the underlying XGBoost Booster object."""
        return self.model_

    
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
            "enable_categorical": self.enable_categorical,
            "threshold": self._threshold,
            "threshold_tuning_stats": self._threshold_tuning_stats,
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
        estimator.enable_categorical = meta.get("enable_categorical", False)
        estimator._threshold = meta.get("threshold", 0.5)
        estimator._threshold_tuning_stats = meta.get("threshold_tuning_stats", None)
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
