"""
PyTorch backend implementation.
Supports CPU and CUDA GPU training for neural network models.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

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

class PyTorchDataLoader(BaseDataLoader):
    """Data loader for PyTorch backend."""

    def from_spark(
        self,
        spark_df: Any,
        label_col: str,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Spark DataFrame to numpy arrays."""
        # Select relevant columns and convert to Pandas
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
        """Create a PyTorch DataLoader."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float().view(-1, 1)

        if sample_weight is not None:
            w_tensor = torch.from_numpy(sample_weight).float().view(-1, 1)
            dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )


# =============================================================================
# Preprocessor
# =============================================================================

class PyTorchPreprocessor(BasePreprocessor):
    """Preprocessor for PyTorch backend using sklearn transformers."""

    def __init__(
        self,
        scaling: str = "standard",  # "standard", "minmax", "none"
        categorical_encoding: str = "onehot",  # "onehot", "label", "ordinal", "native"
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        dim_reduction: Optional[Dict[str, Any]] = None,
    ):
        self.scaling = scaling
        self.categorical_encoding = categorical_encoding
        self.effective_categorical_encoding_ = categorical_encoding
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.dim_reduction = dim_reduction or {}
        self.scaler_ = None
        self.pca_ = None
        self.encoders_: Dict[str, Any] = {}
        self.feature_cols_: Optional[List[str]] = None

    def fit(
        self,
        data: Union[np.ndarray, Any],
        label_col: str,
        feature_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> "PyTorchPreprocessor":
        """Fit preprocessing pipeline."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
        from sklearn.decomposition import PCA
        print('fitting preprocessor')
        # Convert to DataFrame if needed
        if hasattr(data, "toPandas"):
            print('convert to Pandas')
            df = data.select(feature_cols).toPandas()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=feature_cols)
        else:
            df = data[feature_cols].copy()

        self.feature_cols_ = feature_cols
        self.categorical_cols = categorical_cols or self.categorical_cols
        self.numeric_cols = numeric_cols or self.numeric_cols

        # Auto-detect column types if not specified
        if not self.categorical_cols and not self.numeric_cols:
            for col in feature_cols:
                if df[col].dtype == object or df[col].dtype.name == 'category':
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)

        if self.categorical_encoding == "native":
            print('one hot encoding')
            logger.warning("Native categorical encoding is not supported in PyTorch; using onehot encoding.")
            self.effective_categorical_encoding_ = "onehot"
        else:
            self.effective_categorical_encoding_ = self.categorical_encoding

        # Fit categorical encoders
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if self.effective_categorical_encoding_ in ("label", "ordinal"):
                encoder = LabelEncoder()
                col_data = df[col].fillna("__missing__").astype(str)
                # Ensure "__missing__" is always in classes even if no NaN in training
                unique_values = list(col_data.unique())
                if "__missing__" not in unique_values:
                    unique_values.append("__missing__")
                encoder.fit(unique_values)
                self.encoders_[col] = encoder
            elif self.effective_categorical_encoding_ == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                col_data = df[[col]].fillna("__missing__").astype(str)
                encoder.fit(col_data)
                self.encoders_[col] = encoder

        # Process numeric columns for scaling
        if self.numeric_cols:
            numeric_data = df[self.numeric_cols].values.astype(np.float32)
            if self.scaling == "standard":
                self.scaler_ = StandardScaler()
                self.scaler_.fit(numeric_data)
            elif self.scaling == "minmax":
                self.scaler_ = MinMaxScaler()
                self.scaler_.fit(numeric_data)

        # PCA on all transformed data
        if self.dim_reduction.get("enabled", False):
            X_transformed = self._transform_impl(df)
            n_components = int(self.dim_reduction.get("k", min(50, X_transformed.shape[1])))
            self.pca_ = PCA(n_components=n_components)
            self.pca_.fit(X_transformed)

        return self

    def _transform_impl(self, df) -> np.ndarray:
        """Internal transform without PCA."""
        result_arrays = []
        print('transforming data')
        # Process categorical columns
        for col in self.categorical_cols:
            if col not in df.columns or col not in self.encoders_:
                continue
            encoder = self.encoders_[col]
            col_data = df[col].fillna("__missing__").astype(str)

            if self.effective_categorical_encoding_ in ("label", "ordinal"):
                known_classes = set(encoder.classes_)
                col_data = col_data.apply(lambda x: x if x in known_classes else "__missing__")
                encoded = encoder.transform(col_data).reshape(-1, 1)
            else:  # onehot
                encoded = encoder.transform(df[[col]].fillna("__missing__").astype(str))

            result_arrays.append(encoded.astype(np.float32))

        # Process numeric columns
        if self.numeric_cols:
            numeric_data = df[self.numeric_cols].values.astype(np.float32)
            if self.scaler_ is not None:
                numeric_data = self.scaler_.transform(numeric_data)
            result_arrays.append(numeric_data.astype(np.float32))

        if not result_arrays:
            return df.values.astype(np.float32)
        print('transform done')
        return np.hstack(result_arrays).astype(np.float32)

    def transform(self, data: Union[np.ndarray, Any]) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        import pandas as pd

        # Convert to DataFrame if needed
        if hasattr(data, "toPandas"):
            df = data.select(self.feature_cols_).toPandas()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=self.feature_cols_)
        else:
            df = data[self.feature_cols_].copy()

        X = self._transform_impl(df)

        if self.pca_ is not None:
            X = self.pca_.transform(X)

        return X.astype(np.float32)

    def fit_transform(
        self,
        data,
        label_col: str,
        feature_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        self.fit(data, label_col, feature_cols, categorical_cols, numeric_cols)
        return self.transform(data)

    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        import joblib

        os.makedirs(path, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler_,
                "pca": self.pca_,
                "encoders": self.encoders_,
                "feature_cols": self.feature_cols_,
                "categorical_cols": self.categorical_cols,
                "numeric_cols": self.numeric_cols,
                "scaling": self.scaling,
                "categorical_encoding": self.categorical_encoding,
                "dim_reduction": self.dim_reduction,
            },
            os.path.join(path, "preprocessor.joblib"),
        )

    @classmethod
    def load(cls, path: str) -> "PyTorchPreprocessor":
        """Load preprocessor from disk."""
        import joblib

        data = joblib.load(os.path.join(path, "preprocessor.joblib"))
        preprocessor = cls(
            scaling=data["scaling"],
            categorical_encoding=data.get("categorical_encoding", "onehot"),
            categorical_cols=data.get("categorical_cols", []),
            numeric_cols=data.get("numeric_cols", []),
            dim_reduction=data["dim_reduction"],
        )
        if preprocessor.categorical_encoding == "native":
            preprocessor.effective_categorical_encoding_ = "onehot"
        else:
            preprocessor.effective_categorical_encoding_ = preprocessor.categorical_encoding
        preprocessor.scaler_ = data["scaler"]
        preprocessor.pca_ = data["pca"]
        preprocessor.encoders_ = data.get("encoders", {})
        preprocessor.feature_cols_ = data["feature_cols"]
        return preprocessor


# =============================================================================
# Neural Network Models
# =============================================================================

def _build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int = 1,
    dropout: float = 0.2,
    activation: str = "relu",
) -> Any:
    """Build a simple MLP architecture."""
    import torch.nn as nn

    activation_fn = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }.get(activation, nn.ReLU)

    layers: List[Any] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(dropout),
            ]
        )
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class PyTorchMLPEstimator(BaseEstimator):
    """PyTorch MLP estimator for binary classification."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ):
        super().__init__(model_config, device_config)

        params = model_config.get("params", {})
        self.hidden_dims = params.get("hidden_dims", [128, 64])
        self.dropout = params.get("dropout", 0.2)
        self.activation = params.get("activation", "relu")

        self.learning_rate = params.get("learning_rate", 1e-3)
        self.weight_decay = params.get("weight_decay", 1e-4)
        self.epochs = params.get("epochs", 100)
        self.batch_size = params.get("batch_size", 256)
        self.early_stopping_patience = params.get("early_stopping_patience", 10)
        self.use_amp = bool(params.get("use_amp", True))

        self.model_ = None
        self.input_dim_: Optional[int] = None

    def _get_gpu_device(self) -> str:
        dm = get_device_manager()
        gpu_id = self.device_config.gpu_id
        return dm.get_torch_device(gpu_id)

    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        if dm.check_cuda_available():
            return dm.get_torch_device()
        return "cpu"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols=None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        device: Optional[str] = None,
    ) -> TrainResult:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = (device.type == "cuda")

        logger.info(f"Training PyTorch MLP on device: {device}")

        self.input_dim_ = int(X.shape[1])
        self.model_ = _build_mlp(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            dropout=self.dropout,
            activation=self.activation,
        ).to(device)

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float().view(-1, 1)

        w_t = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.ndim != 1 or sw.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must be shape (n_samples,)")
            w_t = torch.from_numpy(sw).float().view(-1, 1)

        train_dataset = TensorDataset(X_t, y_t) if w_t is None else TensorDataset(X_t, y_t, w_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
            drop_last=False,
        )

        val_loader = None
        if eval_set:
            X_val, y_val = eval_set[0]
            Xv = torch.from_numpy(X_val).float()
            yv = torch.from_numpy(y_val).float().view(-1, 1)
            val_loader = DataLoader(
                TensorDataset(Xv, yv),
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda,
                drop_last=False,
            )

        # pos_weight = neg/pos
        y_np = np.asarray(y, dtype=np.float32)
        n_pos = float(y_np.sum())
        n_neg = float(len(y_np) - n_pos)
        pos_weight = n_neg / max(n_pos, 1.0)
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

        def weighted_bce_with_logits(logits, targets, weights=None):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pos_weight_t
            )
            if weights is not None:
                loss = loss * weights
                return loss.sum() / (weights.sum().clamp_min(1e-12))
            return loss.mean()

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and self.use_amp)

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(int(self.epochs)):
            self.model_.train()
            train_loss_sum = 0.0
            n_seen = 0

            for batch in train_loader:
                if w_t is None:
                    batch_X, batch_y = batch
                    batch_w = None
                else:
                    batch_X, batch_y, batch_w = batch

                batch_X = batch_X.to(device, non_blocking=use_cuda)
                batch_y = batch_y.to(device, non_blocking=use_cuda)
                if batch_w is not None:
                    batch_w = batch_w.to(device, non_blocking=use_cuda)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = self.model_(batch_X)
                    loss = weighted_bce_with_logits(logits, batch_y, batch_w)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = batch_X.size(0)
                train_loss_sum += float(loss.item()) * bs
                n_seen += bs

            train_loss = train_loss_sum / max(n_seen, 1)
            history["train_loss"].append(train_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            lr_now = float(optimizer.param_groups[0]["lr"])
            mlflow.log_metric("lr", lr_now, step=epoch)

            val_loss = None
            if val_loader is not None:
                self.model_.eval()
                val_loss_sum = 0.0
                n_val = 0
                y_true: List[np.ndarray] = []
                y_pred: List[np.ndarray] = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device, non_blocking=use_cuda)
                        batch_y = batch_y.to(device, non_blocking=use_cuda)

                        logits = self.model_(batch_X)
                        loss = weighted_bce_with_logits(logits, batch_y, None)

                        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
                        y_pred.append(probs)
                        y_true.append(batch_y.detach().cpu().numpy().ravel())

                        bs = batch_X.size(0)
                        val_loss_sum += float(loss.item()) * bs
                        n_val += bs

                val_loss = val_loss_sum / max(n_val, 1)
                history["val_loss"].append(val_loss)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                y_true_all = np.concatenate(y_true) if y_true else np.array([], dtype=np.float32)
                y_pred_all = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.float32)

                if y_true_all.size > 0 and np.unique(y_true_all).size > 1:
                    auc = roc_auc_score(y_true_all, y_pred_all)
                    mlflow.log_metric("val_auc", float(auc), step=epoch)

                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= int(self.early_stopping_patience):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            if epoch % 10 == 0:
                val_str = f", val_loss: {val_loss:.4f}" if val_loss is not None else ""
                logger.info(f"Epoch {epoch}: train_loss: {train_loss:.4f}{val_str}")

        self._is_fitted = True

        return TrainResult(
            model=self.model_,
            metrics={"train_loss": float(train_loss), "val_loss": float(best_val_loss) if val_loader else None},
            metadata={"epochs_trained": int(epoch) + 1, "history": history, "device": str(device)},
        )

    def predict(self, X: np.ndarray, device: Optional[str] = None) -> PredictResult:
        """Generate class predictions using the current threshold."""
        proba = self.predict_proba(X, device=device)
        preds = (proba.probabilities[:, 1] >= self._threshold).astype(int)
        return PredictResult(
            predictions=preds,
            metadata={'threshold': self._threshold}
        )

    def predict_proba(self, X: np.ndarray, device: Optional[str] = None) -> PredictResult:
        import torch

        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = (device.type == "cuda")

        self.model_.eval()
        self.model_.to(device)

        X_tensor = torch.from_numpy(X).float().to(device, non_blocking=use_cuda)
        with torch.no_grad():
            logits = self.model_(X_tensor).view(-1)
            probas = torch.sigmoid(logits).detach().cpu().numpy()

        probas_2d = np.column_stack([1.0 - probas, probas])
        return PredictResult(predictions=None, probabilities=probas_2d)

    def save(self, path: str) -> None:
        import torch

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model_.state_dict(),
                "model_config": self.model_config,
                "input_dim": self.input_dim_,
                "threshold": self._threshold,
                "threshold_tuning_stats": self._threshold_tuning_stats,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        device_config: Optional[DeviceConfig] = None,
    ) -> "PyTorchMLPEstimator":
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        estimator = cls(checkpoint["model_config"], device_config)
        estimator.input_dim_ = int(checkpoint["input_dim"])

        estimator.model_ = _build_mlp(
            input_dim=estimator.input_dim_,
            hidden_dims=estimator.hidden_dims,
            output_dim=1,
            dropout=estimator.dropout,
            activation=estimator.activation,
        )
        estimator.model_.load_state_dict(checkpoint["model_state_dict"])
        estimator.model_.to(torch.device(device))
        estimator._threshold = checkpoint.get("threshold", 0.5)
        estimator._threshold_tuning_stats = checkpoint.get("threshold_tuning_stats", None)
        estimator._is_fitted = True
        return estimator


# =============================================================================
# Logistic Regression Estimator
# =============================================================================

class PyTorchLogisticRegression(BaseEstimator):
    """PyTorch Logistic Regression for binary classification (single linear layer)."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ):
        super().__init__(model_config, device_config)

        params = model_config.get("params", {})
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.weight_decay = params.get("weight_decay", 1e-4)
        self.epochs = params.get("epochs", 100)
        self.batch_size = params.get("batch_size", 1024)
        self.early_stopping_patience = params.get("early_stopping_patience", 10)
        self.use_amp = bool(params.get("use_amp", True))
        self.incremental_chunk_size = params.get("incremental_chunk_size", 1_000_000)

        self.model_ = None
        self.input_dim_: Optional[int] = None

    def _get_gpu_device(self) -> str:
        dm = get_device_manager()
        gpu_id = self.device_config.gpu_id
        return dm.get_torch_device(gpu_id)

    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        if dm.check_cuda_available():
            return dm.get_torch_device()
        return "cpu"

    def fit_incremental_spark(
        self,
        spark_df: Any,
        label_col: str,
        feature_cols: List[str],
        preprocessor: Any = None,
        chunk_size: Optional[int] = None,
        eval_spark_df: Optional[Any] = None,
    ) -> TrainResult:
        """
        Incrementally train on Spark DataFrame in chunks.
        Avoids loading entire dataset into memory.

        Args:
            spark_df: Training Spark DataFrame
            label_col: Label column name
            feature_cols: Feature column names (raw, before preprocessing)
            preprocessor: Fitted preprocessor to apply per chunk
            chunk_size: Number of rows per chunk (default from config)
            eval_spark_df: Optional validation Spark DataFrame
        """
        import torch
        import torch.nn as nn
        from pyspark.sql import functions as F
        print('fitting incremental logistic regression')
        if chunk_size is None:
            chunk_size = self.incremental_chunk_size

        device = torch.device(self.get_device())
        use_cuda = (device.type == "cuda")

        logger.info(f"Training PyTorch Logistic Regression incrementally on device: {device}")
        logger.info(f"Chunk size: {chunk_size:,} rows")

        # Get total count and positive class weight from Spark (fast)
        stats = spark_df.agg(
            F.count("*").alias("total"),
            F.sum(F.col(label_col).cast("int")).alias("positives"),
        ).collect()[0]

        total_count = stats['total']
        pos_count = stats['positives']
        pos_weight = (total_count - pos_count) / max(pos_count, 1.0)
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

        logger.info(f"Total rows: {total_count:,}, Positive: {pos_count:,}, pos_weight: {pos_weight:.2f}")

        # Initialize model - need to determine input_dim after preprocessing
        # Load small sample to get preprocessed feature count
        sample_df = spark_df.limit(1000).select(feature_cols + [label_col]).toPandas()
        if preprocessor is not None:
            sample_processed = preprocessor.transform(sample_df[feature_cols].values)
            self.input_dim_ = sample_processed.shape[1]
        else:
            self.input_dim_ = len(feature_cols)

        self.model_ = nn.Linear(self.input_dim_, 1).to(device)
        logger.info(f"Model initialized with {self.input_dim_} features (after preprocessing)")

        # Loss function
        def weighted_bce_with_logits(logits, targets):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="mean", pos_weight=pos_weight_t
            )
            return loss

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and self.use_amp)

        # Calculate number of chunks
        num_chunks = (total_count + chunk_size - 1) // chunk_size
        logger.info(f"Training with {num_chunks} chunks over {self.epochs} epochs")

        # Training loop
        for epoch in range(int(self.epochs)):
            self.model_.train()
            epoch_loss_sum = 0.0
            epoch_samples = 0

            # Process chunks using Spark's native iteration
            # Add row_number to enable chunking
            from pyspark.sql import Window
            from pyspark.sql import functions as F

            df_with_idx = spark_df.withColumn("_idx", F.monotonically_increasing_id())

            # Iterate through chunks
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = start_idx + chunk_size

                # Filter chunk by index range
                chunk_df = df_with_idx.filter(
                    (F.col("_idx") >= start_idx) & (F.col("_idx") < end_idx)
                ).drop("_idx")

                # Convert to pandas (only this chunk)
                chunk_pdf = chunk_df.select(feature_cols + [label_col]).toPandas()

                if len(chunk_pdf) == 0:
                    break

                # Apply preprocessing to chunk
                if preprocessor is not None:
                    X_chunk_processed = preprocessor.transform(chunk_pdf[feature_cols].values)
                else:
                    X_chunk_processed = chunk_pdf[feature_cols].values.astype(np.float32)

                # Convert to tensors
                X_chunk = torch.from_numpy(X_chunk_processed.astype(np.float32)).to(device)
                y_chunk = torch.from_numpy(chunk_pdf[label_col].values.astype(np.float32)).view(-1, 1).to(device)

                # Train on chunk in mini-batches
                chunk_size_actual = len(X_chunk)
                for i in range(0, chunk_size_actual, self.batch_size):
                    end_idx = min(i + self.batch_size, chunk_size_actual)
                    X_batch = X_chunk[i:end_idx]
                    y_batch = y_chunk[i:end_idx]

                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        logits = self.model_(X_batch)
                        loss = weighted_bce_with_logits(logits, y_batch)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    batch_size_actual = len(X_batch)
                    epoch_loss_sum += float(loss.item()) * batch_size_actual
                    epoch_samples += batch_size_actual

                # Free memory
                del X_chunk, y_chunk, chunk_pdf
                if use_cuda:
                    torch.cuda.empty_cache()

                if (chunk_idx + 1) % max(1, num_chunks // 10) == 0:
                    curr_loss = epoch_loss_sum / max(epoch_samples, 1)
                    logger.info(f"Epoch {epoch}, Chunk {chunk_idx+1}/{num_chunks}, loss: {curr_loss:.4f}")

            avg_loss = epoch_loss_sum / max(epoch_samples, 1)
            logger.info(f"Epoch {epoch} complete: avg_loss={avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        self._is_fitted = True

        return TrainResult(
            model=self.model_,
            metrics={"train_loss": avg_loss},
            metadata={"input_dim": self.input_dim_, "total_samples": total_count},
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols=None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        device: Optional[str] = None,
    ) -> TrainResult:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        print('fitting logistic regression')
        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = (device.type == "cuda")

        logger.info(f"Training PyTorch Logistic Regression on device: {device}")

        self.input_dim_ = int(X.shape[1])

        # Simple logistic regression: just one linear layer
        self.model_ = nn.Linear(self.input_dim_, 1).to(device)

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float().view(-1, 1)

        w_t = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.ndim != 1 or sw.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must be shape (n_samples,)")
            w_t = torch.from_numpy(sw).float().view(-1, 1)

        train_dataset = TensorDataset(X_t, y_t) if w_t is None else TensorDataset(X_t, y_t, w_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
            drop_last=False,
        )

        val_loader = None
        if eval_set:
            X_val, y_val = eval_set[0]
            Xv = torch.from_numpy(X_val).float()
            yv = torch.from_numpy(y_val).float().view(-1, 1)
            val_loader = DataLoader(
                TensorDataset(Xv, yv),
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda,
                drop_last=False,
            )

        # pos_weight = neg/pos for class imbalance
        y_np = np.asarray(y, dtype=np.float32)
        n_pos = float(y_np.sum())
        n_neg = float(len(y_np) - n_pos)
        pos_weight = n_neg / max(n_pos, 1.0)
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

        def weighted_bce_with_logits(logits, targets, weights=None):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pos_weight_t
            )
            if weights is not None:
                loss = loss * weights
                return loss.sum() / (weights.sum().clamp_min(1e-12))
            return loss.mean()

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and self.use_amp)

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(int(self.epochs)):
            self.model_.train()
            train_loss_sum = 0.0
            n_seen = 0

            for batch in train_loader:
                if w_t is None:
                    batch_X, batch_y = batch
                    batch_w = None
                else:
                    batch_X, batch_y, batch_w = batch

                batch_X = batch_X.to(device, non_blocking=use_cuda)
                batch_y = batch_y.to(device, non_blocking=use_cuda)
                if batch_w is not None:
                    batch_w = batch_w.to(device, non_blocking=use_cuda)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = self.model_(batch_X)
                    loss = weighted_bce_with_logits(logits, batch_y, batch_w)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = batch_X.size(0)
                train_loss_sum += float(loss.item()) * bs
                n_seen += bs

            train_loss = train_loss_sum / max(n_seen, 1)
            history["train_loss"].append(train_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            lr_now = float(optimizer.param_groups[0]["lr"])
            mlflow.log_metric("lr", lr_now, step=epoch)

            val_loss = None
            if val_loader:
                self.model_.eval()
                val_loss_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_X, batch_y = batch
                        batch_X = batch_X.to(device, non_blocking=use_cuda)
                        batch_y = batch_y.to(device, non_blocking=use_cuda)
                        logits = self.model_(batch_X)
                        loss = weighted_bce_with_logits(logits, batch_y, None)
                        bs = batch_X.size(0)
                        val_loss_sum += float(loss.item()) * bs
                        val_n += bs
                val_loss = val_loss_sum / max(val_n, 1)
                history["val_loss"].append(val_loss)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}"
                logger.info(msg)

        self._is_fitted = True

        metrics = {
            "train_loss": history["train_loss"][-1],
            "final_lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_loss is not None:
            metrics["val_loss"] = history["val_loss"][-1]

        return TrainResult(
            model=self.model_,
            metrics=metrics,
            metadata={"history": history, "input_dim": self.input_dim_},
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
        import torch

        device = next(self.model_.parameters()).device
        self.model_.eval()

        X_t = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            logits = self.model_(X_t)
            p1 = torch.sigmoid(logits).cpu().numpy().ravel()

        probas_2d = np.column_stack([1 - p1, p1])
        return PredictResult(predictions=None, probabilities=probas_2d)

    def save(self, path: str) -> None:
        import torch

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model_.state_dict(),
                "model_config": self.model_config,
                "input_dim": self.input_dim_,
                "threshold": self._threshold,
                "threshold_tuning_stats": self._threshold_tuning_stats,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        device_config: Optional[DeviceConfig] = None,
    ) -> "PyTorchLogisticRegression":
        import torch
        import torch.nn as nn

        checkpoint = torch.load(path, map_location="cpu")
        estimator = cls(checkpoint["model_config"], device_config)
        estimator.input_dim_ = int(checkpoint["input_dim"])

        estimator.model_ = nn.Linear(estimator.input_dim_, 1)
        estimator.model_.load_state_dict(checkpoint["model_state_dict"])
        estimator.model_.to(torch.device(device))
        estimator._threshold = checkpoint.get("threshold", 0.5)
        estimator._threshold_tuning_stats = checkpoint.get("threshold_tuning_stats", None)
        estimator._is_fitted = True
        return estimator


# =============================================================================
# Registration
# =============================================================================

def register_pytorch_backend():
    """Register PyTorch backend components with the factory."""
    BackendFactory.register_preprocessor(BackendType.PYTORCH, PyTorchPreprocessor)
    BackendFactory.register_estimator(BackendType.PYTORCH, "mlp", PyTorchMLPEstimator)
    BackendFactory.register_estimator(BackendType.PYTORCH, "logistic_regression", PyTorchLogisticRegression)
    BackendFactory.register_data_loader(BackendType.PYTORCH, PyTorchDataLoader)
    logger.info("PyTorch backend registered")
