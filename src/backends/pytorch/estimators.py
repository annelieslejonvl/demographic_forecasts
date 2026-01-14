"""
PyTorch backend implementation.
Supports CPU and CUDA GPU training for neural network models.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score
import numpy as np
import mlflow
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
        import torch
        
        # Select relevant columns and convert to Pandas
        pdf = spark_df.select(feature_cols + [label_col]).toPandas()
        
        X = pdf[feature_cols].values.astype(np.float32)
        y = pdf[label_col].values.astype(np.float32)
        
        return X, y
    
    def to_spark(self, X: np.ndarray, y: np.ndarray, spark_session: Any) -> Any:
        """Convert numpy arrays to Spark DataFrame."""
        import pandas as pd
        from pyspark.sql import SparkSession
        
        # Create DataFrame with feature columns
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
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
        
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        if sample_weight is not None:
            w_tensor = torch.from_numpy(sample_weight).float()
            dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


# =============================================================================
# Preprocessor
# =============================================================================

class PyTorchPreprocessor(BasePreprocessor):
    """Preprocessor for PyTorch backend using sklearn transformers."""
    
    def __init__(
        self,
        scaling: str = "standard",  # "standard", "minmax", "none"
        dim_reduction: Optional[Dict[str, Any]] = None,
    ):
        self.scaling = scaling
        self.dim_reduction = dim_reduction or {}
        self.scaler_ = None
        self.pca_ = None
        self.feature_cols_: Optional[List[str]] = None
    
    def fit(
        self,
        data: Union[np.ndarray, Any],
        label_col: str,
        feature_cols: List[str],
    ) -> "PyTorchPreprocessor":
        """Fit preprocessing pipeline."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA
        
        # Convert Spark DF to numpy if needed
        if hasattr(data, "toPandas"):
            X = data.select(feature_cols).toPandas().values
        else:
            X = data
        
        self.feature_cols_ = feature_cols
        
        # Scaling
        if self.scaling == "standard":
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        elif self.scaling == "minmax":
            self.scaler_ = MinMaxScaler()
            X = self.scaler_.fit_transform(X)
        
        # Dimensionality reduction
        if self.dim_reduction.get("enabled", False):
            n_components = self.dim_reduction.get("k", min(50, X.shape[1]))
            self.pca_ = PCA(n_components=n_components)
            self.pca_.fit(X)
        
        return self
    
    def transform(self, data: Union[np.ndarray, Any]) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if hasattr(data, "toPandas"):
            X = data.select(self.feature_cols_).toPandas().values
        else:
            X = data
        
        X = X.astype(np.float32)
        
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        if self.pca_ is not None:
            X = self.pca_.transform(X)
        
        return X.astype(np.float32)
    
    def fit_transform(self, data, label_col: str, feature_cols: List[str]) -> np.ndarray:
        # 1x naar numpy
        if hasattr(data, "toPandas"):
            X = data.select(feature_cols).toPandas().to_numpy(dtype=np.float32, copy=True)
        else:
            X = np.asarray(data, dtype=np.float32)

        self.feature_cols_ = feature_cols

        # scaling
        if self.scaling == "standard":
            from sklearn.preprocessing import StandardScaler
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X).astype(np.float32, copy=False)
        elif self.scaling == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_ = MinMaxScaler()
            X = self.scaler_.fit_transform(X).astype(np.float32, copy=False)

        # pca
        if self.dim_reduction.get("enabled", False):
            from sklearn.decomposition import PCA
            n_components = self.dim_reduction.get("k", min(50, X.shape[1]))
            self.pca_ = PCA(n_components=n_components)
            X = self.pca_.fit_transform(X).astype(np.float32, copy=False)

        return X

    
    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler_,
            "pca": self.pca_,
            "feature_cols": self.feature_cols_,
            "scaling": self.scaling,
            "dim_reduction": self.dim_reduction,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "PyTorchPreprocessor":
        """Load preprocessor from disk."""
        import joblib
        data = joblib.load(path)
        preprocessor = cls(
            scaling=data["scaling"],
            dim_reduction=data["dim_reduction"],
        )
        preprocessor.scaler_ = data["scaler"]
        preprocessor.pca_ = data["pca"]
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
    import torch
    import torch.nn as nn
    
    activation_fn = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }.get(activation, nn.ReLU)
    
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            activation_fn(),
            nn.Dropout(dropout),
        ])
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
        
        # Model architecture
        params = model_config.get("params", {})
        self.hidden_dims = params.get("hidden_dims", [128, 64])
        self.dropout = params.get("dropout", 0.2)
        self.activation = params.get("activation", "relu")
        
        # Training params
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.weight_decay = params.get("weight_decay", 1e-4)
        self.epochs = params.get("epochs", 100)
        self.batch_size = params.get("batch_size", 256)
        self.early_stopping_patience = params.get("early_stopping_patience", 10)
        
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
        device: Optional[str] = None,   # NEW: "cpu" | "cuda" | "cuda:0" | etc. (None -> self.get_device())
    ) -> TrainResult:
        """
        Fit the MLP model on CPU or GPU (controlled by `device`).
        - Supports AMP on CUDA
        - Correct pos_weight handling for BCEWithLogitsLoss
        - Handles sample_weight (per-example) via weighted BCE
        - Fixes missing lr logging
        """
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # ---- device
        if device is None:
            device = self.get_device()  # should return "cpu" or "cuda"
        device = torch.device(device)
        use_cuda = (device.type == "cuda")

        logger.info(f"Training PyTorch MLP on device: {device}")

        self.input_dim_ = int(X.shape[1])

        # ---- model
        self.model_ = _build_mlp(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            dropout=self.dropout,
            activation=self.activation,
        ).to(device)

        # ---- tensors
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float().view(-1, 1)

        # sample weights (per-row), optional
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

        # ---- validation
        val_loader = None
        if eval_set:
            X_val, y_val = eval_set[0]
            Xv = torch.from_numpy(X_val).float()
            yv = torch.from_numpy(y_val).float().view(-1, 1)
            val_dataset = TensorDataset(Xv, yv)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda,
                drop_last=False,
            )

        # ---- imbalance handling (pos_weight)
        # Correct: pos_weight should be (neg/pos). Do NOT use sample_weight.mean().
        y_np = np.asarray(y).astype(np.float32)
        n_pos = float(y_np.sum())
        n_neg = float(len(y_np) - n_pos)
        pos_weight = (n_neg / max(n_pos, 1.0))
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

        # We'll compute BCE with logits manually to support per-example weights + pos_weight.
        # This matches BCEWithLogitsLoss(pos_weight=...) when sample_weight is None.
        def weighted_bce_with_logits(logits, targets, weights=None):
            # logits/targets shape: (B,1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pos_weight_t
            )  # (B,1)
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

        # AMP only on CUDA
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and bool(getattr(self, "use_amp", True)))

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(int(self.epochs)):
            # ---- train
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
                    logits = self.model_(batch_X)  # (B,1)
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

            # log LR
            lr_now = float(optimizer.param_groups[0]["lr"])
            mlflow.log_metric("lr", lr_now, step=epoch)

            # ---- validate
            val_loss = None
            if val_loader is not None:
                from sklearn.metrics import roc_auc_score

                self.model_.eval()
                val_loss_sum = 0.0
                n_val = 0
                y_true = []
                y_pred = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device, non_blocking=use_cuda)
                        batch_y = batch_y.to(device, non_blocking=use_cuda)

                        logits = self.model_(batch_X)  # (B,1)
                        loss = weighted_bce_with_logits(logits, batch_y, weights=None)

                        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
                        y_pred.append(probs)
                        y_true.append(batch_y.detach().cpu().numpy().ravel())

                        bs = batch_X.size(0)
                        val_loss_sum += float(loss.item()) * bs
                        n_val += bs

                val_loss = val_loss_sum / max(n_val, 1)
                history["val_loss"].append(val_loss)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                # AUC only if both classes present
                if np.unique(y_true).size > 1:
                    auc = roc_auc_score(y_true, y_pred)
                    mlflow.log_metric("val_auc", float(auc), step=epoch)

                scheduler.step(val_loss)

                # early stopping
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
        proba = self.predict_proba(X, device=device)
        preds = (proba.probabilities[:, 1] >= 0.5).astype(int)
        return PredictResult(predictions=preds)
        
    def predict_proba(
        self,
        X: np.ndarray,
        device: Optional[str] = None,  # NEW
    ) -> PredictResult:
        """Generate probability predictions on CPU or GPU."""
        import numpy as np
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
        import os
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model_.state_dict(),
                "model_config": self.model_config,
                "input_dim": self.input_dim_,
            },
            path,
        )


    

    @classmethod
    def load(cls, path: str, device: str = "cpu", device_config: Optional[DeviceConfig] = None) -> "PyTorchMLPEstimator":
        """
        Load model; `device` controls where the model is placed ("cpu" or "cuda").
        """
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
        estimator._is_fitted = True
        return estimator

# =============================================================================
# Registration
# =============================================================================

def register_pytorch_backend():
    """Register PyTorch backend components with the factory."""
    BackendFactory.register_preprocessor(BackendType.PYTORCH, PyTorchPreprocessor)
    BackendFactory.register_estimator(BackendType.PYTORCH, "mlp", PyTorchMLPEstimator)
    BackendFactory.register_data_loader(BackendType.PYTORCH, PyTorchDataLoader)
    logger.info("PyTorch backend registered")
