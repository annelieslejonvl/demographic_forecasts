"""
PyTorch backend implementation.
Supports CPU and CUDA GPU training for neural network models.
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
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> TrainResult:
        """Fit the MLP model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        device = self.get_device()
        logger.info(f"Training PyTorch MLP on device: {device}")
        
        self.input_dim_ = X.shape[1]
        
        # Build model
        self.model_ = _build_mlp(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            dropout=self.dropout,
            activation=self.activation,
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([sample_weight.mean()]) if sample_weight is not None else None
        ).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        # Data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=(device != "cpu"),
        )
        
        # Validation set
        val_loader = None
        if eval_set:
            X_val, y_val = eval_set[0]
            val_dataset = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float(),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2)
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_loader:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        outputs = self.model_(batch_X).squeeze()
                        val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
                
                val_loss /= len(val_loader.dataset)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                val_str = f", val_loss: {val_loss:.4f}" if val_loader else ""
                logger.info(f"Epoch {epoch}: train_loss: {train_loss:.4f}{val_str}")
        
        self._is_fitted = True
        
        return TrainResult(
            model=self.model_,
            metrics={"train_loss": train_loss, "val_loss": best_val_loss},
            metadata={"epochs_trained": epoch + 1, "history": history},
        )
    
    def predict(self, X: np.ndarray) -> PredictResult:
        """Generate class predictions."""
        proba = self.predict_proba(X)
        predictions = (proba.probabilities >= 0.5).astype(int)
        return PredictResult(predictions=predictions)
    
    def predict_proba(self, X: np.ndarray) -> PredictResult:
        """Generate probability predictions."""
        import torch
        
        device = self.get_device()
        self.model_.eval()
        
        X_tensor = torch.from_numpy(X).float().to(device)
        
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze()
            probas = torch.sigmoid(logits).cpu().numpy()
        
        # Return as 2D array [p(0), p(1)]
        probas_2d = np.column_stack([1 - probas, probas])
        
        return PredictResult(
            predictions=None,
            probabilities=probas_2d,
        )
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model_.state_dict(),
            "model_config": self.model_config,
            "input_dim": self.input_dim_,
        }, path)
    
    @classmethod
    def load(
        cls,
        path: str,
        device_config: Optional[DeviceConfig] = None,
    ) -> "PyTorchMLPEstimator":
        """Load model from disk."""
        import torch
        
        checkpoint = torch.load(path, map_location="cpu")
        
        estimator = cls(checkpoint["model_config"], device_config)
        estimator.input_dim_ = checkpoint["input_dim"]
        
        estimator.model_ = _build_mlp(
            input_dim=estimator.input_dim_,
            hidden_dims=estimator.hidden_dims,
            output_dim=1,
            dropout=estimator.dropout,
            activation=estimator.activation,
        )
        estimator.model_.load_state_dict(checkpoint["model_state_dict"])
        estimator.model_.to(estimator.get_device())
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
