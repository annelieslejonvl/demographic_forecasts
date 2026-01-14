"""
Specification classes for ML pipelines.

DataSpec: Wraps FeatureConfig for backward compatibility with run.py
ModelSpec: Model configuration (backend, type, params, preprocessing)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .features import FeatureConfig, FeatureResolver


@dataclass
class DataSpec:
    """
    Data specification - defines features, label, and column types.
    
    This is a wrapper around FeatureConfig that provides:
    - Backward compatibility with run.py
    - Resolution against DataFrame
    - Easy access to resolved columns
    """
    
    label_col: str = "label"
    drop_cols: List[str] = field(default_factory=list)
    key_cols: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)
    weight_col: Optional[str] = None
    
    # Resolved state
    _resolved: bool = False
    _resolver: Optional[FeatureResolver] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataSpec":
        """Create from dictionary (YAML config)."""
        return cls(
            label_col=d.get("label_col", "label"),
            drop_cols=d.get("drop_cols", []),
            key_cols=d.get("key_cols", []),
            cat_cols=d.get("cat_cols", []),
            num_cols=d.get("num_cols", []),
            weight_col=d.get("weight_col"),
        )
    
    def to_feature_config(self) -> FeatureConfig:
        """Convert to FeatureConfig."""
        return FeatureConfig(
            label_col=self.label_col,
            drop_cols=self.drop_cols,
            key_cols=self.key_cols,
            cat_cols=self.cat_cols,
            num_cols=self.num_cols,
            weight_col=self.weight_col,
            auto_detect_remaining=False,
        )
    
    def resolve(self, df) -> "DataSpec":
        """Resolve specification against DataFrame."""
        feature_config = self.to_feature_config()
        self._resolver = FeatureResolver(feature_config).resolve(df)
        self._resolved = True
        return self
    
    @property
    def feature_cols(self) -> List[str]:
        """Get all feature columns (cat + num)."""
        if self._resolved and self._resolver:
            return self._resolver.feature_cols
        return self.cat_cols + self.num_cols
    
    @property
    def categorical_cols(self) -> List[str]:
        """Get categorical columns."""
        if self._resolved and self._resolver:
            return self._resolver.categorical_cols
        return self.cat_cols
    
    @property
    def numeric_cols(self) -> List[str]:
        """Get numeric columns."""
        if self._resolved and self._resolver:
            return self._resolver.numeric_cols
        return self.num_cols


@dataclass
class ModelSpec:
    """
    Model specification - defines backend, model type, and parameters.
    """
    
    backend: str = "spark"  # spark, xgboost, pytorch
    model_type: str = "classifier"
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Preprocessing
    scaling_enabled: bool = False
    scaling_type: str = "standard"
    dim_reduction_enabled: bool = False
    dim_reduction_k: int = 50
    
    # Device
    device: str = "auto"
    
    # Training
    weight_col: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        """Create from dictionary (YAML config)."""
        model_cfg = d.get("model", {})
        scaling_cfg = d.get("scaling", {})
        dimred_cfg = d.get("dim_reduction", {})
        
        return cls(
            backend=d.get("backend", "spark"),
            model_type=model_cfg.get("type", "classifier"),
            params=model_cfg.get("params", {}),
            scaling_enabled=scaling_cfg.get("enabled", False),
            scaling_type=scaling_cfg.get("type", "standard"),
            dim_reduction_enabled=dimred_cfg.get("enabled", False),
            dim_reduction_k=dimred_cfg.get("k", 50),
            device=d.get("device", "auto"),
            weight_col=d.get("weight_col"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "device": self.device,
            "model": {
                "type": self.model_type,
                "params": self.params,
            },
            "scaling": {
                "enabled": self.scaling_enabled,
                "type": self.scaling_type,
            },
            "dim_reduction": {
                "enabled": self.dim_reduction_enabled,
                "k": self.dim_reduction_k,
            },
            "weight_col": self.weight_col,
        }
