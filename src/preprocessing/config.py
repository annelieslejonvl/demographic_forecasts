"""Preprocessing configuration classes."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ImputationStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    NONE = "none"


class EncodingStrategy(Enum):
    ONEHOT = "onehot"
    LABEL = "label"
    TARGET = "target"
    ORDINAL = "ordinal"
    NONE = "none"


class ScalingStrategy(Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


@dataclass
class ColumnConfig:
    """Configuration for a single column's preprocessing."""
    name: str
    dtype: str = "numeric"  # numeric, categorical, boolean
    imputation: ImputationStrategy = ImputationStrategy.NONE
    imputation_value: Optional[Any] = None
    encoding: EncodingStrategy = EncodingStrategy.NONE
    scaling: ScalingStrategy = ScalingStrategy.NONE
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnConfig":
        return cls(
            name=d["name"],
            dtype=d.get("dtype", "numeric"),
            imputation=ImputationStrategy(d.get("imputation", "none")),
            imputation_value=d.get("imputation_value"),
            encoding=EncodingStrategy(d.get("encoding", "none")),
            scaling=ScalingStrategy(d.get("scaling", "none")),
        )


@dataclass
class PreprocessingConfig:
    """Full preprocessing configuration."""
    
    columns: List[ColumnConfig] = field(default_factory=list)
    
    # Global defaults
    numeric_imputation: ImputationStrategy = ImputationStrategy.MEDIAN
    numeric_scaling: ScalingStrategy = ScalingStrategy.STANDARD
    categorical_encoding: EncodingStrategy = EncodingStrategy.ONEHOT
    categorical_imputation: ImputationStrategy = ImputationStrategy.MODE
    
    # Feature assembly
    output_col: str = "features"
    handle_unknown: str = "keep"
    
    # Dimensionality reduction
    pca_enabled: bool = False
    pca_k: int = 50
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessingConfig":
        columns = [ColumnConfig.from_dict(c) for c in d.get("columns", [])]
        return cls(
            columns=columns,
            numeric_imputation=ImputationStrategy(d.get("numeric_imputation", "median")),
            numeric_scaling=ScalingStrategy(d.get("numeric_scaling", "standard")),
            categorical_encoding=EncodingStrategy(d.get("categorical_encoding", "onehot")),
            categorical_imputation=ImputationStrategy(d.get("categorical_imputation", "mode")),
            output_col=d.get("output_col", "features"),
            handle_unknown=d.get("handle_unknown", "keep"),
            pca_enabled=d.get("pca", {}).get("enabled", False),
            pca_k=d.get("pca", {}).get("k", 50),
        )
    
    @classmethod
    def auto_detect(
        cls,
        df,
        feature_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> "PreprocessingConfig":
        """Auto-detect column types and create config."""
        columns = []
        categorical_cols = categorical_cols or []
        numeric_cols = numeric_cols or []
        
        if hasattr(df, "schema"):
            from pyspark.sql.types import StringType, BooleanType
            
            for col_name in feature_cols:
                field = df.schema[col_name]
                
                if col_name in categorical_cols:
                    dtype = "categorical"
                elif col_name in numeric_cols:
                    dtype = "numeric"
                elif isinstance(field.dataType, StringType):
                    dtype = "categorical"
                elif isinstance(field.dataType, BooleanType):
                    dtype = "boolean"
                else:
                    dtype = "numeric"
                
                columns.append(ColumnConfig(name=col_name, dtype=dtype))
        else:
            import pandas as pd
            
            if isinstance(df, pd.DataFrame):
                for col_name in feature_cols:
                    if col_name in categorical_cols:
                        dtype = "categorical"
                    elif col_name in numeric_cols:
                        dtype = "numeric"
                    elif df[col_name].dtype == object or df[col_name].dtype.name == 'category':
                        dtype = "categorical"
                    elif df[col_name].dtype == bool:
                        dtype = "boolean"
                    else:
                        dtype = "numeric"
                    
                    columns.append(ColumnConfig(name=col_name, dtype=dtype))
        
        return cls(columns=columns)
