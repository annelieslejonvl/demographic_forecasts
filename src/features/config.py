"""
Feature Configuration Module

Defines the feature set for ML pipelines:
- Label column
- Categorical columns
- Numeric columns  
- Columns to drop
- Auto-detection of remaining columns
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """
    Configuration for feature columns.
    
    Example YAML:
        label_col: y_moved
        drop_cols: [year, id]
        cat_cols: [age_group, nationality]
        num_cols: [income, tenure, score]
    """
    
    label_col: str = "label"
    
    # Explicitly defined columns
    cat_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)
    drop_cols: List[str] = field(default_factory=list)
    
    # Key columns (for joins, not features)
    key_cols: List[str] = field(default_factory=list)
    
    # Weight column (optional)
    weight_col: Optional[str] = None
    
    # Auto-detection settings
    auto_detect_remaining: bool = True
    treat_unknown_as: str = "numeric"  # "numeric" or "categorical"
    
    # Validation
    strict: bool = False  # If True, fail on missing columns
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure no overlap between column lists
        all_explicit = set(self.cat_cols) | set(self.num_cols) | set(self.drop_cols)
        
        if self.label_col in all_explicit:
            logger.warning(f"Label column '{self.label_col}' also appears in feature columns")
        
        cat_set = set(self.cat_cols)
        num_set = set(self.num_cols)
        overlap = cat_set & num_set
        if overlap:
            raise ValueError(f"Columns appear in both cat_cols and num_cols: {overlap}")
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureConfig":
        """Create config from dictionary (YAML-compatible)."""
        return cls(
            label_col=d.get("label_col", "label"),
            cat_cols=d.get("cat_cols", []),
            num_cols=d.get("num_cols", []),
            drop_cols=d.get("drop_cols", []),
            key_cols=d.get("key_cols", []),
            weight_col=d.get("weight_col"),
            auto_detect_remaining=d.get("auto_detect_remaining", True),
            treat_unknown_as=d.get("treat_unknown_as", "numeric"),
            strict=d.get("strict", False),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "FeatureConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "label_col": self.label_col,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "drop_cols": self.drop_cols,
            "key_cols": self.key_cols,
            "weight_col": self.weight_col,
            "auto_detect_remaining": self.auto_detect_remaining,
            "treat_unknown_as": self.treat_unknown_as,
            "strict": self.strict,
        }
    
    @property
    def feature_cols(self) -> List[str]:
        """Get all explicitly defined feature columns."""
        return self.cat_cols + self.num_cols
    
    @property
    def all_required_cols(self) -> List[str]:
        """Get all columns that must be present in the data."""
        cols = [self.label_col] + self.cat_cols + self.num_cols + self.key_cols
        if self.weight_col:
            cols.append(self.weight_col)
        return list(set(cols))


class FeatureResolver:
    """
    Resolves feature configuration against actual DataFrame schema.
    
    Handles:
    - Validation of configured columns
    - Auto-detection of remaining columns
    - Type inference for unknown columns
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.resolved_cat_cols: List[str] = []
        self.resolved_num_cols: List[str] = []
        self.resolved_drop_cols: List[str] = []
        self.all_columns: List[str] = []
        self._is_resolved = False
    
    def resolve(self, df) -> "FeatureResolver":
        """
        Resolve configuration against DataFrame.
        
        Args:
            df: Spark DataFrame or pandas DataFrame
        
        Returns:
            self (for chaining)
        """
        # Get all columns from DataFrame
        if hasattr(df, "columns"):
            self.all_columns = list(df.columns)
        else:
            raise ValueError("DataFrame must have 'columns' attribute")
        
        all_cols_set = set(self.all_columns)
        
        # Validate configured columns exist
        self._validate_columns(all_cols_set)
        
        # Start with explicitly configured columns
        self.resolved_cat_cols = list(self.config.cat_cols)
        self.resolved_num_cols = list(self.config.num_cols)
        self.resolved_drop_cols = list(self.config.drop_cols)
        
        # Columns to exclude from features
        exclude_cols = set(
            self.config.drop_cols + 
            self.config.key_cols + 
            [self.config.label_col]
        )
        if self.config.weight_col:
            exclude_cols.add(self.config.weight_col)
        
        # Already assigned columns
        assigned_cols = set(self.config.cat_cols) | set(self.config.num_cols)
        
        # Auto-detect remaining columns
        if self.config.auto_detect_remaining:
            remaining_cols = all_cols_set - exclude_cols - assigned_cols
            
            for col in remaining_cols:
                col_type = self._infer_column_type(df, col)
                
                if col_type == "categorical":
                    self.resolved_cat_cols.append(col)
                elif col_type == "numeric":
                    self.resolved_num_cols.append(col)
                else:
                    # Boolean or other - treat as numeric
                    self.resolved_num_cols.append(col)
        
        self._is_resolved = True
        
        logger.info(
            f"Resolved features: {len(self.resolved_cat_cols)} categorical, "
            f"{len(self.resolved_num_cols)} numeric, "
            f"{len(self.resolved_drop_cols)} dropped"
        )
        
        return self
    
    def _validate_columns(self, all_cols_set: Set[str]) -> None:
        """Validate that configured columns exist in DataFrame."""
        # Check label column
        if self.config.label_col not in all_cols_set:
            raise ValueError(f"Label column '{self.config.label_col}' not found in DataFrame")
        
        # Check configured feature columns
        missing_cat = set(self.config.cat_cols) - all_cols_set
        missing_num = set(self.config.num_cols) - all_cols_set
        missing_drop = set(self.config.drop_cols) - all_cols_set
        missing_key = set(self.config.key_cols) - all_cols_set
        
        all_missing = missing_cat | missing_num | missing_drop | missing_key
        
        if all_missing:
            msg = f"Configured columns not found in DataFrame: {all_missing}"
            if self.config.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                # Remove missing columns from config
                self.config.cat_cols = [c for c in self.config.cat_cols if c in all_cols_set]
                self.config.num_cols = [c for c in self.config.num_cols if c in all_cols_set]
                self.config.drop_cols = [c for c in self.config.drop_cols if c in all_cols_set]
                self.config.key_cols = [c for c in self.config.key_cols if c in all_cols_set]
    
    def _infer_column_type(self, df, col: str) -> str:
        """Infer column type from DataFrame schema."""
        if hasattr(df, "schema"):
            # Spark DataFrame
            from pyspark.sql.types import (
                StringType, 
                BooleanType, 
                NumericType,
                IntegerType,
                LongType,
                FloatType,
                DoubleType,
            )
            
            field = df.schema[col]
            dtype = field.dataType
            
            if isinstance(dtype, StringType):
                return "categorical"
            elif isinstance(dtype, BooleanType):
                return "boolean"
            elif isinstance(dtype, (IntegerType, LongType, FloatType, DoubleType)):
                # Check cardinality for integers - might be encoded categorical
                if isinstance(dtype, (IntegerType, LongType)):
                    distinct_count = df.select(col).distinct().count()
                    if distinct_count <= 20:  # Low cardinality integer -> likely categorical
                        return "categorical"
                return "numeric"
            else:
                return self.config.treat_unknown_as
        else:
            # Pandas DataFrame
            dtype = df[col].dtype
            
            if dtype == object or dtype.name == 'category':
                return "categorical"
            elif dtype == bool:
                return "boolean"
            elif dtype.kind in ('i', 'u'):  # Integer types
                # Check cardinality
                if df[col].nunique() <= 20:
                    return "categorical"
                return "numeric"
            elif dtype.kind == 'f':  # Float types
                return "numeric"
            else:
                return self.config.treat_unknown_as
    
    @property
    def feature_cols(self) -> List[str]:
        """Get all resolved feature columns."""
        if not self._is_resolved:
            raise RuntimeError("Must call resolve() first")
        return self.resolved_cat_cols + self.resolved_num_cols
    
    @property
    def categorical_cols(self) -> List[str]:
        """Get resolved categorical columns."""
        if not self._is_resolved:
            raise RuntimeError("Must call resolve() first")
        return self.resolved_cat_cols
    
    @property
    def numeric_cols(self) -> List[str]:
        """Get resolved numeric columns."""
        if not self._is_resolved:
            raise RuntimeError("Must call resolve() first")
        return self.resolved_num_cols
    
    def get_select_cols(self, include_label: bool = True, include_weight: bool = True) -> List[str]:
        """Get columns to select from DataFrame for modeling."""
        if not self._is_resolved:
            raise RuntimeError("Must call resolve() first")
        
        cols = self.feature_cols.copy()
        
        if include_label:
            cols.append(self.config.label_col)
        
        if include_weight and self.config.weight_col:
            cols.append(self.config.weight_col)
        
        return cols
    
    def select_features(self, df):
        """Select only feature columns from DataFrame."""
        if not self._is_resolved:
            raise RuntimeError("Must call resolve() first")
        
        select_cols = self.get_select_cols()
        
        if hasattr(df, "select"):
            return df.select(select_cols)
        else:
            return df[select_cols]
    
    def drop_columns(self, df):
        """Drop configured columns from DataFrame."""
        if hasattr(df, "drop"):
            # Spark
            return df.drop(*self.resolved_drop_cols)
        else:
            # Pandas
            return df.drop(columns=self.resolved_drop_cols, errors='ignore')
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of resolved configuration."""
        if not self._is_resolved:
            raise RuntimeError("Must call resolve() first")
        
        return {
            "label_col": self.config.label_col,
            "weight_col": self.config.weight_col,
            "n_categorical": len(self.resolved_cat_cols),
            "n_numeric": len(self.resolved_num_cols),
            "n_dropped": len(self.resolved_drop_cols),
            "categorical_cols": self.resolved_cat_cols,
            "numeric_cols": self.resolved_num_cols,
            "dropped_cols": self.resolved_drop_cols,
            "total_features": len(self.feature_cols),
        }
    
    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.summary()
        print("=" * 60)
        print("FEATURE CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Label column: {summary['label_col']}")
        print(f"Weight column: {summary['weight_col']}")
        print(f"Total features: {summary['total_features']}")
        print(f"  - Categorical: {summary['n_categorical']}")
        print(f"  - Numeric: {summary['n_numeric']}")
        print(f"Dropped columns: {summary['n_dropped']}")
        print("=" * 60)


def resolve_features(
    df,
    config: FeatureConfig,
) -> FeatureResolver:
    """
    Convenience function to resolve features.
    
    Args:
        df: Spark or pandas DataFrame
        config: FeatureConfig instance
    
    Returns:
        Resolved FeatureResolver
    """
    resolver = FeatureResolver(config)
    return resolver.resolve(df)


def load_feature_config(path: str) -> FeatureConfig:
    """Load feature configuration from YAML file."""
    return FeatureConfig.from_yaml(path)
