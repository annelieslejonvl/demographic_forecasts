"""Sklearn-based preprocessing for PyTorch and XGBoost backends."""
import json
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .config import (
    EncodingStrategy,
    ImputationStrategy,
    PreprocessingConfig,
    ScalingStrategy,
)

if TYPE_CHECKING:
    from ..features import FeatureConfig

logger = logging.getLogger(__name__)


class SklearnPreprocessor:
    """
    Sklearn-based preprocessor for PyTorch and XGBoost backends.
    
    Handles:
    - SimpleImputer for missing values
    - OneHotEncoder / OrdinalEncoder for categoricals
    - StandardScaler / MinMaxScaler for numerics
    - ColumnTransformer to combine all
    
    Can be initialized with:
    - PreprocessingConfig only (column types must be specified in fit())
    - PreprocessingConfig + FeatureConfig (explicit column definitions)
    """
    
    def __init__(
        self, 
        config: PreprocessingConfig,
        feature_config: Optional["FeatureConfig"] = None,
    ):
        self.config = config
        self.feature_config = feature_config
        self.preprocessor_ = None
        self.feature_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.label_col_: Optional[str] = None
    
    @classmethod
    def from_feature_config(
        cls,
        feature_config: "FeatureConfig",
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ) -> "SklearnPreprocessor":
        """Create preprocessor from FeatureConfig."""
        return cls(
            config=preprocessing_config or PreprocessingConfig(),
            feature_config=feature_config,
        )
    
    def fit(
        self,
        X,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> "SklearnPreprocessor":
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: DataFrame or numpy array
            feature_cols: List of feature columns (optional if using FeatureConfig)
            categorical_cols: List of categorical columns (optional if using FeatureConfig)
            numeric_cols: List of numeric columns (optional if using FeatureConfig)
        """
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            StandardScaler,
            MinMaxScaler,
            RobustScaler,
            OneHotEncoder,
            OrdinalEncoder,
        )
        
        # Resolve columns from FeatureConfig if available
        if self.feature_config is not None:
            categorical_cols = categorical_cols or self.feature_config.cat_cols
            numeric_cols = numeric_cols or self.feature_config.num_cols
            feature_cols = feature_cols or (categorical_cols + numeric_cols)
            self.label_col_ = self.feature_config.label_col
        
        self.feature_cols_ = feature_cols or []
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_cols_)
        
        # Auto-detect column types if not provided
        if categorical_cols is None:
            categorical_cols = []
            for col in self.feature_cols_:
                if X[col].dtype == object or X[col].dtype.name == 'category':
                    categorical_cols.append(col)
        
        if numeric_cols is None:
            numeric_cols = [c for c in self.feature_cols_ if c not in categorical_cols]
        
        self.categorical_cols_ = categorical_cols
        self.numeric_cols_ = numeric_cols
        
        transformers = []
        
        # Numeric pipeline
        if numeric_cols:
            numeric_steps = []
            
            impute_strategy = self.config.numeric_imputation
            if impute_strategy != ImputationStrategy.NONE:
                if impute_strategy == ImputationStrategy.CONSTANT:
                    imputer = SimpleImputer(strategy="constant", fill_value=0)
                else:
                    imputer = SimpleImputer(strategy=impute_strategy.value)
                numeric_steps.append(("imputer", imputer))
            
            scaling = self.config.numeric_scaling
            if scaling == ScalingStrategy.STANDARD:
                numeric_steps.append(("scaler", StandardScaler()))
            elif scaling == ScalingStrategy.MINMAX:
                numeric_steps.append(("scaler", MinMaxScaler()))
            elif scaling == ScalingStrategy.ROBUST:
                numeric_steps.append(("scaler", RobustScaler()))
            
            if numeric_steps:
                numeric_pipeline = Pipeline(numeric_steps)
                transformers.append(("numeric", numeric_pipeline, numeric_cols))
            else:
                transformers.append(("numeric", "passthrough", numeric_cols))
        
        # Categorical pipeline
        if categorical_cols:
            categorical_steps = []
            
            cat_impute = self.config.categorical_imputation
            if cat_impute != ImputationStrategy.NONE:
                if cat_impute == ImputationStrategy.CONSTANT:
                    imputer = SimpleImputer(strategy="constant", fill_value="missing")
                else:
                    imputer = SimpleImputer(strategy="most_frequent")
                categorical_steps.append(("imputer", imputer))
            
            encoding = self.config.categorical_encoding
            if encoding == EncodingStrategy.ONEHOT:
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore" if self.config.handle_unknown == "keep" else "error",
                )
                categorical_steps.append(("encoder", encoder))
            elif encoding in (EncodingStrategy.ORDINAL, EncodingStrategy.LABEL):
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                categorical_steps.append(("encoder", encoder))
            
            if categorical_steps:
                categorical_pipeline = Pipeline(categorical_steps)
                transformers.append(("categorical", categorical_pipeline, categorical_cols))
        
        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        
        self.preprocessor_.fit(X[feature_cols])
        
        try:
            self.feature_names_out_ = list(self.preprocessor_.get_feature_names_out())
        except AttributeError:
            self.feature_names_out_ = feature_cols
        
        return self
    
    def transform(self, X) -> np.ndarray:
        """Transform data."""
        import pandas as pd
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_cols_)
        
        result = self.preprocessor_.transform(X[self.feature_cols_])
        return result.astype(np.float32)
    
    def fit_transform(
        self,
        X,
        feature_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, feature_cols, categorical_cols, numeric_cols)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get output feature names."""
        return self.feature_names_out_
    
    def save(self, path: str) -> None:
        """Save fitted preprocessor."""
        import joblib
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.preprocessor_, os.path.join(path, "preprocessor.joblib"))
        
        meta = {
            "feature_cols": self.feature_cols_,
            "numeric_cols": self.numeric_cols_,
            "categorical_cols": self.categorical_cols_,
            "feature_names_out": self.feature_names_out_,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path: str, config: Optional[PreprocessingConfig] = None) -> "SklearnPreprocessor":
        """Load fitted preprocessor."""
        import joblib
        
        preprocessor = cls(config or PreprocessingConfig())
        preprocessor.preprocessor_ = joblib.load(os.path.join(path, "preprocessor.joblib"))
        
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)
        
        preprocessor.feature_cols_ = meta["feature_cols"]
        preprocessor.numeric_cols_ = meta["numeric_cols"]
        preprocessor.categorical_cols_ = meta["categorical_cols"]
        preprocessor.feature_names_out_ = meta["feature_names_out"]
        
        return preprocessor
