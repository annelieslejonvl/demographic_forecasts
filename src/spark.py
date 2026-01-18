"""Spark ML preprocessing pipeline."""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .config import (
    ColumnConfig,
    EncodingStrategy,
    ImputationStrategy,
    PreprocessingConfig,
    ScalingStrategy,
)

if TYPE_CHECKING:
    from ..features import FeatureConfig, FeatureResolver

logger = logging.getLogger(__name__)


class SparkPreprocessor:
    """
    Spark ML preprocessing pipeline builder.
    
    Handles:
    - StringIndexer for categorical columns
    - OneHotEncoder for indexed categoricals
    - Imputer for missing values
    - StandardScaler/MinMaxScaler for numeric columns
    - VectorAssembler to combine all features
    
    Can be initialized with:
    - PreprocessingConfig only (column types auto-detected)
    - PreprocessingConfig + FeatureConfig (explicit column definitions)
    - PreprocessingConfig + FeatureResolver (pre-resolved columns)
    """
    
    def __init__(
        self, 
        config: PreprocessingConfig,
        feature_config: Optional["FeatureConfig"] = None,
    ):
        self.config = config
        self.feature_config = feature_config
        self.pipeline_ = None
        self.pipeline_model_ = None
        self.feature_cols_: List[str] = []
        self.indexed_cols_: List[str] = []
        self.encoded_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.output_feature_cols_: List[str] = []
        self._column_mappings: Dict[str, Any] = {}
    
    @classmethod
    def from_feature_config(
        cls,
        feature_config: "FeatureConfig",
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ) -> "SparkPreprocessor":
        """Create preprocessor from FeatureConfig."""
        return cls(
            config=preprocessing_config or PreprocessingConfig(),
            feature_config=feature_config,
        )
    
    def _resolve_columns(
        self,
        df,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Resolve categorical and numeric columns.
        
        Priority:
        1. FeatureConfig if provided
        2. PreprocessingConfig.columns if provided
        3. Auto-detect from DataFrame
        """
        if self.feature_config is not None:
            # Use FeatureConfig
            from ..features import FeatureResolver
            resolver = FeatureResolver(self.feature_config).resolve(df)
            return resolver.categorical_cols, resolver.numeric_cols
        
        elif self.config.columns:
            # Use PreprocessingConfig column definitions
            col_config_map = {c.name: c for c in self.config.columns}
            cat_cols = []
            num_cols = []
            
            cols_to_check = feature_cols or list(col_config_map.keys())
            
            for col in cols_to_check:
                cfg = col_config_map.get(col, ColumnConfig(name=col))
                if cfg.dtype == "categorical":
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
            
            return cat_cols, num_cols
        
        else:
            # Auto-detect from DataFrame schema
            config = PreprocessingConfig.auto_detect(df, feature_cols or [])
            col_config_map = {c.name: c for c in config.columns}
            
            cat_cols = [c.name for c in config.columns if c.dtype == "categorical"]
            num_cols = [c.name for c in config.columns if c.dtype == "numeric"]
            
            return cat_cols, num_cols
    
    def build_pipeline(
        self,
        df,
        feature_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Build Spark ML preprocessing pipeline.
        
        Args:
            df: Spark DataFrame
            feature_cols: Optional list of feature columns. If None, uses FeatureConfig.
            label_col: Optional label column. If None, uses FeatureConfig.
        """
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            StringIndexer,
            OneHotEncoder,
            Imputer,
            StandardScaler,
            MinMaxScaler,
            RobustScaler,
            VectorAssembler,
            PCA,
        )
        
        # Resolve label column
        if label_col is None and self.feature_config is not None:
            label_col = self.feature_config.label_col
        elif label_col is None:
            label_col = "label"
        
        # Resolve feature columns
        if feature_cols is None and self.feature_config is not None:
            categorical_cols, numeric_cols = self._resolve_columns(df)
            feature_cols = categorical_cols + numeric_cols
        elif feature_cols is not None:
            categorical_cols, numeric_cols = self._resolve_columns(df, feature_cols)
        else:
            categorical_cols, numeric_cols = [], []
        
        self.categorical_cols_ = categorical_cols
        self.numeric_cols_ = numeric_cols
        self.feature_cols_ = feature_cols
        
        stages = []
        assembled_cols = []
        boolean_cols = []
        
        # Build column config map for per-column settings
        col_config_map = {c.name: c for c in self.config.columns}
        
        # Imputation for numeric columns
        if numeric_cols:
            impute_strategy = self.config.numeric_imputation
            
            if impute_strategy != ImputationStrategy.NONE:
                imputed_cols = [f"{c}_imputed" for c in numeric_cols]
                
                imputer = Imputer(
                    inputCols=numeric_cols,
                    outputCols=imputed_cols,
                    strategy=impute_strategy.value if impute_strategy != ImputationStrategy.CONSTANT else "mean",
                )
                stages.append(imputer)
                numeric_cols_processed = imputed_cols
            else:
                numeric_cols_processed = numeric_cols
        else:
            numeric_cols_processed = []
        
        # Categorical encoding
        indexed_cols = []
        encoded_cols = []
        
        for col in categorical_cols:
            cfg = col_config_map.get(col, ColumnConfig(name=col, dtype="categorical"))
            encoding = cfg.encoding if cfg.encoding != EncodingStrategy.NONE else self.config.categorical_encoding
            
            indexed_col = f"{col}_indexed"
            indexer = StringIndexer(
                inputCol=col,
                outputCol=indexed_col,
                handleInvalid=self.config.handle_unknown,
            )
            stages.append(indexer)
            indexed_cols.append(indexed_col)
            
            if encoding == EncodingStrategy.ONEHOT:
                encoded_col = f"{col}_encoded"
                encoder = OneHotEncoder(
                    inputCol=indexed_col,
                    outputCol=encoded_col,
                    handleInvalid=self.config.handle_unknown,
                )
                stages.append(encoder)
                encoded_cols.append(encoded_col)
        
        self.indexed_cols_ = indexed_cols
        self.encoded_cols_ = encoded_cols
        
        # Assemble and scale numeric features
        if numeric_cols_processed:
            numeric_assembled = "numeric_features"
            numeric_assembler = VectorAssembler(
                inputCols=numeric_cols_processed,
                outputCol=numeric_assembled,
                handleInvalid="keep",
            )
            stages.append(numeric_assembler)
            
            scaling = self.config.numeric_scaling
            if scaling == ScalingStrategy.STANDARD:
                scaled_col = "numeric_scaled"
                scaler = StandardScaler(
                    inputCol=numeric_assembled,
                    outputCol=scaled_col,
                    withMean=True,
                    withStd=True,
                )
                stages.append(scaler)
                assembled_cols.append(scaled_col)
            elif scaling == ScalingStrategy.MINMAX:
                scaled_col = "numeric_scaled"
                scaler = MinMaxScaler(inputCol=numeric_assembled, outputCol=scaled_col)
                stages.append(scaler)
                assembled_cols.append(scaled_col)
            elif scaling == ScalingStrategy.ROBUST:
                scaled_col = "numeric_scaled"
                scaler = RobustScaler(inputCol=numeric_assembled, outputCol=scaled_col)
                stages.append(scaler)
                assembled_cols.append(scaled_col)
            else:
                assembled_cols.append(numeric_assembled)
        
        # Add encoded categoricals
        if encoded_cols:
            assembled_cols.extend(encoded_cols)
        elif indexed_cols and self.config.categorical_encoding in (EncodingStrategy.LABEL, EncodingStrategy.NATIVE):
            assembled_cols.extend(indexed_cols)
        
        if boolean_cols:
            assembled_cols.extend(boolean_cols)
        
        self.output_feature_cols_ = assembled_cols
        
        # Final assembly
        final_assembler = VectorAssembler(
            inputCols=assembled_cols,
            outputCol="assembled_features",
            handleInvalid="keep",
        )
        stages.append(final_assembler)
        
        # Optional PCA
        if self.config.pca_enabled:
            pca = PCA(
                inputCol="assembled_features",
                outputCol=self.config.output_col,
                k=self.config.pca_k,
            )
            stages.append(pca)
        
        output_col = self.config.output_col if self.config.pca_enabled else "assembled_features"
        
        self.pipeline_ = Pipeline(stages=stages)
        
        info = {
            "features_col": output_col,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "indexed_cols": indexed_cols,
            "encoded_cols": encoded_cols,
            "n_stages": len(stages),
        }
        
        return self.pipeline_, info
    
    def fit(self, df, feature_cols: List[str], label_col: str) -> "SparkPreprocessor":
        """Fit the preprocessing pipeline."""
        pipeline, info = self.build_pipeline(df, feature_cols, label_col)
        self.pipeline_model_ = pipeline.fit(df)
        self._column_mappings = info
        return self
    
    def transform(self, df) -> Any:
        """Transform data using fitted pipeline."""
        if self.pipeline_model_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return self.pipeline_model_.transform(df)
    
    def fit_transform(self, df, feature_cols: List[str], label_col: str) -> Any:
        """Fit and transform in one step."""
        self.fit(df, feature_cols, label_col)
        return self.transform(df)
    
    def get_output_col(self) -> str:
        """Get the output features column name."""
        return self.config.output_col if self.config.pca_enabled else "assembled_features"
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        names = []
        names.extend([f"{c}_scaled" for c in self.numeric_cols_])
        names.extend(self.encoded_cols_ or self.indexed_cols_)
        return names
    
    def save(self, path: str) -> None:
        """Save fitted pipeline."""
        os.makedirs(path, exist_ok=True)
        self.pipeline_model_.write().overwrite().save(os.path.join(path, "pipeline"))
        
        meta = {
            "feature_cols": self.feature_cols_,
            "numeric_cols": self.numeric_cols_,
            "indexed_cols": self.indexed_cols_,
            "encoded_cols": self.encoded_cols_,
            "output_col": self.get_output_col(),
            "config": {
                "numeric_imputation": self.config.numeric_imputation.value,
                "numeric_scaling": self.config.numeric_scaling.value,
                "categorical_encoding": self.config.categorical_encoding.value,
                "pca_enabled": self.config.pca_enabled,
                "pca_k": self.config.pca_k,
            },
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SparkPreprocessor":
        """Load fitted pipeline."""
        from pyspark.ml import PipelineModel
        
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)
        
        config = PreprocessingConfig(
            numeric_imputation=ImputationStrategy(meta["config"]["numeric_imputation"]),
            numeric_scaling=ScalingStrategy(meta["config"]["numeric_scaling"]),
            categorical_encoding=EncodingStrategy(meta["config"]["categorical_encoding"]),
            pca_enabled=meta["config"]["pca_enabled"],
            pca_k=meta["config"]["pca_k"],
        )
        
        preprocessor = cls(config)
        preprocessor.pipeline_model_ = PipelineModel.load(os.path.join(path, "pipeline"))
        preprocessor.feature_cols_ = meta["feature_cols"]
        preprocessor.numeric_cols_ = meta["numeric_cols"]
        preprocessor.indexed_cols_ = meta["indexed_cols"]
        preprocessor.encoded_cols_ = meta["encoded_cols"]
        
        return preprocessor
