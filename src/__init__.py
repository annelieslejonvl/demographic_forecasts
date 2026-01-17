"""
Multi-Backend ML Framework

A unified framework for running ML experiments across:
- PyTorch (neural networks)
- XGBoost (gradient boosting)
- Spark ML (distributed training)

Supports data from:
- Unity Catalog (Databricks)
- Parquet files (local, S3, ADLS, GCS)
- Delta tables
- CSV files

Preprocessing:
- Imputation (mean, median, mode, constant)
- Encoding (one-hot, label/string indexer, ordinal)
- Scaling (standard, minmax, robust)
- PCA dimensionality reduction

Sampling:
- Random, stratified, undersample, oversample
- Hard negative mining (top_prob, margin, entropy)

All backends support CPU and GPU acceleration.
"""
from .backends import (
    BackendFactory,
    BackendType,
    DeviceConfig,
    DeviceType,
    UnifiedPipeline,
    PipelineConfig,
    create_pipeline,
    initialize_backends,
)
from .data import (
    DataSource,
    DataSourceConfig,
    DataSourceRegistry,
    DataSourceType,
    load_from_unity_catalog,
    load_from_parquet,
    load_from_delta,
)
from .preprocessing import (
    ColumnConfig,
    PreprocessingConfig,
    SparkPreprocessor,
    SklearnPreprocessor,
    create_preprocessor,
    ImputationStrategy,
    EncodingStrategy,
    ScalingStrategy,
)
from .sampling import (
    SamplingConfig,
    SamplingStrategy,
    SparkSampler,
    NumpySampler,
    HardNegativeMiner,
    create_sampler,
    compute_class_weights,
)
from .features import (
    FeatureConfig,
    FeatureResolver,
    resolve_features,
    load_feature_config,
)
from .utils import get_device_manager

__version__ = "0.5.0"

__all__ = [
    # Backends
    "BackendFactory",
    "BackendType",
    "DeviceConfig",
    "DeviceType",
    "UnifiedPipeline",
    "PipelineConfig",
    "create_pipeline",
    "initialize_backends",
    # Data sources
    "DataSource",
    "DataSourceConfig",
    "DataSourceRegistry",
    "DataSourceType",
    "load_from_unity_catalog",
    "load_from_parquet",
    "load_from_delta",
    # Preprocessing
    "ColumnConfig",
    "PreprocessingConfig",
    "SparkPreprocessor",
    "SklearnPreprocessor",
    "create_preprocessor",
    "ImputationStrategy",
    "EncodingStrategy",
    "ScalingStrategy",
    # Sampling
    "SamplingConfig",
    "SamplingStrategy",
    "SparkSampler",
    "NumpySampler",
    "HardNegativeMiner",
    "create_sampler",
    "compute_class_weights",
    # Features
    "FeatureConfig",
    "FeatureResolver",
    "resolve_features",
    "load_feature_config",
    # Utils
    "get_device_manager",
]
