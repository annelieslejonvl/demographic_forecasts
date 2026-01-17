"""
Unified Data Source Module

Supports loading data from:
- Unity Catalog tables (Databricks)
- Parquet files (local, DBFS, S3, ADLS, GCS)
- Delta tables
- CSV files

Works seamlessly with all backends (PyTorch, XGBoost, Spark).
"""
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Supported data source types."""
    UNITY_CATALOG = "unity_catalog"
    PARQUET = "parquet"
    DELTA = "delta"
    CSV = "csv"
    SPARK_DF = "spark_df"  # Pass-through for existing DataFrames


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    
    source_type: DataSourceType = DataSourceType.PARQUET
    
    # Unity Catalog
    catalog: Optional[str] = None
    schema: Optional[str] = None
    table: Optional[str] = None
    
    # File-based sources
    path: Optional[str] = None
    paths: Optional[List[str]] = None  # Multiple files
    
    # Read options
    file_format: str = "parquet"
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Partitioning
    partition_cols: Optional[List[str]] = None
    partition_filter: Optional[str] = None  # SQL WHERE clause
    
    # Sampling (for large datasets)
    sample_fraction: Optional[float] = None
    sample_seed: int = 42
    
    # Schema
    schema_path: Optional[str] = None  # Path to schema JSON
    infer_schema: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataSourceConfig":
        """Create config from dictionary (YAML-compatible)."""
        source_type = DataSourceType(d.get("source_type", "parquet"))
        
        return cls(
            source_type=source_type,
            catalog=d.get("catalog"),
            schema=d.get("schema"),
            table=d.get("table"),
            path=d.get("path"),
            paths=d.get("paths"),
            file_format=d.get("file_format", "parquet"),
            options=d.get("options", {}),
            partition_cols=d.get("partition_cols"),
            partition_filter=d.get("partition_filter"),
            sample_fraction=d.get("sample_fraction"),
            sample_seed=d.get("sample_seed", 42),
            schema_path=d.get("schema_path"),
            infer_schema=d.get("infer_schema", True),
        )
    
    @classmethod
    def from_unity_catalog(
        cls,
        catalog: str,
        schema: str,
        table: str,
        **kwargs,
    ) -> "DataSourceConfig":
        """Create config for Unity Catalog table."""
        return cls(
            source_type=DataSourceType.UNITY_CATALOG,
            catalog=catalog,
            schema=schema,
            table=table,
            **kwargs,
        )
    
    @classmethod
    def from_parquet(
        cls,
        path: str,
        **kwargs,
    ) -> "DataSourceConfig":
        """Create config for Parquet file(s)."""
        return cls(
            source_type=DataSourceType.PARQUET,
            path=path,
            file_format="parquet",
            **kwargs,
        )
    
    @classmethod
    def from_delta(
        cls,
        path: str,
        **kwargs,
    ) -> "DataSourceConfig":
        """Create config for Delta table."""
        return cls(
            source_type=DataSourceType.DELTA,
            path=path,
            file_format="delta",
            **kwargs,
        )
    
    def get_full_table_name(self) -> str:
        """Get fully qualified Unity Catalog table name."""
        if self.source_type != DataSourceType.UNITY_CATALOG:
            raise ValueError("Not a Unity Catalog source")
        return f"{self.catalog}.{self.schema}.{self.table}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type.value,
            "catalog": self.catalog,
            "schema": self.schema,
            "table": self.table,
            "path": self.path,
            "paths": self.paths,
            "file_format": self.file_format,
            "options": self.options,
            "partition_cols": self.partition_cols,
            "partition_filter": self.partition_filter,
            "sample_fraction": self.sample_fraction,
            "sample_seed": self.sample_seed,
        }


class DataSource:
    """
    Unified data source that loads from Unity Catalog, Parquet, Delta, etc.
    
    Example usage:
        # From Unity Catalog
        source = DataSource.from_unity_catalog("my_catalog", "my_schema", "my_table")
        df = source.load(spark)
        
        # From Parquet
        source = DataSource.from_parquet("/path/to/data.parquet")
        df = source.load(spark)
        
        # Convert to numpy for PyTorch/XGBoost
        X, y = source.to_numpy(spark, feature_cols, label_col)
    """
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self._spark_df: Optional[Any] = None
        self._cached = False
    
    @classmethod
    def from_config(cls, config: Union[DataSourceConfig, Dict[str, Any]]) -> "DataSource":
        """Create DataSource from config dict or object."""
        if isinstance(config, dict):
            config = DataSourceConfig.from_dict(config)
        return cls(config)
    
    @classmethod
    def from_unity_catalog(
        cls,
        catalog: str,
        schema: str,
        table: str,
        **kwargs,
    ) -> "DataSource":
        """Create DataSource for Unity Catalog table."""
        config = DataSourceConfig.from_unity_catalog(catalog, schema, table, **kwargs)
        return cls(config)
    
    @classmethod
    def from_parquet(cls, path: str, **kwargs) -> "DataSource":
        """Create DataSource for Parquet file(s)."""
        config = DataSourceConfig.from_parquet(path, **kwargs)
        return cls(config)
    
    @classmethod
    def from_delta(cls, path: str, **kwargs) -> "DataSource":
        """Create DataSource for Delta table."""
        config = DataSourceConfig.from_delta(path, **kwargs)
        return cls(config)
    
    @classmethod
    def from_spark_df(cls, df: Any) -> "DataSource":
        """Wrap existing Spark DataFrame."""
        config = DataSourceConfig(source_type=DataSourceType.SPARK_DF)
        source = cls(config)
        source._spark_df = df
        return source
    
    def load(self, spark: Any) -> Any:
        """
        Load data as Spark DataFrame.
        
        Args:
            spark: SparkSession instance
        
        Returns:
            Spark DataFrame
        """
        if self._spark_df is not None:
            return self._spark_df
        
        cfg = self.config
        
        if cfg.source_type == DataSourceType.UNITY_CATALOG:
            df = self._load_unity_catalog(spark)
        elif cfg.source_type == DataSourceType.PARQUET:
            df = self._load_parquet(spark)
        elif cfg.source_type == DataSourceType.DELTA:
            df = self._load_delta(spark)
        elif cfg.source_type == DataSourceType.CSV:
            df = self._load_csv(spark)
        else:
            raise ValueError(f"Unsupported source type: {cfg.source_type}")
        
        # Apply partition filter
        if cfg.partition_filter:
            df = df.where(cfg.partition_filter)
        
        # Apply sampling
        if cfg.sample_fraction and cfg.sample_fraction < 1.0:
            df = df.sample(
                fraction=cfg.sample_fraction,
                seed=cfg.sample_seed,
            )
        
        self._spark_df = df
        return df
    
    def _load_unity_catalog(self, spark: Any) -> Any:
        """Load from Unity Catalog table."""
        cfg = self.config
        table_name = cfg.get_full_table_name()
        
        logger.info(f"Loading from Unity Catalog: {table_name}")
        
        # Use spark.table() for Unity Catalog
        df = spark.table(table_name)
        
        return df
    
    def _load_parquet(self, spark: Any) -> Any:
        """Load from Parquet file(s)."""
        cfg = self.config
        
        paths = cfg.paths or [cfg.path]
        logger.info(f"Loading Parquet from: {paths}")
        
        reader = spark.read.format("parquet")
        
        # Apply options
        for key, value in cfg.options.items():
            reader = reader.option(key, value)
        
        # Load single or multiple paths
        if len(paths) == 1:
            df = reader.load(paths[0])
        else:
            df = reader.load(paths)
        
        return df
    
    def _load_delta(self, spark: Any) -> Any:
        """Load from Delta table."""
        cfg = self.config
        
        logger.info(f"Loading Delta from: {cfg.path}")
        
        reader = spark.read.format("delta")
        
        for key, value in cfg.options.items():
            reader = reader.option(key, value)
        
        df = reader.load(cfg.path)
        
        return df
    
    def _load_csv(self, spark: Any) -> Any:
        """Load from CSV file(s)."""
        cfg = self.config
        
        paths = cfg.paths or [cfg.path]
        logger.info(f"Loading CSV from: {paths}")
        
        reader = spark.read.format("csv")
        reader = reader.option("header", cfg.options.get("header", True))
        reader = reader.option("inferSchema", cfg.infer_schema)
        
        for key, value in cfg.options.items():
            reader = reader.option(key, value)
        
        if len(paths) == 1:
            df = reader.load(paths[0])
        else:
            df = reader.load(paths)
        
        return df
    
    def to_pandas(
        self,
        spark: Any,
        columns: Optional[List[str]] = None,
    ) -> Any:
        """
        Convert to Pandas DataFrame.
        
        Args:
            spark: SparkSession
            columns: Optional list of columns to select
        
        Returns:
            Pandas DataFrame
        """
        df = self.load(spark)
        
        if columns:
            df = df.select(columns)
        
        return df.toPandas()
    
    def to_numpy(
        self,
        spark: Any,
        feature_cols: List[str],
        label_col: str,
        weight_col: Optional[str] = None,
    ) -> Tuple:
        """
        Convert to numpy arrays for PyTorch/XGBoost.
        
        Args:
            spark: SparkSession
            feature_cols: Feature column names
            label_col: Label column name
            weight_col: Optional weight column name
        
        Returns:
            Tuple of (X, y, weights) numpy arrays
        """
        import numpy as np
        
        cols = feature_cols + [label_col]
        if weight_col:
            cols.append(weight_col)
        
        pdf = self.to_pandas(spark, cols)
        
        X = pdf[feature_cols].values.astype(np.float32)
        y = pdf[label_col].values.astype(np.float32)
        w = pdf[weight_col].values.astype(np.float32) if weight_col else None
        
        return X, y, w
    
    def cache(self, spark: Any) -> "DataSource":
        """Cache the DataFrame in memory."""
        df = self.load(spark)
        self._spark_df = df.cache()
        self._cached = True
        return self
    
    def unpersist(self) -> "DataSource":
        """Unpersist cached DataFrame."""
        if self._spark_df is not None and self._cached:
            self._spark_df.unpersist()
            self._cached = False
        return self
    
    def get_schema(self, spark: Any) -> Any:
        """Get DataFrame schema."""
        return self.load(spark).schema
    
    def count(self, spark: Any) -> int:
        """Get row count."""
        return self.load(spark).count()
    
    def describe(self, spark: Any) -> Dict[str, Any]:
        """Get data source description."""
        df = self.load(spark)
        return {
            "source_type": self.config.source_type.value,
            "path": self.config.path or self.config.get_full_table_name() if self.config.source_type == DataSourceType.UNITY_CATALOG else None,
            "num_rows": df.count(),
            "num_cols": len(df.columns),
            "columns": df.columns,
            "cached": self._cached,
        }


class DataSourceRegistry:
    """
    Registry for managing multiple data sources.
    
    Example:
        registry = DataSourceRegistry()
        registry.register("train", DataSource.from_unity_catalog(...))
        registry.register("test", DataSource.from_parquet(...))
        
        train_df = registry.load("train", spark)
    """
    
    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
    
    def register(self, name: str, source: DataSource) -> "DataSourceRegistry":
        """Register a data source."""
        self._sources[name] = source
        return self
    
    def get(self, name: str) -> DataSource:
        """Get a registered data source."""
        if name not in self._sources:
            raise KeyError(f"Data source not found: {name}")
        return self._sources[name]
    
    def load(self, name: str, spark: Any) -> Any:
        """Load a registered data source as Spark DataFrame."""
        return self.get(name).load(spark)
    
    def list_sources(self) -> List[str]:
        """List registered source names."""
        return list(self._sources.keys())
    
    @classmethod
    def from_config(cls, config: Dict[str, Dict[str, Any]]) -> "DataSourceRegistry":
        """
        Create registry from config dictionary.
        
        Example config:
            {
                "train": {"source_type": "unity_catalog", "catalog": "...", ...},
                "test": {"source_type": "parquet", "path": "...", ...},
            }
        """
        registry = cls()
        for name, source_config in config.items():
            source = DataSource.from_config(source_config)
            registry.register(name, source)
        return registry


# =============================================================================
# Convenience functions
# =============================================================================

def load_from_unity_catalog(
    spark: Any,
    catalog: str,
    schema: str,
    table: str,
    **kwargs,
) -> Any:
    """Quick function to load from Unity Catalog."""
    source = DataSource.from_unity_catalog(catalog, schema, table, **kwargs)
    return source.load(spark)


def load_from_parquet(
    spark: Any,
    path: str,
    **kwargs,
) -> Any:
    """Quick function to load from Parquet."""
    source = DataSource.from_parquet(path, **kwargs)
    return source.load(spark)


def load_from_delta(
    spark: Any,
    path: str,
    **kwargs,
) -> Any:
    """Quick function to load from Delta table."""
    source = DataSource.from_delta(path, **kwargs)
    return source.load(spark)


def unity_catalog_to_numpy(
    spark: Any,
    catalog: str,
    schema: str,
    table: str,
    feature_cols: List[str],
    label_col: str,
    **kwargs,
) -> Tuple:
    """Load Unity Catalog table directly to numpy arrays."""
    source = DataSource.from_unity_catalog(catalog, schema, table, **kwargs)
    return source.to_numpy(spark, feature_cols, label_col)


def parquet_to_numpy(
    spark: Any,
    path: str,
    feature_cols: List[str],
    label_col: str,
    **kwargs,
) -> Tuple:
    """Load Parquet file directly to numpy arrays."""
    source = DataSource.from_parquet(path, **kwargs)
    return source.to_numpy(spark, feature_cols, label_col)
