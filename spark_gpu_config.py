"""
Optimized Spark configuration for GPU workloads with XGBoost/PyTorch.

Key optimizations:
1. Increased off-heap memory for GPU operations
2. Apache Arrow for 10-100x faster Spark→Pandas conversion
3. G1GC tuning for better concurrent GC
4. Reduced JVM heap to leave more memory for GPU
"""

import os
from pathlib import Path

# Always set JAVA_HOME to JDK 17 (required by Spark 4.x; JDK 25+ is incompatible)
_jdk_path = Path.home() / "jdk-17.0.18+8"
if _jdk_path.exists():
    os.environ["JAVA_HOME"] = str(_jdk_path)

# Hadoop winutils.exe required on Windows
if not os.environ.get("HADOOP_HOME"):
    _hadoop_path = Path.home() / "hadoop"
    if _hadoop_path.exists():
        os.environ["HADOOP_HOME"] = str(_hadoop_path)

from pyspark.sql import SparkSession


def create_gpu_optimized_spark_session(
    app_name: str = "DemographicForecasts",
    driver_memory: str = "12g",
    executor_memory: str = "12g",
    memory_overhead: str = "2g",    # Off-heap memory for GPU operations
    use_arrow: bool = True,
) -> SparkSession:
    """
    Create Spark session optimized for GPU workloads.

    Memory allocation for 32GB system (local mode = single JVM):
    - JVM heap: 12GB (driver/executor share the same JVM)
    - Off-heap: 2GB (GPU/native operations)
    - Code cache: ~512MB
    - Total Spark footprint: ~15GB
    - Remaining for OS + browser + IDE: ~17GB

    Args:
        app_name: Spark application name
        driver_memory: JVM heap for driver
        executor_memory: JVM heap per executor
        memory_overhead: Off-heap memory for GPU/native operations
        use_arrow: Enable Apache Arrow for fast Spark↔Pandas conversion
    """

    builder = (
        SparkSession.builder
        .appName(app_name)

        # Memory configuration
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memoryOverhead", memory_overhead)
        .config("spark.executor.memoryOverhead", memory_overhead)

        # GC tuning for concurrent workloads
        .config("spark.driver.extraJavaOptions",
                "-XX:+UseG1GC "
                "-XX:InitiatingHeapOccupancyPercent=35 "
                "-XX:G1HeapRegionSize=16M "
                "-XX:MaxGCPauseMillis=200 "
                "-XX:+ParallelRefProcEnabled "
                "-XX:ParallelGCThreads=8 "
                "-XX:ConcGCThreads=2 "
                "-XX:ReservedCodeCacheSize=512m "
                "-XX:NonProfiledCodeHeapSize=256m")

        .config("spark.executor.extraJavaOptions",
                "-XX:+UseG1GC "
                "-XX:InitiatingHeapOccupancyPercent=35 "
                "-XX:G1HeapRegionSize=16M "
                "-XX:MaxGCPauseMillis=200 "
                "-XX:+ParallelRefProcEnabled "
                "-XX:ParallelGCThreads=8 "
                "-XX:ConcGCThreads=2 "
                "-XX:ReservedCodeCacheSize=512m "
                "-XX:NonProfiledCodeHeapSize=256m")

        # Memory tuning: maximize in-memory processing
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.5")
        .config("spark.driver.maxResultSize", "4g")

        # Partitions for local mode
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")

        # Adaptive query execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "64MB")

        # Broadcast joins for small tables (huge speedup over sort-merge)
        .config("spark.sql.autoBroadcastJoinThreshold", "256MB")

        # Disable UI (saves memory)
        .config("spark.ui.enabled", "false")
    )

    # Apache Arrow for fast Spark↔Pandas conversion
    if use_arrow:
        builder = (
            builder
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000")
        )

    spark = builder.getOrCreate()

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("ERROR")

    print("=" * 60)
    print("GPU-Optimized Spark Configuration")
    print("=" * 60)
    print(f"Driver memory: {driver_memory} (JVM heap)")
    print(f"Memory overhead: {memory_overhead} (off-heap for GPU)")
    print(f"Apache Arrow: {'Enabled' if use_arrow else 'Disabled'}")
    print(f"GC: G1GC with concurrent collection")
    print("=" * 60)

    return spark


# Example usage
if __name__ == "__main__":
    spark = create_gpu_optimized_spark_session()

    # Test Arrow conversion
    import pandas as pd
    test_df = spark.createDataFrame(pd.DataFrame({"x": range(100)}))
    pandas_df = test_df.toPandas()
    print(f"Test conversion successful: {len(pandas_df)} rows")
