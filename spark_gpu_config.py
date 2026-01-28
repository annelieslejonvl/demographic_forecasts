"""
Optimized Spark configuration for GPU workloads with XGBoost/PyTorch.

Key optimizations:
1. Increased off-heap memory for GPU operations
2. Apache Arrow for 10-100x faster Spark→Pandas conversion
3. G1GC tuning for better concurrent GC
4. Reduced JVM heap to leave more memory for GPU
"""

from pyspark.sql import SparkSession


def create_gpu_optimized_spark_session(
    app_name: str = "DemographicForecasts",
    driver_memory: str = "6g",      # Reduced from 8g to leave room for off-heap
    executor_memory: str = "6g",
    memory_overhead: str = "4g",    # CRITICAL: Off-heap memory for GPU operations
    use_arrow: bool = True,
) -> SparkSession:
    """
    Create Spark session optimized for GPU workloads.

    Memory allocation example for 16GB system with 10GB GPU:
    - JVM heap (driver + executor): 6GB + 6GB = 12GB
    - Off-heap (GPU operations): 4GB
    - System/GPU: remaining ~4GB

    Args:
        app_name: Spark application name
        driver_memory: JVM heap for driver (reduce to leave GPU memory)
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
                "-XX:+UseG1GC "                    # Use G1GC (better for concurrent ops)
                "-XX:InitiatingHeapOccupancyPercent=35 "  # Start GC earlier
                "-XX:G1HeapRegionSize=16M "        # Larger regions for big allocations
                "-XX:MaxGCPauseMillis=200 "        # Target max GC pause
                "-XX:+ParallelRefProcEnabled "     # Parallel reference processing
                "-XX:ParallelGCThreads=8 "         # Parallel GC threads
                "-XX:ConcGCThreads=2")             # Concurrent GC threads

        .config("spark.executor.extraJavaOptions",
                "-XX:+UseG1GC "
                "-XX:InitiatingHeapOccupancyPercent=35 "
                "-XX:G1HeapRegionSize=16M "
                "-XX:MaxGCPauseMillis=200 "
                "-XX:+ParallelRefProcEnabled "
                "-XX:ParallelGCThreads=8 "
                "-XX:ConcGCThreads=2")

        # Reduce partitions for local mode (less overhead)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")

        # Disable UI if not needed (saves memory)
        .config("spark.ui.enabled", "false")

        # Locality wait (give time for data locality)
        .config("spark.locality.wait", "3s")
    )

    # Apache Arrow for fast Spark↔Pandas conversion
    if use_arrow:
        builder = (
            builder
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")  # Batch size
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
