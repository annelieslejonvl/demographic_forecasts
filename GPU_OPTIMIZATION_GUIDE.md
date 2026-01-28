# GPU Performance Optimization Guide

## Problem: GC Locker Warnings and Severe Performance Degradation

When using Spark with GPU backends (XGBoost/PyTorch), you may see warnings like:
```
[warning][gc,alloc] Retried waiting for GCLocker too often allocating 2097152 words
```

This indicates memory contention between JVM garbage collection and GPU native operations.

## Root Cause

1. **Memory Flow**: Spark (JVM) → `.toPandas()` → NumPy → GPU
2. **JVM Heap Pressure**: `.toPandas()` allocates large amounts of JVM heap memory
3. **GC Contention**: GPU native code holds GC locker while JVM tries to garbage collect
4. **Thrashing**: JVM spends excessive time in GC retries, causing severe slowdowns

## Solutions

### 1. Use GPU-Optimized Spark Configuration (Recommended)

**Replace your current Spark initialization:**

```python
# ❌ OLD (causes GC issues)
spark = SparkSession.builder \
    .appName("DemographicForecasts") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
```

**With the optimized configuration:**

```python
# ✅ NEW (optimized for GPU)
from spark_gpu_config import create_gpu_optimized_spark_session

spark = create_gpu_optimized_spark_session(
    driver_memory="6g",      # Reduced to leave room for GPU operations
    memory_overhead="4g",    # Off-heap memory for GPU operations
    use_arrow=True,          # 10-100x faster Spark→Pandas conversion
)
```

### 2. Key Configuration Changes

#### Memory Allocation
- **Reduced JVM heap** (6GB instead of 8GB) to leave memory for GPU operations
- **Added memory overhead** (4GB) for off-heap GPU operations
- **Formula**: Total RAM = JVM heap + Memory overhead + System/GPU memory

Example for 16GB system:
- JVM heap: 6GB (driver) + 6GB (executor) = 12GB
- Memory overhead: 4GB (GPU operations)
- Remaining: 4GB (system + GPU memory)

#### Apache Arrow
- **10-100x faster** Spark↔Pandas conversion
- **Lower memory footprint** during conversion
- **Columnar format** optimized for analytical workloads

#### G1GC Tuning
- **Concurrent collection** to avoid blocking GPU operations
- **Earlier GC trigger** (35% heap occupancy) to prevent memory spikes
- **Parallel reference processing** for faster cleanup

### 3. Additional Optimizations

#### Reduce Batch Size if Still Seeing Issues

In your training code:

```python
# If still experiencing issues, reduce batch size
pipeline.fit(
    train_data=train_df,
    feature_cols=feature_cols,
    batch_size=50_000,  # Reduced from 100,000
)
```

#### Monitor Memory Usage

```python
import psutil
import GPUtil

# Before training
process = psutil.Process()
print(f"CPU Memory: {process.memory_info().rss / 1e9:.2f} GB")

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
```

### 4. Expected Performance Improvements

| Metric | Before | After |
|--------|--------|-------|
| Spark→Pandas conversion | Slow (standard) | 10-100x faster (Arrow) |
| GC warnings | Frequent | Rare/none |
| Training time per batch | ~35-40s | ~10-15s (estimated) |
| Memory pressure | High | Moderate |

### 5. Troubleshooting

#### Still Seeing GC Warnings?

1. **Reduce driver memory further**: Try `driver_memory="4g"`
2. **Increase memory overhead**: Try `memory_overhead="6g"`
3. **Reduce batch size**: Use `batch_size=25_000`
4. **Check GPU memory**: Use `nvidia-smi` to monitor GPU utilization

#### Arrow Conversion Fails?

If you see "Arrow conversion failed" warnings:
```bash
# Install PyArrow
pip install pyarrow
```

#### Out of Memory Errors?

```python
# Use smaller sample fractions
spark_to_numpy(df, feature_cols, label_col, sample_fraction=0.05)  # 5% instead of 10%
```

## Quick Start

```python
# 1. Create optimized Spark session
from spark_gpu_config import create_gpu_optimized_spark_session
spark = create_gpu_optimized_spark_session()

# 2. Load your data
df = spark.read.parquet("data/synth_features")

# 3. Run training (pipeline now automatically uses Arrow)
from run import run_experiments

results = run_experiments(
    df=df,
    experiment_name="moved_models_xgb",
    runs=[
       ("configs/features.yaml", "configs/models/xgboost_classifier.yaml", "configs/datasets/default.yaml"),
    ],
    mode="tune",
    fit=True
)
```

## References

- Apache Arrow: https://arrow.apache.org/docs/python/
- Spark Memory Management: https://spark.apache.org/docs/latest/tuning.html
- G1GC Tuning: https://www.oracle.com/technical-resources/articles/java/g1gc.html
- XGBoost GPU: https://xgboost.readthedocs.io/en/latest/gpu/index.html
