# XGBoost Evaluation - Quick Reference

## What You'll See During Evaluation

### 1. Evaluation Phase Starts
```
============================================================
📊 EVALUATION PHASE
============================================================
```

### 2. Training Set Evaluation (Fast - using sample)
```
📋 Train set (sample)...
   ✓ Train metrics: AUC=0.8234, F1=0.7156
```

### 3. Test Set Evaluation (Full Dataset)
```
📋 Test set (FULL dataset with batched evaluation)...
💻 CPU batch_size=100,000
📊 Counting total rows...
📊 Dataset: 2,456,789 rows → ~25 batches of 100,000
============================================================
⏱️  Batch 1/25 (4.1%) | 100,000/2,456,789 rows | 15,234 rows/s | ETA: 2.7min
   └─ Batch time: 6.56s (preprocess: 2.89s, predict: 3.42s)
⏱️  Batch 5/25 (20.4%) | 500,000/2,456,789 rows | 16,123 rows/s | ETA: 2.0min
   └─ Batch time: 6.21s (preprocess: 2.74s, predict: 3.28s)
⏱️  Batch 10/25 (40.7%) | 1,000,000/2,456,789 rows | 16,897 rows/s | ETA: 1.4min
   └─ Batch time: 5.92s (preprocess: 2.61s, predict: 3.11s)
⏱️  Batch 15/25 (61.1%) | 1,500,000/2,456,789 rows | 17,234 rows/s | ETA: 55s
   └─ Batch time: 5.81s (preprocess: 2.55s, predict: 3.05s)
⏱️  Batch 20/25 (81.4%) | 2,000,000/2,456,789 rows | 17,456 rows/s | ETA: 26s
   └─ Batch time: 5.73s (preprocess: 2.51s, predict: 3.01s)
⏱️  Batch 25/25 (100.0%) | 2,500,000/2,456,789 rows | 17,543 rows/s | ETA: 0s
   └─ Batch time: 5.69s (preprocess: 2.49s, predict: 2.99s)

============================================================
🔗 Concatenating predictions...
✅ Complete! Predicted 2,456,789 rows in 140.1s
   📈 Throughput: 17,543 rows/s
   ⏱️  Breakdown:
      • Preprocessing: 62.5s (44.6%)
      • Prediction: 75.2s (53.7%)
      • Concatenation: 0.8s (0.6%)
      • Other (I/O): 1.6s (1.1%)
============================================================

   ✓ Test metrics: AUC=0.8156, F1=0.7089
```

### 4. Validation Set Evaluation (if applicable)
```
📋 Validation set (FULL dataset with batched evaluation)...
🚀 GPU-optimized batch_size=500,000
📊 Dataset: 678,234 rows → ~2 batches of 500,000
...
   ✓ Valid metrics: AUC=0.8198, F1=0.7123
```

## Understanding the Progress Logs

### Batch Progress Line
```
⏱️  Batch 10/25 (40.7%) | 1,000,000/2,456,789 rows | 16,897 rows/s | ETA: 1.4min
```
- **Batch 10/25**: Current batch / total batches
- **(40.7%)**: Percentage complete
- **1,000,000/2,456,789 rows**: Rows processed / total rows
- **16,897 rows/s**: Current throughput
- **ETA: 1.4min**: Estimated time remaining

### Timing Breakdown
```
   └─ Batch time: 5.92s (preprocess: 2.61s, predict: 3.11s)
```
- **Batch time**: Total time for this batch
- **preprocess**: Time spent transforming features
- **predict**: Time spent in XGBoost prediction

### Final Summary
```
✅ Complete! Predicted 2,456,789 rows in 140.1s
   📈 Throughput: 17,543 rows/s
   ⏱️  Breakdown:
      • Preprocessing: 62.5s (44.6%)
      • Prediction: 75.2s (53.7%)
      • Concatenation: 0.8s (0.6%)
      • Other (I/O): 1.6s (1.1%)
```

## Typical Performance

### CPU
- **Batch size**: 100,000 rows
- **Throughput**: 10,000-20,000 rows/s
- **Memory**: ~400-800 MB per batch

### GPU (RTX 3090)
- **Batch size**: 500,000-1,000,000 rows
- **Throughput**: 50,000-100,000 rows/s
- **Memory**: ~2-4 GB per batch

### GPU (A100)
- **Batch size**: 1,000,000-2,000,000 rows
- **Throughput**: 100,000-200,000 rows/s
- **Memory**: ~4-8 GB per batch

## Performance Indicators

### Good Performance ✅
- Throughput stays stable or increases slightly over batches
- Preprocessing < 50% of total time
- Prediction > 40% of total time
- Other (I/O) < 10% of total time

### Bottlenecks to Watch ⚠️

#### Slow Preprocessing (>60%)
```
⏱️  Breakdown:
   • Preprocessing: 84.3s (60.2%)  ← SLOW
   • Prediction: 42.1s (30.1%)
   • Other: 13.6s (9.7%)
```
**Causes**: Complex categorical encoding, many features, inefficient transformations
**Solutions**: Simplify encoding, reduce features, optimize preprocessor

#### Slow Prediction (>70%)
```
⏱️  Breakdown:
   • Preprocessing: 28.0s (20.0%)
   • Prediction: 98.0s (70.0%)  ← SLOW
   • Other: 14.0s (10.0%)
```
**Causes**: Large model, small batch size, CPU instead of GPU
**Solutions**: Increase batch size, use GPU, reduce model complexity

#### Slow I/O (>20%)
```
⏱️  Breakdown:
   • Preprocessing: 56.0s (40.0%)
   • Prediction: 56.0s (40.0%)
   • Other: 28.0s (20.0%)  ← SLOW
```
**Causes**: Slow Spark→Pandas conversion, small Spark partitions, slow storage
**Solutions**: Enable Apache Arrow, increase partitions, use faster storage

## Quick Diagnostics

### Check Batch Size
Look for this line at the start:
```
💻 CPU batch_size=100,000
```
or
```
🚀 GPU-optimized batch_size=500,000
```

**Too small?** Increase it manually:
```python
y_test, pred_test = predict_in_batches(
    pipeline, test_df, feature_cols, label_col,
    batch_size=500_000  # Increase
)
```

### Monitor Progress
Watch the **rows/s** (throughput):
- Should be consistent across batches
- If decreasing → memory pressure or thermal throttling
- If very low → check for bottlenecks in breakdown

### Estimate Total Time
For future runs:
```
Total time ≈ (number_of_rows / throughput)
```

Example:
- 10M rows @ 20K rows/s = 500 seconds (~8 minutes)
- 10M rows @ 100K rows/s = 100 seconds (~1.7 minutes)

## Keyboard Shortcuts

### In Jupyter/IPython
- **Ctrl+C**: Interrupt evaluation (safe, will stop at next batch)
- **Output scroll**: Shows live progress

### Terminal
- **Ctrl+C**: Interrupt evaluation
- **Ctrl+L**: Clear screen (keeps logs in history)
