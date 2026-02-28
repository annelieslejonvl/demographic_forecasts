"""
Multi-Backend Experiment Runner

Runs ML experiments across PyTorch, XGBoost, and Spark backends with:
- Unified config-driven pipeline
- Time-based data splits
- MLflow experiment tracking
- CPU/GPU support for all backends

Usage:
    from run import run_experiments
    
    results = run_experiments(
        df=spark_dataframe,
        experiment_name="my_experiment",
        runs=[
            ("configs/data/features.yaml", "configs/models/xgboost_classifier.yaml", "configs/datasets/default.yaml"),
            ("configs/data/features.yaml", "configs/models/pytorch_mlp.yaml", "configs/datasets/default.yaml"),
        ],
        mode="tune",
    )
"""
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Always set JAVA_HOME to JDK 17 (required by Spark 4.x; JDK 25+ is incompatible)
_jdk_path = Path.home() / "jdk-17.0.18+8"
if _jdk_path.exists():
    os.environ["JAVA_HOME"] = str(_jdk_path)

# Hadoop winutils.exe required on Windows
if not os.environ.get("HADOOP_HOME"):
    _hadoop_path = Path.home() / "hadoop"
    if _hadoop_path.exists():
        os.environ["HADOOP_HOME"] = str(_hadoop_path)

import mlflow
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.storagelevel import StorageLevel

# Local imports
from src.backends import (
    BackendType,
    DeviceConfig,
    DeviceType,
    UnifiedPipeline,
    PipelineConfig,
    create_pipeline,
)
from src.backends.base import BackendFactory
from src.utils.device import get_device_manager
from src.config_loader import load_yaml
from src.specs import DataSpec, ModelSpec
from src.dataset import DatasetBuilder, DatasetSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics (backend-agnostic where possible)
# =============================================================================

def compute_binary_metrics(
    y_true,
    y_pred_proba,
    backend: BackendType,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    Works with numpy arrays (PyTorch/XGBoost) or Spark DataFrames.
    """
    if backend == BackendType.SPARK:
        return _spark_binary_metrics(y_true, y_pred_proba, threshold)
    else:
        return _numpy_binary_metrics(y_true, y_pred_proba, threshold)


def _numpy_binary_metrics(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics for numpy arrays."""
    import numpy as np
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        precision_score,
        recall_score,
        f1_score,
    )
    
    # Handle 2D probability arrays
    if len(y_pred_proba.shape) == 2:
        p1 = y_pred_proba[:, 1]
    else:
        p1 = y_pred_proba
    
    y_pred = (p1 >= threshold).astype(int)
    
    return {
        "auc_roc": float(roc_auc_score(y_true, p1)),
        "auc_pr": float(average_precision_score(y_true, p1)),
        "brier": float(brier_score_loss(y_true, p1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": len(y_true),
        "pos_rate": float(y_true.mean()),
        "mean_pred_p1": float(p1.mean()),
    }


def _spark_binary_metrics(
    pred_df,
    label_col: str = "label",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics for Spark DataFrames."""
    p1 = vector_to_array(F.col("probability"))[1]
    
    # AUC metrics
    roc_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )
    
    roc = float(roc_eval.evaluate(pred_df))
    pr = float(pr_eval.evaluate(pred_df))
    
    # Brier score
    brier = float(
        pred_df.select(
            F.avg((p1 - F.col(label_col)) ** 2).alias("brier")
        ).collect()[0]["brier"]
    )
    
    # Threshold-based metrics
    dfb = pred_df.select(
        F.col(label_col).cast("int").alias("y"),
        (p1 >= F.lit(threshold)).cast("int").alias("yhat"),
        p1.alias("p1"),
    )
    
    agg = dfb.select(
        F.count("*").alias("n"),
        F.avg("y").alias("pos_rate"),
        F.avg("p1").alias("mean_pred_p1"),
        F.sum((F.col("yhat") == 1).cast("int")).alias("pred_pos"),
        F.sum((F.col("y") == 1).cast("int")).alias("pos"),
        F.sum(((F.col("yhat") == 1) & (F.col("y") == 1)).cast("int")).alias("tp"),
    ).collect()[0]
    
    tp = float(agg["tp"])
    pred_pos = float(agg["pred_pos"])
    pos = float(agg["pos"])
    
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / pos if pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "auc_roc": roc,
        "auc_pr": pr,
        "brier": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": float(agg["n"]),
        "pos_rate": float(agg["pos_rate"]),
        "mean_pred_p1": float(agg["mean_pred_p1"]),
    }


def _compute_fairness_metrics_batched(
    spark_df,
    pipeline,
    feature_cols: List[str],
    label_col: str,
    subgroup_cols: List[str],
    threshold: float,
    split_name: str = "test",
    sample_fraction: Optional[float] = None,
    max_rows: Optional[int] = None,
    seed: int = 42,
):
    """
    Compute fairness metrics per subgroup using batched prediction.

    Args:
        spark_df: Spark DataFrame
        pipeline: Trained pipeline
        feature_cols: Feature column names
        label_col: Label column name
        subgroup_cols: Columns to group by for fairness analysis
        threshold: Classification threshold
        split_name: Name of split (train/valid/test) for logging
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

    # Optionally downsample to avoid driver OOM
    spark_eval_df = _prepare_eval_df(
        spark_df,
        sample_fraction=sample_fraction,
        max_rows=max_rows,
        seed=seed,
    )

    # Get predictions
    y_true, y_pred_proba = predict_in_batches(
        pipeline, spark_eval_df, feature_cols, label_col
    )
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Convert subgroup columns to pandas for grouping
    # IMPORTANT: Limit to same number of rows as predictions to avoid mismatch
    cols_to_select = subgroup_cols + [label_col]
    subgroup_df = spark_eval_df.select(cols_to_select).limit(len(y_true)).toPandas()

    # Validate row count
    if len(subgroup_df) != len(y_true):
        logger.warning(f"Row count mismatch in fairness evaluation: "
                      f"subgroup_df={len(subgroup_df)}, predictions={len(y_true)}")
        # Truncate to match
        min_len = min(len(subgroup_df), len(y_true))
        subgroup_df = subgroup_df.iloc[:min_len]
        y_true = y_true[:min_len]
        y_pred_proba = y_pred_proba[:min_len]
        y_pred = y_pred[:min_len]

    subgroup_df["y_pred_proba"] = y_pred_proba
    subgroup_df["y_pred"] = y_pred

    # Overall metrics
    overall_metrics = {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "auc_pr": float(average_precision_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
    }

    # Log overall metrics
    for metric_name, value in overall_metrics.items():
        mlflow.log_metric(f"{split_name}_overall_{metric_name}", value)

    # Compute per-subgroup metrics
    for col in subgroup_cols:
        if col not in subgroup_df.columns:
            logger.warning(f"Column {col} not in DataFrame, skipping fairness analysis")
            continue

        for group_value in subgroup_df[col].unique():
            # Skip NaN
            if pd.isna(group_value):
                continue

            # Filter to subgroup
            mask = subgroup_df[col] == group_value
            group_size = mask.sum()

            if group_size < 100:  # Skip very small groups
                continue

            y_true_group = subgroup_df.loc[mask, label_col].values
            y_pred_proba_group = subgroup_df.loc[mask, "y_pred_proba"].values
            y_pred_group = subgroup_df.loc[mask, "y_pred"].values

            # Compute metrics
            try:
                group_metrics = {
                    "auc": float(roc_auc_score(y_true_group, y_pred_proba_group)),
                    "auc_pr": float(average_precision_score(y_true_group, y_pred_proba_group)),
                    "precision": float(precision_score(y_true_group, y_pred_group, zero_division=0)),
                    "recall": float(recall_score(y_true_group, y_pred_group, zero_division=0)),
                    "f1": float(f1_score(y_true_group, y_pred_group, zero_division=0)),
                    "n_samples": int(group_size),
                    "n_positive": int(y_true_group.sum()),
                    "positive_rate": float(y_true_group.mean()),
                }

                # Log to MLflow with hierarchical naming
                for metric_name, value in group_metrics.items():
                    # Clean group_value for MLflow (remove special chars)
                    clean_group = str(group_value).replace("/", "_").replace(" ", "_")
                    mlflow.log_metric(f"{split_name}_fairness/{col}/{clean_group}/{metric_name}", value)

                # Print summary for top groups
                if group_size > 1000:  # Only print larger groups
                    print(f"   {col}={group_value}: AUC={group_metrics['auc']:.4f}, "
                          f"F1={group_metrics['f1']:.4f}, n={group_size:,}")

            except Exception as e:
                logger.warning(f"Failed to compute metrics for {col}={group_value}: {e}")
                continue

    return overall_metrics


# =============================================================================
# Data Preparation
# =============================================================================

def add_inverse_prevalence_weights(
    df,
    label_col: str,
    weight_col: str = "weight",
    cap: float = 50.0,
):
    """Add inverse prevalence weights to handle class imbalance."""
    stats = df.agg(
        F.sum(F.when(F.col(label_col) == 1, 1).otherwise(0)).alias("n_pos"),
        F.sum(F.when(F.col(label_col) == 0, 1).otherwise(0)).alias("n_neg"),
    ).collect()[0]
    
    n_pos = float(stats["n_pos"])
    n_neg = float(stats["n_neg"])
    
    if n_pos == 0:
        raise ValueError("No positives in training set")
    
    pos_w = min(n_neg / n_pos, cap) if cap else n_neg / n_pos
    
    df_weighted = df.withColumn(
        weight_col,
        F.when(F.col(label_col) == 1, F.lit(pos_w)).otherwise(F.lit(1.0))
    )
    
    return df_weighted, {"n_pos": n_pos, "n_neg": n_neg, "pos_weight": pos_w}


def _prepare_eval_df(
    spark_df,
    sample_fraction: Optional[float] = None,
    max_rows: Optional[int] = None,
    seed: int = 42,
):
    """Optionally downsample a Spark DataFrame for evaluation to avoid OOM."""
    if sample_fraction:
        return spark_df.sample(fraction=sample_fraction, seed=seed)
    if max_rows:
        total_rows = spark_df.count()
        if total_rows > max_rows:
            return spark_df.sample(fraction=max_rows / total_rows, seed=seed)
    return spark_df


def predict_in_batches(
    pipeline,
    spark_df,
    feature_cols: List[str],
    label_col: str,
    batch_size: int = None,
) -> Tuple:
    """
    Predict on a Spark DataFrame in batches to avoid memory issues.

    Args:
        pipeline: Fitted pipeline with predict_proba method
        spark_df: Input Spark DataFrame
        feature_cols: List of feature column names
        label_col: Label column name
        batch_size: Number of rows per batch (auto-detected if None)

    Returns:
        Tuple of (y_true, y_pred_proba) as numpy arrays
    """
    import numpy as np
    import time
    from src.data.utils import create_batch_iterator_simple

    # Auto-detect optimal batch size based on device
    if batch_size is None:
        device = pipeline.estimator_.get_device()
        if device.startswith("cuda"):
            # GPU: Use larger batches for efficiency
            try:
                from src.utils.gpu_monitor import get_gpu_memory_usage, estimate_batch_size_for_gpu
                gpu_id = int(device.split(":")[-1]) if ":" in device else 0
                gpu_stats = get_gpu_memory_usage(gpu_id)
                batch_size = estimate_batch_size_for_gpu(
                    n_features=len(feature_cols),
                    gpu_memory_gb=gpu_stats["total_mb"] / 1024,
                    safety_factor=0.6,  # Conservative for inference
                )
                print(f"🚀 GPU-optimized batch_size={batch_size:,}")
            except Exception as e:
                batch_size = 500_000
                print(f"⚠️  Could not estimate GPU batch size: {e}")
                print(f"   Using default batch_size={batch_size:,}")
        else:
            # CPU: Keep moderate size
            batch_size = 100_000
            print(f"💻 CPU batch_size={batch_size:,}")

    # Get total row count for progress tracking
    print("📊 Counting total rows...")
    total_rows = spark_df.count()
    estimated_batches = (total_rows + batch_size - 1) // batch_size
    print(f"📊 Dataset: {total_rows:,} rows → ~{estimated_batches} batches of {batch_size:,}")
    print(f"{'='*60}")

    y_true_list = []
    y_pred_list = []

    batch_count = 0
    total_preprocess_time = 0
    total_predict_time = 0
    start_time = time.time()

    batch_iter = create_batch_iterator_simple(
        spark_df,
        batch_size=batch_size,
        feature_cols=feature_cols,
        label_col=label_col,
    )

    for batch_X, batch_y in batch_iter:
        batch_count += 1
        batch_start = time.time()

        # Transform using fitted preprocessor (CRITICAL!)
        preprocess_start = time.time()
        if pipeline.config.backend == BackendType.XGBOOST:
            X_transformed = pipeline.preprocessor_.transform(batch_X[feature_cols])
        else:
            X_transformed = pipeline.preprocessor_.transform(batch_X[feature_cols].values)
        preprocess_time = time.time() - preprocess_start
        total_preprocess_time += preprocess_time

        y_batch = batch_y.values.astype(np.float32)

        # Predict directly on transformed data (skip pipeline._prepare_data)
        predict_start = time.time()
        pred_result = pipeline.estimator_.predict_proba(X_transformed)
        predict_time = time.time() - predict_start
        total_predict_time += predict_time

        y_true_list.append(y_batch)
        y_pred_list.append(pred_result.probabilities)

        batch_time = time.time() - batch_start
        rows_processed = batch_count * batch_size

        # Progress logging with timing details
        if batch_count == 1 or batch_count % 5 == 0 or batch_count == estimated_batches:
            progress_pct = min(100, (rows_processed / total_rows) * 100)
            elapsed = time.time() - start_time
            avg_batch_time = elapsed / batch_count
            remaining_batches = estimated_batches - batch_count
            eta_seconds = remaining_batches * avg_batch_time

            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}min"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"

            rows_per_sec = rows_processed / elapsed if elapsed > 0 else 0

            print(f"⏱️  Batch {batch_count}/{estimated_batches} ({progress_pct:.1f}%) | "
                  f"{rows_processed:,}/{total_rows:,} rows | "
                  f"{rows_per_sec:,.0f} rows/s | "
                  f"ETA: {eta_str}")
            print(f"   └─ Batch time: {batch_time:.2f}s "
                  f"(preprocess: {preprocess_time:.2f}s, predict: {predict_time:.2f}s)")

    # Concatenate all batches
    print(f"\n{'='*60}")
    print("🔗 Concatenating predictions...")
    concat_start = time.time()
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    concat_time = time.time() - concat_start

    total_time = time.time() - start_time
    print(f"✅ Complete! Predicted {len(y_true):,} rows in {total_time:.1f}s")

    # Validate row count
    if len(y_true) != total_rows:
        rows_lost = total_rows - len(y_true)
        pct_lost = (rows_lost / total_rows) * 100
        print(f"⚠️  WARNING: Row count mismatch!")
        print(f"   Expected: {total_rows:,} rows")
        print(f"   Got:      {len(y_true):,} rows")
        print(f"   Lost:     {rows_lost:,} rows ({pct_lost:.2f}%)")
        if pct_lost > 1.0:
            logger.error(f"More than 1% of rows lost during prediction! This may indicate a data processing issue.")

    print(f"   📈 Throughput: {len(y_true)/total_time:,.0f} rows/s")
    print(f"   ⏱️  Breakdown:")
    print(f"      • Preprocessing: {total_preprocess_time:.1f}s ({total_preprocess_time/total_time*100:.1f}%)")
    print(f"      • Prediction: {total_predict_time:.1f}s ({total_predict_time/total_time*100:.1f}%)")
    print(f"      • Concatenation: {concat_time:.1f}s ({concat_time/total_time*100:.1f}%)")
    print(f"      • Other (I/O): {total_time - total_preprocess_time - total_predict_time - concat_time:.1f}s")
    print(f"{'='*60}\n")

    return y_true, y_pred


def spark_to_numpy(
    spark_df,
    feature_cols: List[str],
    label_col: str,
    weight_col: Optional[str] = None,
    limit_rows: Optional[int] = 10_000,  # 👈 Safe default!
    sample_fraction: Optional[float] = None,  # 👈 Nieuwe optie
) -> Tuple:
    """Convert Spark DataFrame to numpy arrays for PyTorch/XGBoost.

    WARNING: This loads data into memory. For large datasets, use
    create_batch_iterator() instead.

    Args:
        limit_rows: Max rows to load (default 10k for safety)
        sample_fraction: Alternative to limit_rows, sample % of data
    """
    import numpy as np

    cols = feature_cols + [label_col]
    if weight_col:
        cols.append(weight_col)

    # Apply sampling/limiting
    if sample_fraction is not None:
        n_rows = spark_df.count()
        print(f"Sampling {sample_fraction:.2%} of {n_rows:,} rows")
        spark_df = spark_df.sample(fraction=sample_fraction, seed=42)

        # Use Arrow if available for faster conversion
        try:
            pdf = spark_df.select(cols).toPandas()
        except Exception as e:
            print(f"⚠️  Arrow conversion failed: {e}")
            pdf = spark_df.select(cols).toPandas()
    elif limit_rows is not None:
        print(f"Limiting to {limit_rows:,} rows")
        try:
            pdf = spark_df.limit(limit_rows).select(cols).toPandas()
        except Exception as e:
            print(f"⚠️  Arrow conversion failed: {e}")
            pdf = spark_df.limit(limit_rows).select(cols).toPandas()
    else:
        # Expliciete check - forceer gebruiker om bewust te zijn
        n_rows = spark_df.count()
        if n_rows > 100_000:
            raise ValueError(
                f"Dataset has {n_rows:,} rows. This will use ~{n_rows * len(cols) * 4 / 1e9:.1f}GB RAM. "
                f"Use limit_rows= or sample_fraction= to avoid memory issues, "
                f"or use create_batch_iterator() for streaming."
            )
        try:
            pdf = spark_df.select(cols).toPandas()
        except Exception as e:
            print(f"⚠️  Arrow conversion failed: {e}")
            pdf = spark_df.select(cols).toPandas()

    print(f"Loaded {len(pdf):,} rows, {len(cols)} cols, "
          f"~{pdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    X = pdf[feature_cols].values.astype(np.float32)
    y = pdf[label_col].values.astype(np.float32).ravel()
    w = pdf[weight_col].values.astype(np.float32) if weight_col else None

    return X, y, w

# =============================================================================
# Run Naming & Config Hashing
# =============================================================================

def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def config_signature(data_cfg: dict, model_cfg: dict) -> str:
    payload = json.dumps({"data": data_cfg, "model": model_cfg}, sort_keys=True)
    return _short_hash(payload, 10)


def run_name_from_cfg(
    data_yaml: str,
    model_yaml: str,
    data_cfg: dict,
    model_cfg: dict,
) -> str:
    """Generate a descriptive run name from config files."""
    backend = model_cfg.get("backend", "spark")
    model_type = model_cfg.get("model", {}).get("type", "model")
    
    scaling = model_cfg.get("scaling", {})
    sc = "sc1" if scaling.get("enabled", False) else "sc0"
    
    dimred = model_cfg.get("dim_reduction", {})
    dr = f"pca{dimred.get('k', '')}" if dimred.get("enabled", False) else "pca0"
    
    data_tag = os.path.splitext(os.path.basename(data_yaml))[0]
    model_tag = os.path.splitext(os.path.basename(model_yaml))[0]
    
    sig = config_signature(data_cfg, model_cfg)
    
    return f"{backend}__{data_tag}__{model_tag}__{model_type}__{sc}__{dr}__{sig}"


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiments(
    df,
    experiment_name: str,
    runs: List[Tuple[str, str, str]],  # (data_yaml, model_yaml, dataset_yaml)
    mode: str = "tune",
    override_params_list: Optional[List[Dict[str, Any]]] = None,
    persist_features: bool = True,
    cache_clean_df: bool = False,
    fit= True,
    enable_hyperparameter_tuning: bool = False,
    n_tuning_trials: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run ML experiments across multiple backends.

    Args:
        df: Input Spark DataFrame
        experiment_name: MLflow experiment name
        runs: List of (data_yaml, model_yaml, dataset_yaml) tuples
        mode: "tune" or "final"
        override_params_list: List of parameter override dicts for hyperparameter search
        persist_features: Cache transformed features
        cache_clean_df: Cache cleaned DataFrame
        fit: Whether to fit model (True) or just prepare data (False)
        enable_hyperparameter_tuning: Enable Optuna hyperparameter tuning (XGBoost only)
        n_tuning_trials: Number of Optuna trials (default: 50)

    Returns:
        List of result dictionaries with metrics for each run
    """
    mlflow.set_experiment(experiment_name)
    
    # Print device info
    dm = get_device_manager()
    dm.print_device_summary()
    
    param_sets = override_params_list or [{}]
    results = []
    
    for data_yaml, model_yaml, dataset_yaml in runs:
        # Load configs
        data_cfg = load_yaml(data_yaml)
        model_cfg = load_yaml(model_yaml)
        dataset_cfg = load_yaml(dataset_yaml)
        
        # Parse specs
        data_spec = DataSpec.from_dict(data_cfg).resolve(df)
        ds_spec = DatasetSpec.from_dict(dataset_cfg)
        
        backend = BackendType(model_cfg.get("backend", "spark"))
        label_col = data_spec.label_col
        feature_cols = data_spec.feature_cols
        
        logger.info(f"Running experiment: backend={backend.value}, model={model_cfg.get('model', {}).get('type')}")
        
        # Build dataset splits
        ds_builder = DatasetBuilder(
            label_col=label_col,
            key_cols=ds_spec.key_cols,
            year_col=ds_spec.split.get("year_col", "year"),
        )
        
        train_df, valid_df, test_df = ds_builder.time_split(
            df,
            tuple(ds_spec.split["train_range"]),
            tuple(ds_spec.split["valid_range"]) if ds_spec.split.get("valid_range") else None,
            tuple(ds_spec.split["test_range"]),
        )
        
        # Apply mode-specific sampling
        mode_cfg = ds_spec.tune if mode == "tune" else ds_spec.final
        sampling_cfg = mode_cfg.get("sampling", {"type": "none"})
        hard_neg_cfg = mode_cfg.get("hard_negatives", {})
        train_df_sampled, samp_info = ds_builder.sample_train(
            train_df, sampling_cfg, hard_neg_cfg=hard_neg_cfg
        )
        
        # Add weights if needed
        if model_cfg.get("weight_col"):
            train_df_sampled, w_info = add_inverse_prevalence_weights(
                train_df_sampled, label_col, model_cfg["weight_col"]
            )
            samp_info.update(w_info)
        
        # Hyperparameter tuning (if enabled)
        tuning_results = None
        if enable_hyperparameter_tuning and backend == BackendType.XGBOOST:
            logger.info(f"\n{'='*80}")
            logger.info("HYPERPARAMETER TUNING")
            logger.info(f"{'='*80}")
            logger.info(f"Running {n_tuning_trials} trials on sampled data...")

            # Convert Spark DataFrames to numpy for efficient tuning
            # This avoids loading from Spark 50 times
            # Use reasonable sample for tuning (max 500k rows for training)
            logger.info("Converting data to numpy for tuning...")

            # Check train size and sample if needed
            train_count = train_df_sampled.count()
            logger.info(f"  Train set size: {train_count:,} rows")

            if train_count > 500_000:
                # Sample to max 500k for tuning (sufficient for finding good params)
                train_sample_frac = 500_000 / train_count
                logger.info(f"  Sampling {train_sample_frac:.2%} for tuning ({500_000:,} rows)")
                X_train_tune, y_train_tune, _ = spark_to_numpy(
                    train_df_sampled,
                    feature_cols,
                    label_col,
                    limit_rows=None,
                    sample_fraction=train_sample_frac,
                )
            else:
                # Use all sampled data if it's already small enough
                X_train_tune, y_train_tune, _ = spark_to_numpy(
                    train_df_sampled,
                    feature_cols,
                    label_col,
                    limit_rows=train_count,
                    sample_fraction=None,
                )

            # Sample validation set for tuning (max 100k for speed)
            val_count = valid_df.count()
            logger.info(f"  Validation set size: {val_count:,} rows")

            if val_count > 100_000:
                val_sample_frac = 100_000 / val_count
                logger.info(f"  Sampling {val_sample_frac:.2%} for tuning ({100_000:,} rows)")
                X_val_tune, y_val_tune, _ = spark_to_numpy(
                    valid_df,
                    feature_cols,
                    label_col,
                    limit_rows=None,
                    sample_fraction=val_sample_frac,
                )
            else:
                X_val_tune, y_val_tune, _ = spark_to_numpy(
                    valid_df,
                    feature_cols,
                    label_col,
                    limit_rows=val_count,
                    sample_fraction=None,
                )

            logger.info(f"  Training samples: {len(y_train_tune):,}")
            logger.info(f"  Validation samples: {len(y_val_tune):,}")

            from src.tuning import quick_tune

            tuning_results, _ = quick_tune(
                train_data=(X_train_tune, y_train_tune),  # Pass as numpy tuple
                val_data=(X_val_tune, y_val_tune),        # Pass as numpy tuple
                feature_cols=feature_cols,
                categorical_cols=data_cfg.get('cat_cols', []),
                numerical_cols=data_cfg.get('num_cols', []),
                label_col=label_col,
                n_trials=n_tuning_trials,
                device=model_cfg.get("device", {}).get("type", "auto"),
                objective_metric="auc_pr",  # Use AUC-PR for imbalanced migration data
            )

            logger.info(f"\nTuning complete! Best AUC: {tuning_results['best_value']:.4f}")
            logger.info(f"Best parameters: {tuning_results['best_params']}")

            # Clean up tuning data
            del X_train_tune, y_train_tune, X_val_tune, y_val_tune
            import gc
            gc.collect()

            # Override model params with tuned values
            if not override_params_list:
                override_params_list = [tuning_results['best_params']]
                param_sets = override_params_list

        # Run for each parameter override
        rn_base = run_name_from_cfg(data_yaml, model_yaml, data_cfg, model_cfg)

        for overrides in param_sets:
            # Merge overrides into model config
            run_model_cfg = copy.deepcopy(model_cfg)
            if overrides:
                run_model_cfg.setdefault("model", {}).setdefault("params", {}).update(overrides)
            
            # Create pipeline
            pipeline_config = PipelineConfig.from_dict(run_model_cfg)
            pipeline_config.label_col = label_col
            pipeline_config.cat_cols = data_cfg.get('cat_cols'  , [])
            pipeline_config.num_cols = data_cfg.get('num_cols'  , [])
            print(pipeline_config.cat_cols)
            pipeline = UnifiedPipeline(pipeline_config)
            
            # Prepare data based on backend
            if backend == BackendType.SPARK:
                # Spark uses DataFrame directly
                if(fit==False):
                    X_train, y_train, feature_cols = pipeline._prepare_data(train_df_sampled, feature_cols,fit=True)
                    return (X_train, y_train), feature_cols , (None)
                    
                train_result = pipeline.fit(
                    train_data=train_df_sampled,
                    feature_cols = feature_cols,
                    eval_data=valid_df,
                )

                # Threshold tuning on validation set
                threshold_config = run_model_cfg.get("threshold_tuning", {})
                if threshold_config.get("enabled", False) and valid_df is not None:
                    strategy = threshold_config.get("strategy", "f1")
                    # Build kwargs for threshold tuning
                    tuning_kwargs = {}
                    if "beta" in threshold_config:
                        tuning_kwargs["beta"] = threshold_config["beta"]
                    if "min_precision" in threshold_config:
                        tuning_kwargs["min_precision"] = threshold_config["min_precision"]
                    if "min_recall" in threshold_config:
                        tuning_kwargs["min_recall"] = threshold_config["min_recall"]

                    logger.info(f"Tuning threshold using {strategy} strategy on validation set")
                    optimal_threshold = pipeline.tune_threshold(valid_df, strategy=strategy, **tuning_kwargs)
                    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
                else:
                    optimal_threshold = 0.5

                pred_train = pipeline.predict(train_df_sampled).predictions
                pred_test = pipeline.predict(test_df).predictions

                train_metrics = _spark_binary_metrics(pred_train, label_col, threshold=optimal_threshold)
                test_metrics = _spark_binary_metrics(pred_test, label_col, threshold=optimal_threshold)

                valid_metrics = None
                if valid_df is not None:
                    pred_valid = pipeline.predict(valid_df).predictions
                    valid_metrics = _spark_binary_metrics(pred_valid, label_col, threshold=optimal_threshold)
            
            else:
                print(train_df_sampled.columns)
                # PyTorch/XGBoost need numpy arrays
                # Training set: can use sampling since we're just training
                if(fit ==False):
                    X_train, y_train, w_train = spark_to_numpy(
                        train_df_sampled, feature_cols, label_col,
                        model_cfg.get("weight_col"),  sample_fraction= 0.1
                    )

                # Validation set for training monitoring and threshold tuning (use sample)
                eval_set = None
                X_val_sample, y_val_sample = None, None
                if valid_df is not None:
                    X_val_sample, y_val_sample, _ = spark_to_numpy(
                        valid_df, feature_cols, label_col,
                        limit_rows=None, sample_fraction=0.2  # 20% sample for training monitoring and threshold tuning
                    )
                    eval_set = [(X_val_sample, y_val_sample)]

                print('fitting')
                if(fit  ==False):
                    print('preparing data')
                    X_train_prep, y_train_prep = pipeline._prepare_data((X_train, y_train), feature_cols, fit=True, max_samples=50_000)
                    # For non-fit mode, also prepare validation data
                    val_data = None
                    if valid_df is not None:
                        val_data = (X_val_sample, y_val_sample)
                    return (X_train_prep, y_train_prep), feature_cols, val_data
                else:

                    train_result = pipeline.fit(
                        train_data = train_df_sampled,
                        feature_cols =feature_cols,
                        eval_data=valid_df if valid_df is not None else None,
                        sample_weight=1 if model_cfg.get("weight_col") else None,
                        use_batches=True, 
                        )

                # Threshold tuning on validation sample (sufficient for threshold selection)
                threshold_config = run_model_cfg.get("threshold_tuning", {})
                if threshold_config.get("enabled", False) and valid_df is not None:
                    strategy = threshold_config.get("strategy", "f1")
                    # Build kwargs for threshold tuning
                    tuning_kwargs = {}
                    if "beta" in threshold_config:
                        tuning_kwargs["beta"] = threshold_config["beta"]
                    if "min_precision" in threshold_config:
                        tuning_kwargs["min_precision"] = threshold_config["min_precision"]
                    if "min_recall" in threshold_config:
                        tuning_kwargs["min_recall"] = threshold_config["min_recall"]

                    logger.info(f"Tuning threshold using {strategy} strategy on validation sample")
                    optimal_threshold = pipeline.tune_threshold(
                        (X_val_sample, y_val_sample), strategy=strategy, **tuning_kwargs
                    )
                    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
                else:
                    optimal_threshold = 0.5

                # Evaluation: Use FULL datasets with batch processing for accurate metrics
                print(f"\n{'='*60}")
                print('📊 EVALUATION PHASE')
                print(f"{'='*60}\n")

                eval_cfg = run_model_cfg.get("evaluation", {})
                eval_sample_fraction = eval_cfg.get("sample_fraction")
                eval_max_rows = eval_cfg.get("max_rows")
                eval_seed = eval_cfg.get("seed", 42)

                # Train metrics (on sample is fine)
                print('📋 Train set (sample)...')
                train_eval_df = _prepare_eval_df(
                    train_df_sampled,
                    sample_fraction=eval_sample_fraction,
                    max_rows=eval_max_rows,
                    seed=eval_seed,
                )
                y_train_eval, pred_train = predict_in_batches(
                    pipeline, train_eval_df, feature_cols, label_col
                )
                train_metrics = _numpy_binary_metrics(y_train_eval, pred_train, threshold=optimal_threshold)
                print(f"   ✓ Train metrics: AUC={train_metrics['auc_roc']:.4f}, F1={train_metrics['f1']:.4f}\n")

                # Test metrics (FULL dataset with batches - NO SAMPLING)
                print('📋 Test set (batched evaluation)...')
                test_eval_df = _prepare_eval_df(
                    test_df,
                    sample_fraction=eval_sample_fraction,
                    max_rows=eval_max_rows,
                    seed=eval_seed,
                )
                y_test, pred_test = predict_in_batches(
                    pipeline, test_eval_df, feature_cols, label_col
                )
                test_metrics = _numpy_binary_metrics(y_test, pred_test, threshold=optimal_threshold)
                print(f"   ✓ Test metrics: AUC={test_metrics['auc_roc']:.4f}, F1={test_metrics['f1']:.4f}\n")

                # Validation metrics (FULL dataset with batches - NO SAMPLING)
                valid_metrics = None
                if valid_df is not None:
                    print('📋 Validation set (batched evaluation)...')
                    valid_eval_df = _prepare_eval_df(
                        valid_df,
                        sample_fraction=eval_sample_fraction,
                        max_rows=eval_max_rows,
                        seed=eval_seed,
                    )
                    y_val, pred_val = predict_in_batches(
                        pipeline, valid_eval_df, feature_cols, label_col
                    )
                    valid_metrics = _numpy_binary_metrics(y_val, pred_val, threshold=optimal_threshold)
                    print(f"   ✓ Valid metrics: AUC={valid_metrics['auc_roc']:.4f}, F1={valid_metrics['f1']:.4f}\n")
            
            # Build run name
            rn = rn_base
            if overrides:
                rn += "__" + "__".join([f"{k}={v}" for k, v in overrides.items()])
            
            # Log to MLflow
            with mlflow.start_run(run_name=rn):
                mlflow.log_param("backend", backend.value)
                mlflow.log_param("mode", mode)
                mlflow.log_param("data_yaml", data_yaml)
                mlflow.log_param("model_yaml", model_yaml)
                mlflow.log_param("device", pipeline.device_config.device_type.value)
                mlflow.log_param("threshold", optimal_threshold)
                mlflow.log_param("threshold_tuned", threshold_config.get("enabled", False))
                mlflow.log_param("hyperparameter_tuned", enable_hyperparameter_tuning)

                # Log tuning results if available
                if tuning_results:
                    mlflow.log_param("n_tuning_trials", n_tuning_trials)
                    mlflow.log_metric("tuning_best_auc", tuning_results['best_value'])
                    for param_name, param_value in tuning_results['best_params'].items():
                        mlflow.log_param(f"tuned_{param_name}", param_value)

                for k, v in samp_info.items():
                    mlflow.log_param(f"sampling_{k}", str(v))
                
                for k, v in (overrides or {}).items():
                    mlflow.log_param(f"override_{k}", str(v))
                
                for k, v in train_metrics.items():
                    mlflow.log_metric(f"train_{k}", v)
                
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                
                if valid_metrics:
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)

                # Fairness evaluation per subgroup
                fairness_config = run_model_cfg.get("fairness_evaluation", {})
                if fairness_config.get("enabled", False):
                    subgroup_cols = fairness_config.get("subgroup_cols", [])
                    if subgroup_cols:
                        print(f"\n{'='*60}")
                        print('⚖️  FAIRNESS EVALUATION')
                        print(f"{'='*60}\n")
                        print(f"Analyzing fairness across: {', '.join(subgroup_cols)}")
                        fairness_sample_fraction = fairness_config.get("sample_fraction")
                        fairness_max_rows = fairness_config.get("max_rows")
                        fairness_seed = fairness_config.get("seed", 42)

                        # Compute fairness metrics on test set
                        fairness_results = _compute_fairness_metrics_batched(
                            spark_df=test_df,
                            pipeline=pipeline,
                            feature_cols=feature_cols,
                            label_col=label_col,
                            subgroup_cols=subgroup_cols,
                            threshold=optimal_threshold,
                            split_name="test",
                            sample_fraction=fairness_sample_fraction,
                            max_rows=fairness_max_rows,
                            seed=fairness_seed,
                        )

                        # Also on validation if available
                        if valid_df is not None:
                            _compute_fairness_metrics_batched(
                                spark_df=valid_df,
                                pipeline=pipeline,
                                feature_cols=feature_cols,
                                label_col=label_col,
                                subgroup_cols=subgroup_cols,
                                threshold=optimal_threshold,
                                split_name="valid",
                                sample_fraction=fairness_sample_fraction,
                                max_rows=fairness_max_rows,
                                seed=fairness_seed,
                            )

                        print(f"\n   ✓ Fairness metrics logged to MLflow")

            # Collect results
            result = {
                "run_name": rn,
                "backend": backend.value,
                "mode": mode,
                **samp_info,
                **{f"override_{k}": v for k, v in (overrides or {}).items()},
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            if valid_metrics:
                result.update({f"valid_{k}": v for k, v in valid_metrics.items()})
            
            results.append(result)
            
            logger.info(f"Completed: {rn} | test_auc_roc={test_metrics['auc_roc']:.4f}")
    
    return results, pipeline, feature_cols, train_df_sampled, valid_df, test_df


# =============================================================================
# Quick Run Functions
# =============================================================================

def quick_train(
    df,
    feature_cols: List[str],
    label_col: str = "label",
    backend: str = "xgboost",
    model_type: str = "classifier",
    device: str = "auto",
    **model_params,
):
    """
    Quick training function for ad-hoc experiments.
    
    Example:
        model = quick_train(df, feature_cols, backend="pytorch", model_type="mlp")
    """
    pipeline = create_pipeline(
        backend=backend,
        model_type=model_type,
        device=device,
        model_params=model_params,
        label_col=label_col,
    )
    
    result = pipeline.fit(df, feature_cols)
    return pipeline, result


if __name__ == "__main__":
    # Example usage
    print("Multi-Backend ML Framework")
    print("=" * 50)
    
    dm = get_device_manager()
    dm.print_device_summary()
    
    print("\nAvailable backends:")
    for backend in BackendFactory.list_backends():
        models = BackendFactory.list_models(backend)
        print(f"  {backend.value}: {', '.join(models)}")
