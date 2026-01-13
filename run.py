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
from typing import Any, Dict, List, Optional, Tuple

import mlflow
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
#from src.config_loader import load_yaml
#from src.specs import DataSpec, ModelSpec
#from src.dataset import DatasetBuilder, DatasetSpec

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


def spark_to_numpy(
    spark_df,
    feature_cols: List[str],
    label_col: str,
    weight_col: Optional[str] = None,
) -> Tuple:
    """Convert Spark DataFrame to numpy arrays for PyTorch/XGBoost."""
    import numpy as np
    
    cols = feature_cols + [label_col]
    if weight_col:
        cols.append(weight_col)
    
    pdf = spark_df.select(cols).toPandas()
    
    X = pdf[feature_cols].values.astype(np.float32)
    y = pdf[label_col].values.astype(np.float32)
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
        train_df_sampled, samp_info = ds_builder.sample_train(train_df, sampling_cfg)
        
        # Add weights if needed
        if model_cfg.get("weight_col"):
            train_df_sampled, w_info = add_inverse_prevalence_weights(
                train_df_sampled, label_col, model_cfg["weight_col"]
            )
            samp_info.update(w_info)
        
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
            pipeline = UnifiedPipeline(pipeline_config)
            
            # Prepare data based on backend
            if backend == BackendType.SPARK:
                # Spark uses DataFrame directly
                train_result = pipeline.fit(
                    train_df_sampled,
                    feature_cols,
                    eval_data=valid_df,
                )
                
                pred_train = pipeline.predict(train_df_sampled).predictions
                pred_test = pipeline.predict(test_df).predictions
                
                train_metrics = _spark_binary_metrics(pred_train, label_col)
                test_metrics = _spark_binary_metrics(pred_test, label_col)
                
                valid_metrics = None
                if valid_df is not None:
                    pred_valid = pipeline.predict(valid_df).predictions
                    valid_metrics = _spark_binary_metrics(pred_valid, label_col)
            
            else:
                # PyTorch/XGBoost need numpy arrays
                X_train, y_train, w_train = spark_to_numpy(
                    train_df_sampled, feature_cols, label_col,
                    model_cfg.get("weight_col"),
                )
                X_test, y_test, _ = spark_to_numpy(test_df, feature_cols, label_col)
                
                eval_set = None
                if valid_df is not None:
                    X_val, y_val, _ = spark_to_numpy(valid_df, feature_cols, label_col)
                    eval_set = [(X_val, y_val)]
                
                train_result = pipeline.fit(
                    X_train,
                    feature_cols,
                    eval_data=X_val if valid_df else None,
                    sample_weight=w_train,
                )
                
                # Note: For non-Spark, we need to handle y separately
                pred_train = pipeline.predict_proba(X_train)
                pred_test = pipeline.predict_proba(X_test)
                
                train_metrics = _numpy_binary_metrics(y_train, pred_train.probabilities)
                test_metrics = _numpy_binary_metrics(y_test, pred_test.probabilities)
                
                valid_metrics = None
                if valid_df is not None:
                    pred_valid = pipeline.predict_proba(X_val)
                    valid_metrics = _numpy_binary_metrics(y_val, pred_valid.probabilities)
            
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
    
    return results


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
