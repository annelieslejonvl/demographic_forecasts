"""
Fairness and subgroup evaluation metrics.
Computes model performance per demographic category.
"""
import logging
from typing import Dict, List, Optional, Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_proba_col: str,
    subgroup_cols: List[str],
    threshold: float = 0.5,
    log_to_mlflow: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute evaluation metrics per subgroup.

    Args:
        df: DataFrame with predictions and subgroup columns
        y_true_col: Column name for true labels
        y_pred_proba_col: Column name for predicted probabilities
        subgroup_cols: List of column names to group by (e.g., ['nationality', 'year'])
        threshold: Classification threshold
        log_to_mlflow: Whether to log metrics to MLflow

    Returns:
        Dictionary with subgroup metrics
    """
    results = {}

    # Overall metrics first
    y_true = df[y_true_col].values
    y_pred_proba = df[y_pred_proba_col].values
    y_pred = (y_pred_proba >= threshold).astype(int)

    overall_metrics = {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "auc_pr": float(average_precision_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
    }

    results["overall"] = overall_metrics

    if log_to_mlflow:
        for metric_name, value in overall_metrics.items():
            mlflow.log_metric(f"overall/{metric_name}", value)

    # Compute metrics per subgroup
    for col in subgroup_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue

        logger.info(f"Computing metrics for subgroup: {col}")
        results[col] = {}

        for group_value in df[col].unique():
            # Skip NaN groups
            if pd.isna(group_value):
                continue

            # Filter to this subgroup
            mask = df[col] == group_value
            group_df = df[mask]

            if len(group_df) < 10:  # Skip very small groups
                continue

            y_true_group = group_df[y_true_col].values
            y_pred_proba_group = group_df[y_pred_proba_col].values
            y_pred_group = (y_pred_proba_group >= threshold).astype(int)

            # Compute metrics
            try:
                group_metrics = {
                    "auc": float(roc_auc_score(y_true_group, y_pred_proba_group)),
                    "auc_pr": float(average_precision_score(y_true_group, y_pred_proba_group)),
                    "precision": float(precision_score(y_true_group, y_pred_group, zero_division=0)),
                    "recall": float(recall_score(y_true_group, y_pred_group, zero_division=0)),
                    "f1": float(f1_score(y_true_group, y_pred_group, zero_division=0)),
                    "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                    "n_samples": len(y_true_group),
                    "n_positive": int(y_true_group.sum()),
                    "positive_rate": float(y_true_group.mean()),
                }

                results[col][str(group_value)] = group_metrics

                # Log to MLflow
                if log_to_mlflow:
                    for metric_name, value in group_metrics.items():
                        mlflow.log_metric(f"{col}/{group_value}/{metric_name}", value)

            except Exception as e:
                logger.warning(f"Failed to compute metrics for {col}={group_value}: {e}")
                continue

    return results


def log_fairness_report(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a fairness report DataFrame and optionally save to CSV.

    Args:
        results: Results from compute_subgroup_metrics
        output_path: Optional path to save CSV report

    Returns:
        DataFrame with fairness metrics
    """
    rows = []

    for subgroup_col, subgroup_data in results.items():
        if subgroup_col == "overall":
            continue

        for group_value, metrics in subgroup_data.items():
            row = {
                "subgroup": subgroup_col,
                "value": group_value,
                **metrics
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Fairness report saved to {output_path}")

    return df


def compute_disparity_metrics(
    results: Dict[str, Dict[str, float]],
    reference_group: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute disparity metrics (ratio of metrics compared to reference group).

    Args:
        results: Results from compute_subgroup_metrics
        reference_group: Reference group for comparison (default: group with highest n_samples)

    Returns:
        Dictionary with disparity ratios
    """
    disparity = {}

    for subgroup_col, subgroup_data in results.items():
        if subgroup_col == "overall":
            continue

        # Find reference group (largest group if not specified)
        if reference_group is None:
            ref_key = max(subgroup_data.keys(), key=lambda k: subgroup_data[k]["n_samples"])
        else:
            ref_key = reference_group

        if ref_key not in subgroup_data:
            logger.warning(f"Reference group {ref_key} not found in {subgroup_col}")
            continue

        ref_metrics = subgroup_data[ref_key]
        disparity[subgroup_col] = {}

        for group_value, metrics in subgroup_data.items():
            if group_value == ref_key:
                continue

            # Compute ratio for each metric
            group_disparity = {}
            for metric in ["auc", "precision", "recall", "f1"]:
                if ref_metrics[metric] > 0:
                    ratio = metrics[metric] / ref_metrics[metric]
                    group_disparity[f"{metric}_ratio"] = float(ratio)

            disparity[subgroup_col][str(group_value)] = group_disparity

    return disparity


def print_fairness_summary(
    results: Dict[str, Dict[str, float]],
    disparity: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Print a summary of fairness metrics to console."""
    print("\n" + "="*80)
    print("FAIRNESS METRICS SUMMARY")
    print("="*80)

    # Overall metrics
    if "overall" in results:
        print("\nOverall Performance:")
        print("-" * 40)
        for metric, value in results["overall"].items():
            if metric in ["auc", "auc_pr", "precision", "recall", "f1", "accuracy"]:
                print(f"  {metric:15s}: {value:.4f}")
            else:
                print(f"  {metric:15s}: {value:,}")

    # Per-subgroup metrics
    for subgroup_col, subgroup_data in results.items():
        if subgroup_col == "overall":
            continue

        print(f"\n{subgroup_col.upper()} Breakdown:")
        print("-" * 40)

        # Sort by sample size
        sorted_groups = sorted(
            subgroup_data.items(),
            key=lambda x: x[1]["n_samples"],
            reverse=True
        )

        for group_value, metrics in sorted_groups[:10]:  # Top 10 groups
            print(f"\n  {group_value} (n={metrics['n_samples']:,}):")
            print(f"    AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    # Disparity metrics
    if disparity:
        print("\n" + "="*80)
        print("DISPARITY ANALYSIS (ratio compared to reference group)")
        print("="*80)

        for subgroup_col, subgroup_disparities in disparity.items():
            print(f"\n{subgroup_col.upper()}:")
            for group_value, ratios in subgroup_disparities.items():
                print(f"  {group_value}:")
                for metric, ratio in ratios.items():
                    flag = "⚠️" if ratio < 0.8 or ratio > 1.2 else "✓"
                    print(f"    {flag} {metric:20s}: {ratio:.3f}")

    print("\n" + "="*80)
