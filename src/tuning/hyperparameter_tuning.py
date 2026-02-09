"""
Hyperparameter Tuning for XGBoost Migration Models using Optuna.

Optimizes model parameters to maximize AUC while maintaining good calibration.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from ..backends.pipeline import UnifiedPipeline, PipelineConfig
from ..backends.base import BackendType, DeviceType

logger = logging.getLogger(__name__)


class XGBoostTuner:
    """
    Hyperparameter tuner for XGBoost models using Optuna.

    Features:
    - Bayesian optimization with TPE sampler
    - Early stopping to save time
    - Multi-metric evaluation (AUC, Brier, calibration)
    - Best model persistence
    """

    def __init__(
        self,
        base_config: PipelineConfig,
        feature_cols: List[str],
        categorical_cols: List[str],
        numerical_cols: List[str],
        objective_metric: str = "auc_roc",  # or "auc_pr", "brier"
        direction: str = "maximize",  # or "minimize" for brier
    ):
        """
        Initialize tuner.

        Args:
            base_config: Base pipeline configuration
            feature_cols: List of feature columns
            categorical_cols: Categorical feature columns
            numerical_cols: Numerical feature columns
            objective_metric: Metric to optimize ("auc_roc", "auc_pr", "brier")
            direction: "maximize" or "minimize"
        """
        self.base_config = base_config
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.objective_metric = objective_metric
        self.direction = direction

        self.best_params_ = None
        self.best_value_ = None
        self.study_ = None

    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters
        """
        return {
            # Tree structure
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),

            # Sampling
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),

            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

            # Learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),

            # Other
            'max_bin': trial.suggest_int('max_bin', 128, 512, step=64),
        }

    def get_conservative_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Conservative search space for faster tuning.

        Focus on most important parameters with narrower ranges.
        """
        return {
            # Most important parameters
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),

            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),

            # Sampling
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),

            # Fixed good defaults
            'min_child_weight': 5,
            'gamma': 0.5,
            'max_bin': 256,
        }

    def objective(
        self,
        trial: optuna.Trial,
        train_data: Any,
        val_data: Any,
        search_space: str = "full",  # "full" or "conservative"
    ) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial
            train_data: Training data (numpy or Spark)
            val_data: Validation data (numpy or Spark)
            search_space: "full" or "conservative"

        Returns:
            Metric value to optimize
        """
        import gc

        # Get hyperparameters from trial
        if search_space == "conservative":
            params = self.get_conservative_search_space(trial)
        else:
            params = self.get_search_space(trial)

        # Add fixed parameters
        params['early_stopping_rounds'] = 50
        params['verbose_eval'] = False

        # Create config with trial parameters
        config = PipelineConfig(
            backend=self.base_config.backend,
            model_type=self.base_config.model_type,
            device=self.base_config.device,
            gpu_id=self.base_config.gpu_id,
            categorical_encoding=self.base_config.categorical_encoding,
            label_col=self.base_config.label_col,
            cat_cols=self.categorical_cols,
            num_cols=self.numerical_cols,
            model_params=params,
        )

        # Create and train pipeline
        try:
            pipeline = UnifiedPipeline(config)

            # For tuning, don't use batching to avoid memory issues across 50 trials
            pipeline.fit(
                train_data=train_data,
                eval_data=val_data,
                feature_cols=self.feature_cols,
                use_batches=False,  # Avoid batching during tuning
                batch_size=None,
            )

            # Predict on validation set
            # Use sampled validation if Spark DataFrame to save memory
            if hasattr(val_data, 'toPandas'):
                # Spark DataFrame - sample 20% for faster evaluation during tuning
                val_sample = val_data.sample(fraction=0.2, seed=42)
                y_val, y_pred = pipeline.predict_proba_batched(
                    val_sample,
                    batch_size=100_000  # Smaller batches
                )
            else:
                # Already numpy - predict directly
                X_val, y_val = val_data
                pred_result = pipeline.estimator_.predict_proba(X_val)
                y_pred = pred_result.probabilities

            # Compute metrics
            metrics = {
                'auc_roc': roc_auc_score(y_val, y_pred[:, 1]),
                'auc_pr': average_precision_score(y_val, y_pred[:, 1]),
                'brier': brier_score_loss(y_val, y_pred[:, 1]),
            }

            # Add calibration metric (how close is mean prediction to true rate)
            calibration_error = abs(y_pred[:, 1].mean() - y_val.mean())
            metrics['calibration_error'] = calibration_error

            # Log all metrics to trial
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)

            # Return objective metric
            objective_value = metrics[self.objective_metric]

            logger.info(
                f"Trial {trial.number}: {self.objective_metric}={objective_value:.4f}, "
                f"AUC={metrics['auc_roc']:.4f}, Brier={metrics['brier']:.4f}, "
                f"Calib_err={calibration_error:.4f}"
            )

            # Clean up to save memory between trials
            del pipeline
            gc.collect()

            return objective_value

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            # Clean up on error
            gc.collect()
            # Return worst possible value
            return 0.0 if self.direction == "maximize" else 999.0

    def tune(
        self,
        train_data: Any,
        val_data: Any,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        search_space: str = "full",
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            train_data: Training data
            val_data: Validation data
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None for no limit)
            search_space: "full" or "conservative"
            n_jobs: Number of parallel jobs (1 = sequential)
            show_progress: Show progress bar

        Returns:
            Dictionary with best parameters and metrics
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        logger.info(f"Objective: {self.direction} {self.objective_metric}")
        logger.info(f"Search space: {search_space}")

        # Create study
        sampler = TPESampler(seed=42, multivariate=True)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        self.study_ = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

        # Run optimization
        self.study_.optimize(
            lambda trial: self.objective(
                trial, train_data, val_data, search_space
            ),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
        )

        # Extract best results
        self.best_params_ = self.study_.best_params
        self.best_value_ = self.study_.best_value

        best_trial = self.study_.best_trial
        best_metrics = {
            'auc_roc': best_trial.user_attrs.get('auc_roc'),
            'auc_pr': best_trial.user_attrs.get('auc_pr'),
            'brier': best_trial.user_attrs.get('brier'),
            'calibration_error': best_trial.user_attrs.get('calibration_error'),
        }

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best {self.objective_metric}: {self.best_value_:.4f}")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"All metrics: {best_metrics}")

        return {
            'best_params': self.best_params_,
            'best_value': self.best_value_,
            'best_metrics': best_metrics,
            'study': self.study_,
        }

    def get_best_pipeline(self) -> UnifiedPipeline:
        """
        Create pipeline with best parameters.

        Returns:
            Configured UnifiedPipeline
        """
        if self.best_params_ is None:
            raise RuntimeError("Must run tune() before getting best pipeline")

        config = PipelineConfig(
            backend=self.base_config.backend,
            model_type=self.base_config.model_type,
            device=self.base_config.device,
            gpu_id=self.base_config.gpu_id,
            categorical_encoding=self.base_config.categorical_encoding,
            label_col=self.base_config.label_col,
            cat_cols=self.categorical_cols,
            num_cols=self.numerical_cols,
            model_params=self.best_params_,
        )

        return UnifiedPipeline(config)

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.

        Args:
            save_path: Path to save plot (None to display)
        """
        if self.study_ is None:
            raise RuntimeError("Must run tune() before plotting")

        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )
            import plotly.io as pio

            # Optimization history
            fig = plot_optimization_history(self.study_)
            if save_path:
                pio.write_html(fig, f"{save_path}_history.html")
            else:
                fig.show()

            # Parameter importances
            fig = plot_param_importances(self.study_)
            if save_path:
                pio.write_html(fig, f"{save_path}_importance.html")
            else:
                fig.show()

            # Parallel coordinate plot
            fig = plot_parallel_coordinate(self.study_)
            if save_path:
                pio.write_html(fig, f"{save_path}_parallel.html")
            else:
                fig.show()

        except ImportError:
            logger.warning("Install plotly for visualization: pip install plotly")


def quick_tune(
    train_data: Any,
    val_data: Any,
    feature_cols: List[str],
    categorical_cols: List[str],
    numerical_cols: List[str],
    label_col: str = "y_moved",
    n_trials: int = 50,
    device: str = "auto",
    objective_metric: str = "auc_pr",  # Default to AUC-PR for imbalanced data
) -> Tuple[Dict[str, Any], UnifiedPipeline]:
    """
    Quick hyperparameter tuning with sensible defaults.

    Args:
        train_data: Training data
        val_data: Validation data
        feature_cols: Feature columns
        categorical_cols: Categorical columns
        numerical_cols: Numerical columns
        label_col: Label column name
        n_trials: Number of trials
        device: Device to use ("cpu", "gpu", "auto")
        objective_metric: Metric to optimize ("auc_pr", "auc_roc", "brier")
                         Default is "auc_pr" which is better for imbalanced data

    Returns:
        Tuple of (results dict, best pipeline)
    """
    # Create base config
    base_config = PipelineConfig(
        backend=BackendType.XGBOOST,
        model_type="classifier",
        device=DeviceType(device),
        categorical_encoding="native",
        label_col=label_col,
    )

    # Determine optimization direction
    direction = "minimize" if objective_metric == "brier" else "maximize"

    # Create tuner
    tuner = XGBoostTuner(
        base_config=base_config,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        objective_metric=objective_metric,
        direction=direction,
    )

    # Run tuning
    results = tuner.tune(
        train_data=train_data,
        val_data=val_data,
        n_trials=n_trials,
        search_space="conservative",  # Faster
        show_progress=True,
    )

    # Get best pipeline
    best_pipeline = tuner.get_best_pipeline()

    return results, best_pipeline


def save_tuning_results(
    results: Dict[str, Any],
    save_path: str,
    include_study: bool = False
):
    """
    Save tuning results to file.

    Args:
        results: Results from tune()
        save_path: Path to save JSON file
        include_study: Save full study object (large file)
    """
    import json

    output = {
        'best_params': results['best_params'],
        'best_value': float(results['best_value']),
        'best_metrics': {
            k: float(v) if v is not None else None
            for k, v in results['best_metrics'].items()
        },
    }

    if include_study:
        # Save study as separate pickle
        import joblib
        study_path = save_path.replace('.json', '_study.pkl')
        joblib.dump(results['study'], study_path)
        output['study_path'] = study_path

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Tuning results saved to {save_path}")
