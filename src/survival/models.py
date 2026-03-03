"""
Survival model training wrappers for XGBoost AFT and Cox.

Provides train_survival_aft and train_survival_cox functions that
follow the same patterns as the classifier training in run_test.py.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .data_prep import create_aft_labels, create_cox_labels


class CheckpointCallback(xgb.callback.TrainingCallback):
    """
    XGBoost callback that saves model checkpoints during training.

    Saves the booster to disk every `period` rounds. Keeps only the
    latest `max_to_keep` checkpoints to avoid filling disk. Also writes
    a small JSON metadata file alongside each checkpoint so training
    can be resumed with full context.

    Usage:
        cb = CheckpointCallback("checkpoints/y_moved", period=50)
        xgb.train(params, dtrain, callbacks=[cb])

        # Resume from latest checkpoint:
        model_path, meta = CheckpointCallback.latest_checkpoint("checkpoints/y_moved")
        xgb.train(params, dtrain, xgb_model=model_path, ...)
    """

    def __init__(
        self,
        directory: str,
        period: int = 50,
        max_to_keep: int = 3,
        event_name: str = "",
    ):
        self.directory = directory
        self.period = period
        self.max_to_keep = max_to_keep
        self.event_name = event_name
        self._saved: list = []

    def after_iteration(self, model, epoch, evals_log):
        """Save checkpoint every `period` rounds."""
        round_num = epoch + 1  # epoch is 0-indexed
        if round_num % self.period != 0:
            return False  # continue training

        os.makedirs(self.directory, exist_ok=True)

        # Save model
        model_path = os.path.join(self.directory, f"checkpoint_{round_num:05d}.json")
        model.save_model(model_path)

        # Save metadata
        meta = {
            "round": round_num,
            "event_name": self.event_name,
            "num_boosted_rounds": model.num_boosted_rounds(),
        }
        # Include eval metrics if available
        if evals_log:
            meta["evals"] = {
                ds: {metric: vals[-1] for metric, vals in metrics.items()}
                for ds, metrics in evals_log.items()
            }
        meta_path = os.path.join(self.directory, f"checkpoint_{round_num:05d}.meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._saved.append(round_num)

        # Prune old checkpoints
        while len(self._saved) > self.max_to_keep:
            old_round = self._saved.pop(0)
            for suffix in [".json", ".meta.json"]:
                old_path = os.path.join(self.directory, f"checkpoint_{old_round:05d}{suffix}")
                try:
                    os.unlink(old_path)
                except OSError:
                    pass

        return False  # continue training

    @staticmethod
    def latest_checkpoint(directory: str) -> Optional[tuple]:
        """
        Find the latest checkpoint in a directory.

        Returns:
            (model_path, meta_dict) or None if no checkpoint found.
        """
        if not os.path.isdir(directory):
            return None

        checkpoints = sorted(
            p for p in os.listdir(directory)
            if p.startswith("checkpoint_") and p.endswith(".json")
            and not p.endswith(".meta.json")
        )
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        model_path = os.path.join(directory, latest)
        meta_path = model_path.replace(".json", ".meta.json")

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        return model_path, meta


def _get_default_aft_params():
    """Default parameters for XGBoost AFT."""
    return {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'aft_loss_distribution': 'normal',
        'aft_loss_distribution_scale': 1.0,
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_weight': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.2,
        'reg_lambda': 3.0,
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42,
    }


def _get_default_cox_params():
    """Default parameters for XGBoost Cox."""
    return {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_weight': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.2,
        'reg_lambda': 3.0,
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42,
    }


def train_survival_aft(
    X_train,
    duration_train,
    event_train,
    X_test=None,
    duration_test=None,
    event_test=None,
    params=None,
    n_estimators=500,
    early_stopping_rounds=30,
    verbose_eval=25,
    checkpoint_dir=None,
    checkpoint_period=50,
    event_name='',
    resume_from=None,
):
    """
    Train XGBoost Accelerated Failure Time (AFT) model.

    Args:
        X_train: Training features (DataFrame or array)
        duration_train: Training durations
        event_train: Training event indicators (1=event, 0=censored)
        X_test: Optional test features for early stopping
        duration_test: Optional test durations
        event_test: Optional test event indicators
        params: Optional XGBoost parameters dict
        n_estimators: Number of boosting rounds
        early_stopping_rounds: Early stopping patience
        verbose_eval: Verbosity interval
        checkpoint_dir: Directory to save checkpoints (None to disable)
        checkpoint_period: Save checkpoint every N rounds
        event_name: Event name for checkpoint metadata
        resume_from: Path to model checkpoint to resume from

    Returns:
        Trained xgb.Booster model
    """
    if params is None:
        params = _get_default_aft_params()

    # Create AFT labels
    y_lower_train, y_upper_train = create_aft_labels(duration_train, event_train)

    # Create DMatrix with AFT bounds
    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info('label_lower_bound', y_lower_train)
    dtrain.set_float_info('label_upper_bound', y_upper_train)

    evals = [(dtrain, 'train')]

    if X_test is not None and duration_test is not None and event_test is not None:
        y_lower_test, y_upper_test = create_aft_labels(duration_test, event_test)
        dtest = xgb.DMatrix(X_test)
        dtest.set_float_info('label_lower_bound', y_lower_test)
        dtest.set_float_info('label_upper_bound', y_upper_test)
        evals.append((dtest, 'test'))

    # Build callbacks
    callbacks = []
    if checkpoint_dir:
        callbacks.append(CheckpointCallback(
            directory=checkpoint_dir,
            period=checkpoint_period,
            event_name=event_name,
        ))

    # Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        xgb_model=resume_from,
        callbacks=callbacks or None,
    )

    return model


def train_survival_cox(
    X_train,
    duration_train,
    event_train,
    X_test=None,
    duration_test=None,
    event_test=None,
    params=None,
    n_estimators=500,
    early_stopping_rounds=30,
    verbose_eval=25,
    checkpoint_dir=None,
    checkpoint_period=50,
    event_name='',
    resume_from=None,
):
    """
    Train XGBoost Cox Proportional Hazards model.

    Args:
        X_train: Training features
        duration_train: Training durations
        event_train: Training event indicators (1=event, 0=censored)
        X_test: Optional test features
        duration_test: Optional test durations
        event_test: Optional test event indicators
        params: Optional XGBoost parameters dict
        n_estimators: Number of boosting rounds
        early_stopping_rounds: Early stopping patience
        verbose_eval: Verbosity interval
        checkpoint_dir: Directory to save checkpoints (None to disable)
        checkpoint_period: Save checkpoint every N rounds
        event_name: Event name for checkpoint metadata
        resume_from: Path to model checkpoint to resume from

    Returns:
        Trained xgb.Booster model
    """
    if params is None:
        params = _get_default_cox_params()

    # Create Cox labels (positive = event, negative = censored)
    y_train = create_cox_labels(duration_train, event_train)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    evals = [(dtrain, 'train')]

    if X_test is not None and duration_test is not None and event_test is not None:
        y_test = create_cox_labels(duration_test, event_test)
        dtest = xgb.DMatrix(X_test, label=y_test)
        evals.append((dtest, 'test'))

    # Build callbacks
    callbacks = []
    if checkpoint_dir:
        callbacks.append(CheckpointCallback(
            directory=checkpoint_dir,
            period=checkpoint_period,
            event_name=event_name,
        ))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        xgb_model=resume_from,
        callbacks=callbacks or None,
    )

    return model


def train_survival_model(
    X_train,
    duration_train,
    event_train,
    X_test=None,
    duration_test=None,
    event_test=None,
    config=None,
    model_type='aft',
    checkpoint_dir=None,
    checkpoint_period=50,
    event_name='',
    resume_from=None,
):
    """
    Unified training function that dispatches to AFT or Cox.

    Args:
        X_train, duration_train, event_train: Training data
        X_test, duration_test, event_test: Optional test data
        config: Model configuration dict (from YAML)
        model_type: 'aft' or 'cox'
        checkpoint_dir: Directory to save checkpoints (None to disable)
        checkpoint_period: Save checkpoint every N rounds
        event_name: Event name for checkpoint metadata
        resume_from: Path to model checkpoint to resume from

    Returns:
        Trained xgb.Booster model
    """
    if config is not None:
        model_params = config.get('model', {}).get('params', {}).copy()
        model_type = config.get('model', {}).get('type', model_type)
        n_estimators = model_params.pop('n_estimators', 500)
        early_stopping_rounds = model_params.pop('early_stopping_rounds', 30)
        verbose_eval = model_params.pop('verbose_eval', 25)
    else:
        model_params = None
        n_estimators = 500
        early_stopping_rounds = 30
        verbose_eval = 25

    ckpt_kwargs = dict(
        checkpoint_dir=checkpoint_dir,
        checkpoint_period=checkpoint_period,
        event_name=event_name,
        resume_from=resume_from,
    )

    if model_type == 'cox':
        return train_survival_cox(
            X_train, duration_train, event_train,
            X_test, duration_test, event_test,
            params=model_params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            **ckpt_kwargs,
        )
    else:
        return train_survival_aft(
            X_train, duration_train, event_train,
            X_test, duration_test, event_test,
            params=model_params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            **ckpt_kwargs,
        )


def train_all_events(
    df_train,
    df_test,
    feature_cols,
    event_cols,
    config=None,
    model_type='aft',
    checkpoint_dir=None,
    checkpoint_period=50,
):
    """
    Train one survival model per event type.

    Args:
        df_train: Training DataFrame (must have {event}_duration and {event}_event_observed columns)
        df_test: Test DataFrame
        feature_cols: List of feature column names
        event_cols: List of event column names (e.g., ['y_moved', 'birth1_event', ...])
        config: Model configuration dict
        model_type: 'aft' or 'cox'
        checkpoint_dir: Base directory for checkpoints (subdir per event)
        checkpoint_period: Save checkpoint every N rounds

    Returns:
        Dict mapping event_col → trained model
    """
    models = {}

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    for event_col in event_cols:
        duration_col = f"{event_col}_duration"
        observed_col = f"{event_col}_event_observed"

        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model for: {event_col}")
        print(f"{'='*60}")

        duration_train = df_train[duration_col].values
        event_train = df_train[observed_col].values
        duration_test = df_test[duration_col].values
        event_test = df_test[observed_col].values

        n_events = event_train.sum()
        n_censored = len(event_train) - n_events
        print(f"  Train: {n_events:,} events, {n_censored:,} censored "
              f"({n_events / len(event_train):.1%} event rate)")

        event_ckpt_dir = os.path.join(checkpoint_dir, event_col) if checkpoint_dir else None

        model = train_survival_model(
            X_train, duration_train, event_train,
            X_test, duration_test, event_test,
            config=config,
            model_type=model_type,
            checkpoint_dir=event_ckpt_dir,
            checkpoint_period=checkpoint_period,
            event_name=event_col,
        )

        models[event_col] = model
        print(f"  Trained {model.num_boosted_rounds()} trees for {event_col}")

        # Memory cleanup
        gc.collect()

    return models


def train_incremental_survival(
    parquet_path,
    total_rows,
    feature_cols,
    event_col,
    config=None,
    model_type='aft',
    target_batch_rows=None,
    checkpoint_dir=None,
    checkpoint_period=50,
):
    """
    Train a survival model incrementally on large datasets.

    Loads data year-by-year and continues training the model
    on each batch.

    Args:
        parquet_path: Path to parquet directory
        total_rows: Total number of rows
        feature_cols: List of feature column names
        event_col: Event column name
        config: Model configuration
        model_type: 'aft' or 'cox'
        target_batch_rows: Target rows per batch
        checkpoint_dir: Directory to save checkpoints (None to disable)
        checkpoint_period: Save checkpoint every N rounds

    Returns:
        Trained xgb.Booster model
    """
    import pyarrow.parquet as pq

    duration_col = f"{event_col}_duration"
    observed_col = f"{event_col}_event_observed"

    if config is not None:
        params = config.get('model', {}).get('params', {}).copy()
        params.pop('n_estimators', None)
        params.pop('early_stopping_rounds', None)
        params.pop('verbose_eval', None)
    else:
        params = _get_default_aft_params() if model_type == 'aft' else _get_default_cox_params()

    # Load test set (years 2024-2025)
    print(f"  Loading test set for {event_col}...")
    test_df = pd.read_parquet(parquet_path, filters=[('year', '>=', 2024)])

    if len(test_df) > 100_000:
        test_df = test_df.sample(n=100_000, random_state=42)

    X_test = test_df[feature_cols]

    if model_type == 'aft':
        y_lower_test, y_upper_test = create_aft_labels(
            test_df[duration_col].values, test_df[observed_col].values
        )
        dtest = xgb.DMatrix(X_test)
        dtest.set_float_info('label_lower_bound', y_lower_test)
        dtest.set_float_info('label_upper_bound', y_upper_test)
    else:
        y_test = create_cox_labels(
            test_df[duration_col].values, test_df[observed_col].values
        )
        dtest = xgb.DMatrix(X_test, label=y_test)

    del test_df
    gc.collect()

    # Train incrementally on year batches
    model = None
    train_years = list(range(2011, 2024))
    rows_per_year = total_rows // 15
    if target_batch_rows is None:
        target_batch_rows = 10_000_000
    years_per_batch = max(1, int(target_batch_rows / rows_per_year))

    for i in range(0, len(train_years), years_per_batch):
        year_batch = train_years[i:i + years_per_batch]
        year_str = f"{year_batch[0]}-{year_batch[-1]}" if len(year_batch) > 1 else str(year_batch[0])

        print(f"  Batch: year {year_str}...", end=" ", flush=True)

        batch = pd.read_parquet(parquet_path, filters=[('year', 'in', year_batch)])
        if len(batch) == 0:
            print("skipped (empty)")
            continue

        X_batch = batch[feature_cols]

        if model_type == 'aft':
            y_lower, y_upper = create_aft_labels(
                batch[duration_col].values, batch[observed_col].values
            )
            dtrain = xgb.DMatrix(X_batch)
            dtrain.set_float_info('label_lower_bound', y_lower)
            dtrain.set_float_info('label_upper_bound', y_upper)
        else:
            y_batch_labels = create_cox_labels(
                batch[duration_col].values, batch[observed_col].values
            )
            dtrain = xgb.DMatrix(X_batch, label=y_batch_labels)

        rounds = 50 if model is None else 20

        callbacks = []
        if checkpoint_dir:
            callbacks.append(CheckpointCallback(
                directory=checkpoint_dir,
                period=checkpoint_period,
                event_name=event_col,
            ))

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            xgb_model=model,
            evals=[(dtest, 'test')],
            verbose_eval=0,
            callbacks=callbacks or None,
        )

        print(f"{len(batch):,} rows, {model.num_boosted_rounds()} trees")

        del batch, X_batch, dtrain
        gc.collect()

    return model
