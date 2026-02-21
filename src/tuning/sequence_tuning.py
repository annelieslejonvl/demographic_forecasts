"""
Hyperparameter tuning for PyTorch sequence models using Optuna.

Optimizes model architecture and training parameters via Bayesian
optimization (TPE sampler) with epoch-level pruning.

Usage:
    python run_with_municipality_sequence.py --reuse --config configs/models/pytorch_seq_gru_fast.yaml \
        --tune --n-trials 50 --search-space conservative
"""
import copy
import gc
import json
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from ..sequence.estimator import PyTorchSequenceEstimator
from ..sequence.vocabulary import LifeEventVocabulary

logger = logging.getLogger(__name__)


class SequenceModelTuner:
    """
    Hyperparameter tuner for PyTorch sequence models using Optuna.

    Features:
    - Bayesian optimization with TPE sampler (multivariate)
    - Epoch-level pruning via MedianPruner on val_mean_auc
    - GPU memory cleanup between trials
    - MLflow integration for trial logging
    - Two search spaces: 'full' and 'conservative'
    - Saves best params to JSON and optionally writes a tuned YAML config
    """

    # Predefined composite metrics: weighted combinations of metrics.
    # Keys must match the epoch_callback metrics dict from estimator.fit().
    COMPOSITE_METRICS = {
        'composite': {
            'val_mean_c_index': 0.4,
            'val_mean_crpss': 0.3,  # skill score: higher is better (0=naive, 1=perfect)
            'val_mean_f1': 0.3,
        },
        'composite_no_aft': {
            'val_mean_ap': 0.4,
            'val_mean_f1': 0.3,
            'val_mean_auc': 0.3,
        },
        'ap_f1': {
            'val_mean_ap': 0.5,
            'val_mean_f1': 0.5,
        },
    }
    # Metrics where lower is better — use (1 - v) in composite
    _INVERT_METRICS: set = set()  # CRPSS is already higher-is-better

    def __init__(
        self,
        base_config: Dict[str, Any],
        vocabulary: LifeEventVocabulary,
        events: List[str],
        horizons: List[int],
        objective_metric: str = "composite",
        direction: str = "maximize",
        tuning_epochs: int = 20,
        seed: int = 42,
    ):
        self.base_config = base_config
        self.vocabulary = vocabulary
        self.events = events
        self.horizons = horizons
        self.objective_metric = objective_metric
        self.direction = direction
        self.tuning_epochs = tuning_epochs
        self.seed = seed

        self.study_: Optional[optuna.Study] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_value_: Optional[float] = None

    # ------------------------------------------------------------------
    # Search spaces
    # ------------------------------------------------------------------

    def get_search_space(
        self, trial: optuna.Trial, encoder_type: str,
    ) -> Dict[str, Any]:
        """Full search space — broad exploration."""
        params: Dict[str, Any] = {}

        # Architecture
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)

        encoder: Dict[str, Any] = {}
        if encoder_type in ('lstm', 'gru'):
            params['embed_dim'] = trial.suggest_categorical(
                'embed_dim', [16, 32, 64, 128],
            )
            encoder['hidden_dim'] = trial.suggest_categorical(
                'hidden_dim', [64, 128, 256, 512],
            )
            encoder['num_layers'] = trial.suggest_int('num_layers', 1, 2)
            encoder['dropout'] = trial.suggest_float('encoder_dropout', 0.1, 0.5)
        elif encoder_type == 'transformer':
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
            valid_dims = [d for d in [32, 64, 128, 256] if d % num_heads == 0]
            params['embed_dim'] = trial.suggest_categorical(
                'embed_dim', valid_dims or [64],
            )
            encoder['num_heads'] = num_heads
            encoder['num_layers'] = trial.suggest_int('num_layers', 1, 6)
            encoder['ff_dim'] = trial.suggest_categorical('ff_dim', [256, 512, 1024])
            encoder['dropout'] = trial.suggest_float('encoder_dropout', 0.05, 0.3)
        params['encoder'] = encoder

        # Prediction head
        n_head_layers = trial.suggest_int('n_head_layers', 1, 3)
        head_dims = []
        for i in range(n_head_layers):
            head_dims.append(
                trial.suggest_categorical(f'head_dim_{i}', [32, 64, 128, 256]),
            )
        params['head_hidden_dims'] = head_dims

        # Training
        params['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True,
        )
        params['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True,
        )
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', [64, 128, 256, 512],
        )

        # Loss configuration
        #params['loss_type'] = trial.suggest_categorical(
        #    'loss_type', ['aft', 'bce', 'focal'],
        #)
        #if params['loss_type'] == 'focal':
        #    params['focal_gamma'] = trial.suggest_float('focal_gamma', 0.5, 5.0)
        params['multi_head'] = trial.suggest_categorical(
            'multi_head', [True, False],
        )
        params['event_weight_mode'] = trial.suggest_categorical(
            'event_weight_mode', ['uniform', 'inverse_rate', 'learned'],
        )

        # Balanced sampling & horizon weights
        params['balanced_sampling'] = trial.suggest_categorical(
            'balanced_sampling', [True, False],
        )
        #params['horizon_weights'] = trial.suggest_categorical(
        #    'horizon_weights', ['none', 'auto', '3_1.5_1'],
        #)
        # Calibration (full space only)
        params['calibration_method'] = trial.suggest_categorical(
            'calibration_method', ['none', 'isotonic', 'platt'],
        )

        return params

    def get_conservative_search_space(
        self, trial: optuna.Trial, encoder_type: str,
    ) -> Dict[str, Any]:
        """Conservative search space — narrower ranges for faster convergence."""
        params: Dict[str, Any] = {}

        params['dropout'] = trial.suggest_float('dropout', 0.15, 0.4)

        encoder: Dict[str, Any] = {}
        if encoder_type in ('lstm', 'gru'):
            params['embed_dim'] = trial.suggest_categorical(
                'embed_dim', [16,32],
            )
            encoder['hidden_dim'] = trial.suggest_categorical(
                'hidden_dim', [32,64],
            )
            encoder['num_layers'] = trial.suggest_int('num_layers', 1, 2)
            encoder['dropout'] = trial.suggest_float('encoder_dropout', 0.2, 0.5)
        elif encoder_type == 'transformer':
            num_heads = trial.suggest_categorical('num_heads', [4, 8])
            valid_dims = [d for d in [64, 128, 256] if d % num_heads == 0]
            params['embed_dim'] = trial.suggest_categorical(
                'embed_dim', valid_dims or [64],
            )
            encoder['num_heads'] = num_heads
            encoder['num_layers'] = trial.suggest_int('num_layers', 2, 4)
            encoder['ff_dim'] = trial.suggest_categorical('ff_dim', [256, 512])
            encoder['dropout'] = trial.suggest_float('encoder_dropout', 0.1, 0.2)
        params['encoder'] = encoder

        params['head_hidden_dims'] = [
            trial.suggest_categorical('head_dim_0', [32, 64, 128]),
        ]

        params['learning_rate'] = trial.suggest_float(
            'learning_rate', 5e-5, 5e-3, log=True,
        )
        params['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-5, 5e-3, log=True,
        )
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', [128, 256, 512],
        )

        # Loss config
        params['loss_type'] = trial.suggest_categorical(
            'loss_type', ['aft'],
        )
        if params['loss_type'] == 'focal':
            params['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)
        params['multi_head'] = trial.suggest_categorical(
            'multi_head', [True],
        )
        params['event_weight_mode'] = trial.suggest_categorical(
            'event_weight_mode', ['uniform', 'inverse_rate'],
        )

        # Balanced sampling & horizon weights (conservative)
        params['balanced_sampling'] = trial.suggest_categorical(
            'balanced_sampling', [True, False],
        )
        if params.get('loss_type') == 'aft':
            params['horizon_weights'] = 'none'
        else:
            params['horizon_weights'] = trial.suggest_categorical(
                'horizon_weights', ['none', 'auto'],
            )
        params['mc_dropout_samples'] = trial.suggest_int('mc_dropout_samples',40,100,10, log=False)


        return params

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_model_config(
        self, trial_params: Dict[str, Any], encoder_type: str,
    ) -> Dict[str, Any]:
        """Build a model config dict for PyTorchSequenceEstimator from trial params."""
        base_params = self.base_config.get('model', {}).get('params', {})

        return {
            'type': self.base_config.get('model', {}).get('type', f'seq_{encoder_type}'),
            'params': {
                # Fixed from base config
                'encoder_type': encoder_type,
                'max_seq_len': base_params.get('max_seq_len', 256),
                'events': self.events,
                'horizons': self.horizons,
                'use_amp': base_params.get('use_amp', True),
                'early_stopping_patience': max(3, self.tuning_epochs // 5),
                'epochs': self.tuning_epochs,
                'lr_find': False,  # Optuna tunes LR instead
                # Trial-suggested params
                'embed_dim': trial_params['embed_dim'],
                'encoder': trial_params['encoder'],
                'head_hidden_dims': trial_params['head_hidden_dims'],
                'dropout': trial_params['dropout'],
                'learning_rate': trial_params['learning_rate'],
                'weight_decay': trial_params['weight_decay'],
                'batch_size': trial_params['batch_size'],
                'loss_type': trial_params['loss_type'],
                'focal_gamma': trial_params.get('focal_gamma', 2.0),
                'multi_head': trial_params['multi_head'],
                'event_weight_mode': trial_params['event_weight_mode'],
                'balanced_sampling': trial_params.get('balanced_sampling', False),
                'horizon_weights': self._decode_horizon_weights(
                    trial_params.get('horizon_weights', 'none'),
                ),
                'calibration_method': self._decode_calibration(
                    trial_params.get('calibration_method', 'none'),
                ),
                'mc_dropout_samples' : trial_params.get('mc_dropout_samples', 0)
            },
        }

    @staticmethod
    def _decode_horizon_weights(value: str):
        """Convert Optuna string encoding back to config value."""
        if value == 'none' or value is None:
            return None
        if value == 'auto':
            return 'auto'
        # e.g. '3_1.5_1' -> [3.0, 1.5, 1.0]
        return [float(x) for x in value.split('_')]

    @staticmethod
    def _decode_calibration(value: str):
        """Convert Optuna string to calibration method or None."""
        if value == 'none' or value is None:
            return None
        return value

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample_train(dataset, fraction: float, seed: int):
        """Stratified subsample of training dataset for a single trial.

        For chunked datasets, selects a fraction of chunks.
        For regular datasets, uses a Subset with stratified sampling
        that preserves the event rate.

        Returns a new dataset (original is not modified for chunked).
        """
        if fraction >= 1.0:
            return dataset

        rng = np.random.RandomState(seed)

        if getattr(dataset, '_using_chunks', False):
            import copy
            ds = copy.copy(dataset)
            n_chunks = len(ds._chunk_sizes)
            n_use = max(1, int(n_chunks * fraction))
            selected = sorted(rng.choice(n_chunks, n_use, replace=False))
            ds._chunk_paths = [dataset._chunk_paths[i] for i in selected]
            ds._chunk_sizes = [dataset._chunk_sizes[i] for i in selected]
            ds._chunk_offsets = []
            offset = 0
            for s in ds._chunk_sizes:
                ds._chunk_offsets.append(offset)
                offset += s
            ds._chunk_offsets.append(offset)
            ds._total_len = offset
            ds._chunk_cache = {}
            ds._chunk_cache_order = []
            return ds
        else:
            from torch.utils.data import Subset

            class _SubsetWithAttrs(Subset):
                def get_pos_weights(self, max_samples=None):
                    return self.dataset.get_pos_weights(max_samples=max_samples)

            n = max(1, int(len(dataset) * fraction))
            idx = rng.choice(len(dataset), n, replace=False).tolist()
            return _SubsetWithAttrs(dataset, idx)

    def objective(
        self,
        trial: optuna.Trial,
        train_dataset,
        eval_dataset,
        search_space: str = "full",
        parent_run_id: Optional[str] = None,
        tuning_fraction: float = 1.0,
    ) -> float:
        """Optuna objective function — train one trial, return metric."""
        import mlflow

        encoder_type = self.base_config.get('model', {}).get('params', {}).get(
            'encoder_type', 'lstm',
        )

        # Suggest hyperparameters
        if search_space == "conservative":
            trial_params = self.get_conservative_search_space(trial, encoder_type)
        else:
            trial_params = self.get_search_space(trial, encoder_type)

        # Per-trial subsample of training data (different seed per trial)
        if tuning_fraction < 1.0:
            trial_train = self._subsample_train(
                train_dataset, tuning_fraction, seed=self.seed + trial.number,
            )
        else:
            trial_train = train_dataset

        model_config = self._build_model_config(trial_params, encoder_type)

        # Track best metric across epochs
        best_metric = float('-inf') if self.direction == "maximize" else float('inf')

        def _compute_objective(metrics: Dict[str, float]) -> float:
            """Compute objective value from epoch metrics."""
            if self.objective_metric in self.COMPOSITE_METRICS:
                weights = self.COMPOSITE_METRICS[self.objective_metric]
                total = 0.0
                all_valid = True
                for key, w in weights.items():
                    v = metrics.get(key, float('nan'))
                    if np.isnan(v):
                        all_valid = False
                        break
                    # Invert lower-is-better metrics (e.g. CRPS)
                    if key in self._INVERT_METRICS:
                        v = 1.0 - v
                    total += w * v
                if all_valid:
                    return total
                # Fallback: if AFT metrics missing (non-AFT trial), use composite_no_aft
                if 'composite_no_aft' in self.COMPOSITE_METRICS:
                    fb_weights = self.COMPOSITE_METRICS['composite_no_aft']
                    fb_total = 0.0
                    for key, w in fb_weights.items():
                        v = metrics.get(key, float('nan'))
                        if np.isnan(v):
                            return float('nan')
                        fb_total += w * v
                    return fb_total
                return float('nan')
            return metrics.get(self.objective_metric, float('nan'))

        def epoch_callback(epoch: int, metrics: Dict[str, float]):
            nonlocal best_metric
            value = _compute_objective(metrics)
            if np.isnan(value):
                return

            if self.direction == "maximize":
                best_metric = max(best_metric, value)
            else:
                best_metric = min(best_metric, value)

            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            with mlflow.start_run(
                run_name=f"tune_trial_{trial.number}",
                nested=True,
                parent_run_id=parent_run_id,
            ):
                # Log all params — flatten nested dicts (encoder), join lists
                flat_params: Dict[str, Any] = {}
                for k, v in trial_params.items():
                    if isinstance(v, dict):
                        for dk, dv in v.items():
                            flat_params[f'{k}_{dk}'] = dv
                    elif isinstance(v, list):
                        flat_params[k] = str(v)
                    else:
                        flat_params[k] = v
                flat_params['trial_number'] = trial.number
                flat_params['tuning_fraction'] = tuning_fraction
                mlflow.log_params(flat_params)

                estimator = PyTorchSequenceEstimator(
                    model_config=model_config,
                    device_config=None,
                )

                result = estimator.fit(
                    train_dataset=trial_train,
                    eval_dataset=eval_dataset,
                    vocabulary=self.vocabulary,
                    epoch_callback=epoch_callback,
                )

                final_metric = best_metric
                epochs_trained = result.metadata.get('epochs_trained', 0)
                n_params = result.metadata.get('n_params', 0)
                val_loss = result.metrics.get('val_loss')
                best_composite = result.metrics.get('best_composite')

                trial.set_user_attr('val_loss', val_loss)
                trial.set_user_attr('epochs_trained', epochs_trained)
                trial.set_user_attr('n_params', n_params)
                trial.set_user_attr(self.objective_metric, final_metric)

                # Log final trial metrics explicitly so they show in MLflow child run
                mlflow.log_metric('trial_best_composite', final_metric)
                mlflow.log_metric('trial_epochs_trained', epochs_trained)
                mlflow.log_metric('trial_n_params', n_params)
                if val_loss is not None:
                    mlflow.log_metric('trial_val_loss', val_loss)
                if best_composite is not None:
                    mlflow.log_metric('trial_best_composite_restored', best_composite)

                logger.info(
                    f"Trial {trial.number}: {self.objective_metric}={final_metric:.4f}, "
                    f"epochs={epochs_trained}, "
                    f"params={n_params:,}"
                )
                print(
                    f"  Trial {trial.number}: "
                    f"{self.objective_metric}={final_metric:.4f}, "
                    f"epochs={epochs_trained}"
                )

                del estimator, result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return final_metric

        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            print(f"  Trial {trial.number}: pruned")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0 if self.direction == "maximize" else 999.0

    # ------------------------------------------------------------------
    # Main tuning entry point
    # ------------------------------------------------------------------

    def tune(
        self,
        train_dataset,
        eval_dataset,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        search_space: str = "full",
        show_progress: bool = True,
        n_jobs: int = 1,
        tuning_fraction: float = 1.0,
    ) -> Dict[str, Any]:
        """Run the full Optuna study.

        Args:
            tuning_fraction: Train each trial on this fraction of data (0-1).
                Each trial gets a different random subset. Enables parallel
                trials on a single GPU with small models (GRU embed=16-32).
                Val set is always used in full for comparable metrics.
        """
        import mlflow

        # Disable Optuna's logging entirely — its StreamHandler conflicts
        # with TeeOutput (redirected sys.stdout/stderr) and crashes on emit().
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna_logger = optuna.logging.get_logger("optuna")
        optuna_logger.handlers.clear()
        optuna_logger.addHandler(logging.NullHandler())

        logger.info(f"Starting sequence model tuning: {n_trials} trials")
        logger.info(f"Objective: {self.direction} {self.objective_metric}")
        logger.info(f"Search space: {search_space}, tuning_epochs: {self.tuning_epochs}")

        frac_str = f", {tuning_fraction:.0%} data/trial" if tuning_fraction < 1.0 else ""
        print(f"\n  Optuna: {n_trials} trials, "
              f"{self.direction} {self.objective_metric}, "
              f"search_space={search_space}, n_jobs={n_jobs}{frac_str}")

        sampler = TPESampler(seed=self.seed, multivariate=True)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=6,  # don't prune until epoch 7 — rare events need time to show signal
            interval_steps=2,  # check every 2 epochs instead of every epoch
        )

        self.study_ = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name="sequence_model_tuning",
        )

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("demographic_forecasts_sequence_tuning")

        with mlflow.start_run(run_name=f"tuning_{n_trials}trials_seed{self.seed}"):
            mlflow.log_params({
                'n_trials': n_trials,
                'search_space': search_space,
                'objective_metric': self.objective_metric,
                'tuning_epochs': self.tuning_epochs,
                'train_size': len(train_dataset),
                'val_size': len(eval_dataset),
                'tpe_seed': self.seed,
                'n_jobs': n_jobs,
                'tuning_fraction': tuning_fraction,
            })

            # Capture parent run ID before optimize() — worker threads have their
            # own MLflow thread-local context so nested=True won't work there.
            parent_run_id = mlflow.active_run().info.run_id

            # Disable progress bar — it conflicts with TeeOutput stdout redirect
            self.study_.optimize(
                lambda trial: self.objective(
                    trial, train_dataset, eval_dataset, search_space, parent_run_id,
                    tuning_fraction=tuning_fraction,
                ),
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=False,
            )

            # Gather results
            self.best_params_ = self.study_.best_params
            self.best_value_ = self.study_.best_value
            best_trial = self.study_.best_trial

            best_metrics = {
                'val_loss': best_trial.user_attrs.get('val_loss'),
                'epochs_trained': best_trial.user_attrs.get('epochs_trained'),
                'n_params': best_trial.user_attrs.get('n_params'),
            }

            mlflow.log_metric(f"best_{self.objective_metric}", self.best_value_)
            mlflow.log_params({
                f"best_{k}": v for k, v in self.best_params_.items()
                if not isinstance(v, (dict, list))
            })

        n_complete = len([
            t for t in self.study_.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        n_pruned = len([
            t for t in self.study_.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ])
        n_failed = len([
            t for t in self.study_.trials
            if t.state == optuna.trial.TrialState.FAIL
        ])

        logger.info(
            f"Tuning complete: {n_complete} complete, "
            f"{n_pruned} pruned, {n_failed} failed"
        )
        logger.info(f"Best {self.objective_metric}: {self.best_value_:.4f}")
        logger.info(f"Best params: {self.best_params_}")

        return {
            'best_params': self.best_params_,
            'best_value': self.best_value_,
            'best_metrics': best_metrics,
            'study': self.study_,
            'n_complete': n_complete,
            'n_pruned': n_pruned,
            'n_failed': n_failed,
        }

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(results: Dict[str, Any], save_path: str):
        """Save tuning results to JSON."""
        output = {
            'best_params': results['best_params'],
            'best_value': float(results['best_value']),
            'best_metrics': {
                k: float(v) if v is not None else None
                for k, v in results.get('best_metrics', {}).items()
            },
            'n_complete': results.get('n_complete', 0),
            'n_pruned': results.get('n_pruned', 0),
            'n_failed': results.get('n_failed', 0),
        }
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Tuning results saved to {save_path}")

    @staticmethod
    def save_tuned_config(
        results: Dict[str, Any],
        base_config: Dict[str, Any],
        output_path: str,
    ):
        """Merge best params into base YAML config and write to file."""
        import yaml

        config = copy.deepcopy(base_config)
        best = results['best_params']
        params = config.setdefault('model', {}).setdefault('params', {})

        # Map flat Optuna params back to nested config
        for key in ('embed_dim', 'dropout', 'learning_rate', 'weight_decay',
                     'batch_size', 'loss_type', 'focal_gamma',
                     'multi_head', 'event_weight_mode',
                     'balanced_sampling'):
            if key in best:
                params[key] = best[key]

        # Decode horizon_weights from Optuna string encoding
        if 'horizon_weights' in best:
            params['horizon_weights'] = SequenceModelTuner._decode_horizon_weights(
                best['horizon_weights'],
            )
        # Decode calibration_method
        if 'calibration_method' in best:
            params['calibration_method'] = SequenceModelTuner._decode_calibration(
                best['calibration_method'],
            )

        # Encoder params
        encoder = params.setdefault('encoder', {})
        for key in ('hidden_dim', 'num_layers', 'num_heads', 'ff_dim'):
            if key in best:
                encoder[key] = best[key]
        if 'encoder_dropout' in best:
            encoder['dropout'] = best['encoder_dropout']

        # Head dims — reconstruct from head_dim_0, head_dim_1, ...
        head_dims = []
        for i in range(10):
            key = f'head_dim_{i}'
            if key in best:
                head_dims.append(best[key])
        if head_dims:
            params['head_hidden_dims'] = head_dims

        # Restore full training epochs (not the reduced tuning epochs)
        orig_params = base_config.get('model', {}).get('params', {})
        params['epochs'] = orig_params.get('epochs', 50)
        params['early_stopping_patience'] = orig_params.get(
            'early_stopping_patience', 10,
        )
        # Re-enable LR finder if it was on in the base config
        if orig_params.get('lr_find', False):
            params['lr_find'] = True

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Tuned config saved to {output_path}")
        print(f"  Tuned config written to {output_path}")
