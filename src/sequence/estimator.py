"""
PyTorch sequence model estimator integrating with the BaseEstimator interface.

Wraps SequenceModel (LSTM/GRU/Transformer) into the BackendFactory pattern
used by the rest of the project.
"""
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..backends.base import (
    BaseEstimator,
    DeviceConfig,
    PredictResult,
    TrainResult,
)
from ..utils.device import get_device_manager
from .dataset import SequenceDataset, sequence_collate_fn
from .losses import AFTLoss, DeepHitLoss, EventBCELoss, FocalLoss, LearnedWeightedLoss
from .models import SequenceModel
from .vocabulary import LifeEventVocabulary, EVENT_TOKEN_MAP

logger = logging.getLogger(__name__)

DEFAULT_EVENTS = list(EVENT_TOKEN_MAP.keys())
DEFAULT_HORIZONS = [1, 3, 5]

# ---------------------------------------------------------------------------
# LR Finder
# ---------------------------------------------------------------------------

class LRFinderResult:
    """Stores results from a learning rate range test."""

    def __init__(self, lrs: List[float], losses: List[float], suggested_lr: float):
        self.lrs = lrs
        self.losses = losses
        self.suggested_lr = suggested_lr

    def plot(self, save_path: Optional[str] = None, skip_start: int = 10, skip_end: int = 5):
        """Plot loss vs learning rate and save to file."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses, linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss (smoothed)')
        ax.set_title('LR Finder')

        # Mark minimum loss point
        all_min_idx = int(np.argmin(self.losses))
        if all_min_idx < len(self.lrs):
            ax.axvline(x=self.lrs[all_min_idx], color='blue', linestyle=':',
                        alpha=0.6, label=f'Min loss @ {self.lrs[all_min_idx]:.2e}')

        # Mark suggested LR
        ax.axvline(x=self.suggested_lr, color='r', linestyle='--',
                    linewidth=2, label=f'Suggested LR: {self.suggested_lr:.2e}')
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info(f"LR finder plot saved to {save_path}")
        plt.close(fig)
        return fig

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _progress_iter(items: Iterable, desc: str, total: Optional[int] = None) -> Iterable:
    if tqdm is None:
        return items
    if total is None and hasattr(items, "__len__"):
        try:
            total = len(items)
        except Exception:
            total = None
    return tqdm(items, desc=desc, total=total, leave=True, dynamic_ncols=True)


class PyTorchSequenceEstimator(BaseEstimator):
    """
    PyTorch sequence model estimator for multi-label event prediction.

    Supports LSTM, GRU, and Transformer encoders via the encoder_type config.
    Registered as seq_lstm, seq_gru, seq_transformer in BackendFactory.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_config: Optional[DeviceConfig] = None,
    ):
        super().__init__(model_config, device_config)

        params = model_config.get('params', {})
        self.encoder_type = params.get('encoder_type', 'lstm')
        self.embed_dim = params.get('embed_dim', 128)
        self.max_seq_len = params.get('max_seq_len', 256)
        self.encoder_config = params.get('encoder', {})
        self.head_hidden_dims = params.get('head_hidden_dims', [128, 64])
        self.dropout = params.get('dropout', 0.1)

        self.learning_rate = params.get('learning_rate', 1e-3)
        self.weight_decay = params.get('weight_decay', 1e-4)
        self.epochs = params.get('epochs', 50)
        self.batch_size = params.get('batch_size', 64)
        self.early_stopping_patience = params.get('early_stopping_patience', 10)
        self.early_stopping_metric = params.get('early_stopping_metric', 'composite')
        self.use_amp = bool(params.get('use_amp', True))

        # LR finder config
        self.lr_find = bool(params.get('lr_find', False))
        lr_find_cfg = params.get('lr_find_config', {})
        self.lr_find_min = float(lr_find_cfg.get('lr_min', 1e-7))
        self.lr_find_max = float(lr_find_cfg.get('lr_max', 1.0))
        self.lr_find_steps = int(lr_find_cfg.get('num_steps', 200))
        self.lr_find_smooth = float(lr_find_cfg.get('smooth_factor', 0.05))
        self.lr_find_diverge = float(lr_find_cfg.get('diverge_threshold', 4.0))
        self.lr_find_fraction = float(lr_find_cfg.get('fraction', 0.1))

        self.loss_type = params.get('loss_type', 'bce')
        self.focal_gamma = float(params.get('focal_gamma', 2.0))
        self.multi_head = bool(params.get('multi_head', False))
        self.event_weight_mode = params.get('event_weight_mode', 'uniform')
        self.use_numeric_features = bool(params.get('use_numeric_features', False))
        self.numeric_inject = params.get('numeric_inject', 'add')
        if self.numeric_inject not in {'add', 'concat'}:
            logger.warning(
                "numeric_inject='%s' requested, but only 'add'|'concat' are supported; "
                "falling back to 'add'.",
                self.numeric_inject,
            )
            self.numeric_inject = 'add'

        # Horizon weighting: [3.0, 1.5, 1.0] | "auto" | null
        hw_config = params.get('horizon_weights', None)
        if hw_config == 'auto':
            self.horizon_weight_mode = 'auto'
            self.horizon_weights_manual = None
        elif isinstance(hw_config, list):
            self.horizon_weight_mode = 'manual'
            self.horizon_weights_manual = hw_config
        else:
            self.horizon_weight_mode = None
            self.horizon_weights_manual = None
        if self.loss_type == 'deephit' and self.horizon_weight_mode is not None:
            self.deephit_alpha = params.get('deephit_alpha', 0.1)
            self.deephit_sigma_rank = params.get('deephit_sigma_rank', 0.1)
            self.n_rank_pairs = params.get('deephit_n_rank_pairs', 128)
        # Balanced sampling
        self.balanced_sampling = bool(params.get('balanced_sampling', False))

        # Probability calibration
        self.calibration_method = params.get('calibration_method', None)

        # MC Dropout UQ
        self.mc_dropout_samples = int(params.get('mc_dropout_samples', 0))

        self.events = params.get('events', DEFAULT_EVENTS)
        self.horizons = params.get('horizons', DEFAULT_HORIZONS)

        self.model_: Optional[SequenceModel] = None
        self.vocabulary_: Optional[LifeEventVocabulary] = None
        self.optimal_thresholds_: Optional[np.ndarray] = None  # (n_events * n_horizons,)

    def _aft_to_probs(self, output: torch.Tensor) -> torch.Tensor:
        """Convert AFT output (mu, log_sigma) to per-horizon probabilities.

        If ``sigma_scales_`` is set (from ``calibrate_sigma_per_event``),
        each event's sigma is multiplied by its calibrated scale factor.

        Returns (batch, n_events * n_horizons) tensor matching BCE layout.
        """
        n_events = len(self.events)
        mu = output[:, :n_events]
        log_sigma = output[:, n_events:]
        sigma = torch.exp(log_sigma).clamp(min=0.01)

        # Apply per-event sigma calibration if available
        if hasattr(self, 'sigma_scales_') and self.sigma_scales_:
            scales = torch.tensor(
                [self.sigma_scales_.get(ei, 1.0) for ei in range(n_events)],
                dtype=torch.float32, device=output.device,
            )
            sigma = sigma * scales.unsqueeze(0)

        log_h = torch.log(torch.tensor(
            self.horizons, dtype=torch.float32, device=output.device,
        ))
        z = (log_h.unsqueeze(0).unsqueeze(0) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        probs = torch.sigmoid(z)  # (batch, n_events, n_horizons)
        return probs.reshape(output.size(0), -1)

    def _deephit_to_probs(self, output: torch.Tensor) -> torch.Tensor:
        """Convert DeepHit logits to cumulative horizon probabilities.

        Applies per-event softmax over (n_horizons + 1) bins (intervals +
        censored), then cumulative-sums the interval PMF to get CDF-like
        probabilities P(T <= h) at each horizon.

        Returns (batch, n_events * n_horizons) tensor matching BCE layout.
        """
        n_events = len(self.events)
        n_horizons = len(self.horizons)
        logits_3d = output.view(-1, n_events, n_horizons)
        # Append implicit censored logit (0 = reference category)
        censored = torch.zeros(
            output.size(0), n_events, 1,
            device=output.device, dtype=output.dtype,
        )
        logits_ext = torch.cat([logits_3d, censored], dim=-1)
        pmf = torch.softmax(logits_ext, dim=-1)[:, :, :n_horizons]
        # Cumulative sum → CDF: P(T <= h)
        cdf = pmf.cumsum(dim=-1)
        return cdf.reshape(output.size(0), -1)

    def _init_aft_biases(self, train_dataset, n_events, n_horizons):
        """Initialize AFT head output biases from population event statistics.

        For each event, computes the population-level log-median-time and
        log-sigma from training targets, then sets the output layer bias
        so the model starts near the marginal distribution instead of
        mu=0, log_sigma=0 for all events.
        """
        from src.sequence.evaluation import _targets_to_survival

        # Sample targets
        max_samples = min(200_000, len(train_dataset))
        loader = DataLoader(
            train_dataset, batch_size=4096, shuffle=False,
            collate_fn=sequence_collate_fn, drop_last=False,
        )
        all_tgt = []
        n = 0
        for batch in loader:
            all_tgt.append(batch['targets'].numpy())
            n += all_tgt[-1].shape[0]
            if n >= max_samples:
                break
        targets = np.concatenate(all_tgt, axis=0)[:max_samples]

        head = self.model_.head  # MultiEventHead
        for ei, event in enumerate(self.events):
            cols = [ei * n_horizons + hi for hi in range(n_horizons)]
            event_tgt = targets[:, cols]
            duration, event_ind = _targets_to_survival(event_tgt, self.horizons)

            event_mask = event_ind.astype(bool)
            if event_mask.sum() < 10:
                logger.info(f"  AFT init {event}: too few events, using defaults")
                continue

            log_dur = np.log(np.maximum(duration[event_mask], 1e-8))
            mu_init = float(np.mean(log_dur))
            sigma_init = float(max(np.std(log_dur), 0.3))
            log_sigma_init = float(np.log(sigma_init))

            # Set the bias of the last linear layer in this event's head
            last_layer = head.heads[ei][-1]  # nn.Linear(hidden, 2)
            with torch.no_grad():
                last_layer.bias[0] = mu_init       # mu bias
                last_layer.bias[1] = log_sigma_init  # log_sigma bias

            med_time = np.exp(mu_init)
            evt_rate = float(event_mask.mean())
            logger.info(
                f"  AFT init {event}: mu={mu_init:.2f} (median={med_time:.2f}yr), "
                f"log_sigma={log_sigma_init:.2f} (sigma={sigma_init:.2f}), "
                f"event_rate={evt_rate:.3f}"
            )

    def _compute_event_weights(self, train_dataset) -> torch.Tensor:
        """Compute inverse-prevalence per-event weights from training data."""
        from torch.utils.data import IterableDataset
        n_events = len(self.events)
        n_horizons = len(self.horizons)
        max_samples = None
        if getattr(train_dataset, '_using_chunks', False):
            max_samples = min(500_000, len(train_dataset))
        elif isinstance(train_dataset, IterableDataset):
            # Streaming dataset: cap to avoid scanning the entire dataset
            max_samples = 10_000
            logger.info(f"Streaming dataset: capping event weight computation to {max_samples:,} samples")
        pw = train_dataset.get_pos_weights(max_samples=max_samples)

        # Use longest horizon rate per event
        rates = []
        for ei in range(n_events):
            col = ei * n_horizons + (n_horizons - 1)
            pw_val = pw[col].item()
            rate = 1.0 / (1.0 + pw_val) if pw_val > 0 else 0.01
            rates.append(rate)

        weights = torch.tensor([1.0 / r for r in rates], dtype=torch.float32)
        weights = weights * n_events / weights.sum()  # normalize: sum = n_events

        for ei, event in enumerate(self.events):
            logger.info(f"  Event weight {event}: {weights[ei]:.3f} (rate={rates[ei]:.3%})")
        return weights

    def _compute_horizon_weights(self, train_dataset, device: torch.device) -> Optional[torch.Tensor]:
        """Compute horizon weights tensor based on config.

        Returns:
            Tensor of shape (n_horizons,) or None if no weighting.
        """
        if self.horizon_weight_mode is None:
            return None

        n_events = len(self.events)
        n_horizons = len(self.horizons)

        if self.horizon_weight_mode == 'manual':
            hw = torch.tensor(self.horizon_weights_manual, dtype=torch.float32)
            if len(hw) != n_horizons:
                raise ValueError(
                    f"horizon_weights has {len(hw)} values but {n_horizons} horizons configured"
                )
            logger.info(f"Using manual horizon_weights: {self.horizon_weights_manual}")
            return hw.to(device)

        # auto: inverse prevalence per horizon, averaged across events
        from torch.utils.data import IterableDataset
        max_samples = None
        if getattr(train_dataset, '_using_chunks', False):
            max_samples = min(500_000, len(train_dataset))
        elif isinstance(train_dataset, IterableDataset):
            max_samples = 10_000
        pw = train_dataset.get_pos_weights(max_samples=max_samples)

        horizon_rates = []
        for hi in range(n_horizons):
            event_rates = []
            for ei in range(n_events):
                col = ei * n_horizons + hi
                pw_val = pw[col].item()
                rate = 1.0 / (1.0 + pw_val) if pw_val > 0 else 0.01
                event_rates.append(rate)
            horizon_rates.append(np.mean(event_rates))

        # Inverse rate, normalize so shortest horizon gets highest weight
        hw = torch.tensor([1.0 / r for r in horizon_rates], dtype=torch.float32)
        hw = hw / hw.min()  # normalize so minimum weight = 1.0
        logger.info(
            f"Auto horizon_weights (inverse prevalence): "
            + ", ".join(f"{h}yr={w:.2f} (rate={r:.4f})" for h, w, r in zip(self.horizons, hw, horizon_rates))
        )
        return hw.to(device)

    def _build_loss_fn(
        self,
        device: torch.device,
        pos_weight: Optional[torch.Tensor] = None,
        event_weights: Optional[torch.Tensor] = None,
        horizon_weights: Optional[torch.Tensor] = None,
    ):
        """Build loss function based on loss_type and event_weight_mode."""
        n_events = len(self.events)
        n_horizons = len(self.horizons)
        params = self.model_config.get('params', {})
        if self.loss_type == 'aft':
            if self.event_weight_mode == 'learned':
                base = AFTLoss(
                    self.horizons, n_events,
                    horizon_weights=horizon_weights,
                    reduction='per_event',
                ).to(device)
                return LearnedWeightedLoss(base, n_events).to(device)
            return AFTLoss(
                self.horizons, n_events,
                event_weights=event_weights,
                horizon_weights=horizon_weights,
            ).to(device)
        elif self.loss_type == 'deephit':
            alpha = self.deephit_alpha if hasattr(self, 'deephit_alpha') else 0.1
            sigma_rank = self.deephit_sigma_rank if hasattr(self, 'deephit_sigma_rank') else 0.1
            n_rank_pairs = self.n_rank_pairs if hasattr(self, 'n_rank_pairs') else 128
            if self.event_weight_mode == 'learned':
                base = DeepHitLoss(
                    self.horizons, n_events,
                    alpha=alpha, sigma_rank=sigma_rank,
                    n_rank_pairs=n_rank_pairs,
                    horizon_weights=horizon_weights,
                    reduction='per_event',
                ).to(device)
                return LearnedWeightedLoss(base, n_events).to(device)
            return DeepHitLoss(
                self.horizons, n_events,
                alpha=alpha, sigma_rank=sigma_rank,
                n_rank_pairs=n_rank_pairs,
                event_weights=event_weights,
                horizon_weights=horizon_weights,
            ).to(device)
        elif self.loss_type == 'focal':
            if self.event_weight_mode == 'learned':
                base = FocalLoss(
                    n_events, n_horizons, gamma=self.focal_gamma,
                    pos_weight=pos_weight,
                    horizon_weights=horizon_weights,
                    reduction='per_event',
                ).to(device)
                return LearnedWeightedLoss(base, n_events).to(device)
            return FocalLoss(
                n_events, n_horizons, gamma=self.focal_gamma,
                pos_weight=pos_weight,
                event_weights=event_weights,
                horizon_weights=horizon_weights,
            ).to(device)
        else:
            if self.event_weight_mode == 'learned':
                base = EventBCELoss(
                    n_events, n_horizons,
                    pos_weight=pos_weight,
                    horizon_weights=horizon_weights,
                    reduction='per_event',
                ).to(device)
                return LearnedWeightedLoss(base, n_events).to(device)
            return EventBCELoss(
                n_events, n_horizons,
                pos_weight=pos_weight,
                event_weights=event_weights,
                horizon_weights=horizon_weights,
            ).to(device)

    def find_lr(
        self,
        train_dataset: SequenceDataset,
        vocabulary: LifeEventVocabulary,
        device: Optional[str] = None,
        lr_min: float = 1e-7,
        lr_max: float = 1.0,
        num_steps: int = 200,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 4.0,
        fraction: float = 0.1,
        save_path: Optional[str] = None,
    ) -> LRFinderResult:
        """
        Learning rate range test (Smith 2017).

        Trains for `num_steps` mini-batches with LR increasing exponentially
        from `lr_min` to `lr_max`. Records smoothed loss at each step.
        Only uses `fraction` of the training data.

        Args:
            train_dataset: Training SequenceDataset.
            vocabulary: LifeEventVocabulary.
            device: Torch device string.
            lr_min: Starting learning rate.
            lr_max: Ending learning rate.
            num_steps: Number of mini-batch steps to run.
            smooth_factor: EMA smoothing factor (lower = smoother).
            diverge_threshold: Stop if loss exceeds diverge_threshold * best_loss.
            fraction: Fraction of training data to use (0-1].
            save_path: Path to save the LR-vs-loss plot.

        Returns:
            LRFinderResult with suggested learning rate.
        """
        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = device.type == "cuda"

        n_events = len(self.events)
        n_horizons = len(self.horizons)

        # Build a fresh model
        from .vocabulary import N_NUMERIC_FEATURES
        n_numeric = N_NUMERIC_FEATURES if self.use_numeric_features else 0

        model = SequenceModel(
            vocab_size=vocabulary.vocab_size,
            embed_dim=self.embed_dim,
            encoder_type=self.encoder_type,
            encoder_config=self.encoder_config,
            n_events=n_events,
            n_horizons=n_horizons,
            max_seq_len=self.max_seq_len,
            head_hidden_dims=self.head_hidden_dims,
            dropout=self.dropout,
            loss_type=self.loss_type,
            multi_head=self.multi_head,
            n_numeric_features=n_numeric,
            numeric_inject=self.numeric_inject,
        ).to(device)
        model.train()

        # DataLoader — subsample to `fraction` of the data
        from torch.utils.data import SubsetRandomSampler
        num_workers = 0
        n_total = len(train_dataset)
        n_subset = max(self.batch_size, int(n_total * fraction))

        if getattr(train_dataset, '_using_chunks', False):
            # For chunked datasets: select a subset of chunks to avoid
            # cross-chunk random access (expensive torch.load per chunk)
            from .dataset import ChunkShuffledSampler
            n_chunks = len(train_dataset._chunk_sizes)
            n_chunks_use = max(1, int(n_chunks * fraction))
            rng = np.random.RandomState(42)
            selected = sorted(rng.choice(n_chunks, n_chunks_use, replace=False))
            sub_offsets = [train_dataset._chunk_offsets[i] for i in selected]
            sub_sizes = [train_dataset._chunk_sizes[i] for i in selected]
            train_sampler = ChunkShuffledSampler(
                chunk_offsets=sub_offsets,
                chunk_sizes=sub_sizes,
            )
            n_subset = sum(sub_sizes)
            logger.info(f"LR Finder: using {n_chunks_use}/{n_chunks} chunks ({n_subset:,} samples)")
        else:
            rng = np.random.RandomState(42)
            indices = rng.choice(n_total, n_subset, replace=False).tolist()
            train_sampler = SubsetRandomSampler(indices)

        # Cap num_steps to available batches
        max_batches = n_subset // self.batch_size
        if num_steps > max_batches and max_batches > 0:
            num_steps = max_batches
            logger.info(f"LR Finder: capped num_steps to {num_steps} (data limited)")

        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=sequence_collate_fn,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # Event weights (for LR finder, use inverse_rate even for 'learned' mode)
        event_weights = None
        ew_mode = self.event_weight_mode
        if ew_mode in ('inverse_rate', 'learned'):
            event_weights = self._compute_event_weights(train_dataset).to(device)

        # Horizon weights
        horizon_weights = self._compute_horizon_weights(train_dataset, device)

        # Loss function (use inverse_rate as fallback for learned — no time to learn in sweep)
        saved_mode = self.event_weight_mode
        if self.event_weight_mode == 'learned':
            self.event_weight_mode = 'inverse_rate'  # temporary override
        if self.loss_type in ('aft', 'deephit'):
            loss_fn = self._build_loss_fn(device, event_weights=event_weights, horizon_weights=horizon_weights)
        else:
            max_pw = min(50_000, len(train_dataset)) if not isinstance(train_dataset, IterableDataset) else 50_000
            pos_weight = train_dataset.get_pos_weights(max_samples=max_pw).to(device)
            loss_fn = self._build_loss_fn(device, pos_weight=pos_weight, event_weights=event_weights, horizon_weights=horizon_weights)
        self.event_weight_mode = saved_mode

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr_min, weight_decay=self.weight_decay,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=use_cuda and self.use_amp)

        # Exponential LR schedule: lr = lr_min * (lr_max/lr_min)^(step/num_steps)
        mult = (lr_max / lr_min) ** (1.0 / num_steps)

        lrs: List[float] = []
        losses: List[float] = []
        best_loss = float('inf')
        smoothed_loss = 0.0

        data_iter = iter(loader)
        logger.info(f"LR Finder: sweeping {lr_min:.1e} -> {lr_max:.1e} over {num_steps} steps ({fraction:.0%} of data, {n_subset:,} samples)")
        print(f"  LR Finder: {lr_min:.1e} -> {lr_max:.1e}, {num_steps} steps ({fraction:.0%} of data, {n_subset:,} samples)")

        for step in range(num_steps):
            # Get next batch (loop over dataset if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device, non_blocking=use_cuda)
            attention_mask = batch['attention_mask'].to(device, non_blocking=use_cuda)
            targets = batch['targets'].to(device, non_blocking=use_cuda)
            numeric_features = batch.get('numeric_features')
            if numeric_features is not None:
                numeric_features = numeric_features.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                logits = model(input_ids, attention_mask, numeric_features=numeric_features)
                loss = loss_fn(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())

            # Exponential moving average smoothing
            if step == 0:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_factor * loss_val + (1 - smooth_factor) * smoothed_loss

            # Bias correction for EMA
            corrected_loss = smoothed_loss / (1 - (1 - smooth_factor) ** (step + 1))

            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            losses.append(corrected_loss)

            if corrected_loss < best_loss:
                best_loss = corrected_loss

            # Stop if loss diverges
            if step > 10 and corrected_loss > diverge_threshold * best_loss:
                logger.info(f"LR Finder: stopping early at step {step}, loss diverged")
                break

            # Update LR for next step
            for pg in optimizer.param_groups:
                pg['lr'] *= mult

            if (step + 1) % 50 == 0:
                print(f"    step {step+1}/{num_steps}: lr={current_lr:.2e}, loss={corrected_loss:.4f}")

        # Find suggested LR:
        # 1. Find LR at minimum loss (before divergence)
        # 2. Find LR at steepest descent (most useful learning signal)
        # 3. Use steepest descent as suggestion (backed off slightly from min)
        n = len(lrs)

        min_loss_idx = int(np.argmin(losses))
        lr_at_min = lrs[min_loss_idx]

        # Steepest descent: find where loss drops fastest on log-LR scale.
        # Only look between 20% of steps and the minimum loss point.
        suggestion_grad = lr_at_min / 3.0  # fallback: back off from min
        if n > 40:
            log_lrs = np.log10(lrs)
            # Skip flat region at start (LR too low to learn)
            skip = max(10, n // 5)
            end = max(skip + 10, min(min_loss_idx + 1, n - 5))
            if end > skip + 5:
                raw_grad = np.gradient(losses[skip:end], log_lrs[skip:end])
                # Smooth with sliding window to suppress noise
                win = max(5, len(raw_grad) // 20)
                if len(raw_grad) > win:
                    kernel = np.ones(win) / win
                    smooth_grad = np.convolve(raw_grad, kernel, mode='valid')
                    grad_min_idx = int(np.argmin(smooth_grad)) + skip + win // 2
                else:
                    grad_min_idx = int(np.argmin(raw_grad)) + skip
                if 0 <= grad_min_idx < n:
                    suggestion_grad = lrs[grad_min_idx]

        # Use the steepest descent LR — it sits in the sweet spot
        # between "too low to learn" and "about to diverge"
        suggested_lr = suggestion_grad

        # Clamp: at least 1e-6, at most the LR at minimum loss
        suggested_lr = max(1e-6, min(suggested_lr, lr_at_min))

        logger.info(f"LR Finder: min_loss@{lr_at_min:.2e}, "
                     f"steepest@{suggestion_grad:.2e}, suggested={suggested_lr:.2e}")

        result = LRFinderResult(lrs=lrs, losses=losses, suggested_lr=suggested_lr)

        logger.info(f"LR Finder suggested LR: {suggested_lr:.2e}")
        print(f"  LR Finder: min_loss@{lr_at_min:.2e}, steepest@{suggestion_grad:.2e} -> suggested={suggested_lr:.2e}")

        if save_path:
            result.plot(save_path=save_path)

        try:
            mlflow.log_metric("lr_finder_suggested", suggested_lr)
        except Exception:
            pass

        # Clean up
        del model, optimizer, scaler
        torch.cuda.empty_cache() if use_cuda else None

        return result

    def _get_gpu_device(self) -> str:
        dm = get_device_manager()
        return dm.get_torch_device(self.device_config.gpu_id)

    def _auto_detect_device(self) -> str:
        dm = get_device_manager()
        if dm.check_cuda_available():
            return dm.get_torch_device()
        return "cpu"

    def fit(
        self,
        X: Any = None,
        y: Any = None,
        sample_weight: Optional[Any] = None,
        eval_set: Optional[Any] = None,
        train_dataset: Optional[SequenceDataset] = None,
        eval_dataset: Optional[SequenceDataset] = None,
        vocabulary: Optional[LifeEventVocabulary] = None,
        device: Optional[str] = None,
        epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        initial_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_period: int = 5,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> TrainResult:
        """
        Train the sequence model.

        Args:
            train_dataset: SequenceDataset for training.
            eval_dataset: SequenceDataset for validation.
            vocabulary: LifeEventVocabulary (required for model construction).
            device: Override device selection.
            initial_state_dict: Pre-trained weights to warm-start from (federated learning).
            checkpoint_dir: Directory for epoch checkpoints (None to disable).
            checkpoint_period: Save checkpoint every N epochs (default 5).
            resume_from_checkpoint: Path to checkpoint directory to resume from.
            X, y, sample_weight, eval_set: For BaseEstimator compatibility (unused).
        """
        if train_dataset is None:
            raise ValueError("train_dataset is required for sequence model training")
        if vocabulary is None:
            raise ValueError("vocabulary is required for sequence model training")

        self.vocabulary_ = vocabulary

        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = device.type == "cuda"

        # LR finder (runs before training with a throwaway model)
        if self.lr_find:
            lr_plot_path = kwargs.get('lr_find_plot_path', 'lr_finder.png')
            lr_result = self.find_lr(
                train_dataset=train_dataset,
                vocabulary=vocabulary,
                device=str(device),
                lr_min=self.lr_find_min,
                lr_max=self.lr_find_max,
                num_steps=self.lr_find_steps,
                smooth_factor=self.lr_find_smooth,
                diverge_threshold=self.lr_find_diverge,
                fraction=self.lr_find_fraction,
                save_path=lr_plot_path,
            )
            old_lr = self.learning_rate
            self.learning_rate = lr_result.suggested_lr
            logger.info(f"LR Finder: updated learning_rate {old_lr:.2e} -> {self.learning_rate:.2e}")
            print(f"  LR Finder: {old_lr:.2e} -> {self.learning_rate:.2e}")

        logger.info(f"Training sequence model ({self.encoder_type}) on device: {device}")

        # Build model
        n_events = len(self.events)
        n_horizons = len(self.horizons)

        from .vocabulary import N_NUMERIC_FEATURES
        n_numeric = N_NUMERIC_FEATURES if self.use_numeric_features else 0

        self.model_ = SequenceModel(
            vocab_size=vocabulary.vocab_size,
            embed_dim=self.embed_dim,
            encoder_type=self.encoder_type,
            encoder_config=self.encoder_config,
            n_events=n_events,
            n_horizons=n_horizons,
            max_seq_len=self.max_seq_len,
            head_hidden_dims=self.head_hidden_dims,
            dropout=self.dropout,
            loss_type=self.loss_type,
            multi_head=self.multi_head,
            n_numeric_features=n_numeric,
            numeric_inject=self.numeric_inject,
        ).to(device)

        n_params = sum(p.numel() for p in self.model_.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        # Warm-start from pre-trained weights (federated learning)
        if initial_state_dict is not None:
            self.model_.load_state_dict(initial_state_dict)
            logger.info("Loaded initial weights (federated warm-start)")

        # Initialize AFT output biases from population-level event statistics
        elif self.loss_type == 'aft' and self.multi_head:
            self._init_aft_biases(train_dataset, n_events, n_horizons)

        # DataLoaders
        is_iterable_train = isinstance(train_dataset, IterableDataset)
        num_workers = int(self.device_config.num_workers) if self.device_config else 0
        # Chunked datasets use lazy loading - workers would each load chunks
        # separately and waste memory. Use 0 workers for chunked mode.
        if getattr(train_dataset, '_using_chunks', False) and num_workers > 0:
            logger.info(f"Chunked dataset detected: setting num_workers=0 (was {num_workers})")
            num_workers = 0
        # For chunked datasets, use ChunkShuffledSampler to avoid
        # random cross-chunk access (each torch.load is expensive)
        train_sampler = None
        balanced_sampler = None
        train_shuffle = False if is_iterable_train else True
        if getattr(train_dataset, '_using_chunks', False):
            if self.balanced_sampling:
                from .dataset import BalancedChunkSampler
                chunk_rates = train_dataset.get_chunk_event_rates(
                    n_events=n_events, n_horizons=n_horizons, target_horizon_idx=0,
                )
                balanced_sampler = BalancedChunkSampler(
                    chunk_offsets=train_dataset._chunk_offsets[:-1],
                    chunk_sizes=train_dataset._chunk_sizes,
                    chunk_event_rates=chunk_rates,
                )
                train_sampler = balanced_sampler
                logger.info(
                    f"Using BalancedChunkSampler ({len(train_dataset._chunk_sizes)} chunks, "
                    f"effective epoch size: {len(balanced_sampler)})"
                )
            else:
                from .dataset import ChunkShuffledSampler
                train_sampler = ChunkShuffledSampler(
                    chunk_offsets=train_dataset._chunk_offsets[:-1],  # exclude sentinel
                    chunk_sizes=train_dataset._chunk_sizes,
                )
                logger.info(f"Using ChunkShuffledSampler ({len(train_dataset._chunk_sizes)} chunks)")
            train_shuffle = False  # sampler handles shuffling
        elif self.balanced_sampling and not is_iterable_train:
            # Non-chunked: use WeightedRandomSampler
            from torch.utils.data import WeightedRandomSampler
            targets_np = train_dataset._targets.numpy() if hasattr(train_dataset, '_targets') else None
            if targets_np is not None:
                # Per-sample weight: oversample persons with any positive 1-year event
                any_pos = np.zeros(len(targets_np), dtype=bool)
                for ei in range(n_events):
                    col = ei * n_horizons  # 1-year horizon = index 0
                    any_pos |= (targets_np[:, col] > 0.5)
                pos_rate = any_pos.mean()
                sample_weights = np.where(any_pos, 1.0 / max(pos_rate, 1e-6), 1.0)
                sample_weights = sample_weights / sample_weights.mean()  # normalize mean=1
                train_sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights.astype(np.float64)),
                    num_samples=len(targets_np),
                    replacement=True,
                )
                train_shuffle = False
                logger.info(f"Using WeightedRandomSampler (pos_rate={pos_rate:.4f})")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=sequence_collate_fn,
            pin_memory=use_cuda,
            drop_last=False,
        )

        val_loader = None
        if eval_dataset is not None:
            val_loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=sequence_collate_fn,
                pin_memory=use_cuda,
                drop_last=False,
            )

        # Event weights — use eval_dataset if available (already in memory),
        # otherwise fall back to train_dataset
        weight_source = eval_dataset if eval_dataset is not None else train_dataset
        event_weights = None
        if self.event_weight_mode in ('inverse_rate', 'learned'):
            event_weights = self._compute_event_weights(weight_source).to(device)

        # Horizon weights
        horizon_weights = self._compute_horizon_weights(weight_source, device)

        # Loss function
        pos_weight = None
        if self.loss_type == 'aft':
            loss_fn = self._build_loss_fn(device, event_weights=event_weights, horizon_weights=horizon_weights)
            logger.info(f"Using AFT loss (log-logistic, horizons={self.horizons})")
        elif self.loss_type == 'deephit':
            loss_fn = self._build_loss_fn(device, event_weights=event_weights, horizon_weights=horizon_weights)
            logger.info(f"Using DeepHit loss (alpha={self.deephit_alpha}, horizons={self.horizons})")
        else:  # bce or focal
            if is_iterable_train and eval_dataset is None:
                pos_params = self.model_config.get('params', {})
                max_samples = int(pos_params.get('pos_weight_max_samples', 100000))
                if max_samples <= 0:
                    pos_weight = torch.ones(n_events * n_horizons, dtype=torch.float32, device=device)
                    logger.info("Skipping pos_weight computation for streaming dataset")
                else:
                    logger.info(f"Estimating pos_weight from {max_samples} samples (streaming)")
                    pos_weight = train_dataset.get_pos_weights(max_samples=max_samples).to(device)
            else:
                pw_source = eval_dataset if (is_iterable_train and eval_dataset is not None) else train_dataset
                max_pw_samples = None
                if getattr(pw_source, '_using_chunks', False):
                    max_pw_samples = min(500_000, len(pw_source))
                pos_weight = pw_source.get_pos_weights(max_samples=max_pw_samples).to(device)
            loss_fn = self._build_loss_fn(device, pos_weight=pos_weight, event_weights=event_weights, horizon_weights=horizon_weights)

        logger.info(f"Loss: {self.loss_type}, event_weight_mode: {self.event_weight_mode}, multi_head: {self.multi_head}")
        if horizon_weights is not None:
            logger.info(f"Horizon weights: {horizon_weights.cpu().tolist()}")

        # Store importance weights on dataset for collate to pick up
        if balanced_sampler is not None:
            # Build per-sample importance weight array from chunk-level weights
            n_total = len(train_dataset)
            iw = np.ones(n_total, dtype=np.float32)
            for ci in range(balanced_sampler.n_chunks):
                offset = train_dataset._chunk_offsets[ci]
                size = train_dataset._chunk_sizes[ci]
                iw[offset:offset + size] = balanced_sampler.chunk_importance_weights[ci]
            train_dataset._importance_weights = torch.from_numpy(iw)
            logger.info(
                f"Balanced sampling IW: min={iw.min():.3f}, max={iw.max():.3f}, "
                f"mean={iw.mean():.3f}"
            )
        else:
            train_dataset._importance_weights = None

        # Optimizer — include learned loss params if applicable
        all_params = list(self.model_.parameters())
        if isinstance(loss_fn, LearnedWeightedLoss):
            all_params += list(loss_fn.parameters())
            logger.info(f"Learned task weighting: {loss_fn.n_events} learnable log_var params added to optimizer")

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=use_cuda and self.use_amp)

        best_val_loss = float('inf')
        best_composite = -float('inf')
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            ckpt_file = resume_from_checkpoint
            if os.path.isdir(ckpt_file):
                ckpt_file = self.latest_training_checkpoint(ckpt_file)
            if ckpt_file and os.path.isfile(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                self.model_.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                scaler.load_state_dict(ckpt['scaler_state_dict'])
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                best_composite = ckpt.get('best_composite', -float('inf'))
                patience_counter = ckpt.get('patience_counter', 0)
                history = ckpt.get('history', history)
                start_epoch = ckpt['epoch'] + 1
                logger.info(f"Resumed training from checkpoint epoch {ckpt['epoch']} ({ckpt_file})")
                print(f"  Resumed from checkpoint epoch {ckpt['epoch']}")
                del ckpt
            else:
                logger.warning(f"No checkpoint found at {resume_from_checkpoint}, starting fresh")

        # Early stopping metric: 'composite' (0.4*AP + 0.3*F1 + 0.3*AUC),
        # 'val_loss', 'val_mean_auc', 'val_mean_ap', 'val_mean_f1'
        es_metric = self.early_stopping_metric
        es_higher_is_better = (es_metric != 'val_loss')
        es_value = float('nan')
        logger.info(f"Early stopping metric: {es_metric} ({'higher' if es_higher_is_better else 'lower'} is better)")

        epoch_iter = _progress_iter(range(start_epoch, int(self.epochs)), desc="Epochs", total=int(self.epochs) - start_epoch)
        for epoch in epoch_iter:
            # Training
            self.model_.train()
            train_loss_sum = 0.0
            n_seen = 0

            batch_iter = _progress_iter(train_loader, desc=f"Train Epoch {epoch}")
            for batch in batch_iter:
                input_ids = batch['input_ids'].to(device, non_blocking=use_cuda)
                attention_mask = batch['attention_mask'].to(device, non_blocking=use_cuda)
                targets = batch['targets'].to(device, non_blocking=use_cuda)
                sample_weights = batch.get('sample_weight')
                if sample_weights is not None:
                    sample_weights = sample_weights.to(device, non_blocking=use_cuda)
                numeric_features = batch.get('numeric_features')
                if numeric_features is not None:
                    numeric_features = numeric_features.to(device, non_blocking=use_cuda)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                    logits = self.model_(input_ids, attention_mask, numeric_features=numeric_features)
                    loss = loss_fn(logits, targets, sample_weights=sample_weights)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                bs = input_ids.size(0)
                train_loss_sum += float(loss.item()) * bs
                n_seen += bs

                # Log progress periodically for streaming datasets
                if n_seen % (self.batch_size * 100) == 0 and n_seen > 0:
                    running_loss = train_loss_sum / n_seen
                    logger.info(f"  Epoch {epoch}: {n_seen:,} samples processed, running_loss={running_loss:.4f}")

            train_loss = train_loss_sum / max(n_seen, 1)
            history['train_loss'].append(train_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lr", float(optimizer.param_groups[0]['lr']), step=epoch)

            # Validation
            val_loss = None
            val_metrics_str = ""
            if val_loader is not None:
                self.model_.eval()
                val_loss_sum = 0.0
                n_val = 0
                all_val_probs = []
                all_val_targets = []
                all_val_aft_raw = []
                with torch.no_grad():
                    val_iter = _progress_iter(val_loader, desc=f"Val Epoch {epoch}")
                    for batch in val_iter:
                        input_ids = batch['input_ids'].to(device, non_blocking=use_cuda)
                        attention_mask = batch['attention_mask'].to(device, non_blocking=use_cuda)
                        targets = batch['targets'].to(device, non_blocking=use_cuda)
                        numeric_features = batch.get('numeric_features')
                        if numeric_features is not None:
                            numeric_features = numeric_features.to(device, non_blocking=use_cuda)

                        logits = self.model_(input_ids, attention_mask, numeric_features=numeric_features)
                        loss = loss_fn(logits, targets)

                        bs = input_ids.size(0)
                        val_loss_sum += float(loss.item()) * bs
                        n_val += bs

                        if self.loss_type == 'aft':
                            probs = self._aft_to_probs(logits).cpu()
                            all_val_aft_raw.append(logits.cpu())
                        elif self.loss_type == 'deephit':
                            probs = self._deephit_to_probs(logits).cpu()
                            all_val_aft_raw.append(logits.cpu())
                        else:
                            probs = torch.sigmoid(logits).cpu()
                        all_val_probs.append(probs)
                        all_val_targets.append(batch['targets'].cpu() if targets.device.type != 'cpu' else batch['targets'])

                val_loss = val_loss_sum / max(n_val, 1)
                history['val_loss'].append(val_loss)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                # Compute per-event AUC & AP
                from sklearn.metrics import roc_auc_score, average_precision_score
                val_probs = torch.cat(all_val_probs).numpy()
                val_targets = torch.cat(all_val_targets).numpy()
                del all_val_probs, all_val_targets

                from sklearn.metrics import f1_score, brier_score_loss
                from src.sequence.evaluation import _targets_to_survival
                from src.survival.evaluation import brier_score_at_horizons
                epoch_aucs = []
                epoch_aps = []
                epoch_f1s = []
                epoch_briers = []
                epoch_ipcw_briers = []
                for ei, event in enumerate(self.events):
                    # Compute IPCW Brier for this event (all horizons at once)
                    event_cols = [ei * n_horizons + hi for hi in range(n_horizons)]
                    event_targets = val_targets[:, event_cols]
                    duration, event_ind = _targets_to_survival(event_targets, list(self.horizons))
                    proba_dict = {
                        h: val_probs[:, ei * n_horizons + hi]
                        for hi, h in enumerate(self.horizons)
                    }
                    ipcw_bs = brier_score_at_horizons(
                        duration, event_ind,
                        proba_dict, list(self.horizons),
                        duration_train=duration, event_train=event_ind,
                    )

                    for hi, horizon in enumerate(self.horizons):
                        col = ei * n_horizons + hi
                        y_t = val_targets[:, col]
                        y_p = val_probs[:, col]
                        n_pos = int(y_t.sum())
                        if n_pos > 0 and n_pos < len(y_t):
                            auc = float(roc_auc_score(y_t, y_p))
                            ap = float(average_precision_score(y_t, y_p))
                            brier = float(brier_score_loss(y_t, y_p))
                            ipcw_brier_val = ipcw_bs.get(horizon, float('nan'))
                            # Find optimal F1 threshold — fixed 0.5 is wrong
                            # for rare events where probabilities are low
                            base_rate = n_pos / len(y_t)
                            candidates = np.unique(np.concatenate([
                                np.linspace(max(0.005, base_rate * 0.2),
                                            min(0.95, base_rate * 5), 30),
                                np.array([0.5, base_rate]),
                            ]))
                            best_f1 = 0.0
                            for thr in candidates:
                                _f1 = f1_score(y_t, (y_p >= thr).astype(int),
                                               zero_division=0)
                                if _f1 > best_f1:
                                    best_f1 = _f1
                            f1 = float(best_f1)
                            epoch_aucs.append(auc)
                            epoch_aps.append(ap)
                            epoch_f1s.append(f1)
                            epoch_briers.append(brier)
                            if not np.isnan(ipcw_brier_val):
                                epoch_ipcw_briers.append(ipcw_brier_val)
                            mlflow.log_metric(f"val_auc_{event}_{horizon}yr", auc, step=epoch)
                            mlflow.log_metric(f"val_ap_{event}_{horizon}yr", ap, step=epoch)
                            mlflow.log_metric(f"val_f1_{event}_{horizon}yr", f1, step=epoch)
                            mlflow.log_metric(f"val_brier_{event}_{horizon}yr", brier, step=epoch)
                            if not np.isnan(ipcw_brier_val):
                                mlflow.log_metric(f"val_ipcw_brier_{event}_{horizon}yr", ipcw_brier_val, step=epoch)

                mean_auc = float(np.mean(epoch_aucs)) if epoch_aucs else float('nan')
                mean_ap = float(np.mean(epoch_aps)) if epoch_aps else float('nan')
                mean_f1 = float(np.mean(epoch_f1s)) if epoch_f1s else float('nan')
                mean_brier = float(np.mean(epoch_briers)) if epoch_briers else float('nan')
                mean_ipcw_brier = float(np.mean(epoch_ipcw_briers)) if epoch_ipcw_briers else float('nan')
                mlflow.log_metric("val_mean_auc", mean_auc, step=epoch)
                mlflow.log_metric("val_mean_ap", mean_ap, step=epoch)
                mlflow.log_metric("val_mean_f1", mean_f1, step=epoch)
                mlflow.log_metric("val_mean_brier", mean_brier, step=epoch)
                mlflow.log_metric("val_mean_ipcw_brier", mean_ipcw_brier, step=epoch)
                # AFT-native metrics per epoch (C-index, CRPS, TD-AUC)
                # Uses fast sampled C-index (O(k) instead of O(n²))
                aft_metrics_str = ""
                mean_c = float('nan')
                mean_td = float('nan')
                mean_crps = float('nan')
                mean_crpss = float('nan')
                if self.loss_type in ('aft', 'deephit') and all_val_aft_raw:
                    try:
                        from src.sequence.evaluation import (
                            _log_logistic_cdf, _targets_to_survival,
                            _fast_c_index, _fast_crps,
                        )
                        from sklearn.metrics import roc_auc_score as _roc_auc

                        aft_raw = torch.cat(all_val_aft_raw).numpy()
                        n_total = aft_raw.shape[0]
                        n_ev = len(self.events)

                        # Stratified subsample for speed — preserves event rates
                        aft_max_samples = 100_000
                        if n_total > aft_max_samples:
                            rng = np.random.RandomState(epoch)
                            any_pos = (val_targets > 0.5).any(axis=1)
                            pos_idx = np.where(any_pos)[0]
                            neg_idx = np.where(~any_pos)[0]
                            n_pos_keep = min(len(pos_idx), aft_max_samples // 2)
                            n_neg_keep = min(len(neg_idx), aft_max_samples - n_pos_keep)
                            pos_chosen = rng.choice(pos_idx, n_pos_keep, replace=False) if n_pos_keep < len(pos_idx) else pos_idx
                            neg_chosen = rng.choice(neg_idx, n_neg_keep, replace=False) if n_neg_keep < len(neg_idx) else neg_idx
                            idx = np.concatenate([pos_chosen, neg_chosen])
                            aft_raw = aft_raw[idx]
                            val_tgt_sub = val_targets[idx]
                        else:
                            val_tgt_sub = val_targets

                        max_h = float(max(self.horizons))
                        epoch_c_indices = []
                        epoch_td_aucs = []
                        epoch_crps = []
                        epoch_crpss = []

                        if self.loss_type == 'aft':
                            mu_all = aft_raw[:, :n_ev]
                            sigma_all = np.exp(aft_raw[:, n_ev:]).clip(min=0.01)
                        elif self.loss_type == 'deephit':
                            # Convert raw logits → CDF for each event at each horizon
                            dh_logits = torch.tensor(aft_raw, dtype=torch.float32)
                            dh_cdf = self._deephit_to_probs(dh_logits).numpy()
                            # Reshape: (n, n_events, n_horizons)
                            dh_cdf_3d = dh_cdf.reshape(-1, n_ev, n_horizons)

                        for ei, event in enumerate(self.events):
                            cols = [ei * n_horizons + hi for hi in range(n_horizons)]
                            event_tgt = val_tgt_sub[:, cols]
                            duration, event_ind = _targets_to_survival(event_tgt, self.horizons)

                            n_pos_e = int(event_ind.sum())
                            if n_pos_e < 5 or (len(event_ind) - n_pos_e) < 5:
                                continue

                            if self.loss_type == 'aft':
                                mu_e = mu_all[:, ei]
                                sigma_e = sigma_all[:, ei]

                                # Fast sampled C-index
                                risk = -np.exp(mu_e)
                                c_idx = _fast_c_index(event_ind, duration, risk)
                                if not np.isnan(c_idx):
                                    epoch_c_indices.append(c_idx)
                                    mlflow.log_metric(f"val_c_index_{event}", c_idx, step=epoch)

                                # Fast CRPS with skill score
                                crps_e, crps_naive_e, skill_e = _fast_crps(
                                    mu_e, sigma_e, duration, event_ind,
                                    max_horizon=max_h, distribution='logistic',
                                    return_skill=True,
                                )
                                epoch_crps.append(crps_e)
                                epoch_crpss.append(skill_e)
                                mlflow.log_metric(f"val_crps_{event}", crps_e, step=epoch)
                                mlflow.log_metric(f"val_crpss_{event}", skill_e, step=epoch)

                                if epoch < 3 or skill_e < -0.1:
                                    med_time = np.exp(mu_e)
                                    evt_rate = float(event_ind.mean())
                                    logger.info(
                                        f"  [{event}] mu: mean={mu_e.mean():.2f} std={mu_e.std():.2f} | "
                                        f"sigma: mean={sigma_e.mean():.2f} std={sigma_e.std():.2f} | "
                                        f"median_time: mean={med_time.mean():.2f} p50={np.median(med_time):.2f} | "
                                        f"event_rate={evt_rate:.3f} | "
                                        f"CRPS={crps_e:.4f} naive={crps_naive_e:.4f} skill={skill_e:.4f}"
                                    )

                                # TD-AUC at each horizon
                                for hi, h in enumerate(self.horizons):
                                    y_h = event_tgt[:, hi]
                                    n_p = int(y_h.sum())
                                    if 0 < n_p < len(y_h):
                                        risk_h = _log_logistic_cdf(float(h), mu_e, sigma_e)
                                        auc_h = float(_roc_auc(y_h, risk_h))
                                        epoch_td_aucs.append(auc_h)
                                        mlflow.log_metric(f"val_td_auc_{event}_{h}yr", auc_h, step=epoch)

                            elif self.loss_type == 'deephit':
                                cdf_e = dh_cdf_3d[:, ei, :]  # (n, n_horizons)

                                # C-index: use CDF at max horizon as risk score
                                risk = cdf_e[:, -1]
                                c_idx = _fast_c_index(event_ind, duration, risk)
                                if not np.isnan(c_idx):
                                    epoch_c_indices.append(c_idx)
                                    mlflow.log_metric(f"val_c_index_{event}", c_idx, step=epoch)

                                # TD-AUC at each horizon using learned CDF
                                for hi, h in enumerate(self.horizons):
                                    y_h = event_tgt[:, hi]
                                    n_p = int(y_h.sum())
                                    if 0 < n_p < len(y_h):
                                        risk_h = cdf_e[:, hi]
                                        auc_h = float(_roc_auc(y_h, risk_h))
                                        epoch_td_aucs.append(auc_h)
                                        mlflow.log_metric(f"val_td_auc_{event}_{h}yr", auc_h, step=epoch)

                        mean_c = float(np.mean(epoch_c_indices)) if epoch_c_indices else float('nan')
                        mean_td = float(np.mean(epoch_td_aucs)) if epoch_td_aucs else float('nan')
                        mean_crps = float(np.mean(epoch_crps)) if epoch_crps else float('nan')
                        mean_crpss = float(np.mean(np.clip(epoch_crpss, 0, None))) if epoch_crpss else float('nan')
                        if not np.isnan(mean_c):
                            mlflow.log_metric("val_mean_c_index", mean_c, step=epoch)
                        if not np.isnan(mean_td):
                            mlflow.log_metric("val_mean_td_auc", mean_td, step=epoch)
                        if not np.isnan(mean_crps):
                            mlflow.log_metric("val_mean_crps", mean_crps, step=epoch)
                        if not np.isnan(mean_crpss):
                            mlflow.log_metric("val_mean_crpss", mean_crpss, step=epoch)
                        if self.loss_type == 'aft':
                            aft_metrics_str = f", C-idx: {mean_c:.4f}, CRPSS: {mean_crpss:.4f}, CRPS: {mean_crps:.4f}, TD-AUC: {mean_td:.4f}"
                        else:
                            aft_metrics_str = f", C-idx: {mean_c:.4f}, TD-AUC: {mean_td:.4f}"
                    except Exception as e:
                        logger.warning(f"Survival epoch metrics failed: {e}")
                del all_val_aft_raw

                # Language-model-style metrics (next-event accuracy, perplexity, temporal)
                lm_metrics_str = ""
                lm_top1 = float('nan')
                lm_top3 = float('nan')
                lm_mrr = float('nan')
                lm_ppl = float('nan')
                lm_tc = float('nan')
                try:
                    from src.sequence.evaluation import evaluate_lm_metrics_fast

                    lm = evaluate_lm_metrics_fast(
                        val_probs, val_targets, self.events, self.horizons,
                    )
                    lm_top1 = lm['lm_top1_acc']
                    lm_top3 = lm['lm_top3_acc']
                    lm_mrr = lm['lm_mrr']
                    lm_ppl = lm['lm_perplexity']
                    lm_tc = lm['lm_temporal_consistency']

                    for k_lm, v_lm in lm.items():
                        if not np.isnan(v_lm):
                            mlflow.log_metric(f"val_{k_lm}", v_lm, step=epoch)

                    lm_metrics_str = (
                        f", LM(top1={lm_top1:.3f}, top3={lm_top3:.3f}, "
                        f"MRR={lm_mrr:.3f}, ppl={lm_ppl:.2f}, TC={lm_tc:.4f})"
                    )
                except Exception as e:
                    logger.warning(f"LM epoch metrics failed: {e}")

                del val_probs, val_targets

                brier_str = f", mean_Brier: {mean_brier:.4f}" if not np.isnan(mean_brier) else ""
                ipcw_brier_str = f", mean_IPCW_Brier: {mean_ipcw_brier:.4f}" if not np.isnan(mean_ipcw_brier) else ""
                val_metrics_str = f", mean_AUC: {mean_auc:.4f}, mean_AP: {mean_ap:.4f}, mean_F1: {mean_f1:.4f}{brier_str}{ipcw_brier_str}{aft_metrics_str}{lm_metrics_str}"

                # Epoch callback (e.g. for Optuna pruning)
                if epoch_callback is not None:
                    epoch_callback(epoch, {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_mean_auc': mean_auc,
                        'val_mean_ap': mean_ap,
                        'val_mean_f1': mean_f1,
                        'val_mean_brier': mean_brier,
                        'val_mean_ipcw_brier': mean_ipcw_brier,
                        'val_mean_c_index': mean_c,
                        'val_mean_td_auc': mean_td,
                        'val_mean_crps': mean_crps,
                        'val_mean_crpss': mean_crpss,
                        'val_lm_top1_acc': lm_top1,
                        'val_lm_top3_acc': lm_top3,
                        'val_lm_mrr': lm_mrr,
                        'val_lm_perplexity': lm_ppl,
                        'val_lm_temporal_consistency': lm_tc,
                    })

                scheduler.step(val_loss)

                # Compute composite metric for early stopping
                if es_metric == 'composite':
                    has_survival = not np.isnan(mean_c) and not np.isnan(mean_td)
                    has_aft = has_survival and not np.isnan(mean_crpss)
                    if has_aft:
                        # AFT composite: ranking + calibration + classification
                        # CRPSS = skill score (higher=better, 0=naive, 1=perfect)
                        # Comparable across events regardless of base rate
                        _vals = [v for v in [mean_c, mean_crpss, mean_f1] if not np.isnan(v)]
                        if len(_vals) == 3:
                            es_value = 0.4 * mean_c + 0.3 * mean_crpss + 0.3 * mean_f1
                        else:
                            es_value = np.nanmean(_vals) if _vals else float('nan')
                    elif has_survival:
                        # DeepHit composite: C-index + TD-AUC + F1
                        _vals = [v for v in [mean_c, mean_td, mean_f1] if not np.isnan(v)]
                        if len(_vals) == 3:
                            es_value = 0.4 * mean_c + 0.3 * mean_td + 0.3 * mean_f1
                        else:
                            es_value = np.nanmean(_vals) if _vals else float('nan')
                    else:
                        # Fallback for non-AFT models
                        _vals = [v for v in [mean_ap, mean_f1, mean_auc] if not np.isnan(v)]
                        if len(_vals) == 3:
                            es_value = 0.4 * mean_ap + 0.3 * mean_f1 + 0.3 * mean_auc
                        else:
                            es_value = np.nanmean(_vals) if _vals else float('nan')
                elif es_metric == 'val_loss':
                    es_value = val_loss
                elif es_metric == 'val_mean_auc':
                    es_value = mean_auc
                elif es_metric == 'val_mean_ap':
                    es_value = mean_ap
                elif es_metric == 'val_mean_f1':
                    es_value = mean_f1
                else:
                    es_value = val_loss
                    es_higher_is_better = False

                mlflow.log_metric("es_metric_value", es_value, step=epoch)

                # Track best and apply patience
                if not np.isnan(es_value):
                    if es_higher_is_better:
                        improved = es_value > best_composite + 1e-6
                    else:
                        improved = es_value < best_val_loss - 1e-6

                    if improved:
                        if es_higher_is_better:
                            best_composite = es_value
                        else:
                            best_val_loss = es_value
                        patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= int(self.early_stopping_patience):
                            logger.info(f"Early stopping at epoch {epoch} (best {es_metric}: "
                                        f"{best_composite if es_higher_is_better else best_val_loss:.4f})")
                            break

            val_str = f", val_loss: {val_loss:.4f}" if val_loss is not None else ""
            es_str = f", es({es_metric}): {es_value:.4f} [pat={patience_counter}]" if val_loader is not None and not np.isnan(es_value) else ""
            logger.info(f"Epoch {epoch}: train_loss: {train_loss:.4f}{val_str}{val_metrics_str}{es_str}")
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}{val_str}{val_metrics_str}{es_str}")

            # Save training checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_period == 0:
                self._save_training_checkpoint(
                    directory=checkpoint_dir,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    best_composite=best_composite,
                    patience_counter=patience_counter,
                    history=history,
                )

        # Log learned event weights if using Kendall weighting
        if isinstance(loss_fn, LearnedWeightedLoss):
            learned_w = loss_fn.get_weights()
            for ei, event in enumerate(self.events):
                w_val = float(learned_w[ei])
                logger.info(f"  Learned weight {event}: {w_val:.4f}")
                mlflow.log_metric(f"learned_weight_{event}", w_val)

        # Restore best model weights
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
            best_es_val = best_composite if es_higher_is_better else best_val_loss
            logger.info(f"Restored best model (best {es_metric}: {best_es_val:.4f})")
        else:
            logger.info("No improvement tracked — using final model weights")

        self._is_fitted = True

        return TrainResult(
            model=self.model_,
            metrics={
                'train_loss': float(train_loss),
                'val_loss': float(best_val_loss) if val_loader else None,
                'best_composite': float(best_composite) if best_composite > -float('inf') else None,
            },
            metadata={
                'epochs_trained': int(epoch) + 1,
                'history': history,
                'device': str(device),
                'encoder_type': self.encoder_type,
                'n_params': n_params,
                'vocab_size': vocabulary.vocab_size,
            },
        )

    def calibrate_thresholds(
        self,
        dataset: SequenceDataset,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Find optimal F1-maximizing threshold per event/horizon on a validation set.

        Stores result in self.optimal_thresholds_ and returns the array.
        Shape: (n_events * n_horizons,).
        """
        from sklearn.metrics import f1_score
        proba_result = self.predict_proba(dataset=dataset, device=device)
        probs = proba_result.probabilities  # (n_samples, n_events * n_horizons)

        # We need targets
        n_events = len(self.events)
        n_horizons = len(self.horizons)
        n_cols = n_events * n_horizons

        # Collect targets from dataset
        all_targets = []
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            collate_fn=sequence_collate_fn, drop_last=False,
        )
        for batch in loader:
            all_targets.append(batch['targets'].numpy())
        targets = np.concatenate(all_targets, axis=0)

        thresholds = np.full(n_cols, 0.5)
        for col in range(n_cols):
            ei, hi = divmod(col, n_horizons)
            event = self.events[ei]
            horizon = self.horizons[hi]
            y_t = targets[:, col]
            y_p = probs[:, col]
            n_pos = int(y_t.sum())
            if n_pos == 0 or n_pos == len(y_t):
                continue

            best_f1, best_thresh = 0.0, 0.5
            for thresh in np.linspace(0.01, 0.99, 200):
                f1 = f1_score(y_t, (y_p >= thresh).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            thresholds[col] = best_thresh
            logger.info(f"  Optimal threshold {event}_{horizon}yr: {best_thresh:.3f} (F1={best_f1:.4f})")

        self.optimal_thresholds_ = thresholds
        mlflow.log_metric("mean_optimal_threshold", float(thresholds.mean()))
        for col in range(n_cols):
            ei, hi = divmod(col, n_horizons)
            mlflow.log_metric(
                f"threshold_{self.events[ei]}_{self.horizons[hi]}yr",
                float(thresholds[col]),
            )

        logger.info(f"Calibrated thresholds: mean={thresholds.mean():.3f}, "
                     f"range=[{thresholds.min():.3f}, {thresholds.max():.3f}]")
        return thresholds

    def calibrate_group_thresholds(
        self,
        dataset: SequenceDataset,
        group_series: np.ndarray,
        device: Optional[str] = None,
        min_group_size: int = 100,
    ) -> Dict[str, Dict]:
        """Find per-group, per-event/horizon thresholds that minimise group-rate MSE.

        For each group (e.g. municipality) and each event/horizon column,
        searches for the threshold t that minimises
        (observed_rate - mean(prob >= t))^2 within that group.

        For groups smaller than *min_group_size*, the global optimal
        threshold (from ``calibrate_thresholds``) is used as fallback.

        Args:
            dataset: Validation SequenceDataset (same used for calibrate_thresholds).
            group_series: Array of group labels, length == len(dataset).
                          E.g. refnis codes.
            device: Torch device.
            min_group_size: Min group size for per-group tuning.

        Returns:
            Dict mapping ``group_value -> {col_idx: threshold}`` and stores it
            in ``self.group_thresholds_``.
        """
        import pandas as pd
        from sklearn.metrics import f1_score

        proba_result = self.predict_proba(dataset=dataset, device=device)
        probs = proba_result.probabilities

        n_events = len(self.events)
        n_horizons = len(self.horizons)
        n_cols = n_events * n_horizons

        # Collect targets
        all_targets = []
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            collate_fn=sequence_collate_fn, drop_last=False,
        )
        for batch in loader:
            all_targets.append(batch['targets'].numpy())
        targets = np.concatenate(all_targets, axis=0)

        # Global fallback thresholds
        global_thresh = getattr(self, 'optimal_thresholds_', np.full(n_cols, 0.5))

        groups = np.asarray(group_series)
        unique_groups = np.unique(groups)
        thresholds_per_group = {}
        candidates = np.linspace(0.005, 0.95, 100)

        for gval in unique_groups:
            mask = groups == gval
            n_g = int(mask.sum())
            if n_g < min_group_size:
                thresholds_per_group[gval] = {col: float(global_thresh[col]) for col in range(n_cols)}
                continue

            g_probs = probs[mask]
            g_targets = targets[mask]
            g_thresholds = {}

            for col in range(n_cols):
                y_t = g_targets[:, col]
                y_p = g_probs[:, col]
                obs_rate = float(y_t.mean())

                best_mse = float('inf')
                best_thr = float(global_thresh[col])
                for thr in candidates:
                    pred_rate = float((y_p >= thr).mean())
                    mse = (obs_rate - pred_rate) ** 2
                    if mse < best_mse:
                        best_mse = mse
                        best_thr = float(thr)
                g_thresholds[col] = best_thr

            thresholds_per_group[gval] = g_thresholds

        self.group_thresholds_ = thresholds_per_group

        # Log summary
        n_tuned = sum(1 for g in unique_groups if int((groups == g).sum()) >= min_group_size)
        logger.info(f"Calibrated per-group thresholds: {n_tuned} groups tuned, "
                     f"{len(unique_groups) - n_tuned} used global fallback")
        mlflow.log_metric("n_groups_threshold_tuned", n_tuned)

        return thresholds_per_group

    def calibrate_sigma_per_event(
        self,
        dataset: SequenceDataset,
        device: Optional[str] = None,
        n_candidates: int = 30,
        max_samples: int = 100_000,
    ) -> Dict[int, float]:
        """Post-hoc per-event sigma scaling to minimise CRPS.

        For each event, searches for a scalar multiplier ``s`` such that
        ``sigma_calibrated = s * sigma_predicted`` yields the lowest CRPS
        on the supplied (validation) data.

        This is the continuous-distribution analogue of per-event F1
        threshold optimisation: it adjusts the *width* of each event's
        predicted survival distribution for best calibration.

        Args:
            dataset: Validation SequenceDataset.
            device: Torch device.
            n_candidates: Number of scale factors to try per event.
            max_samples: Max samples for CRPS evaluation speed.

        Returns:
            Dict mapping event index → optimal scale factor.
            Also stored in ``self.sigma_scales_``.
        """
        from src.sequence.evaluation import _fast_crps, _targets_to_survival

        if device is None:
            device = self.get_device()
        device_t = torch.device(device)
        use_cuda = device_t.type == "cuda"

        self.model_.eval()
        self.model_.to(device_t)

        n_events = len(self.events)
        n_horizons = len(self.horizons)
        max_h = float(max(self.horizons))

        # Collect raw AFT outputs and targets
        num_workers = int(self.device_config.num_workers) if self.device_config else 0
        if getattr(dataset, '_using_chunks', False) and num_workers > 0:
            num_workers = 0
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            collate_fn=sequence_collate_fn, pin_memory=use_cuda,
            drop_last=False, num_workers=num_workers,
        )

        all_aft_raw = []
        all_targets = []
        n_collected = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device_t, non_blocking=use_cuda)
                attn_mask = batch['attention_mask'].to(device_t, non_blocking=use_cuda)
                logits = self.model_(input_ids, attn_mask)
                all_aft_raw.append(logits.cpu().numpy())
                all_targets.append(batch['targets'].numpy())
                n_collected += logits.size(0)
                if n_collected >= max_samples:
                    break

        aft_raw = np.concatenate(all_aft_raw, axis=0)[:max_samples]
        targets = np.concatenate(all_targets, axis=0)[:max_samples]

        mu_all = aft_raw[:, :n_events]
        sigma_all = np.exp(aft_raw[:, n_events:]).clip(min=0.01)

        # Scale candidates: 0.3x to 3x original sigma
        candidates = np.geomspace(0.3, 3.0, n_candidates)

        sigma_scales = {}
        for ei, event in enumerate(self.events):
            mu_e = mu_all[:, ei]
            sigma_e = sigma_all[:, ei]

            cols = [ei * n_horizons + hi for hi in range(n_horizons)]
            event_tgt = targets[:, cols]
            duration, event_ind = _targets_to_survival(event_tgt, self.horizons)

            n_pos = int(event_ind.sum())
            if n_pos < 5 or (len(event_ind) - n_pos) < 5:
                sigma_scales[ei] = 1.0
                continue

            best_crps = float('inf')
            best_scale = 1.0
            for s in candidates:
                crps_s = _fast_crps(
                    mu_e, sigma_e * s, duration, event_ind,
                    max_horizon=max_h, distribution='logistic',
                )
                if crps_s < best_crps:
                    best_crps = crps_s
                    best_scale = float(s)

            sigma_scales[ei] = best_scale
            # Also report the skill score at the best scale
            _, _, skill = _fast_crps(
                mu_e, sigma_e * best_scale, duration, event_ind,
                max_horizon=max_h, distribution='logistic',
                return_skill=True,
            )
            logger.info(
                f"  Sigma calibration {event}: scale={best_scale:.3f}, "
                f"CRPS={best_crps:.4f}, CRPSS={skill:.4f}"
            )
            mlflow.log_metric(f"sigma_scale_{event}", best_scale)
            mlflow.log_metric(f"crps_calibrated_{event}", best_crps)
            mlflow.log_metric(f"crpss_calibrated_{event}", skill)

        self.sigma_scales_ = sigma_scales
        logger.info(f"Per-event sigma scales: {sigma_scales}")
        return sigma_scales

    def calibrate_probabilities(
        self,
        dataset: SequenceDataset,
        method: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[int, Any]:
        """Fit probability calibrators on a validation set.

        Per event-horizon column: fit IsotonicRegression or Platt scaling
        on (predicted_prob, true_label). Skips columns with < 10 positives.

        Stores fitted calibrators in self.calibrators_ dict.

        Args:
            dataset: Validation SequenceDataset.
            method: 'isotonic' or 'platt'. Defaults to self.calibration_method.
            device: Torch device.

        Returns:
            Dict mapping column index to fitted calibrator.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss

        method = method or self.calibration_method
        if method is None:
            logger.info("No calibration method specified — skipping")
            return {}

        # Get raw probabilities (without applying existing calibrators)
        old_calibrators = getattr(self, 'calibrators_', None)
        self.calibrators_ = {}  # temporarily clear to get raw probs
        proba_result = self.predict_proba(dataset=dataset, device=device)
        probs = proba_result.probabilities

        # Collect targets
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            collate_fn=sequence_collate_fn, drop_last=False,
        )
        all_targets = []
        for batch in loader:
            all_targets.append(batch['targets'].numpy())
        targets = np.concatenate(all_targets, axis=0)

        n_events = len(self.events)
        n_horizons = len(self.horizons)
        n_cols = n_events * n_horizons
        calibrators = {}

        for col in range(n_cols):
            ei, hi = divmod(col, n_horizons)
            event = self.events[ei]
            horizon = self.horizons[hi]
            y_t = targets[:, col]
            y_p = probs[:, col]
            n_pos = int(y_t.sum())

            if n_pos < 10:
                logger.info(f"  {event}_{horizon}yr: skipped (n_pos={n_pos} < 10)")
                continue

            brier_before = brier_score_loss(y_t, y_p)

            if method == 'isotonic':
                cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                cal.fit(y_p, y_t)
                y_cal = cal.predict(y_p)
            elif method == 'platt':
                lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                lr.fit(y_p.reshape(-1, 1), y_t)
                y_cal = lr.predict_proba(y_p.reshape(-1, 1))[:, 1]
                cal = lr
            else:
                raise ValueError(f"Unknown calibration method: {method}")

            brier_after = brier_score_loss(y_t, y_cal)
            calibrators[col] = cal
            logger.info(
                f"  {event}_{horizon}yr: Brier {brier_before:.4f} -> {brier_after:.4f} "
                f"({method}, n_pos={n_pos})"
            )
            mlflow.log_metric(f"brier_before_cal_{event}_{horizon}yr", brier_before)
            mlflow.log_metric(f"brier_after_cal_{event}_{horizon}yr", brier_after)

        self.calibrators_ = calibrators
        n_cal = len(calibrators)
        logger.info(f"Calibration: fitted {n_cal}/{n_cols} columns using {method}")
        mlflow.log_metric("n_calibrated_columns", n_cal)
        return calibrators

    def predict(self, X: Any = None, dataset: Optional[SequenceDataset] = None, device: Optional[str] = None) -> PredictResult:
        """Generate binary predictions using calibrated or default thresholds."""
        proba = self.predict_proba(X=X, dataset=dataset, device=device)
        if self.optimal_thresholds_ is not None:
            thresholds = self.optimal_thresholds_
            preds = (proba.probabilities >= thresholds[np.newaxis, :]).astype(int)
        else:
            thresholds = 0.5
            preds = (proba.probabilities >= 0.5).astype(int)
        return PredictResult(predictions=preds, metadata={'thresholds': thresholds})

    def predict_proba(
        self,
        X: Any = None,
        dataset: Optional[SequenceDataset] = None,
        device: Optional[str] = None,
    ) -> PredictResult:
        """
        Generate probability predictions.

        Returns PredictResult where probabilities has shape
        (n_samples, n_events * n_horizons).
        """
        if dataset is None:
            raise ValueError("dataset is required for sequence model prediction")

        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = device.type == "cuda"

        self.model_.eval()
        self.model_.to(device)

        num_workers = int(self.device_config.num_workers) if self.device_config else 0
        if getattr(dataset, '_using_chunks', False) and num_workers > 0:
            num_workers = 0
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=sequence_collate_fn,
            pin_memory=use_cuda,
            drop_last=False,
        )

        all_probs = []
        all_sids = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device, non_blocking=use_cuda)
                attention_mask = batch['attention_mask'].to(device, non_blocking=use_cuda)
                numeric_features = batch.get('numeric_features')
                if numeric_features is not None:
                    numeric_features = numeric_features.to(device, non_blocking=use_cuda)

                logits = self.model_(input_ids, attention_mask, numeric_features=numeric_features)
                if self.loss_type == 'aft':
                    probs = self._aft_to_probs(logits).cpu().numpy()
                elif self.loss_type == 'deephit':
                    probs = self._deephit_to_probs(logits).cpu().numpy()
                else:
                    probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                if 'sid' in batch:
                    all_sids.append(batch['sid'])

        probabilities = np.concatenate(all_probs, axis=0)
        metadata = {}
        if all_sids:
            metadata['sids'] = np.concatenate(all_sids, axis=0)

        # Apply calibrators if available
        if hasattr(self, 'calibrators_') and self.calibrators_:
            probabilities = self._apply_calibrators(probabilities)

        return PredictResult(predictions=None, probabilities=probabilities, metadata=metadata)

    def predict_proba_mc(
        self,
        dataset: SequenceDataset,
        n_samples: int = 30,
        device: Optional[str] = None,
    ) -> PredictResult:
        """MC Dropout prediction: run N forward passes with head dropout enabled.

        Returns PredictResult where probabilities = mean across samples,
        and metadata includes 'mc_std' (per-sample uncertainty) and 'mc_samples'.
        """
        if dataset is None:
            raise ValueError("dataset is required for sequence model prediction")
        if self.model_ is None:
            raise RuntimeError("Model not trained — call fit() first")

        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = device.type == "cuda"

        self.model_.to(device)
        self.model_.enable_head_dropout()

        num_workers = int(self.device_config.num_workers) if self.device_config else 0
        if getattr(dataset, '_using_chunks', False) and num_workers > 0:
            num_workers = 0
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=sequence_collate_fn,
            pin_memory=use_cuda,
            drop_last=False,
        )

        # Collect probabilities from each MC sample
        all_samples = []  # list of (n_data, n_outputs) arrays
        all_sids = None

        for sample_idx in range(n_samples):
            batch_probs = []
            batch_sids = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(device, non_blocking=use_cuda)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=use_cuda)
                    numeric_features = batch.get('numeric_features')
                    if numeric_features is not None:
                        numeric_features = numeric_features.to(device, non_blocking=use_cuda)

                    logits = self.model_(input_ids, attention_mask, numeric_features=numeric_features)
                    if self.loss_type == 'aft':
                        probs = self._aft_to_probs(logits).cpu().numpy()
                    elif self.loss_type == 'deephit':
                        probs = self._deephit_to_probs(logits).cpu().numpy()
                    else:
                        probs = torch.sigmoid(logits).cpu().numpy()
                    batch_probs.append(probs)
                    if sample_idx == 0 and 'sid' in batch:
                        batch_sids.append(batch['sid'])

            all_samples.append(np.concatenate(batch_probs, axis=0))
            if sample_idx == 0 and batch_sids:
                all_sids = np.concatenate(batch_sids, axis=0)

        # Restore standard eval mode
        self.model_.disable_head_dropout()

        # Stack: (n_samples, n_data, n_outputs)
        stacked = np.stack(all_samples, axis=0)
        mean_probs = stacked.mean(axis=0)
        std_probs = stacked.std(axis=0)

        # Apply calibrators to mean if available
        if hasattr(self, 'calibrators_') and self.calibrators_:
            mean_probs = self._apply_calibrators(mean_probs)

        metadata = {
            'mc_std': std_probs,
            'mc_samples': n_samples,
        }
        if all_sids is not None:
            metadata['sids'] = all_sids

        logger.info(
            f"MC Dropout: {n_samples} samples, "
            f"mean uncertainty (std): {std_probs.mean():.4f}"
        )
        return PredictResult(predictions=None, probabilities=mean_probs, metadata=metadata)

    def _apply_calibrators(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply fitted calibrators to raw probabilities."""
        calibrated = probabilities.copy()
        for col_idx, calibrator in self.calibrators_.items():
            calibrated[:, col_idx] = calibrator.predict(probabilities[:, col_idx])
        return calibrated

    def predict_aft_params(
        self,
        dataset: SequenceDataset,
        device: Optional[str] = None,
    ) -> PredictResult:
        """Extract raw AFT parameters (mu, log_sigma) without converting to
        horizon probabilities.

        Only valid when loss_type='aft'. Returns the raw model output:
        (n_samples, 2 * n_events) where first n_events columns are mu
        (location) and next n_events are log_sigma (log-scale).

        These parameters define a log-logistic distribution per event per
        person, enabling prediction intervals and distribution-native
        survival metrics (C-index, CRPS, IBS).

        Returns:
            PredictResult with probabilities=(n_samples, 2*n_events)
            containing raw [mu_0..mu_n, log_sigma_0..log_sigma_n].
        """
        if self.loss_type != 'aft':
            raise ValueError("predict_aft_params requires loss_type='aft'")
        if dataset is None:
            raise ValueError("dataset is required")
        if self.model_ is None:
            raise RuntimeError("Model not trained — call fit() first")

        if device is None:
            device = self.get_device()
        device = torch.device(device)
        use_cuda = device.type == "cuda"

        self.model_.eval()
        self.model_.to(device)

        num_workers = int(self.device_config.num_workers) if self.device_config else 0
        if getattr(dataset, '_using_chunks', False) and num_workers > 0:
            num_workers = 0
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=sequence_collate_fn,
            pin_memory=use_cuda,
            drop_last=False,
        )

        all_params = []
        all_sids = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device, non_blocking=use_cuda)
                attention_mask = batch['attention_mask'].to(device, non_blocking=use_cuda)
                raw_output = self.model_(input_ids, attention_mask)
                all_params.append(raw_output.cpu().numpy())
                if 'sid' in batch:
                    all_sids.append(batch['sid'])

        params = np.concatenate(all_params, axis=0)
        metadata = {}
        if all_sids:
            metadata['sids'] = np.concatenate(all_sids, axis=0)

        return PredictResult(predictions=None, probabilities=params, metadata=metadata)

    def predict_intervals(
        self,
        dataset: SequenceDataset,
        confidence_levels: Optional[List[float]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, dict]:
        """Generate prediction intervals from the AFT model.

        Per person per event, produces:
        - Predicted median time-to-event
        - Credible intervals at specified confidence levels
        - Per-horizon event probabilities

        Only valid when loss_type='aft'.

        Args:
            dataset: SequenceDataset to predict on.
            confidence_levels: CI widths, e.g. [0.5, 0.8, 0.9].
            device: Torch device.

        Returns:
            Dict per event, see evaluation.compute_prediction_intervals.
        """
        from .evaluation import compute_prediction_intervals

        aft_result = self.predict_aft_params(dataset=dataset, device=device)
        intervals = compute_prediction_intervals(
            aft_params=aft_result.probabilities,
            events=self.events,
            horizons=self.horizons,
            confidence_levels=confidence_levels,
        )

        # Attach sids if available
        sids = aft_result.metadata.get('sids')
        if sids is not None:
            for event in intervals:
                intervals[event]['sids'] = sids

        return intervals

    def predict_as_dict(
        self,
        dataset: SequenceDataset,
        device: Optional[str] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate predictions in the same dict format as predict_all_events()
        from the survival pipeline.

        Returns:
            {event: {'prob_1yr': array, 'prob_3yr': array, 'prob_5yr': array,
                     'risk_score': array}}
        """
        proba_result = self.predict_proba(dataset=dataset, device=device)
        probs = proba_result.probabilities  # (n_samples, n_events * n_horizons)

        n_horizons = len(self.horizons)
        result = {}
        for ei, event in enumerate(self.events):
            event_probs = {}
            for hi, horizon in enumerate(self.horizons):
                col_idx = ei * n_horizons + hi
                event_probs[f'prob_{horizon}yr'] = probs[:, col_idx]

            # Risk score: negative of longest-horizon survival probability
            longest_horizon_prob = probs[:, ei * n_horizons + (n_horizons - 1)]
            event_probs['risk_score'] = longest_horizon_prob

            result[event] = event_probs

        return result

    def _save_training_checkpoint(
        self,
        directory: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        best_val_loss: float,
        best_composite: float,
        patience_counter: int,
        history: dict,
        max_to_keep: int = 3,
    ) -> str:
        """Save a training checkpoint for resuming interrupted training."""
        os.makedirs(directory, exist_ok=True)

        ckpt_name = f"epoch_{epoch:04d}"
        ckpt_path = os.path.join(directory, f"{ckpt_name}.pt")

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model_.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_composite': best_composite,
                'patience_counter': patience_counter,
                'history': history,
            },
            ckpt_path,
        )

        # Save metadata as JSON
        meta = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_composite': best_composite,
            'patience_counter': patience_counter,
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        }
        meta_path = os.path.join(directory, f"{ckpt_name}.meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Prune old checkpoints
        existing = sorted(
            p for p in os.listdir(directory)
            if p.startswith("epoch_") and p.endswith(".pt")
        )
        while len(existing) > max_to_keep:
            old = existing.pop(0)
            for suffix in ["", ".meta.json"]:
                old_path = os.path.join(directory, old.replace(".pt", "") + suffix)
                if suffix == "":
                    old_path = os.path.join(directory, old)
                try:
                    os.unlink(old_path)
                except OSError:
                    pass

        logger.info(f"Saved training checkpoint: {ckpt_path}")
        return ckpt_path

    @staticmethod
    def latest_training_checkpoint(directory: str) -> Optional[str]:
        """
        Find the latest training checkpoint in a directory.

        Returns:
            Path to the .pt checkpoint file, or None if no checkpoint found.
        """
        if not os.path.isdir(directory):
            return None

        checkpoints = sorted(
            p for p in os.listdir(directory)
            if p.startswith("epoch_") and p.endswith(".pt")
        )
        if not checkpoints:
            return None

        return os.path.join(directory, checkpoints[-1])

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        from .vocabulary import N_NUMERIC_FEATURES
        n_numeric = N_NUMERIC_FEATURES if self.use_numeric_features else 0

        torch.save(
            {
                'model_state_dict': self.model_.state_dict(),
                'model_config': self.model_config,
                'events': self.events,
                'horizons': self.horizons,
                'encoder_type': self.encoder_type,
                'embed_dim': self.embed_dim,
                'max_seq_len': self.max_seq_len,
                'encoder_config': self.encoder_config,
                'head_hidden_dims': self.head_hidden_dims,
                'dropout': self.dropout,
                'loss_type': self.loss_type,
                'focal_gamma': self.focal_gamma,
                'multi_head': self.multi_head,
                'event_weight_mode': self.event_weight_mode,
                'optimal_thresholds': self.optimal_thresholds_,
                'group_thresholds': getattr(self, 'group_thresholds_', None),
                'sigma_scales': getattr(self, 'sigma_scales_', None),
                'vocab_size': self.vocabulary_.vocab_size if self.vocabulary_ else None,
                'use_numeric_features': self.use_numeric_features,
                'numeric_inject': self.numeric_inject,
                'n_numeric_features': n_numeric,
            },
            os.path.join(path, "model.pt"),
        )

        if self.vocabulary_ is not None:
            self.vocabulary_.save(os.path.join(path, "vocabulary.joblib"))

        # Save calibrators if present
        if hasattr(self, 'calibrators_') and self.calibrators_:
            import joblib
            joblib.dump(self.calibrators_, os.path.join(path, "calibrators.joblib"))
            logger.info(f"Saved {len(self.calibrators_)} calibrators")

        logger.info(f"Sequence model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        device_config: Optional[DeviceConfig] = None,
    ) -> "PyTorchSequenceEstimator":
        checkpoint = torch.load(
            os.path.join(path, "model.pt"), map_location="cpu"
        )

        estimator = cls(checkpoint['model_config'], device_config)
        estimator.events = checkpoint['events']
        estimator.horizons = checkpoint['horizons']
        estimator.encoder_type = checkpoint['encoder_type']
        estimator.embed_dim = checkpoint['embed_dim']
        estimator.max_seq_len = checkpoint['max_seq_len']
        estimator.encoder_config = checkpoint['encoder_config']
        estimator.head_hidden_dims = checkpoint['head_hidden_dims']
        estimator.dropout = checkpoint['dropout']
        estimator.loss_type = checkpoint.get('loss_type', 'bce')
        estimator.focal_gamma = checkpoint.get('focal_gamma', 2.0)
        estimator.multi_head = checkpoint.get('multi_head', False)
        estimator.event_weight_mode = checkpoint.get('event_weight_mode', 'uniform')
        estimator.optimal_thresholds_ = checkpoint.get('optimal_thresholds', None)
        estimator.group_thresholds_ = checkpoint.get('group_thresholds', None)
        estimator.sigma_scales_ = checkpoint.get('sigma_scales', None)
        estimator.use_numeric_features = checkpoint.get('use_numeric_features', False)
        estimator.numeric_inject = checkpoint.get('numeric_inject', 'add')
        if estimator.numeric_inject not in {'add', 'concat'}:
            logger.warning(
                "Checkpoint numeric_inject='%s' is not supported; using 'add'.",
                estimator.numeric_inject,
            )
            estimator.numeric_inject = 'add'
        n_numeric = checkpoint.get('n_numeric_features', 0)

        vocab_path = os.path.join(path, "vocabulary.joblib")
        if os.path.exists(vocab_path):
            estimator.vocabulary_ = LifeEventVocabulary.load(vocab_path)

        estimator.model_ = SequenceModel(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=estimator.embed_dim,
            encoder_type=estimator.encoder_type,
            encoder_config=estimator.encoder_config,
            n_events=len(estimator.events),
            n_horizons=len(estimator.horizons),
            max_seq_len=estimator.max_seq_len,
            head_hidden_dims=estimator.head_hidden_dims,
            dropout=estimator.dropout,
            loss_type=estimator.loss_type,
            multi_head=estimator.multi_head,
            n_numeric_features=n_numeric,
            numeric_inject=estimator.numeric_inject,
        )
        estimator.model_.load_state_dict(checkpoint['model_state_dict'])
        estimator.model_.to(torch.device(device))
        estimator._is_fitted = True

        # Load calibrators if saved
        cal_path = os.path.join(path, "calibrators.joblib")
        if os.path.exists(cal_path):
            import joblib
            estimator.calibrators_ = joblib.load(cal_path)
            logger.info(f"Loaded {len(estimator.calibrators_)} calibrators")
        else:
            estimator.calibrators_ = {}

        logger.info(f"Sequence model loaded from {path}")
        return estimator
