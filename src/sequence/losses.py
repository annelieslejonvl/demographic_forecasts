"""
Loss functions for sequence-based demographic event prediction.

AFTLoss: Accelerated Failure Time loss (log-logistic, interval censoring).
EventBCELoss: BCE with per-event structure for event weighting.
FocalLoss: Focal loss (Lin et al. 2017) for class-imbalanced binary targets.
LearnedWeightedLoss: Kendall et al. 2018 uncertainty-based task weighting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from typing import List, Optional


class AFTLoss(nn.Module):
    """
    Accelerated Failure Time loss with log-logistic distribution.

    Binary cumulative horizon targets define which time interval the event
    occurred in:
        h1=1               -> event in [0, h1]       -> prob = F(h1)
        h1=0, h3=1         -> event in (h1, h3]      -> prob = F(h3) - F(h1)
        h3=0, h5=1         -> event in (h3, h5]      -> prob = F(h5) - F(h3)
        all 0               -> right-censored at h5   -> prob = 1 - F(h5)

    CDF:  F(t) = sigmoid((log(t) - mu) / sigma)

    Model output: (batch, 2 * n_events) -- first n_events are mu (location),
    next n_events are log_sigma (log-scale).
    """

    def __init__(
        self,
        horizons: List[int],
        n_events: int,
        sigma_min: float = 0.01,
        event_weights: Optional[torch.Tensor] = None,
        horizon_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.n_events = n_events
        self.n_horizons = len(horizons)
        self.sigma_min = sigma_min
        self.reduction = reduction
        self.register_buffer(
            'log_horizons',
            torch.log(torch.tensor(horizons, dtype=torch.float32)),
        )
        if event_weights is not None:
            self.register_buffer('event_weights', event_weights)
        else:
            self.event_weights = None
        if horizon_weights is not None:
            # Pad with 1.0 for the right-censored interval
            padded = torch.cat([horizon_weights, torch.ones(1, device=horizon_weights.device)])
            self.register_buffer('horizon_weights', padded)
        else:
            self.horizon_weights = None

    def forward(
        self,
        output: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = output.size(0)
        mu = output[:, :self.n_events]
        log_sigma = output[:, self.n_events:]
        sigma = torch.exp(log_sigma).clamp(min=self.sigma_min)

        z = (self.log_horizons.unsqueeze(0).unsqueeze(0) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        F_cdf = torch.sigmoid(z)  # (batch, n_events, n_horizons)

        zeros = torch.zeros(batch, self.n_events, 1, device=F_cdf.device, dtype=F_cdf.dtype)
        ones = torch.ones(batch, self.n_events, 1, device=F_cdf.device, dtype=F_cdf.dtype)
        F_ext = torch.cat([zeros, F_cdf, ones], dim=-1)
        interval_probs = F_ext[:, :, 1:] - F_ext[:, :, :-1]

        targets_3d = targets.view(batch, self.n_events, self.n_horizons)
        any_event = targets_3d.any(dim=-1)
        first_idx = targets_3d.long().argmax(dim=-1)
        interval_idx = torch.where(
            any_event, first_idx,
            torch.full_like(first_idx, self.n_horizons),
        )

        log_probs = torch.log(interval_probs.clamp(min=1e-8))
        nll = -log_probs.gather(dim=-1, index=interval_idx.unsqueeze(-1)).squeeze(-1)  # (batch, n_events)

        # Horizon-dependent weighting: weight NLL by which interval the event falls in
        if self.horizon_weights is not None:
            hw = self.horizon_weights[interval_idx]  # (batch, n_events)
            nll = nll * hw

        if self.event_weights is not None:
            nll = nll * self.event_weights.unsqueeze(0)

        # Importance weight correction for balanced sampling
        if sample_weights is not None:
            nll = nll * sample_weights.unsqueeze(-1)

        if self.reduction == 'per_event':
            return nll.mean(dim=0)  # (n_events,)
        return nll.mean()


class EventBCELoss(nn.Module):
    """
    BCE loss with per-event structure for event weighting.

    Computes per-element BCE (with optional pos_weight), reshapes to
    (batch, n_events, n_horizons), optionally applies per-event and
    per-horizon weights, then reduces.

    With event_weights=None and reduction='mean', produces identical
    results to F.binary_cross_entropy_with_logits.
    """

    def __init__(
        self,
        n_events: int,
        n_horizons: int,
        pos_weight: Optional[torch.Tensor] = None,
        event_weights: Optional[torch.Tensor] = None,
        horizon_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.n_events = n_events
        self.n_horizons = n_horizons
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
        if event_weights is not None:
            self.register_buffer('event_weights', event_weights)
        else:
            self.event_weights = None
        if horizon_weights is not None:
            self.register_buffer('horizon_weights', horizon_weights)
        else:
            self.horizon_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        element_loss = F_torch.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none',
        )
        batch = logits.size(0)
        loss_3d = element_loss.view(batch, self.n_events, self.n_horizons)

        # Horizon weighting: emphasize short horizons
        if self.horizon_weights is not None:
            loss_3d = loss_3d * self.horizon_weights  # broadcasts (n_horizons,)

        per_event = loss_3d.mean(dim=-1)  # (batch, n_events)

        if self.event_weights is not None:
            per_event = per_event * self.event_weights.unsqueeze(0)

        # Importance weight correction for balanced sampling
        if sample_weights is not None:
            per_event = per_event * sample_weights.unsqueeze(-1)

        if self.reduction == 'per_event':
            return per_event.mean(dim=0)  # (n_events,)
        return per_event.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017) with per-event structure.

    FL = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else (1-p).

    gamma > 0 down-weights easy (well-classified) examples so the model
    focuses on hard, misclassified ones.  Especially useful for highly
    imbalanced targets like short-horizon demographic events.

    Operates on raw logits (same interface as EventBCELoss).
    """

    def __init__(
        self,
        n_events: int,
        n_horizons: int,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        event_weights: Optional[torch.Tensor] = None,
        horizon_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.n_events = n_events
        self.n_horizons = n_horizons
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
        if event_weights is not None:
            self.register_buffer('event_weights', event_weights)
        else:
            self.event_weights = None
        if horizon_weights is not None:
            self.register_buffer('horizon_weights', horizon_weights)
        else:
            self.horizon_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Numerically stable focal loss via logsigmoid
        p = torch.sigmoid(logits)
        # BCE per element (no reduction)
        bce = F_torch.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
        )

        # p_t = probability assigned to the true class
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        # Apply pos_weight: scale positive examples (same as BCE pos_weight)
        if self.pos_weight is not None:
            weight = targets * (self.pos_weight - 1) + 1
            loss = loss * weight

        batch = logits.size(0)
        loss_3d = loss.view(batch, self.n_events, self.n_horizons)

        # Horizon weighting: emphasize short horizons
        if self.horizon_weights is not None:
            loss_3d = loss_3d * self.horizon_weights

        per_event = loss_3d.mean(dim=-1)  # (batch, n_events)

        if self.event_weights is not None:
            per_event = per_event * self.event_weights.unsqueeze(0)

        # Importance weight correction for balanced sampling
        if sample_weights is not None:
            per_event = per_event * sample_weights.unsqueeze(-1)

        if self.reduction == 'per_event':
            return per_event.mean(dim=0)
        return per_event.mean()


class LearnedWeightedLoss(nn.Module):
    """
    Learned multi-task loss weighting (Kendall et al. 2018).

    Wraps a base loss (reduction='per_event') and applies:
        L = mean_i[ (1/2) * exp(-s_i) * L_i + (1/2) * s_i ]
    where s_i = log(sigma_i^2) is a learnable parameter per event.

    Events with high noise get large sigma (downweighted).
    Events with clean signal get small sigma (upweighted).
    The log(sigma) regularizer prevents all sigmas from going to infinity.
    """

    def __init__(self, base_loss: nn.Module, n_events: int):
        super().__init__()
        self.base_loss = base_loss
        self.log_var = nn.Parameter(torch.zeros(n_events))
        self.n_events = n_events

    def forward(
        self,
        output: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_event = self.base_loss(output, targets, sample_weights=sample_weights)  # (n_events,)
        precision = torch.exp(-self.log_var)
        weighted = 0.5 * precision * per_event + 0.5 * self.log_var
        return weighted.mean()

    def get_weights(self) -> torch.Tensor:
        """Return effective per-event weights (higher = more important)."""
        with torch.no_grad():
            return torch.exp(-self.log_var)
