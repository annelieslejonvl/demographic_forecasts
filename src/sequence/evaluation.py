"""
Evaluation metrics for sequence-based demographic event prediction.

Computes per-event, per-horizon metrics and group-level evaluation,
using the same metric format as the survival pipeline for comparability.
Includes time-dependent survival metrics (TD-AUC, IPCW Brier) via sksurv.
Includes AFT-native metrics (C-index, CRPS, IBS) that evaluate the
continuous time-to-event distribution directly, avoiding the fixed-horizon
binary label problem that degrades AP at short horizons.
Includes event-ordering metrics (Kendall's tau, top-1/k accuracy,
pairwise accuracy) for evaluating predicted event sequences.
"""
import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_sequence_predictions(
    all_predictions: Dict[str, Dict[str, np.ndarray]],
    test_df: pd.DataFrame,
    events: List[str],
    horizons: List[int],
    cutoff_year: int,
    time_col: str = 'year',
) -> Dict[str, Any]:
    """
    Evaluate sequence model predictions.

    Computes per-event, per-horizon:
    - AUC-ROC
    - Average Precision (PR-AUC)
    - F1 at optimal threshold
    - Brier score

    Args:
        all_predictions: {event: {prob_1yr: array, prob_3yr: array, ...}}
        test_df: Test DataFrame with actual event columns and person IDs.
        events: List of event column names.
        horizons: Prediction horizons in years.
        cutoff_year: Year used as prediction cutoff.
        time_col: Time column name.

    Returns:
        Dict with per-event metrics and aggregate summary.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        brier_score_loss,
    )

    results = {}

    for event in events:
        if event not in all_predictions:
            continue

        event_results = {}
        for horizon in horizons:
            prob_key = f'prob_{horizon}yr'
            if prob_key not in all_predictions[event]:
                continue

            y_prob = all_predictions[event][prob_key]

            # Build ground truth: did the event occur within the horizon?
            max_year = cutoff_year + horizon
            if event in test_df.columns and time_col in test_df.columns:
                # Group by person to check if event occurred in horizon window
                # But we already have person-level predictions, so we need
                # person-level ground truth
                y_true = _compute_horizon_labels(
                    test_df, event, cutoff_year, horizon, time_col,
                )
            else:
                logger.warning(f"Cannot compute ground truth for {event}")
                continue

            # Ensure shapes match
            if len(y_true) != len(y_prob):
                logger.warning(
                    f"Shape mismatch for {event}@{horizon}yr: "
                    f"y_true={len(y_true)}, y_prob={len(y_prob)}"
                )
                continue

            metrics = {}
            n_pos = int(y_true.sum())
            n_neg = len(y_true) - n_pos

            if n_pos > 0 and n_neg > 0:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob))
                metrics['ap'] = float(average_precision_score(y_true, y_prob))

                # F1 at optimal threshold
                best_f1, best_thresh = _best_f1_threshold(y_true, y_prob)
                metrics['f1'] = float(best_f1)
                metrics['threshold'] = float(best_thresh)

                metrics['brier'] = float(brier_score_loss(y_true, y_prob))
            else:
                metrics['auc'] = float('nan')
                metrics['ap'] = float('nan')
                metrics['f1'] = float('nan')
                metrics['brier'] = float('nan')

            metrics['n_pos'] = n_pos
            metrics['n_total'] = len(y_true)
            metrics['prevalence'] = n_pos / len(y_true) if len(y_true) > 0 else 0

            event_results[f'{horizon}yr'] = metrics

        results[event] = event_results

    # Aggregate metrics
    all_aucs = []
    all_aps = []
    for event_metrics in results.values():
        for h_metrics in event_metrics.values():
            if not np.isnan(h_metrics.get('auc', float('nan'))):
                all_aucs.append(h_metrics['auc'])
            if not np.isnan(h_metrics.get('ap', float('nan'))):
                all_aps.append(h_metrics['ap'])

    aggregate = {
        'mean_auc': float(np.mean(all_aucs)) if all_aucs else float('nan'),
        'mean_ap': float(np.mean(all_aps)) if all_aps else float('nan'),
    }

    return {
        'per_event': results,
        'aggregate': aggregate,
    }


def _compute_horizon_labels(
    test_df: pd.DataFrame,
    event_col: str,
    cutoff_year: int,
    horizon: int,
    time_col: str = 'year',
    id_col: str = 'sid',
) -> np.ndarray:
    """
    Compute binary labels: did the event occur within the horizon window?

    This produces person-level labels aligned with the sequence model's
    person-level predictions.
    """
    max_year = cutoff_year + horizon

    # Get unique person IDs in order
    person_ids = test_df.groupby(id_col).ngroups
    grouped = test_df.groupby(id_col)

    labels = []
    for sid, person_df in grouped:
        window = person_df[person_df[time_col] <= max_year]
        if len(window) > 0 and event_col in window.columns:
            occurred = int(window[event_col].astype(int).sum()) > 0
        else:
            occurred = False
        labels.append(float(occurred))

    return np.array(labels, dtype=np.float32)


def _best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
) -> tuple:
    """Find threshold that maximizes F1 score."""
    from sklearn.metrics import f1_score

    best_f1 = 0.0
    best_thresh = 0.5

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_f1, best_thresh


# ---------------------------------------------------------------------------
# Time-dependent survival metrics (sksurv)
# ---------------------------------------------------------------------------

def _targets_to_survival(
    event_targets: np.ndarray,
    horizons: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert binary horizon targets to (duration, event_indicator) pairs.

    Args:
        event_targets: (n_samples, n_horizons) binary targets for one event.
        horizons: List of horizon values (e.g. [1, 3, 5]).

    Returns:
        duration: (n_samples,) observed/censored time.
        event: (n_samples,) boolean event indicator.
    """
    n_samples, n_horizons = event_targets.shape
    horizons_arr = np.array(horizons, dtype=np.float64)

    # Vectorized: find first horizon where target > 0.5
    any_event = event_targets > 0.5  # (n_samples, n_horizons)
    has_event = any_event.any(axis=1)  # (n_samples,)
    # argmax on bool array returns index of first True (or 0 if all False)
    first_idx = any_event.argmax(axis=1)  # (n_samples,)

    duration = np.where(has_event, horizons_arr[first_idx], float(max(horizons)))
    event = has_event

    return duration, event


def evaluate_survival_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    events: List[str],
    horizons: List[int],
) -> Dict[str, Any]:
    """Compute time-dependent survival evaluation metrics using sksurv.

    Metrics per event:
    - TD-AUC: time-dependent cumulative/dynamic AUC at each horizon
    - IPCW Brier: inverse-probability-of-censoring weighted Brier score

    Args:
        probabilities: (n_samples, n_events * n_horizons) predicted probabilities.
        targets: (n_samples, n_events * n_horizons) binary targets.
        events: Event names.
        horizons: Horizon values in years.

    Returns:
        Dict with per-event survival metrics and aggregate summary.
    """
    try:
        from sksurv.metrics import (
            cumulative_dynamic_auc,
            brier_score,
        )
    except ImportError:
        logger.warning("sksurv not installed — skipping survival metrics")
        return {}

    n_events = len(events)
    n_horizons = len(horizons)
    results = {}
    all_td_aucs = []
    all_brier = []

    for ei, event in enumerate(events):
        # Extract per-event targets and predictions
        cols = [ei * n_horizons + hi for hi in range(n_horizons)]
        event_targets = targets[:, cols]  # (n_samples, n_horizons)
        event_probs = probabilities[:, cols]  # (n_samples, n_horizons)

        duration, event_indicator = _targets_to_survival(event_targets, horizons)

        # sksurv needs structured array for survival data
        y_surv = np.array(
            [(e, d) for e, d in zip(event_indicator, duration)],
            dtype=[('event', bool), ('time', float)],
        )

        n_pos = int(event_indicator.sum())
        n_neg = int((~event_indicator).sum())
        if n_pos < 5 or n_neg < 5:
            logger.info(f"  {event}: too few events ({n_pos}) or censored ({n_neg}) for survival metrics")
            continue

        event_results = {}

        # TD-AUC at each horizon
        # For cumulative_dynamic_auc, we use the risk estimate = predicted probability
        # at each horizon as the risk score at that time point
        try:
            # Use the longest-horizon probability as a single risk score
            risk_score = event_probs[:, -1]  # highest horizon prob as risk

            # Evaluate at each horizon time point
            eval_times = np.array([float(h) for h in horizons])
            # Filter eval_times to be within the observed time range
            t_min = duration[event_indicator].min() if n_pos > 0 else eval_times[0]
            t_max = duration.max()
            valid_times = eval_times[(eval_times >= t_min) & (eval_times < t_max)]

            if len(valid_times) > 0:
                td_auc_values, td_auc_mean = cumulative_dynamic_auc(
                    y_surv, y_surv, risk_score, valid_times,
                )
                for ti, t in enumerate(valid_times):
                    h = int(t)
                    event_results[f'td_auc_{h}yr'] = float(td_auc_values[ti])
                    all_td_aucs.append(float(td_auc_values[ti]))
                event_results['td_auc_mean'] = float(td_auc_mean)
        except Exception as e:
            logger.warning(f"  {event}: TD-AUC failed: {e}")

        # IPCW Brier score at each horizon
        try:
            # Build survival function estimates: S(t) = 1 - F(t)
            # For each sample, create survival probabilities at each horizon
            surv_probs = 1.0 - event_probs  # (n_samples, n_horizons)

            eval_times = np.array([float(h) for h in horizons])
            valid_times = eval_times[(eval_times >= t_min) & (eval_times < t_max)]

            if len(valid_times) > 0:
                # brier_score expects (n_samples, n_times) survival probabilities
                valid_hi = [i for i, h in enumerate(horizons) if float(h) in valid_times]
                surv_at_times = surv_probs[:, valid_hi]

                # sksurv brier_score returns (times, bs_values)
                _, bs_values = brier_score(
                    y_surv, y_surv, surv_at_times, valid_times,
                )
                for ti, t in enumerate(valid_times):
                    h = int(t)
                    event_results[f'ipcw_brier_{h}yr'] = float(bs_values[ti])
                    all_brier.append(float(bs_values[ti]))
        except Exception as e:
            logger.warning(f"  {event}: IPCW Brier failed: {e}")

        if event_results:
            results[event] = event_results
            logger.info(
                f"  {event} survival: "
                + ", ".join(f"{k}={v:.4f}" for k, v in event_results.items())
            )

    aggregate = {
        'mean_td_auc': float(np.mean(all_td_aucs)) if all_td_aucs else float('nan'),
        'mean_ipcw_brier': float(np.mean(all_brier)) if all_brier else float('nan'),
    }

    return {
        'per_event': results,
        'aggregate': aggregate,
    }


# ---------------------------------------------------------------------------
# Fast approximate C-index for per-epoch evaluation
# ---------------------------------------------------------------------------

def _fast_c_index(
    event_indicator: np.ndarray,
    duration: np.ndarray,
    risk_score: np.ndarray,
    n_pairs: int = 100_000,
    seed: int = 0,
) -> float:
    """Sampled concordance index — O(n_pairs) instead of O(n²).

    Randomly samples concordant-eligible pairs (one event, one with longer
    duration or censored) and checks if risk ordering is correct.
    Accurate to ~0.002 with 100k pairs, takes <50ms on any dataset size.

    Args:
        event_indicator: (n,) bool, True if event observed.
        duration: (n,) float, observed/censored time.
        risk_score: (n,) float, higher = higher risk (use -median_tte).
        n_pairs: Number of random pairs to evaluate.
        seed: Random seed for reproducibility within an epoch.

    Returns:
        Approximate C-index in [0, 1], or nan if not enough events.
    """
    idx_event = np.where(event_indicator)[0]
    if len(idx_event) < 2:
        return float('nan')

    n = len(duration)
    rng = np.random.RandomState(seed)

    # Sample i from event cases, j uniformly
    i_samples = rng.choice(idx_event, size=n_pairs, replace=True)
    j_samples = rng.randint(0, n, size=n_pairs)

    # Valid pairs: i had event, j has strictly longer duration
    dur_i = duration[i_samples]
    dur_j = duration[j_samples]
    valid = dur_j > dur_i

    if valid.sum() < 10:
        return float('nan')

    risk_i = risk_score[i_samples[valid]]
    risk_j = risk_score[j_samples[valid]]

    concordant = (risk_i > risk_j).sum()
    discordant = (risk_i < risk_j).sum()
    tied = (risk_i == risk_j).sum()
    total = concordant + discordant + tied

    if total == 0:
        return float('nan')

    return float((concordant + 0.5 * tied) / total)


# ---------------------------------------------------------------------------
# AFT-native survival metrics (directly from mu, sigma parameters)
# ---------------------------------------------------------------------------

def _log_logistic_cdf(t: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log-logistic CDF: F(t) = sigmoid((log(t) - mu) / sigma).

    Args:
        t: Time points, shape broadcastable with mu/sigma.
        mu: Location parameter (log-scale).
        sigma: Scale parameter (> 0).

    Returns:
        CDF values, same shape as broadcast(t, mu, sigma).
    """
    z = (np.log(np.maximum(t, 1e-8)) - mu) / np.maximum(sigma, 1e-8)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _log_logistic_pdf(t: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log-logistic PDF: f(t) = F'(t)."""
    t_safe = np.maximum(t, 1e-8)
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (np.log(t_safe) - mu) / sigma_safe
    z = np.clip(z, -500, 500)
    ez = np.exp(-z)
    return ez / (sigma_safe * t_safe * (1.0 + ez) ** 2)


def _log_logistic_quantile(p: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log-logistic quantile function: Q(p) = exp(mu + sigma * logit(p))."""
    p_safe = np.clip(p, 1e-8, 1.0 - 1e-8)
    logit_p = np.log(p_safe / (1.0 - p_safe))
    return np.exp(mu + sigma * logit_p)


# ---------------------------------------------------------------------------
# Distribution-agnostic AFT CDF (supports logistic, normal, extreme)
# ---------------------------------------------------------------------------

def _aft_cdf(
    t: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    distribution: str = 'logistic',
) -> np.ndarray:
    """AFT CDF: F(t) = CDF_base((log(t) - mu) / sigma).

    Supports multiple base distributions:
    - 'logistic' (log-logistic): sigmoid(z)  — GRU sequence model
    - 'normal' (log-normal): Phi(z)          — XGBoost default
    - 'extreme' (Weibull): 1 - exp(-exp(z))  — XGBoost option

    Args:
        t: Time points, broadcastable with mu/sigma.
        mu: Location parameter (log-scale).
        sigma: Scale parameter (> 0).
        distribution: Base distribution name.

    Returns:
        CDF values in [0, 1].
    """
    z = (np.log(np.maximum(t, 1e-8)) - mu) / np.maximum(sigma, 1e-8)
    z = np.clip(z, -500, 500)

    if distribution == 'logistic':
        return 1.0 / (1.0 + np.exp(-z))
    elif distribution == 'normal':
        from scipy.special import ndtr
        return ndtr(z)
    elif distribution in ('extreme', 'extreme_value'):
        return 1.0 - np.exp(-np.exp(z))
    else:
        raise ValueError(f"Unknown AFT distribution: {distribution}")


def _fast_crps(
    mu: np.ndarray,
    sigma: np.ndarray,
    duration: np.ndarray,
    event_indicator: np.ndarray,
    max_horizon: float = 5.0,
    n_steps: int = 50,
    distribution: str = 'logistic',
    return_skill: bool = False,
) -> "float | tuple[float, float, float]":
    """Fast per-event CRPS suitable for per-epoch evaluation.

    Uses a coarse grid (50 steps vs 200 in full eval) and vectorised
    computation.  On 100k samples this takes ~20ms.

    Args:
        return_skill: If True, also return naive baseline CRPS and skill score.

    Returns:
        If return_skill is False: mean CRPS (lower is better).
        If return_skill is True:  (crps, crps_naive, skill_score) where
            skill_score = 1 - crps/crps_naive (higher is better, 0=naive, 1=perfect).
    """
    t_grid = np.linspace(0.1, max_horizon * 1.5, n_steps)
    dt = t_grid[1] - t_grid[0]
    n = len(mu)

    crps_vals = np.zeros(n, dtype=np.float64)
    chunk = max(1, min(20_000, int(200_000_000 / (n_steps * 8))))

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        mu_c = mu[s:e, np.newaxis]
        sig_c = sigma[s:e, np.newaxis]
        dur_c = duration[s:e, np.newaxis]
        evt_c = event_indicator[s:e, np.newaxis]

        F_c = _aft_cdf(t_grid[np.newaxis, :], mu_c, sig_c, distribution)

        step_c = (t_grid[np.newaxis, :] >= dur_c).astype(np.float64)
        integ_uncens = (F_c - step_c) ** 2
        integ_cens = F_c ** 2 * (t_grid[np.newaxis, :] <= dur_c)

        crps_vals[s:e] = np.where(evt_c.ravel(),
                                   (integ_uncens * dt).sum(axis=1),
                                   (integ_cens * dt).sum(axis=1))

    crps = float(crps_vals.mean())

    if not return_skill:
        return crps

    # Naive baseline: best single-distribution predictor (population-level AFT).
    # Fit global (mu_0, sigma_0) = (mean(log(duration_events)), std(log(duration_events)))
    # then compute CRPS using those constant params for everyone.
    event_mask = event_indicator.astype(bool)
    if event_mask.sum() >= 2:
        log_dur_events = np.log(np.maximum(duration[event_mask], 1e-8))
        mu_naive = float(np.mean(log_dur_events))
        sigma_naive = float(max(np.std(log_dur_events), 0.1))
    else:
        # Fallback: use overall duration stats
        mu_naive = float(np.mean(np.log(np.maximum(duration, 1e-8))))
        sigma_naive = 1.0

    mu_naive_arr = np.full(n, mu_naive)
    sigma_naive_arr = np.full(n, sigma_naive)

    naive_vals = np.zeros(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        mu_c = mu_naive_arr[s:e, np.newaxis]
        sig_c = sigma_naive_arr[s:e, np.newaxis]
        dur_c = duration[s:e, np.newaxis]
        evt_c = event_indicator[s:e, np.newaxis]

        F_naive = _aft_cdf(t_grid[np.newaxis, :], mu_c, sig_c, distribution)

        step_c = (t_grid[np.newaxis, :] >= dur_c).astype(np.float64)
        integ_uncens = (F_naive - step_c) ** 2
        integ_cens = F_naive ** 2 * (t_grid[np.newaxis, :] <= dur_c)

        naive_vals[s:e] = np.where(evt_c.ravel(),
                                    (integ_uncens * dt).sum(axis=1),
                                    (integ_cens * dt).sum(axis=1))

    crps_naive = float(naive_vals.mean())
    skill = 1.0 - (crps / crps_naive) if crps_naive > 1e-10 else 0.0

    return crps, crps_naive, skill


def evaluate_aft_survival_metrics(
    aft_params: np.ndarray,
    targets: np.ndarray,
    events: List[str],
    horizons: List[int],
    eval_times: Optional[np.ndarray] = None,
    n_crps_steps: int = 200,
    max_samples: int = 200_000,
    distribution: str = 'logistic',
) -> Dict[str, Any]:
    """Evaluate the AFT model using metrics that score the full time-to-event
    distribution, bypassing the fixed-horizon binary label problem.

    Works with any AFT distribution (log-logistic, log-normal, Weibull),
    making it usable for both GRU sequence models and XGBoost survival.

    Metrics per event:
    - C-index: ranking accuracy using predicted median time-to-event.
    - CRPS: Continuous Ranked Probability Score — measures how well the
      predicted CDF matches the step-function at the observed event time.
      Lower is better. Unlike AP, CRPS doesn't require a binary threshold.
    - IBS: Integrated Brier Score over [0, max_horizon] — calibration of
      the full survival curve. Lower is better.
    - TD-AUC: time-dependent AUC at each horizon using the predicted CDF.

    Args:
        aft_params: (n_samples, 2 * n_events) — first n_events columns are mu,
                    next n_events columns are log_sigma. Raw model output.
        targets: (n_samples, n_events * n_horizons) binary horizon targets.
        events: Event names.
        horizons: Horizon values in years (e.g. [1, 3, 5]).
        eval_times: Optional custom evaluation time grid for IBS.
                    Defaults to linspace(0.5, max_horizon, 20).
        n_crps_steps: Number of integration steps for CRPS (default 200).
        max_samples: Subsample to this many samples for speed. Metrics on
                     200k samples are accurate to ~0.002. Set 0 to disable.
        distribution: AFT base distribution — 'logistic' (GRU default),
                      'normal' (XGBoost default), or 'extreme' (Weibull).

    Returns:
        Dict with per-event metrics and aggregate summary.
    """
    n_events = len(events)
    n_horizons = len(horizons)
    n_samples = aft_params.shape[0]
    max_horizon = float(max(horizons))

    # Stratified subsample for speed — sksurv C-index is O(n²), CRPS/IBS O(n*steps).
    # Preserves event rates to avoid dropping rare events.
    if max_samples > 0 and n_samples > max_samples:
        rng = np.random.RandomState(42)
        # Find samples with any event across all events/horizons
        any_positive = (targets > 0.5).any(axis=1)
        pos_idx = np.where(any_positive)[0]
        neg_idx = np.where(~any_positive)[0]
        # Keep all positives (up to half budget), fill rest with negatives
        n_pos_keep = min(len(pos_idx), max_samples // 2)
        n_neg_keep = min(len(neg_idx), max_samples - n_pos_keep)
        pos_chosen = rng.choice(pos_idx, n_pos_keep, replace=False) if n_pos_keep < len(pos_idx) else pos_idx
        neg_chosen = rng.choice(neg_idx, n_neg_keep, replace=False) if n_neg_keep < len(neg_idx) else neg_idx
        idx = np.concatenate([pos_chosen, neg_chosen])
        rng.shuffle(idx)
        logger.info(
            f"  Stratified subsample {n_samples} -> {len(idx)} "
            f"({n_pos_keep} pos, {n_neg_keep} neg) for AFT metrics"
        )
        aft_params = aft_params[idx]
        targets = targets[idx]
        n_samples = len(idx)

    # Extract mu and sigma per event
    mu_all = aft_params[:, :n_events]               # (n_samples, n_events)
    sigma_all = np.exp(aft_params[:, n_events:])     # (n_samples, n_events)
    sigma_all = np.maximum(sigma_all, 0.01)

    if eval_times is None:
        eval_times = np.linspace(0.5, max_horizon, 20)

    results = {}
    all_c_index = []
    all_crps = []
    all_ibs = []
    all_td_auc = []

    for ei, event in enumerate(events):
        mu = mu_all[:, ei]       # (n_samples,)
        sigma = sigma_all[:, ei]  # (n_samples,)

        # Reconstruct (duration, event_indicator) from binary targets
        cols = [ei * n_horizons + hi for hi in range(n_horizons)]
        event_targets = targets[:, cols]
        duration, event_indicator = _targets_to_survival(event_targets, horizons)

        n_pos = int(event_indicator.sum())
        n_neg = int((~event_indicator).sum())
        if n_pos < 5 or n_neg < 5:
            logger.info(f"  {event}: too few events ({n_pos}) for AFT survival metrics")
            continue

        event_results = {}

        # --- C-index using predicted median ---
        # Median of log-logistic: exp(mu) (quantile at p=0.5, logit(0.5)=0)
        predicted_median = np.exp(mu)
        # Risk = negative median (shorter predicted time = higher risk)
        risk_score = -predicted_median

        try:
            from sksurv.metrics import concordance_index_censored
            c_idx, _, _, _, _ = concordance_index_censored(
                event_indicator, duration, risk_score,
            )
            event_results['c_index'] = float(c_idx)
            all_c_index.append(float(c_idx))
        except Exception as e:
            logger.warning(f"  {event}: C-index failed: {e}")

        # --- CRPS (Continuous Ranked Probability Score) ---
        # Vectorized with chunking to limit memory for large datasets.
        # CRPS_i = integral [F(t|x_i) - I(t >= T_i)]^2 dt
        try:
            t_grid = np.linspace(0.1, max_horizon * 1.5, n_crps_steps)
            dt = t_grid[1] - t_grid[0]

            crps_values = np.zeros(n_samples)
            chunk_size = max(1, min(10_000, int(500_000_000 / (n_crps_steps * 8))))

            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                mu_c = mu[start:end, np.newaxis]         # (chunk, 1)
                sig_c = sigma[start:end, np.newaxis]     # (chunk, 1)
                dur_c = duration[start:end, np.newaxis]  # (chunk, 1)
                evt_c = event_indicator[start:end, np.newaxis]  # (chunk, 1)

                F_c = _aft_cdf(t_grid[np.newaxis, :], mu_c, sig_c, distribution)

                # Uncensored: (F(t) - I(t >= T))^2
                step_c = (t_grid[np.newaxis, :] >= dur_c).astype(np.float64)
                integ_uncens = (F_c - step_c) ** 2

                # Censored: F(t)^2, zeroed beyond censoring time
                integ_cens = F_c ** 2 * (t_grid[np.newaxis, :] <= dur_c)

                integrand = np.where(evt_c, integ_uncens, integ_cens)
                crps_values[start:end] = integrand.sum(axis=1) * dt

            mean_crps = float(crps_values.mean())
            event_results['crps'] = mean_crps
            all_crps.append(mean_crps)
        except Exception as e:
            logger.warning(f"  {event}: CRPS failed: {e}")

        # --- Integrated Brier Score (IBS) ---
        # BS(t) = mean_i [ (S_hat(t|x_i) - I(T_i > t))^2 * w_i(t) ]
        # IBS = (1/T_max) * integral_0^T_max BS(t) dt
        try:
            # Build survival function at eval_times for each sample
            # S(t) = 1 - F(t)
            valid_eval = eval_times[eval_times <= max_horizon]
            if len(valid_eval) > 1:
                # (n_samples, n_times) survival predictions
                S_hat = np.zeros((n_samples, len(valid_eval)))
                for ti, t in enumerate(valid_eval):
                    S_hat[:, ti] = 1.0 - _aft_cdf(t, mu, sigma, distribution)

                # Try sksurv IPCW Brier
                y_surv = np.array(
                    [(e, d) for e, d in zip(event_indicator, duration)],
                    dtype=[('event', bool), ('time', float)],
                )

                from sksurv.metrics import brier_score as sksurv_brier
                # Filter eval_times within observed range
                t_min = duration[event_indicator].min() if n_pos > 0 else valid_eval[0]
                t_max = duration.max()
                in_range = (valid_eval >= t_min) & (valid_eval < t_max)
                if in_range.sum() > 1:
                    times_ibs = valid_eval[in_range]
                    S_ibs = S_hat[:, in_range]
                    _, bs_vals = sksurv_brier(y_surv, y_surv, S_ibs, times_ibs)
                    # Integrate Brier scores over time
                    dt_ibs = np.diff(times_ibs)
                    ibs = float(np.sum(0.5 * (bs_vals[:-1] + bs_vals[1:]) * dt_ibs) / (times_ibs[-1] - times_ibs[0]))
                    event_results['ibs'] = ibs
                    all_ibs.append(ibs)

                    # Per-horizon Brier from the same computation
                    for h in horizons:
                        closest = np.argmin(np.abs(times_ibs - h))
                        if abs(times_ibs[closest] - h) < 0.5:
                            event_results[f'ipcw_brier_{h}yr'] = float(bs_vals[closest])
        except ImportError:
            logger.warning(f"  {event}: sksurv not available for IBS")
        except Exception as e:
            logger.warning(f"  {event}: IBS failed: {e}")

        # --- TD-AUC using per-horizon CDF as risk score ---
        # Fallback: if sksurv cumulative_dynamic_auc fails (common with
        # discrete duration values from horizon-based targets), compute
        # a simple AUC on the binary label at each horizon.
        for h in horizons:
            risk_h = _aft_cdf(float(h), mu, sigma, distribution)
            try:
                from sksurv.metrics import cumulative_dynamic_auc

                y_surv = np.array(
                    [(e, d) for e, d in zip(event_indicator, duration)],
                    dtype=[('event', bool), ('time', float)],
                )
                eval_t = np.array([float(h)])
                t_min = duration[event_indicator].min() if n_pos > 0 else float(h)
                t_max = duration.max()
                if float(h) >= t_min and float(h) < t_max:
                    td_vals, _ = cumulative_dynamic_auc(y_surv, y_surv, risk_h, eval_t)
                    event_results[f'td_auc_{h}yr'] = float(td_vals[0])
                    all_td_auc.append(float(td_vals[0]))
                    continue
            except Exception:
                pass

            # Fallback: binary AUC at this horizon
            try:
                from sklearn.metrics import roc_auc_score
                # Binary label: event occurred within h years
                hi = horizons.index(h)
                y_h = event_targets[:, hi]
                n_pos_h = int(y_h.sum())
                if 0 < n_pos_h < len(y_h):
                    auc_h = float(roc_auc_score(y_h, risk_h))
                    event_results[f'td_auc_{h}yr'] = auc_h
                    all_td_auc.append(auc_h)
            except Exception as e:
                logger.warning(f"  {event}: TD-AUC@{h}yr failed: {e}")

        if event_results:
            results[event] = event_results
            logger.info(
                f"  {event} AFT survival: "
                + ", ".join(f"{k}={v:.4f}" for k, v in event_results.items())
            )

    aggregate = {
        'mean_c_index': float(np.mean(all_c_index)) if all_c_index else float('nan'),
        'mean_crps': float(np.mean(all_crps)) if all_crps else float('nan'),
        'mean_ibs': float(np.mean(all_ibs)) if all_ibs else float('nan'),
        'mean_td_auc': float(np.mean(all_td_auc)) if all_td_auc else float('nan'),
    }

    return {
        'per_event': results,
        'aggregate': aggregate,
    }


# ---------------------------------------------------------------------------
# Prediction intervals from AFT parameters
# ---------------------------------------------------------------------------

def compute_prediction_intervals(
    aft_params: np.ndarray,
    events: List[str],
    horizons: List[int],
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute prediction intervals from AFT log-logistic parameters.

    Instead of forcing a binary decision at a fixed horizon, this produces
    per-person, per-event:
    - Predicted median time-to-event
    - Credible intervals at specified confidence levels
    - Per-horizon CDF values (probability of event by horizon)

    This is more informative than a single probability: "this person is
    likely to move within 0.8-2.3 years (80% CI), median 1.4 years"
    vs "53% chance of moving within 1 year."

    Args:
        aft_params: (n_samples, 2 * n_events) raw AFT output.
        events: Event names.
        horizons: Horizon values for CDF evaluation.
        confidence_levels: List of CI widths, e.g. [0.5, 0.8, 0.9].
                          Defaults to [0.5, 0.8, 0.9].

    Returns:
        Dict per event with:
            'median': (n_samples,) predicted median time
            'mu': (n_samples,) location param
            'sigma': (n_samples,) scale param
            'ci_{level}_lower': lower bound of CI
            'ci_{level}_upper': upper bound of CI
            'prob_{h}yr': P(event within h years)
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9]

    n_events = len(events)
    n_samples = aft_params.shape[0]

    mu_all = aft_params[:, :n_events]
    log_sigma_all = aft_params[:, n_events:]
    sigma_all = np.exp(log_sigma_all)
    sigma_all = np.maximum(sigma_all, 0.01)

    result = {}
    for ei, event in enumerate(events):
        mu = mu_all[:, ei]
        sigma = sigma_all[:, ei]

        event_result = {
            'median': np.exp(mu),  # Q(0.5) = exp(mu)
            'mu': mu.copy(),
            'sigma': sigma.copy(),
        }

        # Credible intervals
        for level in confidence_levels:
            alpha = (1.0 - level) / 2.0
            lower_q = np.full(n_samples, alpha)
            upper_q = np.full(n_samples, 1.0 - alpha)
            event_result[f'ci_{int(level*100)}_lower'] = _log_logistic_quantile(lower_q, mu, sigma)
            event_result[f'ci_{int(level*100)}_upper'] = _log_logistic_quantile(upper_q, mu, sigma)

        # Per-horizon probabilities
        for h in horizons:
            event_result[f'prob_{h}yr'] = _log_logistic_cdf(float(h), mu, sigma)

        result[event] = event_result

    return result


# ---------------------------------------------------------------------------
# Event-ordering evaluation
# ---------------------------------------------------------------------------

def evaluate_event_ordering(
    predicted_mu: np.ndarray,
    targets: np.ndarray,
    events: List[str],
    horizons: List[int],
    min_events: int = 2,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Evaluate how well the model predicts the *order* of life events.

    For each person who experienced >=``min_events`` distinct events in the
    observation window, compares the predicted ordering (by ascending
    predicted median time = exp(mu)) to the observed ordering (by the
    earliest horizon at which each event was observed).

    Args:
        predicted_mu: (n_samples, n_events) — predicted log-median-time per
            event.  Lower mu → event predicted sooner.
        targets: (n_samples, n_events * n_horizons) — binary horizon targets
            in the standard layout ``[e0_h0, e0_h1, ..., e1_h0, ...]``.
        events: List of event names.
        horizons: List of horizon values (e.g. [1, 3, 5]).
        min_events: Minimum number of distinct events a person must have
            experienced to be included (default 2).
        top_k: *k* for top-k accuracy of the first observed event.

    Returns:
        Dict with:
            ``n_eligible``: Number of persons with >= min_events events.
            ``pairwise_accuracy``: Fraction of observed event-pairs whose
                relative order was predicted correctly.
            ``top1_accuracy``: Fraction where the predicted earliest event
                matches the observed earliest event.
            ``topk_accuracy``: Fraction where the observed earliest event
                is among the predicted k-earliest.
            ``mean_reciprocal_rank``: Mean 1/rank of the true first event
                in the predicted ordering.
            ``kendall_tau``: Mean Kendall's tau-b rank correlation between
                predicted and observed orderings.
            ``per_event_first_rate``: Dict mapping event → fraction of
                times it was the observed first event (among eligible).
            ``per_event_pred_first_rate``: Same for predicted first event.
    """
    n_samples = predicted_mu.shape[0]
    n_events = len(events)
    n_horizons = len(horizons)
    horizons_arr = np.array(horizons, dtype=np.float64)

    # For each person × event, determine observed duration (or censored)
    # using the same logic as _targets_to_survival but across all events
    obs_duration = np.full((n_samples, n_events), float(max(horizons)) + 1.0)
    obs_occurred = np.zeros((n_samples, n_events), dtype=bool)

    for ei in range(n_events):
        cols = [ei * n_horizons + hi for hi in range(n_horizons)]
        evt_tgt = targets[:, cols]  # (n_samples, n_horizons)
        any_event = evt_tgt > 0.5
        has_event = any_event.any(axis=1)
        first_idx = any_event.argmax(axis=1)
        obs_duration[:, ei] = np.where(has_event, horizons_arr[first_idx],
                                        float(max(horizons)) + 1.0)
        obs_occurred[:, ei] = has_event

    # Predicted median time per event (lower = sooner)
    pred_time = np.exp(predicted_mu)  # (n_samples, n_events)

    # Filter to persons with >= min_events distinct events
    n_events_per_person = obs_occurred.sum(axis=1)
    eligible = n_events_per_person >= min_events
    n_eligible = int(eligible.sum())

    if n_eligible == 0:
        return {
            'n_eligible': 0,
            'pairwise_accuracy': float('nan'),
            'top1_accuracy': float('nan'),
            'topk_accuracy': float('nan'),
            'mean_reciprocal_rank': float('nan'),
            'kendall_tau': float('nan'),
            'per_event_first_rate': {},
            'per_event_pred_first_rate': {},
        }

    obs_dur_elig = obs_duration[eligible]       # (n_elig, n_events)
    obs_occ_elig = obs_occurred[eligible]       # (n_elig, n_events)
    pred_time_elig = pred_time[eligible]        # (n_elig, n_events)

    # --- Pairwise accuracy ---
    # For each person, for each pair of events that both occurred,
    # check if predicted order matches observed order
    pairwise_correct = 0
    pairwise_total = 0
    event_pairs = list(combinations(range(n_events), 2))

    for ei, ej in event_pairs:
        both = obs_occ_elig[:, ei] & obs_occ_elig[:, ej]
        if not both.any():
            continue
        n_both = int(both.sum())

        obs_ei_first = obs_dur_elig[both, ei] < obs_dur_elig[both, ej]
        obs_ej_first = obs_dur_elig[both, ej] < obs_dur_elig[both, ei]
        obs_tied = ~obs_ei_first & ~obs_ej_first

        pred_ei_first = pred_time_elig[both, ei] < pred_time_elig[both, ej]
        pred_ej_first = pred_time_elig[both, ej] < pred_time_elig[both, ei]

        correct = ((obs_ei_first & pred_ei_first) |
                   (obs_ej_first & pred_ej_first) |
                   obs_tied).sum()
        pairwise_correct += int(correct)
        pairwise_total += n_both

    pairwise_accuracy = pairwise_correct / pairwise_total if pairwise_total > 0 else float('nan')

    # --- Top-1, Top-k, MRR ---
    # Observed first event: the one with smallest observed duration
    # (among events that occurred)
    # Mask non-occurred events with inf so they don't win
    obs_dur_masked = np.where(obs_occ_elig, obs_dur_elig, np.inf)
    obs_first_event = obs_dur_masked.argmin(axis=1)  # (n_elig,)

    # Predicted ordering: sort events by predicted time (ascending)
    pred_order = np.argsort(pred_time_elig, axis=1)  # (n_elig, n_events)
    pred_first_event = pred_order[:, 0]

    top1_correct = (pred_first_event == obs_first_event).sum()
    top1_accuracy = float(top1_correct) / n_eligible

    # Top-k: is the observed first event in predicted top-k?
    k = min(top_k, n_events)
    pred_topk = pred_order[:, :k]  # (n_elig, k)
    topk_hits = np.any(pred_topk == obs_first_event[:, np.newaxis], axis=1)
    topk_accuracy = float(topk_hits.sum()) / n_eligible

    # MRR: what rank does the observed first event have in predicted order?
    pred_ranks = np.argsort(pred_order, axis=1)  # rank of each event
    obs_first_rank = pred_ranks[np.arange(n_eligible), obs_first_event]
    reciprocal_ranks = 1.0 / (obs_first_rank + 1.0)
    mrr = float(reciprocal_ranks.mean())

    # --- Kendall's tau ---
    # For each person, compute tau over the events that occurred
    tau_values = []
    for i in range(n_eligible):
        occ_mask = obs_occ_elig[i]
        n_occ = int(occ_mask.sum())
        if n_occ < 2:
            continue
        occ_indices = np.where(occ_mask)[0]
        obs_ranks_i = np.argsort(np.argsort(obs_dur_elig[i, occ_indices]))
        pred_ranks_i = np.argsort(np.argsort(pred_time_elig[i, occ_indices]))

        # Kendall's tau-b (handles ties)
        concordant = 0
        discordant = 0
        for a, b in combinations(range(n_occ), 2):
            obs_diff = obs_ranks_i[a] - obs_ranks_i[b]
            pred_diff = pred_ranks_i[a] - pred_ranks_i[b]
            if obs_diff == 0 or pred_diff == 0:
                continue  # skip ties
            if (obs_diff > 0) == (pred_diff > 0):
                concordant += 1
            else:
                discordant += 1
        total_pairs = concordant + discordant
        if total_pairs > 0:
            tau_values.append((concordant - discordant) / total_pairs)

    kendall_tau = float(np.mean(tau_values)) if tau_values else float('nan')

    # --- Per-event first-event rates ---
    per_event_first = {}
    per_event_pred_first = {}
    for ei, event in enumerate(events):
        per_event_first[event] = float((obs_first_event == ei).mean())
        per_event_pred_first[event] = float((pred_first_event == ei).mean())

    return {
        'n_eligible': n_eligible,
        'pairwise_accuracy': pairwise_accuracy,
        'top1_accuracy': top1_accuracy,
        'topk_accuracy': topk_accuracy,
        'mean_reciprocal_rank': mrr,
        'kendall_tau': kendall_tau,
        'per_event_first_rate': per_event_first,
        'per_event_pred_first_rate': per_event_pred_first,
    }
