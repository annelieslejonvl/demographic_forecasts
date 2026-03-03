"""
Generate survival predictions: P(event within 1, 3, 5 years).

Converts raw XGBoost AFT/Cox predictions into interpretable
event probabilities at specific time horizons.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional
from scipy.stats import norm, logistic


def _get_aft_distribution(dist_name):
    """Get the CDF function for the AFT distribution."""
    dist_name = dist_name.lower()
    if dist_name == 'normal':
        return norm
    elif dist_name == 'logistic':
        return logistic
    elif dist_name in ('extreme', 'extreme_value'):
        # Extreme value (Gumbel) distribution
        # CDF(x) = 1 - exp(-exp(x))
        class ExtremeValue:
            @staticmethod
            def cdf(x, loc=0, scale=1):
                z = (x - loc) / scale
                return 1.0 - np.exp(-np.exp(z))
        return ExtremeValue()
    else:
        raise ValueError(f"Unknown AFT distribution: {dist_name}. "
                         f"Use 'normal', 'logistic', or 'extreme'.")


def predict_survival_aft(
    model,
    X,
    horizons=(1, 3, 5),
    distribution='normal',
    sigma=1.0,
):
    """
    Generate survival predictions from an XGBoost AFT model.

    AFT model predicts log(T) where T is time-to-event.
    The survival function is: S(t) = 1 - CDF(log(t); predicted_log_T, sigma)
    Event probability: P(event within H) = CDF(log(H); predicted_log_T, sigma)

    Args:
        model: Trained xgb.Booster AFT model
        X: Features (DataFrame or array)
        horizons: Time horizons in years
        distribution: AFT distribution ('normal', 'logistic', 'extreme')
        sigma: Scale parameter of the AFT distribution

    Returns:
        predictions: Dict with:
            'predicted_log_time': raw model output
            'predicted_median_time': exp(predicted_log_time)
            'risk_score': -predicted_log_time (higher = more risk)
            For each horizon h: 'prob_{h}yr' array of P(event within h years)
    """
    if isinstance(X, pd.DataFrame):
        dmatrix = xgb.DMatrix(X)
    else:
        dmatrix = xgb.DMatrix(X)

    # Raw prediction: log(time-to-event)
    predicted_log_time = model.predict(dmatrix)

    # Median predicted time to event
    predicted_median_time = np.exp(predicted_log_time)

    # Risk score (higher = event sooner = more risk)
    risk_score = -predicted_log_time

    # Get the distribution CDF
    dist = _get_aft_distribution(distribution)

    # Compute event probabilities at each horizon
    predictions = {
        'predicted_log_time': predicted_log_time,
        'predicted_median_time': predicted_median_time,
        'risk_score': risk_score,
    }

    for h in horizons:
        # P(event within H years) = CDF(log(H); predicted_log_T, sigma)
        log_h = np.log(h) if h > 0 else -np.inf
        prob = dist.cdf(log_h, loc=predicted_log_time, scale=sigma)

        # Clip to valid range
        prob = np.clip(prob, 0.0, 1.0)
        predictions[f'prob_{h}yr'] = prob

    return predictions


def predict_survival_aft_dmatrix(
    model, dmatrix, horizons=(1, 3, 5), distribution='normal', sigma=1.0,
):
    """Like predict_survival_aft but accepts a pre-built xgb.DMatrix."""
    predicted_log_time = model.predict(dmatrix)
    predicted_median_time = np.exp(predicted_log_time)
    risk_score = -predicted_log_time
    dist = _get_aft_distribution(distribution)
    predictions = {
        'predicted_log_time': predicted_log_time,
        'predicted_median_time': predicted_median_time,
        'risk_score': risk_score,
    }
    for h in horizons:
        log_h = np.log(h) if h > 0 else -np.inf
        prob = dist.cdf(log_h, loc=predicted_log_time, scale=sigma)
        prob = np.clip(prob, 0.0, 1.0)
        predictions[f'prob_{h}yr'] = prob
    return predictions


def predict_survival_cox_dmatrix(
    model, dmatrix, baseline_hazard, horizons=(1, 3, 5),
):
    """Like predict_survival_cox but accepts a pre-built xgb.DMatrix."""
    predicted_log_hr = model.predict(dmatrix)
    risk_score = predicted_log_hr
    predictions = {
        'predicted_log_hr': predicted_log_hr,
        'risk_score': risk_score,
    }
    for h in horizons:
        if isinstance(baseline_hazard, dict):
            available_times = sorted(baseline_hazard.keys())
            nearest_t = min(available_times, key=lambda t: abs(t - h))
            H0_t = baseline_hazard[nearest_t]
        elif isinstance(baseline_hazard, pd.Series):
            nearest_idx = (baseline_hazard.index - h).abs().argmin()
            H0_t = baseline_hazard.iloc[nearest_idx]
        else:
            H0_t = float(baseline_hazard)
        survival = np.exp(-H0_t * np.exp(predicted_log_hr))
        prob = 1.0 - survival
        prob = np.clip(prob, 0.0, 1.0)
        predictions[f'prob_{h}yr'] = prob
    return predictions


def predict_survival_cox(
    model,
    X,
    baseline_hazard,
    horizons=(1, 3, 5),
):
    """
    Generate survival predictions from an XGBoost Cox model.

    Cox model predicts log hazard ratio. Combined with baseline hazard:
    S(t|x) = S_0(t)^exp(pred)
    P(event within H) = 1 - S(H|x)

    Args:
        model: Trained xgb.Booster Cox model
        X: Features
        baseline_hazard: Dict or DataFrame with baseline cumulative hazard
                        at each time point. Keys/index = time, values = H_0(t).
        horizons: Time horizons

    Returns:
        predictions: Dict similar to AFT predictions
    """
    dmatrix = xgb.DMatrix(X)

    # Raw prediction: log hazard ratio
    predicted_log_hr = model.predict(dmatrix)
    risk_score = predicted_log_hr  # Higher = more risk (already correct for Cox)

    predictions = {
        'predicted_log_hr': predicted_log_hr,
        'risk_score': risk_score,
    }

    for h in horizons:
        # Get baseline cumulative hazard at horizon h
        if isinstance(baseline_hazard, dict):
            # Find nearest available time
            available_times = sorted(baseline_hazard.keys())
            nearest_t = min(available_times, key=lambda t: abs(t - h))
            H0_t = baseline_hazard[nearest_t]
        elif isinstance(baseline_hazard, pd.Series):
            nearest_idx = (baseline_hazard.index - h).abs().argmin()
            H0_t = baseline_hazard.iloc[nearest_idx]
        else:
            H0_t = float(baseline_hazard)

        # S(t|x) = exp(-H_0(t) * exp(pred))
        survival = np.exp(-H0_t * np.exp(predicted_log_hr))
        prob = 1.0 - survival
        prob = np.clip(prob, 0.0, 1.0)
        predictions[f'prob_{h}yr'] = prob

    return predictions


def estimate_baseline_hazard(duration_train, event_train, time_points=None):
    """
    Estimate baseline cumulative hazard using the Nelson-Aalen estimator.

    Used for Cox model predictions.

    Args:
        duration_train: Training durations
        event_train: Training event indicators
        time_points: Optional specific time points. If None, uses unique durations.

    Returns:
        pd.Series with index=time, values=cumulative hazard H_0(t)
    """
    duration = np.asarray(duration_train, dtype=np.float64)
    event = np.asarray(event_train, dtype=np.int32)

    # Sort by duration
    order = np.argsort(duration)
    duration_sorted = duration[order]
    event_sorted = event[order]

    unique_times = np.unique(duration_sorted)
    cumulative_hazard = np.zeros(len(unique_times))

    running_hazard = 0.0
    for i, t in enumerate(unique_times):
        # Number at risk at time t
        at_risk = np.sum(duration_sorted >= t)
        # Number of events at time t
        n_events = np.sum((duration_sorted == t) & (event_sorted == 1))

        if at_risk > 0:
            running_hazard += n_events / at_risk

        cumulative_hazard[i] = running_hazard

    return pd.Series(cumulative_hazard, index=unique_times)


def predict_event_probabilities(
    model,
    X,
    horizons=(1, 3, 5),
    model_type='aft',
    distribution='normal',
    sigma=1.0,
    baseline_hazard=None,
):
    """
    Unified prediction function for both AFT and Cox models.

    Args:
        model: Trained xgb.Booster
        X: Features
        horizons: Time horizons in years
        model_type: 'aft' or 'cox'
        distribution: AFT distribution name (only for AFT)
        sigma: AFT scale parameter (only for AFT)
        baseline_hazard: Baseline cumulative hazard (only for Cox)

    Returns:
        predictions: Dict with risk scores and probabilities per horizon
    """
    if model_type == 'aft':
        return predict_survival_aft(
            model, X, horizons=horizons,
            distribution=distribution, sigma=sigma,
        )
    elif model_type == 'cox':
        if baseline_hazard is None:
            raise ValueError("baseline_hazard required for Cox predictions")
        return predict_survival_cox(
            model, X, baseline_hazard, horizons=horizons,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'aft' or 'cox'.")


def predict_all_events(
    models,
    X,
    horizons=(1, 3, 5),
    model_type='aft',
    distribution='normal',
    sigma=1.0,
    baseline_hazards=None,
):
    """
    Generate predictions for all event types.

    Args:
        models: Dict mapping event_col → trained model
        X: Features (same for all events)
        horizons: Time horizons
        model_type: 'aft' or 'cox'
        distribution: AFT distribution (for AFT)
        sigma: AFT scale parameter (for AFT)
        baseline_hazards: Dict mapping event_col → baseline hazard (for Cox)

    Returns:
        Dict mapping event_col → predictions dict
    """
    all_predictions = {}

    for event_col, model in models.items():
        print(f"  Predicting {event_col}...")

        bh = None
        if baseline_hazards and event_col in baseline_hazards:
            bh = baseline_hazards[event_col]

        preds = predict_event_probabilities(
            model, X,
            horizons=horizons,
            model_type=model_type,
            distribution=distribution,
            sigma=sigma,
            baseline_hazard=bh,
        )
        all_predictions[event_col] = preds

    return all_predictions


def predictions_to_dataframe(
    all_predictions,
    horizons=(1, 3, 5),
    index=None,
):
    """
    Convert predictions dict to a tidy DataFrame.

    Args:
        all_predictions: Dict from predict_all_events
        horizons: Time horizons
        index: Optional index for the DataFrame

    Returns:
        DataFrame with columns like:
            y_moved_prob_1yr, y_moved_prob_3yr, y_moved_prob_5yr,
            y_moved_median_time, birth1_event_prob_1yr, ...
    """
    result = {}

    for event_col, preds in all_predictions.items():
        # Probabilities at each horizon
        for h in horizons:
            key = f'prob_{h}yr'
            if key in preds:
                result[f'{event_col}_prob_{h}yr'] = preds[key]

        # Median predicted time (AFT only)
        if 'predicted_median_time' in preds:
            result[f'{event_col}_median_time'] = preds['predicted_median_time']

        # Risk score
        if 'risk_score' in preds:
            result[f'{event_col}_risk'] = preds['risk_score']

    df = pd.DataFrame(result, index=index)
    return df
