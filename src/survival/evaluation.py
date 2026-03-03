"""
Survival-specific evaluation metrics.

Provides concordance index, time-dependent AUC, Brier score,
and calibration analysis for survival models.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime

def concordance_index(
    duration,
    event_observed,
    predicted_risk,
    max_samples=20000,
    random_state=42,
):
    """
    Compute Harrell's concordance index (C-index).

    Measures the model's ability to correctly rank individuals
    by their risk of experiencing the event.

    C-index = 0.5 → random, 1.0 → perfect discrimination.

    Args:
        duration: Array of observed durations
        event_observed: Array of event indicators (1=event, 0=censored)
        predicted_risk: Array of predicted risk scores (higher = more risk).
                       For AFT: use negative predicted time (shorter time = higher risk).
                       For Cox: use predicted log hazard ratio directly.

    Returns:
        c_index: float between 0 and 1
    """
    event_bool = np.asarray(event_observed, dtype=bool)
    if not np.any(event_bool):
        return 0.5
    try:
        from sksurv.metrics import concordance_index_censored
        # sksurv expects boolean event indicator
        c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
            event_bool, duration, predicted_risk
        )
        return c_index
    except ImportError as exc:
        raise RuntimeError(
            "scikit-survival is required for concordance_index; "
            "install it to run evaluation."
        ) from exc


def time_dependent_auc(
    duration_train,
    event_train,
    duration_test,
    event_test,
    predicted_risk,
    horizons,
    max_train_samples=50000,
    max_test_samples=50000,
):
    """
    Compute time-dependent AUC at specific time horizons.

    Uses the cumulative/dynamic AUC definition: at time t, discriminate
    between individuals who had the event by time t vs those who survived.

    Args:
        duration_train: Training durations (for Kaplan-Meier estimation)
        event_train: Training event indicators
        duration_test: Test durations
        event_test: Test event indicators
        predicted_risk: Predicted risk scores for test set
        horizons: List of time horizons (e.g., [1, 3, 5])
        max_train_samples: Max samples from train (to speed up, default 50k)
        max_test_samples: Max samples from test (to speed up, default 50k)

    Returns:
        Dict mapping horizon → AUC value
    """
    # Subsample if datasets are very large (cumulative_dynamic_auc is O(n^2))
    if len(duration_train) > max_train_samples:
        rng = np.random.RandomState(42)
        train_idx = rng.choice(len(duration_train), max_train_samples, replace=False)
        duration_train = np.asarray(duration_train)[train_idx]
        event_train = np.asarray(event_train)[train_idx]

    if len(duration_test) > max_test_samples:
        rng = np.random.RandomState(42)
        test_idx = rng.choice(len(duration_test), max_test_samples, replace=False)
        duration_test = np.asarray(duration_test)[test_idx]
        event_test = np.asarray(event_test)[test_idx]
        predicted_risk = np.asarray(predicted_risk)[test_idx]

    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv

        # Create structured arrays for sksurv
        y_train = Surv.from_arrays(
            event=np.asarray(event_train, dtype=bool),
            time=np.asarray(duration_train, dtype=np.float64),
        )
        y_test = Surv.from_arrays(
            event=np.asarray(event_test, dtype=bool),
            time=np.asarray(duration_test, dtype=np.float64),
        )

        # Filter horizons to valid range (sksurv requires strict < max follow-up)
        valid_horizons = [h for h in horizons if h < duration_test.max()]
        if not valid_horizons:
            return {h: np.nan for h in horizons}

        # Try all valid horizons first; if IPCW fails (censoring survival = 0),
        # progressively drop the longest horizon until it works.
        result = {h: np.nan for h in horizons}
        remaining = list(valid_horizons)
        while remaining:
            try:
                aucs, mean_auc = cumulative_dynamic_auc(
                    y_train, y_test, predicted_risk, np.array(remaining)
                )
                for h, auc_val in zip(remaining, aucs):
                    result[h] = float(auc_val)
                break
            except ValueError:
                # Censoring survival hits 0 at longest horizon — drop it
                remaining.pop()
        return result

    except ImportError:
        # Fallback: simple AUC at each horizon
        from sklearn.metrics import roc_auc_score

        result = {}
        duration_test = np.asarray(duration_test)
        event_test = np.asarray(event_test)

        for h in horizons:
            # Binary label: event within horizon
            y_binary = ((event_test == 1) & (duration_test <= h)).astype(int)

            # Only compute if both classes present
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                result[h] = np.nan
            else:
                try:
                    result[h] = float(roc_auc_score(y_binary, predicted_risk))
                except ValueError:
                    result[h] = np.nan

        return result


def brier_score_at_horizons(
    duration_test,
    event_test,
    predicted_proba,
    horizons,
    duration_train=None,
    event_train=None,
    max_train_samples=50000,
    max_test_samples=50000,
):
    """
    Compute Brier score at specific time horizons.

    Brier score measures calibration: how close predicted probabilities
    are to observed outcomes. Lower is better.

    Args:
        duration_test: Test durations
        event_test: Test event indicators
        predicted_proba: Dict mapping horizon → predicted P(event within horizon)
        horizons: List of horizons
        duration_train: Optional training durations (for IPCW correction)
        event_train: Optional training event indicators
        max_train_samples: Max samples from train (default 50k)
        max_test_samples: Max samples from test (default 50k)

    Returns:
        Dict mapping horizon → Brier score
    """
    # Subsample if datasets are very large
    test_idx = None
    if len(duration_test) > max_test_samples:
        rng = np.random.RandomState(42)
        test_idx = rng.choice(len(duration_test), max_test_samples, replace=False)
        duration_test = np.asarray(duration_test)[test_idx]
        event_test = np.asarray(event_test)[test_idx]

    if duration_train is not None and len(duration_train) > max_train_samples:
        rng = np.random.RandomState(42)
        train_idx = rng.choice(len(duration_train), max_train_samples, replace=False)
        duration_train = np.asarray(duration_train)[train_idx]
        event_train = np.asarray(event_train)[train_idx]

    try:
        from sksurv.metrics import brier_score as sksurv_brier
        from sksurv.util import Surv

        if duration_train is not None and event_train is not None:
            y_train = Surv.from_arrays(
                event=np.asarray(event_train, dtype=bool),
                time=np.asarray(duration_train, dtype=np.float64),
            )
        else:
            y_train = None

        y_test = Surv.from_arrays(
            event=np.asarray(event_test, dtype=bool),
            time=np.asarray(duration_test, dtype=np.float64),
        )

        result = {}
        for h in horizons:
            if h not in predicted_proba:
                result[h] = np.nan
                continue
            # Get predictions (subsample if needed)
            proba_h = np.asarray(predicted_proba[h])
            if test_idx is not None:
                proba_h = proba_h[test_idx]
            # Survival probability = 1 - event probability
            surv_prob = 1.0 - proba_h
            try:
                times, bs = sksurv_brier(
                    y_train if y_train is not None else y_test,
                    y_test,
                    surv_prob,
                    times=[h],
                )
                result[h] = float(bs[0])
            except Exception:
                result[h] = np.nan

        return result

    except ImportError:
        # Fallback: simple Brier score (without IPCW correction)
        result = {}
        duration_test = np.asarray(duration_test)
        event_test = np.asarray(event_test)

        for h in horizons:
            if h not in predicted_proba:
                result[h] = np.nan
                continue

            proba = np.asarray(predicted_proba[h])
            if test_idx is not None:
                proba = proba[test_idx]

            # Binary outcome: event within horizon
            y_binary = ((event_test == 1) & (duration_test <= h)).astype(float)

            # Exclude individuals censored before horizon (uncertain outcome)
            valid = ~((event_test == 0) & (duration_test < h))
            if valid.sum() == 0:
                result[h] = np.nan
            else:
                bs = np.mean((proba[valid] - y_binary[valid]) ** 2)
                result[h] = float(bs)

        return result


def calibration_table(
    duration,
    event_observed,
    predicted_proba,
    horizon,
    n_bins=10,
):
    """
    Create calibration table: predicted vs observed event rates.

    Args:
        duration: Test durations
        event_observed: Test event indicators
        predicted_proba: Predicted P(event within horizon)
        horizon: Time horizon
        n_bins: Number of calibration bins

    Returns:
        DataFrame with columns: bin, mean_predicted, observed_rate, count
    """
    duration = np.asarray(duration)
    event_observed = np.asarray(event_observed)
    predicted_proba = np.asarray(predicted_proba)

    # Binary outcome at this horizon
    y_binary = ((event_observed == 1) & (duration <= horizon)).astype(float)

    # Exclude censored-before-horizon
    valid = ~((event_observed == 0) & (duration < horizon))

    pred_valid = predicted_proba[valid]
    y_valid = y_binary[valid]

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(pred_valid, bin_edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() > 0:
            rows.append({
                'bin': b,
                'bin_lower': bin_edges[b],
                'bin_upper': bin_edges[b + 1],
                'mean_predicted': float(pred_valid[mask].mean()),
                'observed_rate': float(y_valid[mask].mean()),
                'count': int(mask.sum()),
            })

    return pd.DataFrame(rows)


def evaluate_survival_model(
    predicted_risk,
    predicted_proba,
    duration_test,
    event_test,
    horizons=(1, 3, 5),
    duration_train=None,
    event_train=None,
    event_name='event',
    cindex_max_samples=200000,
    cindex_random_state=42,
):
    """
    Full evaluation suite for a survival model.

    Args:
        predicted_risk: Risk scores (higher = event sooner).
                       For AFT: use -predicted_log_time.
                       For Cox: use predicted log hazard ratio.
        predicted_proba: Dict mapping horizon → P(event within horizon)
        duration_test: Test durations
        event_test: Test event indicators
        horizons: Time horizons to evaluate
        duration_train: Optional training durations
        event_train: Optional training event indicators
        event_name: Name of event (for display)

    Returns:
        Dict with all metrics
    """
    from datetime import datetime

    t_start = datetime.now()
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Starting evaluation for {event_name}")

    duration_test = np.asarray(duration_test, dtype=np.float64)
    event_test = np.asarray(event_test, dtype=np.int32)
    predicted_risk = np.asarray(predicted_risk, dtype=np.float64)

    metrics = {}

    # C-index
    t0 = datetime.now()
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Starting cindex for {event_name}")
    if cindex_max_samples is not None and len(duration_test) > cindex_max_samples:
        rng = np.random.RandomState(cindex_random_state)
        idx = rng.choice(len(duration_test), cindex_max_samples, replace=False)
        c_duration = duration_test[idx]
        c_event = event_test[idx]
        c_risk = predicted_risk[idx]
    else:
        c_duration = duration_test
        c_event = event_test
        c_risk = predicted_risk

    c_idx = concordance_index(c_duration, c_event, c_risk)
    metrics['c_index'] = c_idx
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] C-index: {c_idx:.4f} (+{(datetime.now() - t0).total_seconds():.1f}s)")

    # Time-dependent AUC
    t0 = datetime.now()
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Computing time-dependent AUC...")
    if duration_train is not None and event_train is not None:
        td_auc = time_dependent_auc(
            duration_train, event_train,
            duration_test, event_test,
            predicted_risk, list(horizons)
        )
    else:
        td_auc = time_dependent_auc(
            duration_test, event_test,
            duration_test, event_test,
            predicted_risk, list(horizons)
        )
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Time-dependent AUC done (+{(datetime.now() - t0).total_seconds():.1f}s)")

    for h, auc_val in td_auc.items():
        metrics[f'auc_{h}yr'] = auc_val
        print(f"      AUC @{h}yr: {auc_val:.4f}" if not np.isnan(auc_val) else f"      AUC @{h}yr: N/A")

    # Brier score
    t0 = datetime.now()
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Computing Brier scores...")
    bs = brier_score_at_horizons(
        duration_test, event_test,
        predicted_proba, list(horizons),
        duration_train=duration_train,
        event_train=event_train,
    )
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Brier scores done (+{(datetime.now() - t0).total_seconds():.1f}s)")

    for h, bs_val in bs.items():
        metrics[f'brier_{h}yr'] = bs_val
        print(f"      Brier @{h}yr: {bs_val:.4f}" if not np.isnan(bs_val) else f"      Brier @{h}yr: N/A")

    # F1, AP, AUC-PR at each horizon
    t0 = datetime.now()
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Computing F1, AP, AUC-PR...")
    from sklearn.metrics import (
        f1_score, average_precision_score,
        precision_recall_curve, auc as sk_auc,
    )

    for h in horizons:
        if h not in predicted_proba:
            continue
        y_true = ((event_test == 1) & (duration_test <= h)).astype(int)
        y_prob = np.asarray(predicted_proba[h], dtype=np.float64)
        n_pos = int(y_true.sum())
        if n_pos < 5 or n_pos >= len(y_true):
            metrics[f'f1_{h}yr'] = float('nan')
            metrics[f'ap_{h}yr'] = float('nan')
            metrics[f'auc_pr_{h}yr'] = float('nan')
            continue

        # AP (average precision = area under PR curve, interpolated)
        ap = float(average_precision_score(y_true, y_prob))
        metrics[f'ap_{h}yr'] = ap

        # AUC-PR (trapezoidal area under precision-recall curve)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = float(sk_auc(recall, precision))
        metrics[f'auc_pr_{h}yr'] = auc_pr

        # F1 with optimal threshold search
        base_rate = n_pos / len(y_true)
        candidates = np.unique(np.concatenate([
            np.linspace(max(0.005, base_rate * 0.2),
                        min(0.95, base_rate * 5), 30),
            np.array([0.5, base_rate]),
        ]))
        best_f1 = 0.0
        for thr in candidates:
            _f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
            if _f1 > best_f1:
                best_f1 = _f1
        metrics[f'f1_{h}yr'] = float(best_f1)
        print(f"      @{h}yr: F1={best_f1:.4f}, AP={ap:.4f}, AUC-PR={auc_pr:.4f}")

    print(f"    [{datetime.now().strftime('%H:%M:%S')}] F1/AP/AUC-PR done (+{(datetime.now() - t0).total_seconds():.1f}s)")

    # Calibration tables
    t0 = datetime.now()
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Computing calibration tables...")
    calibration = {}
    for h in horizons:
        if h in predicted_proba:
            cal = calibration_table(
                duration_test, event_test, predicted_proba[h], h
            )
            calibration[h] = cal

    metrics['calibration'] = calibration
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Calibration done (+{(datetime.now() - t0).total_seconds():.1f}s)")

    total_time = (datetime.now() - t_start).total_seconds()
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Total evaluation time: {total_time:.1f}s")

    return metrics


def evaluate_aft_shared_metrics(
    all_predictions,
    test_df,
    events,
    horizons=(1, 3, 5),
    distribution='normal',
    sigma=1.0,
    max_samples=200_000,
):
    """Evaluate XGBoost survival using the shared AFT metric suite.

    Bridges XGBoost per-event predictions into the format expected by
    ``src.sequence.evaluation.evaluate_aft_survival_metrics`` so that
    XGBoost and GRU sequence models produce directly comparable metrics
    (C-index, CRPS, IBS, TD-AUC).

    Args:
        all_predictions: Dict mapping event_col -> predictions dict
                         (from ``predict_all_events``). Must contain
                         ``predicted_log_time`` per event.
        test_df: Test DataFrame with ``{event}_duration`` and
                 ``{event}_event_observed`` columns for each event.
        events: List of event column names.
        horizons: Prediction horizons in years.
        distribution: AFT base distribution ('normal', 'logistic', 'extreme').
        sigma: Global scale parameter (from config).
        max_samples: Subsample budget for metric computation.

    Returns:
        Dict with ``per_event`` and ``aggregate`` keys, same as
        ``evaluate_aft_survival_metrics``.
    """
    from src.sequence.evaluation import evaluate_aft_survival_metrics

    n_samples = len(test_df)
    n_events = len(events)
    n_horizons = len(horizons)

    # Pack (mu, log_sigma) into (n_samples, 2*n_events)
    # XGBoost: mu = predicted_log_time, sigma is globally fixed
    aft_params = np.zeros((n_samples, 2 * n_events), dtype=np.float64)
    for ei, event in enumerate(events):
        preds = all_predictions[event]
        aft_params[:, ei] = preds['predicted_log_time']            # mu
        aft_params[:, n_events + ei] = np.log(sigma)               # log_sigma (constant)

    # Build binary horizon targets (n_samples, n_events * n_horizons)
    targets = np.zeros((n_samples, n_events * n_horizons), dtype=np.float32)
    for ei, event in enumerate(events):
        duration_col = f"{event}_duration"
        observed_col = f"{event}_event_observed"
        dur = test_df[duration_col].values
        evt = test_df[observed_col].values
        for hi, h in enumerate(horizons):
            col = ei * n_horizons + hi
            targets[:, col] = ((evt == 1) & (dur <= h)).astype(np.float32)

    return evaluate_aft_survival_metrics(
        aft_params, targets, events, horizons,
        max_samples=max_samples,
        distribution=distribution,
    )


def stack_survival_predictions(
    all_predictions,
    test_df,
    events,
    horizons=(1, 3, 5),
):
    """Convert XGBoost per-event prediction dicts to stacked arrays.

    Bridges the XGBoost format ``{event -> {prob_1yr, prob_3yr, ...}}``
    into the ``(n_samples, n_events * n_horizons)`` layout used by
    ``evaluate_lm_metrics`` and ``evaluate_grouped_metrics``.

    Args:
        all_predictions: Dict mapping event -> predictions dict with
            ``prob_{h}yr`` keys.
        test_df: Test DataFrame with ``{event}_duration`` and
            ``{event}_event_observed`` columns.
        events: List of event column names.
        horizons: Prediction horizons in years.

    Returns:
        (probabilities, targets) tuple, each ``(n_samples, n_events * n_horizons)``
        in event-major layout.
    """
    n_samples = len(test_df)
    n_events = len(events)
    n_horizons = len(horizons)

    probabilities = np.zeros((n_samples, n_events * n_horizons), dtype=np.float64)
    targets = np.zeros((n_samples, n_events * n_horizons), dtype=np.float32)

    for ei, event in enumerate(events):
        duration_col = f"{event}_duration"
        observed_col = f"{event}_event_observed"
        dur = test_df[duration_col].values
        evt = test_df[observed_col].values

        for hi, h in enumerate(horizons):
            col = ei * n_horizons + hi
            prob_key = f'prob_{h}yr'
            if prob_key in all_predictions[event]:
                probabilities[:, col] = all_predictions[event][prob_key]
            targets[:, col] = ((evt == 1) & (dur <= h)).astype(np.float32)

    return probabilities, targets


def evaluate_by_group(
    df_test,
    predicted_risk,
    predicted_proba,
    duration_col,
    event_col,
    group_cols,
    horizons=(1, 3, 5),
    min_group_size=100,
    use_polars=True,
):
    """
    Evaluate survival predictions by demographic group.

    Computes observed vs predicted event rates for each group
    at each horizon, similar to transition rate evaluation in the
    classifier pipeline (evaluate_transition_rates_by_group).

    Args:
        df_test: Test DataFrame with group columns
        predicted_risk: Risk scores
        predicted_proba: Dict mapping horizon → P(event within horizon)
        duration_col: Duration column name
        event_col: Event observed column name
        group_cols: List of grouping columns (e.g., ['gender', 'age_group', 'refnis'])
        horizons: Time horizons
        min_group_size: Minimum group size to include
        use_polars: If True, use Polars for 2-3x speedup (requires polars package)

    Returns:
        (group_df, summary) tuple:
            group_df: DataFrame with per-group observed/predicted rates and errors
            summary: Dict with weighted/unweighted MAE, MSE, RMSE per horizon
    """
    # Try Polars optimization
    if use_polars:
        try:
            import polars as pl
            return _evaluate_by_group_polars(
                df_test, predicted_risk, predicted_proba,
                duration_col, event_col, group_cols, horizons, min_group_size
            )
        except ImportError:
            pass  # Fall back to pandas

    # Pandas implementation
    valid_group_cols = [c for c in group_cols if c in df_test.columns]
    if not valid_group_cols:
        return pd.DataFrame(), {}

    needed_cols = valid_group_cols + [duration_col, event_col]
    df_pos = df_test[needed_cols].copy()
    df_pos['_row_pos'] = np.arange(len(df_pos))
    duration = df_pos[duration_col].values
    event = df_pos[event_col].values

    rows = []

    grouped = df_pos.groupby(valid_group_cols)

    for group_key, group_df in grouped:
        if len(group_df) < min_group_size:
            continue

        idx = group_df['_row_pos'].values
        group_duration = duration[idx]
        group_event = event[idx]

        row = {}
        if isinstance(group_key, tuple):
            for col, val in zip(valid_group_cols, group_key):
                row[col] = val
        else:
            row[valid_group_cols[0]] = group_key

        row['count'] = len(group_df)

        for h in horizons:
            # Observed rate: fraction with event within horizon
            observed = ((group_event == 1) & (group_duration <= h)).mean()
            row[f'observed_rate_{h}yr'] = observed

            # Predicted rate
            if h in predicted_proba:
                pred = predicted_proba[h][idx].mean()
                row[f'predicted_rate_{h}yr'] = pred
                row[f'abs_error_{h}yr'] = abs(pred - observed)
                row[f'sq_error_{h}yr'] = (pred - observed) ** 2

        rows.append(row)

    group_df = pd.DataFrame(rows)

    # Compute summary statistics (comparable to classifier's group eval)
    summary = {}
    if not group_df.empty:
        weights = group_df['count'].values
        summary['groups'] = int(len(group_df))
        summary['min_group_size'] = int(min_group_size)

        for h in horizons:
            abs_col = f'abs_error_{h}yr'
            sq_col = f'sq_error_{h}yr'
            if abs_col not in group_df.columns:
                continue

            abs_err = group_df[abs_col].values
            sq_err = group_df[sq_col].values

            summary[f'mae_weighted_{h}yr'] = float(np.average(abs_err, weights=weights))
            summary[f'mse_weighted_{h}yr'] = float(np.average(sq_err, weights=weights))
            summary[f'rmse_weighted_{h}yr'] = float(np.sqrt(np.average(sq_err, weights=weights)))
            summary[f'mae_unweighted_{h}yr'] = float(abs_err.mean())
            summary[f'mse_unweighted_{h}yr'] = float(sq_err.mean())
            summary[f'rmse_unweighted_{h}yr'] = float(np.sqrt(sq_err.mean()))

    return group_df, summary


def _evaluate_by_group_polars(
    df_test,
    predicted_risk,
    predicted_proba,
    duration_col,
    event_col,
    group_cols,
    horizons,
    min_group_size,
):
    """
    Polars-optimized group evaluation (2-3x faster groupby).
    """
    import polars as pl

    valid_group_cols = [c for c in group_cols if c in df_test.columns]
    if not valid_group_cols:
        return pd.DataFrame(), {}

    # Convert to Polars
    needed_cols = valid_group_cols + [duration_col, event_col]
    if isinstance(df_test, pd.DataFrame):
        pl_df = pl.from_pandas(df_test[needed_cols].reset_index(drop=True))
    else:
        pl_df = df_test.select(needed_cols)

    # Add predicted probabilities as columns
    for h in horizons:
        if h in predicted_proba:
            pl_df = pl_df.with_columns([
                pl.Series(f'pred_{h}yr', predicted_proba[h])
            ])

    # Compute observed + predicted per horizon
    exprs = []
    for h in horizons:
        # Observed: event within horizon
        obs_expr = ((pl.col(event_col) == 1) & (pl.col(duration_col) <= h)).cast(pl.Float64).alias(f'obs_{h}yr')
        exprs.append(obs_expr)

    pl_df = pl_df.with_columns(exprs)

    # Group and aggregate
    agg_exprs = [pl.count().alias('count')]
    for h in horizons:
        # Observed rate
        agg_exprs.append(pl.col(f'obs_{h}yr').mean().alias(f'observed_rate_{h}yr'))

        # Predicted rate
        if h in predicted_proba:
            agg_exprs.append(pl.col(f'pred_{h}yr').mean().alias(f'predicted_rate_{h}yr'))

    group_df = pl_df.group_by(valid_group_cols).agg(agg_exprs)

    # Filter by min_group_size
    group_df = group_df.filter(pl.col('count') >= min_group_size)

    # Compute errors
    error_exprs = []
    for h in horizons:
        if h in predicted_proba:
            obs_col = f'observed_rate_{h}yr'
            pred_col = f'predicted_rate_{h}yr'
            error_exprs.append(
                (pl.col(pred_col) - pl.col(obs_col)).abs().alias(f'abs_error_{h}yr')
            )
            error_exprs.append(
                ((pl.col(pred_col) - pl.col(obs_col)) ** 2).alias(f'sq_error_{h}yr')
            )

    if error_exprs:
        group_df = group_df.with_columns(error_exprs)

    # Convert to pandas for compatibility
    group_df = group_df.to_pandas()

    # Compute summary statistics
    summary = {}
    if not group_df.empty:
        weights = group_df['count'].values
        summary['groups'] = int(len(group_df))
        summary['min_group_size'] = int(min_group_size)

        for h in horizons:
            abs_col = f'abs_error_{h}yr'
            sq_col = f'sq_error_{h}yr'
            if abs_col not in group_df.columns:
                continue

            abs_err = group_df[abs_col].values
            sq_err = group_df[sq_col].values

            summary[f'mae_weighted_{h}yr'] = float(np.average(abs_err, weights=weights))
            summary[f'mse_weighted_{h}yr'] = float(np.average(sq_err, weights=weights))
            summary[f'rmse_weighted_{h}yr'] = float(np.sqrt(np.average(sq_err, weights=weights)))
            summary[f'mae_unweighted_{h}yr'] = float(abs_err.mean())
            summary[f'mse_unweighted_{h}yr'] = float(sq_err.mean())
            summary[f'rmse_unweighted_{h}yr'] = float(np.sqrt(sq_err.mean()))

    return group_df, summary


class StreamingGroupEvaluator:
    """
    Memory-efficient streaming group-level evaluation.

    Accumulates per-group sums and counts across batches without
    holding the full dataset in memory. After all batches are fed,
    computes group-level rates, errors, and RMSE summary.

    Usage:
        evaluator = StreamingGroupEvaluator(group_cols, horizons)
        for batch_df, batch_proba in batches:
            evaluator.add_batch(batch_df, batch_proba, duration_col, event_col)
        group_df, summary = evaluator.finalize(min_group_size=100)
    """

    def __init__(self, group_cols, horizons=(1, 3, 5), use_polars=True):
        self.group_cols = list(group_cols)
        self.horizons = list(horizons)
        # Accumulators: group_key -> {count, event_sum_Hyr, pred_sum_Hyr}
        self._accum = {}
        self.use_polars = use_polars

        # Check if polars is available
        if self.use_polars:
            try:
                import polars as pl
                self._pl = pl
            except ImportError:
                self.use_polars = False

    def add_batch(self, batch_df, batch_proba, duration_col, event_col):
        """
        Accumulate one batch of data.

        Args:
            batch_df: DataFrame with group columns, duration, and event columns
            batch_proba: Dict mapping horizon -> P(event within horizon) array,
                         aligned with batch_df index
            duration_col: Duration column name in batch_df
            event_col: Event observed column name in batch_df
        """
        if self.use_polars:
            self._add_batch_polars(batch_df, batch_proba, duration_col, event_col)
        else:
            self._add_batch_pandas(batch_df, batch_proba, duration_col, event_col)

    def _add_batch_polars(self, batch_df, batch_proba, duration_col, event_col):
        """Polars-optimized batch accumulation (2-3x faster groupby)."""
        valid_cols = [c for c in self.group_cols if c in batch_df.columns]
        if not valid_cols:
            return

        # Convert to polars for faster groupby
        pl = self._pl

        # Build a combined DataFrame with batch_df columns + predicted probabilities
        if isinstance(batch_df, pd.DataFrame):
            needed_cols = valid_cols + [duration_col, event_col]
            temp_df = batch_df[needed_cols].copy()

            # Add predicted probabilities as columns to pandas df first
            for h in self.horizons:
                if h in batch_proba:
                    temp_df[f'pred_{h}'] = batch_proba[h]

            # Convert to Polars
            pl_df = pl.from_pandas(temp_df.reset_index(drop=True))
        else:
            pl_df = batch_df.clone()
            for h in self.horizons:
                if h in batch_proba:
                    pl_df = pl_df.with_columns([
                        pl.Series(f'pred_{h}', batch_proba[h])
                    ])

        # Compute observed for each horizon
        exprs = []
        for h in self.horizons:
            # Observed: event occurred within horizon
            obs_expr = ((pl.col(event_col) == 1) & (pl.col(duration_col) <= h)).cast(pl.Float64).alias(f'obs_{h}')
            exprs.append(obs_expr)

        pl_df = pl_df.with_columns(exprs)

        # Group and aggregate in one shot (fast!)
        agg_exprs = [pl.count().alias('count')]
        for h in self.horizons:
            agg_exprs.append(pl.col(f'obs_{h}').sum().alias(f'obs_sum_{h}'))
            if h in batch_proba:
                agg_exprs.append(pl.col(f'pred_{h}').sum().alias(f'pred_sum_{h}'))

        grouped = pl_df.group_by(valid_cols).agg(agg_exprs)

        # Convert to dict and accumulate
        for row in grouped.iter_rows(named=True):
            # Extract group key
            gk = tuple(row[c] for c in valid_cols)

            if gk not in self._accum:
                self._accum[gk] = {'count': 0}
                for h in self.horizons:
                    self._accum[gk][f'obs_sum_{h}'] = 0.0
                    self._accum[gk][f'pred_sum_{h}'] = 0.0

            self._accum[gk]['count'] += row['count']
            for h in self.horizons:
                self._accum[gk][f'obs_sum_{h}'] += row[f'obs_sum_{h}']
                if h in batch_proba:
                    self._accum[gk][f'pred_sum_{h}'] += row[f'pred_sum_{h}']

    def _add_batch_pandas(self, batch_df, batch_proba, duration_col, event_col):
        """Pandas implementation (fallback)."""
        valid_cols = [c for c in self.group_cols if c in batch_df.columns]
        if not valid_cols:
            return

        duration = batch_df[duration_col].values
        event = batch_df[event_col].values

        # Build group keys as a single column for fast grouping
        if len(valid_cols) == 1:
            keys = batch_df[valid_cols[0]].values
        else:
            keys = list(zip(*(batch_df[c].values for c in valid_cols)))

        # Pre-compute observed binary per horizon
        observed_by_h = {}
        for h in self.horizons:
            observed_by_h[h] = ((event == 1) & (duration <= h)).astype(np.float64)

        # Group by key and accumulate
        key_series = pd.Series(range(len(batch_df)), index=batch_df.index)
        grouped = batch_df.groupby(valid_cols)

        for group_key, group_idx_df in grouped:
            # Normalize key to tuple
            gk = group_key if isinstance(group_key, tuple) else (group_key,)

            pos = group_idx_df.index
            n = len(pos)

            if gk not in self._accum:
                self._accum[gk] = {'count': 0}
                for h in self.horizons:
                    self._accum[gk][f'obs_sum_{h}'] = 0.0
                    self._accum[gk][f'pred_sum_{h}'] = 0.0

            self._accum[gk]['count'] += n

            # Use positional indexing for numpy arrays
            # Convert index positions to iloc positions
            iloc_pos = batch_df.index.get_indexer(pos)

            for h in self.horizons:
                self._accum[gk][f'obs_sum_{h}'] += observed_by_h[h][iloc_pos].sum()
                if h in batch_proba:
                    self._accum[gk][f'pred_sum_{h}'] += batch_proba[h][iloc_pos].sum()

    def finalize(self, min_group_size=100):
        """
        Compute final group-level metrics and RMSE summary.

        Returns:
            (group_df, summary) tuple, same format as evaluate_by_group
        """
        valid_cols = self.group_cols

        rows = []
        for gk, acc in self._accum.items():
            if acc['count'] < min_group_size:
                continue

            row = {}
            if len(valid_cols) == 1:
                row[valid_cols[0]] = gk[0]
            else:
                for col, val in zip(valid_cols, gk):
                    row[col] = val

            row['count'] = acc['count']

            for h in self.horizons:
                observed_rate = acc[f'obs_sum_{h}'] / acc['count']
                row[f'observed_rate_{h}yr'] = observed_rate

                if acc[f'pred_sum_{h}'] > 0 or f'pred_sum_{h}' in acc:
                    pred_rate = acc[f'pred_sum_{h}'] / acc['count']
                    row[f'predicted_rate_{h}yr'] = pred_rate
                    row[f'abs_error_{h}yr'] = abs(pred_rate - observed_rate)
                    row[f'sq_error_{h}yr'] = (pred_rate - observed_rate) ** 2

            rows.append(row)

        group_df = pd.DataFrame(rows)

        # Compute summary (same as evaluate_by_group)
        summary = {}
        if not group_df.empty:
            weights = group_df['count'].values
            summary['groups'] = int(len(group_df))
            summary['min_group_size'] = int(min_group_size)

            for h in self.horizons:
                abs_col = f'abs_error_{h}yr'
                sq_col = f'sq_error_{h}yr'
                if abs_col not in group_df.columns:
                    continue

                abs_err = group_df[abs_col].values
                sq_err = group_df[sq_col].values

                summary[f'mae_weighted_{h}yr'] = float(np.average(abs_err, weights=weights))
                summary[f'mse_weighted_{h}yr'] = float(np.average(sq_err, weights=weights))
                summary[f'rmse_weighted_{h}yr'] = float(np.sqrt(np.average(sq_err, weights=weights)))
                summary[f'mae_unweighted_{h}yr'] = float(abs_err.mean())
                summary[f'mse_unweighted_{h}yr'] = float(sq_err.mean())
                summary[f'rmse_unweighted_{h}yr'] = float(np.sqrt(sq_err.mean()))

        # Free accumulators
        self._accum.clear()

        return group_df, summary
