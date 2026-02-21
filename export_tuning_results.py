"""
Export MLflow hyperparameter tuning child runs to CSV/Excel for trend analysis.

Usage:
    python export_tuning_results.py [--experiment EXPERIMENT] [--output OUTPUT] [--parent-run PARENT_RUN_ID]

Produces:
    - tuning_trials.csv          — one row per trial with all params + final metrics
    - tuning_epochs.csv          — per-trial per-epoch metrics (for trend plots)
    - tuning_summary.txt         — best params + improvement progression
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd

MLFLOW_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "demographic_forecasts_sequence_tuning"


def get_mlflow_client(tracking_uri: str):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.MlflowClient(tracking_uri=tracking_uri)


def get_parent_runs(client, experiment_name: str):
    """Return all parent tuning runs for the experiment."""
    from mlflow.entities import ViewType
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. "
                         f"Available experiments:\n" +
                         "\n".join(e.name for e in client.search_experiments()))
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId = ''",
        run_view_type=ViewType.ALL,
        order_by=["start_time DESC"],
    )
    return runs, experiment.experiment_id


def get_child_runs(client, experiment_id: str, parent_run_id: str):
    """Return all child (trial) runs for a parent tuning run."""
    from mlflow.entities import ViewType
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        run_view_type=ViewType.ALL,
        order_by=["params.trial_number ASC"],
    )
    return runs


def extract_trial_summary(client, child_runs) -> pd.DataFrame:
    """One row per trial with params + best/final metrics."""
    rows = []
    for run in child_runs:
        row = {}
        # Run info
        row['trial_number'] = int(run.data.params.get('trial_number', -1))
        row['run_id'] = run.info.run_id
        row['status'] = run.info.status
        duration_s = None
        if run.info.end_time and run.info.start_time:
            duration_s = (run.info.end_time - run.info.start_time) / 1000.0
        row['duration_s'] = duration_s

        # Hyperparameters
        for k, v in run.data.params.items():
            if k == 'trial_number':
                continue
            try:
                row[f'param_{k}'] = float(v)
            except (ValueError, TypeError):
                row[f'param_{k}'] = v

        # Final/best metrics (take max over steps for higher-is-better, min for loss)
        metrics = run.data.metrics
        for k, v in metrics.items():
            row[f'metric_{k}'] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and 'trial_number' in df.columns:
        df = df.sort_values('trial_number').reset_index(drop=True)
    return df


def extract_epoch_metrics(client, child_runs) -> pd.DataFrame:
    """Per-trial per-epoch metrics — useful for learning curve trend plots."""
    rows = []
    EPOCH_METRICS = [
        'val_mean_auc', 'val_mean_ap', 'val_mean_f1',
        'val_loss', 'train_loss', 'es_metric_value',
    ]
    for run in child_runs:
        trial_num = int(run.data.params.get('trial_number', -1))
        loss_type = run.data.params.get('loss_type', 'unknown')
        lr = run.data.params.get('learning_rate', None)

        for metric_name in EPOCH_METRICS:
            try:
                history = client.get_metric_history(run.info.run_id, metric_name)
                for point in history:
                    rows.append({
                        'trial_number': trial_num,
                        'run_id': run.info.run_id,
                        'loss_type': loss_type,
                        'learning_rate': lr,
                        'epoch': point.step,
                        'metric': metric_name,
                        'value': point.value,
                    })
            except Exception:
                pass

    return pd.DataFrame(rows)


def print_summary(trial_df: pd.DataFrame, output_prefix: str):
    """Print human-readable summary to stdout and file."""
    lines = []
    lines.append("=" * 70)
    lines.append("HYPERPARAMETER TUNING RESULTS SUMMARY")
    lines.append("=" * 70)

    if trial_df.empty:
        lines.append("No completed trials found.")
        print("\n".join(lines))
        return

    n_trials = len(trial_df)
    n_complete = (trial_df['status'] == 'FINISHED').sum()
    lines.append(f"Total trials: {n_trials}  |  Completed: {n_complete}")

    # Best trial
    metric_col = next(
        (c for c in ['metric_es_metric_value', 'metric_val_mean_auc']
         if c in trial_df.columns),
        None
    )
    if metric_col:
        best_idx = trial_df[metric_col].idxmax()
        best = trial_df.loc[best_idx]
        lines.append(f"\nBest trial: #{int(best.get('trial_number', best_idx))}")
        lines.append(f"  {metric_col}: {best[metric_col]:.4f}")

        # Print best params
        param_cols = [c for c in trial_df.columns if c.startswith('param_')]
        lines.append("  Parameters:")
        for col in param_cols:
            val = best[col]
            name = col.replace('param_', '')
            lines.append(f"    {name}: {val}")

    # Improvement over trials (running best)
    if metric_col and 'trial_number' in trial_df.columns:
        sorted_df = trial_df.sort_values('trial_number')
        running_best = sorted_df[metric_col].expanding().max()
        lines.append("\nImprovement over trials (running best):")
        prev = None
        for i, (trial_num, val) in enumerate(zip(sorted_df['trial_number'], running_best)):
            if pd.isna(val):
                continue
            if prev is None or val > prev + 0.0001:
                lines.append(f"  Trial {int(trial_num):3d}: {val:.4f}  ← new best")
                prev = val

    # Top 5 by metric
    if metric_col:
        lines.append(f"\nTop 5 trials by {metric_col.replace('metric_', '')}:")
        top5 = trial_df.nlargest(5, metric_col)
        for _, row in top5.iterrows():
            param_str = "  ".join(
                f"{c.replace('param_', '')}={row[c]}"
                for c in ['param_loss_type', 'param_learning_rate',
                          'param_batch_size', 'param_hidden_dim']
                if c in trial_df.columns
            )
            lines.append(f"  #{int(row.get('trial_number', -1)):3d}: "
                         f"{row[metric_col]:.4f}  |  {param_str}")

    lines.append("=" * 70)
    summary = "\n".join(lines)
    print(summary)

    summary_path = f"{output_prefix}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Export MLflow tuning child runs to CSV")
    parser.add_argument('--tracking-uri', default=MLFLOW_URI,
                        help=f'MLflow tracking URI (default: {MLFLOW_URI})')
    parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT,
                        help=f'MLflow experiment name (default: {DEFAULT_EXPERIMENT})')
    parser.add_argument('--parent-run', default=None,
                        help='Specific parent run ID to export (default: most recent)')
    parser.add_argument('--output', default='tuning_export',
                        help='Output file prefix (default: tuning_export)')
    parser.add_argument('--no-epochs', action='store_true',
                        help='Skip per-epoch metrics export (faster)')
    args = parser.parse_args()

    print(f"Connecting to MLflow at {args.tracking_uri} ...")
    client = get_mlflow_client(args.tracking_uri)

    parent_runs, experiment_id = get_parent_runs(client, args.experiment)

    if not parent_runs:
        print(f"No runs found in experiment '{args.experiment}'")
        return

    if args.parent_run:
        parent_run_id = args.parent_run
        print(f"Using parent run: {parent_run_id}")
    else:
        parent_run = parent_runs[0]
        parent_run_id = parent_run.info.run_id
        run_name = parent_run.data.tags.get('mlflow.runName', parent_run_id[:8])
        print(f"Using most recent parent run: {run_name} ({parent_run_id[:8]}...)")
        print(f"  All available parent runs:")
        for r in parent_runs[:10]:
            name = r.data.tags.get('mlflow.runName', r.info.run_id[:8])
            n_trials = r.data.params.get('n_trials', '?')
            print(f"    {r.info.run_id[:8]}  {name}  n_trials={n_trials}")

    child_runs = get_child_runs(client, experiment_id, parent_run_id)
    print(f"\nFound {len(child_runs)} child runs (trials)")

    if not child_runs:
        print("No child runs found. Make sure the parent run ID is correct.")
        return

    # Extract trial summary
    print("Extracting trial summaries...")
    trial_df = extract_trial_summary(client, child_runs)
    trials_path = f"{args.output}_trials.csv"
    trial_df.to_csv(trials_path, index=False)
    print(f"Trial summary saved: {trials_path}  ({len(trial_df)} rows)")

    # Extract epoch metrics
    if not args.no_epochs:
        print("Extracting per-epoch metrics (use --no-epochs to skip)...")
        epoch_df = extract_epoch_metrics(client, child_runs)
        if not epoch_df.empty:
            epochs_path = f"{args.output}_epochs.csv"
            epoch_df.to_csv(epochs_path, index=False)
            print(f"Epoch metrics saved: {epochs_path}  "
                  f"({len(epoch_df)} rows, "
                  f"{epoch_df['trial_number'].nunique()} trials, "
                  f"{epoch_df['metric'].nunique()} metrics)")

            # Pivot for easier reading: trials × epochs for each metric
            for metric_name in epoch_df['metric'].unique():
                sub = epoch_df[epoch_df['metric'] == metric_name]
                try:
                    pivot = sub.pivot(index='trial_number', columns='epoch', values='value')
                    pivot_path = f"{args.output}_pivot_{metric_name}.csv"
                    pivot.to_csv(pivot_path)
                    print(f"  Pivot saved: {pivot_path}")
                except Exception:
                    pass
        else:
            print("  No epoch metrics found (trials may have been pruned quickly)")

    # Print and save summary
    print()
    print_summary(trial_df, args.output)

    print(f"\nAll exports written with prefix: {args.output}")


if __name__ == '__main__':
    main()
