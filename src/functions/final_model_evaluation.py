"""Final model evaluation utilities and plots."""
from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def align_actual_predicted(
    df_energy: pd.DataFrame,
    df_predictions: pd.DataFrame,
    use_filter: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pred = df_predictions.copy()
    pred['datetime_parsed'] = pd.to_datetime(pred['datetime'], errors='coerce')
    pred = pred.dropna(subset=['datetime_parsed']).copy()
    pred['dt_str'] = pred['datetime_parsed'].dt.strftime('%Y-%m-%d %H:%M:%S')

    if use_filter and 'total_filter' in df_energy.columns:
        actual = df_energy[['total_filter']].copy()
    else:
        actual = df_energy[['total']].copy()

    if not isinstance(actual.index, pd.DatetimeIndex):
        actual.index = pd.to_datetime(actual.index, errors='coerce')
    actual = actual.dropna().copy()
    actual['dt_str'] = actual.index.strftime('%Y-%m-%d %H:%M:%S')

    merged = actual.merge(pred[['dt_str', 'consumption']], on='dt_str', how='inner')

    actual_values = merged[actual.columns[0]].values if not merged.empty else np.array([])
    predicted_values = merged['consumption'].values if not merged.empty else np.array([])

    return actual_values, predicted_values


def calculate_regression_metrics(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict[str, Any]:
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)

    epsilon = 1e-10
    non_zero_mask = np.abs(actual_values) > epsilon
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((actual_values[non_zero_mask] - predicted_values[non_zero_mask]) / actual_values[non_zero_mask])) * 100
    else:
        mape = np.nan

    mean_actual = np.mean(actual_values)
    cv_rmse = (rmse / mean_actual) * 100 if mean_actual else np.nan

    residuals = actual_values - predicted_values

    metrics = {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape,
        'cv_rmse': cv_rmse,
        'mean_residual': np.mean(residuals), 'std_residual': np.std(residuals),
        'max_error': np.max(np.abs(residuals)),
        'actual_mean': np.mean(actual_values), 'actual_std': np.std(actual_values),
        'predicted_mean': np.mean(predicted_values), 'predicted_std': np.std(predicted_values),
        'residuals': residuals,
    }
    return metrics


def print_regression_metrics(metrics: dict[str, Any]) -> None:
    print("\n" + "="*60)
    print("REGRESSION MODEL EVALUATION METRICS")
    print("="*60)
    print(f"  MAE:        {metrics['mae']:.2f} kW")
    print(f"  RMSE:       {metrics['rmse']:.2f} kW")
    print(f"  R2:         {metrics['r2']:.4f}")
    if not np.isnan(metrics['mape']):
        print(f"  MAPE:       {metrics['mape']:.2f}%")
    print(f"  CV(RMSE):   {metrics['cv_rmse']:.2f}%")
    print("="*60 + "\n")


def create_evaluation_plots(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    metrics: dict[str, Any],
    output_path: str,
) -> str:
    residuals = metrics['residuals']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Actual vs Predicted scatter
    ax = axes[0, 0]
    ax.scatter(actual_values, predicted_values, s=8, alpha=0.5, color='steelblue')
    lims = [min(actual_values.min(), predicted_values.min()),
            max(actual_values.max(), predicted_values.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
    ax.set_xlabel('Actual (kW)')
    ax.set_ylabel('Predicted (kW)')
    ax.set_title('Actual vs Predicted')
    ax.legend(fontsize=8)

    # 2. Residuals histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=30, color='seagreen', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Residual (kW)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residuals Distribution')

    # 3. Residuals over time
    ax = axes[1, 0]
    ax.scatter(range(len(residuals)), residuals, s=4, alpha=0.5, color='darkorange')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Residual (kW)')
    ax.set_title('Residuals over Time')

    # 4. Metrics bar chart
    ax = axes[1, 1]
    names = ['MAE', 'RMSE', 'R2']
    values = [metrics['mae'], metrics['rmse'], metrics['r2']]
    colors = ['steelblue', 'seagreen', 'mediumpurple']
    bars = ax.bar(names, values, color=colors, alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('Error Metrics')

    fig.suptitle('Model Evaluation Diagnostics', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_comparison_plot(
    df_energy: pd.DataFrame,
    df_predictions: pd.DataFrame,
    output_path: str,
    use_filter: bool = False,
) -> str:
    actual_series = df_energy[['total']].copy()
    if use_filter and 'total_filter' in df_energy.columns:
        filtered_series = df_energy[['total_filter']].copy()
    else:
        filtered_series = None

    predicted_series = df_predictions.copy()
    if 'consumption' in predicted_series.columns:
        predicted_series = predicted_series[['datetime', 'consumption']]
        predicted_series.columns = ['datetime', 'predicted']
    elif 'predicted' in predicted_series.columns:
        predicted_series = predicted_series[['datetime', 'predicted']]
    predicted_series.set_index('datetime', inplace=True)

    if actual_series.index.tz is not None:
        actual_series.index = actual_series.index.tz_localize(None)
    if filtered_series is not None and filtered_series.index.tz is not None:
        filtered_series.index = filtered_series.index.tz_localize(None)
    if predicted_series.index.tz is not None:
        predicted_series.index = predicted_series.index.tz_localize(None)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual_series.index, actual_series['total'], color='steelblue', linewidth=1, label='Actual', alpha=0.8)
    if filtered_series is not None:
        ax.plot(filtered_series.index, filtered_series['total_filter'], color='seagreen', linewidth=1, label='Filtered', alpha=0.7)
    ax.plot(predicted_series.index, predicted_series['predicted'], color='red', linewidth=1, label='Predicted', alpha=0.8)
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Load (kW)')
    ax.set_title('Actual vs Predicted Load')
    ax.legend(loc='upper left', fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def evaluate_model(
    df_energy: pd.DataFrame,
    df_predictions: pd.DataFrame,
    output_dir: str,
    k_selected: int,
    use_filter: bool = False,
    print_metrics: bool = True,
) -> dict[str, Any]:
    actual_values, predicted_values = align_actual_predicted(df_energy, df_predictions, use_filter=use_filter)

    if len(actual_values) == 0 or len(predicted_values) == 0:
        if print_metrics:
            print("WARNING: No overlapping timestamps found between actual data and predictions.")
        return {}

    metrics = calculate_regression_metrics(actual_values, predicted_values)

    if print_metrics:
        print_regression_metrics(metrics)

    import os
    eval_plot_path = os.path.join(output_dir, f"model_evaluation_metrics_k_{k_selected}.png")
    create_evaluation_plots(actual_values, predicted_values, metrics, eval_plot_path)
    if print_metrics:
        print(f"Evaluation metrics plot saved to: {eval_plot_path}")

    comparison_plot_path = os.path.join(output_dir, f"load_predictions_comparison_k_{k_selected}.png")
    create_comparison_plot(df_energy, df_predictions, comparison_plot_path, use_filter=use_filter)
    if print_metrics:
        print(f"Comparison plot saved to: {comparison_plot_path}")

    return metrics
