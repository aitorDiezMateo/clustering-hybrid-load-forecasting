from typing import Any, Mapping, Optional, Sequence

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_3d_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model: Any,
    x1: str = "radiation",
    x2: str = "temperature",
    save_plot: bool = False,
    path_output: Optional[str] = None,
    name_output: Optional[str] = None,
    title: Optional[str] = "3D Regression Plot",
    show_model_plane: bool = True,
    model_metrics: Optional[Mapping[str, Any]] = None,
    show: bool = False,
) -> None:
    """Plot 3D regression results and optionally save to disk."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if title is not None:
        ax.set_title(title)
    
    if model_metrics is not None:
        # Create metrics text string from dictionary
        metrics_text = []
        for key, value in model_metrics.items():
            if isinstance(value, float):
                metrics_text.append(f"{key}: {value:.3f}")
            else:
                metrics_text.append(f"{key}: {value}")
        
        metrics_string = "\n".join(metrics_text)
        
        # Add text box in upper left corner
        ax.text2D(0.02, 0.98, metrics_string, 
                  transform=ax.transAxes, 
                  fontsize=10,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    
    ax.scatter(X[x1], X[x2], y, c='r', marker='o', label='Actual')
    ax.scatter(X[x1], X[x2], y_pred, c='b', marker='o', label='Predicted')
    ax.set_xlabel(x1)
    ax.set_ylabel(x2)
    ax.set_zlabel('Energy Consumption')
    
    # Add legend in the top right corner
    ax.legend(loc='upper right')
    
    if show_model_plane:
        x1_range = np.linspace(X[x1].min(), X[x1].max(), 50)
        x2_range = np.linspace(X[x2].min(), X[x2].max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

        grid_points = pd.DataFrame({
            x1: x1_grid.ravel(),
            x2: x2_grid.ravel()
        })

        y_grid_pred = model.predict(grid_points)

        y_grid_pred = y_grid_pred.reshape(x1_grid.shape)

        ax.plot_surface(x1_grid, x2_grid, y_grid_pred, color='green', alpha=0.5)
    
    if save_plot and path_output is not None and name_output is not None:
        os.makedirs(path_output, exist_ok=True)
        plt.savefig(os.path.join(path_output, name_output))

    # Close figure to free memory
    plt.close(fig)

def plot_residuals(
    y: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    save_plot: bool = False,
    path_output: Optional[str] = None,
    name_output: Optional[str] = None,
    title: str = "Residuals Plot",
    show: bool = False,
) -> None:
    """Plot residuals versus predictions and optionally save to disk."""
    residuals = y - y_pred
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, residuals, c='r', marker='o')
    ax.axhline(y=0, color='black', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    
    if save_plot and path_output is not None and name_output is not None:
        os.makedirs(path_output, exist_ok=True)
        plt.savefig(os.path.join(path_output, name_output))

    # Close figure to free memory
    plt.close(fig)

def actuals_vs_predicted(
    y: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    save_plot: bool = False,
    path_output: Optional[str] = None,
    name_output: Optional[str] = None,
    title: str = "Actuals vs Predicted",
    show: bool = False,
) -> None:
    """Plot actual versus predicted values and optionally save to disk."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(y, y_pred, c='r', marker='o')
    
    # Add diagonal line where x=y (perfect prediction line)
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction (y=x)')
    
    ax.set_xlabel('Actuals')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()
    
    if save_plot and path_output is not None and name_output is not None:
        os.makedirs(path_output, exist_ok=True)
        plt.savefig(os.path.join(path_output, name_output))

    # Close figure to free memory
    plt.close(fig)

def var_vs_time(
    df: pd.DataFrame,
    var: str = "total",
    time_col: str = "Date",
    title: str = "Variable vs Time",
    color: str = 'r',
    show: bool = False,
) -> None:
    """Plot a variable over time."""
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.plot(df[time_col], df[var], c=color)
    ax.set_xlabel(time_col)
    ax.set_ylabel(var)
    ax.set_title(title)
    plt.close(fig)

def interactive_time_series(
    df: pd.DataFrame,
    vars: Sequence[str] = ("total", "temperature", "radiation"),
    time_col: str = "Date",
    title: str = "Interactive Time Series",
    show: bool = False,
    save_plot: bool = False,
    path_output: Optional[str] = None,
    output_file: str = "interactive_time_series.html",
) -> None:
    """Create an interactive time series plot and optionally save to HTML."""
    # Ensure datetime index for resampling
    df_local = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_local[time_col]):
        df_local[time_col] = pd.to_datetime(df_local[time_col])
    df_local = df_local.set_index(time_col).sort_index()
    
    fig = go.Figure()
    for var in vars:
        yaxis = "y2" if var == "radiation" else "y"
        fig.add_trace(go.Scatter(x=df_local.index, y=df_local[var], name=var, yaxis=yaxis, mode="lines"))

    fig.update_layout(
        title=dict(
            text=title,
            y=0.98,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        xaxis_title="Date",
        yaxis_title="Value",
        yaxis2=dict(
            title=dict(text="Radiation", font=dict(color="#ff7f0e")),
            tickfont=dict(color="#ff7f0e"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        hovermode="x",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        legend=dict(
            x=1.02,
            y=0.8,
            xanchor='left',
            yanchor='top',
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=80, r=90, t=90, b=40)
    )

    if save_plot and path_output is not None:
        os.makedirs(path_output, exist_ok=True)
        fig.write_html(os.path.join(path_output, output_file))



def plot_var_profiles_per_week(
    df: pd.DataFrame,
    date_col: str = "Date",
    var: str = "total",
    title: str = "Variable vs Time",
    show: bool = False,
) -> None:
    """Plot weekly profiles for a variable."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Obtener semana, día de la semana y posición en la semana (minutos desde lunes 00:00)
    df["week"] = df[date_col].dt.isocalendar().week
    df["weekday"] = df[date_col].dt.weekday  # 0=lunes, 6=domingo
    df["minutes_in_day"] = df[date_col].dt.hour * 60 + df[date_col].dt.minute
    df["minutes_since_week_start"] = df["weekday"] * 1440 + df["minutes_in_day"]
    
    fig = plt.figure(figsize=(15, 7))
    
    # Graficar cada semana superpuesta a lo largo de la semana
    for week in df["week"].unique():
        df_week = (
            df[df["week"] == week]
            .sort_values(["minutes_since_week_start"])  # ordenar por posición en la semana
        )
        plt.plot(df_week["minutes_since_week_start"], df_week[var], label=f"Week {week}")
    
    # Ticks en medianoche de cada día, etiquetados Mon..Sun
    day_start_positions = [d * 1440 for d in range(7)]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.xticks(day_start_positions, day_labels)
    
    plt.xlabel("Day of Week")
    plt.ylabel(var)
    plt.title(title)
    plt.legend()
    plt.close(fig)

def var_vs_time_agg_daily(
    df: pd.DataFrame,
    var: str = "total",
    time_col: str = "Date",
    title: str = "Variable vs Time",
    color: str = 'r',
    agg_func: str = "mean",
    show: bool = False,
) -> None:
    """Plot a daily aggregated time series."""
    
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    df = df.resample("D").agg({var: agg_func})
    df = df.reset_index()
    
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.plot(df[time_col], df[var], c=color)
    ax.set_xlabel(time_col)
    ax.set_ylabel(var)
    ax.set_title(title)
    plt.close(fig)

def correlation_scatter_plot_temp_load(
    df: pd.DataFrame,
    temp_col: str = "temperature",
    load_col: str = "total",
    time_col: str = "Date",
    title: str = "Variable vs Time",
    show: bool = False,
) -> None:
    """Plot temperature versus load with weekday coloring."""
    
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    df = df.resample("D").agg({temp_col: "mean", load_col: "sum"})
    df["weekday"] = df.index.dayofweek
    df["weekday"] = df["weekday"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    df = df.reset_index()

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)

    # Scatter por día de la semana con leyenda por grupo
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in weekday_order:
        group = df[df["weekday"] == day]
        if not group.empty:
            ax.scatter(group[temp_col], group[load_col], label=day, alpha=0.75)

    ax.legend(title="Weekday")
    ax.set_xlabel("Daily Mean Temperature")
    ax.set_ylabel("Daily Cumulative Load")
    
    # Línea de tendencia global usando todos los puntos
    x_all = df[temp_col].to_numpy()
    y_all = df[load_col].to_numpy()
    mask = (~np.isnan(x_all)) & (~np.isnan(y_all))
    if mask.any():
        slope, intercept = np.polyfit(x_all[mask], y_all[mask], 1)
        x_line = np.linspace(x_all[mask].min(), x_all[mask].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='black', linestyle='--', linewidth=2, label='Trend line')
        ax.legend(title="Weekday")

    ax.set_title(title)
    plt.close(fig)

def plot_daily_energy_profile(
    df_energy_normalized: pd.DataFrame,
    selected_date: Optional[str | pd.Timestamp] = None,
    show: bool = False,
) -> None:
    """
    Plot daily energy profile with total and cumulative sum on dual y-axes.
    
    Parameters:
    -----------
    df_energy_normalized : pandas.DataFrame
        DataFrame with energy data including 'total' and 'total_cumsum_normalized' columns
    selected_date : str or datetime, optional
        Date to plot in format 'YYYY-MM-DD' or datetime object. If None, plots the first day.
    show : bool, default=False
        Deprecated. Plot display is disabled; the figure is closed after rendering.
    """
    # If no date specified, use the first day
    if selected_date is None:
        selected_date = df_energy_normalized.index[0].date()
    else:
        # Convert string to datetime if needed
        if isinstance(selected_date, str):
            selected_date = pd.to_datetime(selected_date).date()
        elif hasattr(selected_date, 'date'):
            selected_date = selected_date.date()
    
    # Find the indices for the selected date
    date_mask = df_energy_normalized.index.date == selected_date
    date_indices = df_energy_normalized.index[date_mask]
    
    if len(date_indices) == 0:
        available_dates = df_energy_normalized.index.date
        print(f"Date {selected_date} not found in data. Available dates: {available_dates[0]} to {available_dates[-1]}")
        return
    
    # Get the start and end indices for the selected date
    start_idx = df_energy_normalized.index.get_loc(date_indices[0])
    end_idx = start_idx + len(date_indices)
    
    # Start plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot 'total' on primary y-axis
    ax1.plot(df_energy_normalized.index[start_idx:end_idx], 
             df_energy_normalized["total"].iloc[start_idx:end_idx], 
             color='tab:blue', label='Total')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Total', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot 'total_cumsum' on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df_energy_normalized.index[start_idx:end_idx], 
             df_energy_normalized["total_cumsum_normalized"].iloc[start_idx:end_idx], 
             color='tab:orange', label='Total CumSum')
    ax2.set_ylabel('Total CumSum', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.title(f'Full Day Energy Profile - {selected_date.strftime("%d-%m-%Y")}')
    plt.tight_layout()
    
    plt.close(fig)
