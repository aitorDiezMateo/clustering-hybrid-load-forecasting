"""
Benchmark Models Evaluation
===========================
Evaluates baseline and statistical models: Last Day, Last Week, ARIMA, ARIMAX, Prophet, Chronos.
"""
from __future__ import annotations

import os
import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from chronos import BaseChronosPipeline
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.deterministic import DeterministicProcess
from workalendar.europe import Murcia
warnings.filterwarnings('ignore')

_CAL = Murcia()


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["weekday"] = df.index.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_holiday"] = df.index.to_series().apply(
        lambda x: int(_CAL.is_holiday(x.date() if hasattr(x, "date") else x))
    )
    return df


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
) -> dict[str, Any] | None:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]

    if len(actual_clean) == 0:
        print(f"Warning: No valid predictions for {model_name}")
        return None

    mae = mean_absolute_error(actual_clean, predicted_clean)
    mse = mean_squared_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_clean, predicted_clean)

    epsilon = 1e-10
    non_zero_mask = np.abs(actual_clean) > epsilon
    mape = (
        np.mean(np.abs((actual_clean[non_zero_mask] - predicted_clean[non_zero_mask]) / actual_clean[non_zero_mask])) * 100
        if non_zero_mask.sum() > 0 else np.nan
    )
    wmape = (
        np.sum(np.abs(actual_clean - predicted_clean)) / np.sum(np.abs(actual_clean)) * 100
        if np.sum(np.abs(actual_clean)) > 0 else np.nan
    )

    return {
        'Model': model_name, 'R2': r2, 'MAE': mae, 'RMSE': rmse,
        'MAPE': mape, 'WMAPE': wmape,
        'Valid_Samples': len(actual_clean), 'Total_Samples': len(actual)
    }


def run(config: dict) -> None:
    cfg = config.get("benchmark", {})
    USE_FILTER_VALUES = cfg.get("use_filter_values", [True])
    CHRONOS_MODEL = cfg.get("chronos_model", "amazon/chronos-bolt-mini")
    test_start_date = cfg.get("test_start_date", "2021-01-06")
    n_splits_cv = cfg.get("n_splits_cv", 2)

    root_dir = config.get("_project_root", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "data")
    output_dir = os.path.join(root_dir, "output")
    benchmark_output_dir = os.path.join(output_dir, "benchmark")
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(benchmark_output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    STEPS_PER_DAY = 96
    STEPS_PER_WEEK = 96 * 7

    print(f"Loading Chronos model: {CHRONOS_MODEL}...")
    try:
        chronos_pipeline = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu")
        print("Chronos model loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load Chronos model: {e}")
        chronos_pipeline = None

    print(f"\n{'='*80}")
    print("BENCHMARK MODELS EVALUATION")
    print(f"{'='*80}\n")

    for USE_FILTER in USE_FILTER_VALUES:

        print(f"\n{'#'*80}")
        print(f"# PROCESSING WITH USE_FILTER = {USE_FILTER}")
        print(f"{'#'*80}\n")

        try:
            print("Loading energy consumption data...")
            df_energy = pd.read_csv(
                os.path.join(data_dir, "data.csv"),
                parse_dates=["Date"], usecols=["Date", "total", "total_filter", "temperature", "radiation"]
            )
            df_energy.set_index("Date", inplace=True)

            if df_energy.index.tz is not None:
                df_energy.index = df_energy.index.tz_localize(None)
            df_energy = df_energy.sort_index()

            if USE_FILTER:
                df_energy["total"] = df_energy["total_filter"]
            df_energy.drop(columns=["total_filter"], inplace=True, errors="ignore")

            print(f"Data loaded: {len(df_energy)} observations")
            print(f"Date range: {df_energy.index.min()} to {df_energy.index.max()}\n")

            df_train = df_energy[df_energy.index < test_start_date].copy()
            df_test = df_energy[df_energy.index >= test_start_date].copy()

            print(f"Train set: {len(df_train)} observations")
            print(f"Test set: {len(df_test)} observations\n")

            actual_test = df_test['total'].values

            # ---- BENCHMARK 1: Last Day ----
            print(f"{'='*60}\nBENCHMARK 1: Last Day (Lag 96)\n{'='*60}")
            df_full = pd.concat([df_train, df_test])
            df_test['pred_last_day'] = df_full['total'].shift(STEPS_PER_DAY).loc[df_test.index]
            pred_last_day = df_test['pred_last_day'].values
            metrics_prev_day = calculate_metrics(actual_test, pred_last_day, 'Last Day')
            if metrics_prev_day:
                print(f"R2 = {metrics_prev_day['R2']:.4f}, MAE = {metrics_prev_day['MAE']:.2f}, RMSE = {metrics_prev_day['RMSE']:.2f}, WMAPE = {metrics_prev_day['WMAPE']:.2f}%\n")

            # ---- BENCHMARK 2: Last Week ----
            print(f"{'='*60}\nBENCHMARK 2: Last Week (Lag 672)\n{'='*60}")
            df_test['pred_last_week'] = df_full['total'].shift(STEPS_PER_WEEK).loc[df_test.index]
            pred_last_week = df_test['pred_last_week'].values
            metrics_same_weekday = calculate_metrics(actual_test, pred_last_week, 'Last Week')
            if metrics_same_weekday:
                print(f"R2 = {metrics_same_weekday['R2']:.4f}, MAE = {metrics_same_weekday['MAE']:.2f}, RMSE = {metrics_same_weekday['RMSE']:.2f}, WMAPE = {metrics_same_weekday['WMAPE']:.2f}%\n")

            # ---- BENCHMARK 3: ARIMA + Fourier ----
            print(f"{'='*60}\nBENCHMARK 3: ARIMA + Fourier\n{'='*60}")
            try:
                df_total = pd.concat([df_train, df_test])
                dp = DeterministicProcess(index=df_total.index, period=STEPS_PER_WEEK, fourier=10, drop=True)
                X_fourier = dp.in_sample()
                X_fourier_train = X_fourier.loc[df_train.index]
                X_fourier_test = X_fourier.loc[df_test.index]

                history = df_train['total'].tolist()
                history_exog = X_fourier_train.values.tolist()
                predictions_fourier = []

                unique_days = df_test.index.normalize().unique()
                print(f"Rolling forecast for {len(unique_days)} days...")

                for day in unique_days:
                    daily_mask = (df_test.index.normalize() == day)
                    day_steps = int(daily_mask.sum())
                    if day_steps == 0:
                        continue
                    exog_today = X_fourier_test.loc[daily_mask]
                    try:
                        model = ARIMA(history, exog=history_exog, order=(2, 0, 1))
                        model_fit = model.fit()
                        daily_forecast = model_fit.forecast(steps=day_steps, exog=exog_today)
                        predictions_fourier.extend(daily_forecast)
                    except Exception as e:
                        print(f"  Warning: ARIMA+Fourier failed for {day.date()}: {e}")
                        predictions_fourier.extend([np.nan] * day_steps)
                    history.extend(df_test.loc[daily_mask, 'total'].tolist())
                    history_exog.extend(exog_today.values.tolist())

                df_test['pred_arima_fourier'] = predictions_fourier[:len(df_test)]
                metrics_arima_fourier = calculate_metrics(actual_test, df_test['pred_arima_fourier'].values, 'ARIMA + Fourier')
                if metrics_arima_fourier:
                    print(f"R2 = {metrics_arima_fourier['R2']:.4f}, MAE = {metrics_arima_fourier['MAE']:.2f}, RMSE = {metrics_arima_fourier['RMSE']:.2f}, WMAPE = {metrics_arima_fourier['WMAPE']:.2f}%\n")
            except Exception as e:
                print(f"ARIMA + Fourier failed: {e}")
                df_test['pred_arima_fourier'] = np.nan
                metrics_arima_fourier = {'Model': 'ARIMA + Fourier', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'WMAPE': np.nan, 'Valid_Samples': 0, 'Total_Samples': len(actual_test)}

            # ---- BENCHMARK 4: ARIMAX ----
            print(f"{'='*60}\nBENCHMARK 4: ARIMAX + Weather + Calendar\n{'='*60}")
            try:
                df_train_cal = _add_calendar_features(df_train.copy())
                df_test_cal = _add_calendar_features(df_test.copy())

                df_total_full = pd.concat([df_train_cal, df_test_cal])
                dp_full = DeterministicProcess(index=df_total_full.index, period=STEPS_PER_DAY, fourier=5, drop=True)
                X_fourier_full = dp_full.in_sample()
                X_fourier_train_full = X_fourier_full.loc[df_train.index]
                X_fourier_test_full = X_fourier_full.loc[df_test.index]

                weather_cols = ["temperature", "radiation"]
                scaler_weather = StandardScaler()
                weather_train_scaled = pd.DataFrame(scaler_weather.fit_transform(df_train_cal[weather_cols]), index=df_train.index, columns=weather_cols)
                weather_test_scaled = pd.DataFrame(scaler_weather.transform(df_test_cal[weather_cols]), index=df_test.index, columns=weather_cols)

                cal_cols = ["weekday", "is_weekend", "is_holiday"]
                scaler_calendar = StandardScaler()
                calendar_train_scaled = pd.DataFrame(scaler_calendar.fit_transform(df_train_cal[cal_cols]), index=df_train.index, columns=cal_cols)
                calendar_test_scaled = pd.DataFrame(scaler_calendar.transform(df_test_cal[cal_cols]), index=df_test.index, columns=cal_cols)

                X_train_combined = pd.concat([X_fourier_train_full, weather_train_scaled, calendar_train_scaled], axis=1)
                X_test_combined = pd.concat([X_fourier_test_full, weather_test_scaled, calendar_test_scaled], axis=1)

                history_arimax = df_train['total'].tolist()
                history_exog_arimax = X_train_combined.values.tolist()
                predictions_arimax = []

                unique_days_arimax = df_test.index.normalize().unique()
                print(f"Rolling forecast for {len(unique_days_arimax)} days...")

                for day in unique_days_arimax:
                    daily_mask = (df_test.index.normalize() == day)
                    day_steps = int(daily_mask.sum())
                    if day_steps == 0:
                        continue
                    exog_today = X_test_combined.loc[daily_mask]
                    try:
                        model = ARIMA(history_arimax, exog=history_exog_arimax, order=(2, 0, 1))
                        model_fit = model.fit()
                        daily_forecast = model_fit.forecast(steps=day_steps, exog=exog_today)
                        predictions_arimax.extend(daily_forecast)
                    except Exception as e:
                        print(f"  Warning: ARIMAX failed for {day.date()}: {e}")
                        predictions_arimax.extend([np.nan] * day_steps)
                    history_arimax.extend(df_test.loc[daily_mask, 'total'].tolist())
                    history_exog_arimax.extend(exog_today.values.tolist())

                df_test['pred_arimax'] = predictions_arimax[:len(df_test)]
                metrics_arimax = calculate_metrics(actual_test, df_test['pred_arimax'].values, 'ARIMAX Full')
                if metrics_arimax:
                    print(f"R2 = {metrics_arimax['R2']:.4f}, MAE = {metrics_arimax['MAE']:.2f}, RMSE = {metrics_arimax['RMSE']:.2f}, WMAPE = {metrics_arimax['WMAPE']:.2f}%\n")
            except Exception as e:
                print(f"ARIMAX failed: {e}")
                df_test['pred_arimax'] = np.nan
                metrics_arimax = {'Model': 'ARIMAX Full', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'WMAPE': np.nan, 'Valid_Samples': 0, 'Total_Samples': len(actual_test)}

            # ---- BENCHMARK 5: Prophet ----
            print(f"{'='*60}\nBENCHMARK 5: Prophet\n{'='*60}")
            try:
                df_train_prophet = _add_calendar_features(df_train.copy())
                df_test_prophet = _add_calendar_features(df_test.copy())

                regressor_cols = ["temperature", "radiation", "weekday", "is_weekend", "is_holiday"]
                scaler_prophet = StandardScaler()
                train_scaled = pd.DataFrame(scaler_prophet.fit_transform(df_train_prophet[regressor_cols]), index=df_train_prophet.index, columns=regressor_cols)
                test_scaled = pd.DataFrame(scaler_prophet.transform(df_test_prophet[regressor_cols]), index=df_test_prophet.index, columns=regressor_cols)

                history_df = pd.DataFrame({"ds": df_train_prophet.index, "y": df_train_prophet["total"].values}).set_index("ds")
                history_df = history_df.join(train_scaled).reset_index()

                predictions_prophet = []
                unique_days_prophet = df_test_prophet.index.normalize().unique()
                print(f"Rolling forecast for {len(unique_days_prophet)} days...")

                for day in unique_days_prophet:
                    daily_mask = (df_test_prophet.index.normalize() == day)
                    day_steps = int(daily_mask.sum())
                    if day_steps == 0:
                        continue
                    model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=False)
                    for col in regressor_cols:
                        model.add_regressor(col)
                    model.fit(history_df)
                    future_day = pd.DataFrame({"ds": df_test_prophet.index[daily_mask]}).set_index("ds")
                    future_day = future_day.join(test_scaled.loc[daily_mask]).reset_index()
                    forecast = model.predict(future_day)
                    predictions_prophet.extend(forecast["yhat"].values)
                    actuals_day = df_test_prophet.loc[daily_mask, "total"].values
                    history_update = pd.DataFrame({"ds": df_test_prophet.index[daily_mask], "y": actuals_day}).set_index("ds")
                    history_update = history_update.join(test_scaled.loc[daily_mask])
                    history_df = pd.concat([history_df, history_update.reset_index()], axis=0, ignore_index=True)

                df_test['pred_prophet'] = predictions_prophet[:len(df_test)]
                metrics_prophet = calculate_metrics(actual_test, df_test['pred_prophet'].values, 'Prophet')
                if metrics_prophet:
                    print(f"R2 = {metrics_prophet['R2']:.4f}, MAE = {metrics_prophet['MAE']:.2f}, RMSE = {metrics_prophet['RMSE']:.2f}, WMAPE = {metrics_prophet['WMAPE']:.2f}%\n")
            except Exception as e:
                print(f"Prophet failed: {e}")
                df_test['pred_prophet'] = np.nan
                metrics_prophet = {'Model': 'Prophet', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'WMAPE': np.nan, 'Valid_Samples': 0, 'Total_Samples': len(actual_test)}

            # ---- BENCHMARK 6: Chronos ----
            print(f"{'='*60}\nBENCHMARK 6: Chronos (Zero-Shot Foundation Model)\n{'='*60}")
            if chronos_pipeline is None:
                print("Chronos model not available, skipping...")
                metrics_chronos = {'Model': 'Chronos', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'WMAPE': np.nan, 'Valid_Samples': 0, 'Total_Samples': len(actual_test)}
            else:
                try:
                    context = torch.tensor(df_train['total'].values, dtype=torch.float32)
                    n_test_steps = len(df_test)
                    print(f"Predicting {n_test_steps} steps ahead...")
                    forecast = chronos_pipeline.predict(context, prediction_length=n_test_steps)
                    predictions_chronos = forecast[0, 0, :].numpy()
                    if len(predictions_chronos) < n_test_steps:
                        predictions_chronos = np.pad(predictions_chronos, (0, n_test_steps - len(predictions_chronos)), mode='edge')
                    df_test['pred_chronos'] = predictions_chronos
                    metrics_chronos = calculate_metrics(actual_test, df_test['pred_chronos'].values, 'Chronos')
                    if metrics_chronos:
                        print(f"R2 = {metrics_chronos['R2']:.4f}, MAE = {metrics_chronos['MAE']:.2f}, RMSE = {metrics_chronos['RMSE']:.2f}, WMAPE = {metrics_chronos['WMAPE']:.2f}%\n")
                except Exception as e:
                    print(f"Chronos failed: {e}")
                    traceback.print_exc()
                    df_test['pred_chronos'] = np.nan
                    metrics_chronos = {'Model': 'Chronos', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'WMAPE': np.nan, 'Valid_Samples': 0, 'Total_Samples': len(actual_test)}

            # ---- Summary ----
            print(f"{'='*60}\nSUMMARY\n{'='*60}")
            all_metrics = [m for m in [metrics_prev_day, metrics_same_weekday, metrics_arima_fourier, metrics_arimax, metrics_prophet, metrics_chronos] if m]
            results_df = pd.DataFrame(all_metrics).sort_values('RMSE', ascending=True).reset_index(drop=True)
            print(results_df.to_string(index=False))
            print()

            filter_suffix = "_filter" if USE_FILTER else "_no_filter"
            output_path = os.path.join(output_dir, f"benchmark_results{filter_suffix}.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

            predictions_path = os.path.join(output_dir, f"benchmark_predictions{filter_suffix}.csv")
            df_test_output = df_test[['total', 'pred_last_day', 'pred_last_week', 'pred_arima_fourier', 'pred_arimax', 'pred_prophet', 'pred_chronos']].copy()
            df_test_output.columns = ['Actual', 'Last_Day', 'Last_Week', 'ARIMA_Fourier', 'ARIMAX_Full', 'Prophet', 'Chronos']
            if df_test_output.index.tz is not None:
                df_test_output.index = df_test_output.index.tz_localize(None)
            df_test_output.to_csv(predictions_path)
            print(f"Predictions saved to: {predictions_path}\n")

            # ---- TimeSeriesSplit CV for all benchmarks ----
            print(f"\n{'='*60}")
            print(f"TIME SERIES CV BENCHMARKS ({n_splits_cv} splits)")
            print(f"{'='*60}")

            tscv = TimeSeriesSplit(n_splits=n_splits_cv)
            cv_bench_metrics: list[dict] = []

            for cv_idx, (cv_tr_idx, cv_te_idx) in enumerate(tscv.split(df_energy), 1):
                df_cv_tr = df_energy.iloc[cv_tr_idx].copy()
                df_cv_te = df_energy.iloc[cv_te_idx].copy()
                cv_actual = df_cv_te['total'].values

                print(f"\n  Split {cv_idx}/{n_splits_cv}: "
                      f"train={len(df_cv_tr)} obs, test={len(df_cv_te)} obs "
                      f"({df_cv_te.index.min().date()} - {df_cv_te.index.max().date()})")

                df_cv_full = pd.concat([df_cv_tr, df_cv_te])

                # Last Day
                cv_pred_ld = df_cv_full['total'].shift(STEPS_PER_DAY).loc[df_cv_te.index].values
                m_ld = calculate_metrics(cv_actual, cv_pred_ld, 'Last Day')
                if m_ld:
                    cv_bench_metrics.append({'Split': cv_idx, 'Model': 'Last Day',
                        'R2': m_ld['R2'], 'MAE': m_ld['MAE'], 'RMSE': m_ld['RMSE'], 'WMAPE': m_ld['WMAPE']})

                # Last Week
                cv_pred_lw = df_cv_full['total'].shift(STEPS_PER_WEEK).loc[df_cv_te.index].values
                m_lw = calculate_metrics(cv_actual, cv_pred_lw, 'Last Week')
                if m_lw:
                    cv_bench_metrics.append({'Split': cv_idx, 'Model': 'Last Week',
                        'R2': m_lw['R2'], 'MAE': m_lw['MAE'], 'RMSE': m_lw['RMSE'], 'WMAPE': m_lw['WMAPE']})

                # ARIMA + Fourier
                try:
                    dp_cv = DeterministicProcess(index=df_cv_full.index, period=STEPS_PER_WEEK, fourier=10, drop=True)
                    X_f_cv = dp_cv.in_sample()
                    hist_af = df_cv_tr['total'].tolist()
                    hist_exog_af = X_f_cv.loc[df_cv_tr.index].values.tolist()
                    preds_af: list[float] = []
                    for day in df_cv_te.index.normalize().unique():
                        mask = df_cv_te.index.normalize() == day
                        n_steps = int(mask.sum())
                        exog_d = X_f_cv.loc[df_cv_te.index[mask]]
                        try:
                            fit = ARIMA(hist_af, exog=hist_exog_af, order=(2, 0, 1)).fit()
                            preds_af.extend(fit.forecast(steps=n_steps, exog=exog_d))
                        except Exception:
                            preds_af.extend([np.nan] * n_steps)
                        hist_af.extend(df_cv_te.loc[mask, 'total'].tolist())
                        hist_exog_af.extend(exog_d.values.tolist())
                    m_af = calculate_metrics(cv_actual, np.array(preds_af[:len(df_cv_te)]), 'ARIMA + Fourier')
                    if m_af:
                        cv_bench_metrics.append({'Split': cv_idx, 'Model': 'ARIMA + Fourier',
                            'R2': m_af['R2'], 'MAE': m_af['MAE'], 'RMSE': m_af['RMSE'], 'WMAPE': m_af['WMAPE']})
                except Exception as e:
                    print(f"    ARIMA+Fourier split {cv_idx} failed: {e}")

                # ARIMAX
                try:
                    df_cv_tr_cal = _add_calendar_features(df_cv_tr.copy())
                    df_cv_te_cal = _add_calendar_features(df_cv_te.copy())
                    df_cv_full_cal = pd.concat([df_cv_tr_cal, df_cv_te_cal])
                    dp_ax = DeterministicProcess(index=df_cv_full_cal.index, period=STEPS_PER_DAY, fourier=5, drop=True)
                    X_f_ax = dp_ax.in_sample()
                    weather_cols_cv = ["temperature", "radiation"]
                    cal_cols_cv = ["weekday", "is_weekend", "is_holiday"]
                    sc_w = StandardScaler()
                    w_tr = pd.DataFrame(sc_w.fit_transform(df_cv_tr_cal[weather_cols_cv]), index=df_cv_tr.index, columns=weather_cols_cv)
                    w_te = pd.DataFrame(sc_w.transform(df_cv_te_cal[weather_cols_cv]), index=df_cv_te.index, columns=weather_cols_cv)
                    sc_c = StandardScaler()
                    c_tr = pd.DataFrame(sc_c.fit_transform(df_cv_tr_cal[cal_cols_cv]), index=df_cv_tr.index, columns=cal_cols_cv)
                    c_te = pd.DataFrame(sc_c.transform(df_cv_te_cal[cal_cols_cv]), index=df_cv_te.index, columns=cal_cols_cv)
                    X_tr_ax = pd.concat([X_f_ax.loc[df_cv_tr.index], w_tr, c_tr], axis=1)
                    X_te_ax = pd.concat([X_f_ax.loc[df_cv_te.index], w_te, c_te], axis=1)
                    hist_ax = df_cv_tr['total'].tolist()
                    hist_exog_ax = X_tr_ax.values.tolist()
                    preds_ax: list[float] = []
                    for day in df_cv_te.index.normalize().unique():
                        mask = df_cv_te.index.normalize() == day
                        n_steps = int(mask.sum())
                        exog_d = X_te_ax.loc[df_cv_te.index[mask]]
                        try:
                            fit = ARIMA(hist_ax, exog=hist_exog_ax, order=(2, 0, 1)).fit()
                            preds_ax.extend(fit.forecast(steps=n_steps, exog=exog_d))
                        except Exception:
                            preds_ax.extend([np.nan] * n_steps)
                        hist_ax.extend(df_cv_te.loc[mask, 'total'].tolist())
                        hist_exog_ax.extend(exog_d.values.tolist())
                    m_ax = calculate_metrics(cv_actual, np.array(preds_ax[:len(df_cv_te)]), 'ARIMAX Full')
                    if m_ax:
                        cv_bench_metrics.append({'Split': cv_idx, 'Model': 'ARIMAX Full',
                            'R2': m_ax['R2'], 'MAE': m_ax['MAE'], 'RMSE': m_ax['RMSE'], 'WMAPE': m_ax['WMAPE']})
                except Exception as e:
                    print(f"    ARIMAX split {cv_idx} failed: {e}")

                # Prophet
                try:
                    df_cv_tr_pr = _add_calendar_features(df_cv_tr.copy())
                    df_cv_te_pr = _add_calendar_features(df_cv_te.copy())
                    reg_cols_cv = ["temperature", "radiation", "weekday", "is_weekend", "is_holiday"]
                    sc_pr = StandardScaler()
                    tr_s = pd.DataFrame(sc_pr.fit_transform(df_cv_tr_pr[reg_cols_cv]), index=df_cv_tr_pr.index, columns=reg_cols_cv)
                    te_s = pd.DataFrame(sc_pr.transform(df_cv_te_pr[reg_cols_cv]), index=df_cv_te_pr.index, columns=reg_cols_cv)
                    hist_pr = (pd.DataFrame({"ds": df_cv_tr_pr.index, "y": df_cv_tr_pr["total"].values})
                               .set_index("ds").join(tr_s).reset_index())
                    preds_pr: list[float] = []
                    for day in df_cv_te_pr.index.normalize().unique():
                        mask = df_cv_te_pr.index.normalize() == day
                        m_pr_model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=False)
                        for col in reg_cols_cv:
                            m_pr_model.add_regressor(col)
                        m_pr_model.fit(hist_pr)
                        future_d = (pd.DataFrame({"ds": df_cv_te_pr.index[mask]})
                                    .set_index("ds").join(te_s.loc[mask]).reset_index())
                        preds_pr.extend(m_pr_model.predict(future_d)["yhat"].values)
                        upd = (pd.DataFrame({"ds": df_cv_te_pr.index[mask], "y": df_cv_te_pr.loc[mask, "total"].values})
                               .set_index("ds").join(te_s.loc[mask]))
                        hist_pr = pd.concat([hist_pr, upd.reset_index()], ignore_index=True)
                    m_pr = calculate_metrics(cv_actual, np.array(preds_pr[:len(df_cv_te)]), 'Prophet')
                    if m_pr:
                        cv_bench_metrics.append({'Split': cv_idx, 'Model': 'Prophet',
                            'R2': m_pr['R2'], 'MAE': m_pr['MAE'], 'RMSE': m_pr['RMSE'], 'WMAPE': m_pr['WMAPE']})
                except Exception as e:
                    print(f"    Prophet split {cv_idx} failed: {e}")

                # Chronos
                if chronos_pipeline is not None:
                    try:
                        ctx = torch.tensor(df_cv_tr['total'].values, dtype=torch.float32)
                        fc = chronos_pipeline.predict(ctx, prediction_length=len(df_cv_te))
                        preds_ch = fc[0, 0, :len(df_cv_te)].numpy()
                        if len(preds_ch) < len(df_cv_te):
                            preds_ch = np.pad(preds_ch, (0, len(df_cv_te) - len(preds_ch)), mode='edge')
                        m_ch = calculate_metrics(cv_actual, preds_ch, 'Chronos')
                        if m_ch:
                            cv_bench_metrics.append({'Split': cv_idx, 'Model': 'Chronos',
                                'R2': m_ch['R2'], 'MAE': m_ch['MAE'], 'RMSE': m_ch['RMSE'], 'WMAPE': m_ch['WMAPE']})
                    except Exception as e:
                        print(f"    Chronos split {cv_idx} failed: {e}")

            if cv_bench_metrics:
                cv_df = pd.DataFrame(cv_bench_metrics)
                cv_agg = cv_df.groupby('Model').agg(
                    R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
                    MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
                    RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
                    WMAPE_mean=('WMAPE', 'mean'), WMAPE_std=('WMAPE', 'std'),
                ).reset_index().sort_values('WMAPE_mean')

                print(f"\n{'='*60}\nCV SUMMARY ({n_splits_cv} splits averaged)\n{'='*60}")
                print(cv_agg.to_string(index=False))

                cv_avg_path = os.path.join(benchmark_output_dir, f"benchmark_cv_averaged{filter_suffix}.csv")
                cv_det_path = os.path.join(benchmark_output_dir, f"benchmark_cv_detailed{filter_suffix}.csv")
                cv_agg.to_csv(cv_avg_path, index=False)
                cv_df.to_csv(cv_det_path, index=False)
                print(f"CV results saved to: {cv_avg_path}")

            print("Generating visualization...")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df_energy.index, df_energy['total'], color='black', linewidth=1.5, label='Actual', alpha=0.8)
            ax.axvline(x=pd.Timestamp(test_start_date), color='red', linestyle='--', linewidth=1.5, label='Test Start')
            ax.plot(df_test.index, df_test['pred_last_day'], color='blue', linewidth=1, linestyle=':', label='Last Day', alpha=0.7)
            ax.plot(df_test.index, df_test['pred_arimax'], color='red', linewidth=1.5, linestyle='--', label='ARIMAX', alpha=0.7)
            ax.plot(df_test.index, df_test['pred_prophet'], color='brown', linewidth=1, linestyle=':', label='Prophet', alpha=0.7)
            ax.plot(df_test.index, df_test['pred_chronos'], color='purple', linewidth=1, label='Chronos', alpha=0.7)

            hybrid_pred_path = os.path.join(
                output_dir, "full_predictions_with_ma",
                "filter_True_k_3_config_1", "predictions.csv"
            )
            if os.path.exists(hybrid_pred_path):
                df_hybrid = pd.read_csv(hybrid_pred_path, parse_dates=['datetime'])
                df_hybrid.set_index('datetime', inplace=True)
                ax.plot(df_hybrid.index, df_hybrid['predicted'], color='green', linewidth=1.5, label='Hybrid Model', alpha=0.8)

            ax.set_xlabel('Date')
            ax.set_ylabel('Energy Consumption (kW)')
            ax.set_title('Model Comparison - Energy Consumption')
            ax.legend(loc='upper left', fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            plt.tight_layout()
            viz_path = os.path.join(benchmark_output_dir, "benchmark_time_series.png")
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
            fig.savefig(os.path.join(figures_dir, "models_comparison.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Visualization saved to: {viz_path}")

        except FileNotFoundError as e:
            print(f"\n{'!'*80}\nSKIPPING USE_FILTER={USE_FILTER}\nFile not found: {e}\n{'!'*80}\n")
        except Exception as e:
            print(f"\n{'!'*80}\nERROR WITH USE_FILTER={USE_FILTER}\nError: {e}\n{'!'*80}\n")
            traceback.print_exc()

    print("\n--- Benchmark Evaluation Complete ---")


_STANDALONE_CONFIG = {
    "benchmark": {
        "use_filter_values": [True],
        "chronos_model": "amazon/chronos-bolt-mini",
        "test_start_date": "2021-01-06",
        "n_splits_cv": 2,
    }
}

if __name__ == "__main__":
    run(_STANDALONE_CONFIG)
