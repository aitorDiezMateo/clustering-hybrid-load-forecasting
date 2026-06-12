from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from workalendar.europe.spain import Murcia

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from functions.regression_optimization import (
    optimize_all_regressors,
    evaluate_optimized_regressors,
    print_regression_results_table,
    print_regression_evaluation_table
)


def run(config: dict) -> None:
    cfg = config.get("max_load_regressor", {})
    USE_MA_FEATURES = cfg.get("use_ma_features", True)
    USE_FILTER_VALUES = cfg.get("use_filter_values", [True])
    n_trials = cfg.get("n_trials", 5)
    cv_folds = cfg.get("cv_folds", 2)
    exclude_models = cfg.get("exclude_models", ["CatBoost"])

    root_dir = config.get("_project_root", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "data")
    clustered_dir = os.path.join(root_dir, "output", "clustered_data")
    output_dir = os.path.join(
        root_dir, "output",
        "regression_model_with_ma" if USE_MA_FEATURES else "regression_model"
    )
    optimization_results_dir = os.path.join(output_dir, "optimization_results")
    evaluation_results_dir = os.path.join(output_dir, "evaluation_results")
    os.makedirs(clustered_dir, exist_ok=True)
    os.makedirs(optimization_results_dir, exist_ok=True)
    os.makedirs(evaluation_results_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"USE MA FEATURES: {USE_MA_FEATURES}")
    print(f"TOTAL CONFIGURATIONS TO PROCESS: {len(USE_FILTER_VALUES)}")
    print(f"{'='*80}\n")

    all_configurations_summary = []

    for config_idx, USE_FILTER in enumerate(USE_FILTER_VALUES, 1):

        print(f"\n{'#'*80}")
        print(f"# PROCESSING CONFIGURATION {config_idx}/{len(USE_FILTER_VALUES)}")
        print(f"# USE_FILTER = {USE_FILTER}")
        print(f"{'#'*80}\n")

        try:
            df_energy = pd.read_csv(
                os.path.join(data_dir, "data.csv"),
                parse_dates=["Date"],
                usecols=["Date", "total", "temperature", "radiation", "total_filter"]
            )
            df_energy.set_index("Date", inplace=True)

            if USE_FILTER:
                df_energy["total"] = df_energy["total_filter"]

            if USE_MA_FEATURES:
                print("Creating moving average climate features...")
                df_energy['temp_ma_24h'] = df_energy['temperature'].rolling(window=96, min_periods=1, center=False).mean()
                df_energy['temp_ma_48h'] = df_energy['temperature'].rolling(window=192, min_periods=1, center=False).mean()
                df_energy['temp_ma_72h'] = df_energy['temperature'].rolling(window=288, min_periods=1, center=False).mean()
                df_energy['radiation_sum_24h'] = df_energy['radiation'].rolling(window=96, min_periods=1, center=False).sum()
                df_energy['radiation_sum_48h'] = df_energy['radiation'].rolling(window=192, min_periods=1, center=False).sum()
                print("  - Temperature MA 24h/48h/72h: created")
                print("  - Radiation sum 24h/48h: created")

            if USE_MA_FEATURES:
                df_energy_daily = df_energy.resample("D").agg(
                    temperature=pd.NamedAgg(column='temperature', aggfunc='mean'),
                    load_max=pd.NamedAgg(column='total', aggfunc='max'),
                    radiation_sum=pd.NamedAgg(column='radiation', aggfunc='sum'),
                    temp_ma_24h=pd.NamedAgg(column='temp_ma_24h', aggfunc='last'),
                    temp_ma_48h=pd.NamedAgg(column='temp_ma_48h', aggfunc='last'),
                    temp_ma_72h=pd.NamedAgg(column='temp_ma_72h', aggfunc='last'),
                    radiation_sum_24h=pd.NamedAgg(column='radiation_sum_24h', aggfunc='last'),
                    radiation_sum_48h=pd.NamedAgg(column='radiation_sum_48h', aggfunc='last')
                )
            else:
                df_energy_daily = df_energy.resample("D").agg(
                    temperature=pd.NamedAgg(column='temperature', aggfunc='mean'),
                    load_max=pd.NamedAgg(column='total', aggfunc='max'),
                    radiation_sum=pd.NamedAgg(column='radiation', aggfunc='sum')
                )

            df_energy_daily["temperature"] = df_energy_daily["temperature"].round(2)
            df_energy_daily.index = pd.to_datetime(df_energy_daily.index)
            df_energy_daily = df_energy_daily.sort_index()

            cal = Murcia()
            df_energy_daily['dayofyear'] = df_energy_daily.index.dayofyear
            df_energy_daily['month'] = df_energy_daily.index.month
            df_energy_daily['weekday'] = df_energy_daily.index.weekday
            df_energy_daily['is_weekend'] = df_energy_daily['weekday'].isin([5, 6]).astype(int)
            df_energy_daily['is_holiday'] = df_energy_daily.index.to_series().apply(lambda x: int(cal.is_holiday(x)))
            df_energy_daily.index = df_energy_daily.index.date

            X = df_energy_daily.drop(columns=['load_max'])
            y = df_energy_daily['load_max']

            X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
            X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
            X = X.drop(columns=['month'])
            X['weekday_sin'] = np.sin(2 * np.pi * X['weekday'] / 7)
            X['weekday_cos'] = np.cos(2 * np.pi * X['weekday'] / 7)
            X = X.drop(columns=['weekday'])
            X['dayofyear_sin'] = np.sin(2 * np.pi * X['dayofyear'] / 365)
            X['dayofyear_cos'] = np.cos(2 * np.pi * X['dayofyear'] / 365)
            X = X.drop(columns=['dayofyear'])

            scaler = StandardScaler()
            climate_features = (
                ['temperature', 'radiation_sum', 'temp_ma_24h', 'temp_ma_48h',
                 'temp_ma_72h', 'radiation_sum_24h', 'radiation_sum_48h']
                if USE_MA_FEATURES else ['temperature', 'radiation_sum']
            )
            X[climate_features] = scaler.fit_transform(X[climate_features])

            print("\n" + "="*60)
            print("STARTING HYPERPARAMETER OPTIMIZATION FOR REGRESSION")
            print("="*60)

            results_df = optimize_all_regressors(X=X, y=y, n_trials=n_trials, cv=cv_folds, verbose=True, exclude_models=exclude_models)
            print_regression_results_table(results_df)

            evaluation_df = evaluate_optimized_regressors(X, y, results_df, cv=cv_folds)
            print_regression_evaluation_table(evaluation_df)

            results_output_path = os.path.join(optimization_results_dir, f"optimization_results_filter_{USE_FILTER}.csv")
            evaluation_output_path = os.path.join(evaluation_results_dir, f"evaluation_results_filter_{USE_FILTER}.csv")
            results_to_save = results_df.copy()
            results_to_save['Best Params'] = results_to_save['Best Params'].apply(json.dumps)
            results_to_save.to_csv(results_output_path, index=False)

            eval_to_save = evaluation_df.copy()
            eval_to_save['Best Params'] = eval_to_save['Best Params'].apply(json.dumps)
            eval_to_save.to_csv(evaluation_output_path, index=False)

            print(f"\nResults saved to:\n  - {results_output_path}\n  - {evaluation_output_path}")

            best_model_row = evaluation_df.iloc[0]
            all_configurations_summary.append({
                'USE_FILTER': USE_FILTER,
                'Best_Model': best_model_row['Model'],
                'R2': best_model_row['R2'],
                'MAE': best_model_row['MAE'],
                'MSE': best_model_row['MSE'],
                'RMSE': best_model_row['RMSE'],
                'MAPE': best_model_row['MAPE'],
                'R2_Std': best_model_row['R2 Std']
            })

            print(f"\n{'='*80}")
            print(f"CONFIGURATION {config_idx}/{len(USE_FILTER_VALUES)} COMPLETED SUCCESSFULLY")
            print(f"Best Model: {best_model_row['Model']} (MSE: {best_model_row['MSE']:.4f})")
            print(f"{'='*80}\n")

        except FileNotFoundError as e:
            print(f"\n{'!'*80}\nSKIPPING CONFIGURATION {config_idx}/{len(USE_FILTER_VALUES)}\nFile not found: {e}\n{'!'*80}\n")
        except Exception as e:
            print(f"\n{'!'*80}\nERROR IN CONFIGURATION {config_idx}/{len(USE_FILTER_VALUES)}\nError: {e}\n{'!'*80}\n")

    print(f"\n{'#'*80}\n# ALL CONFIGURATIONS PROCESSED\n{'#'*80}\n")

    if all_configurations_summary:
        summary_df = pd.DataFrame(all_configurations_summary)
        summary_df = summary_df.sort_values('MSE', ascending=True).reset_index(drop=True)
        summary_path = os.path.join(output_dir, "best_regressors_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}\nSUMMARY OF BEST REGRESSION MODELS\n{'='*80}")
        print(summary_df.to_string(index=False))
        print(f"\n{'='*80}\nSummary saved to: {summary_path}\n{'='*80}\n")
    else:
        print("\nNo successful configurations to summarize.\n")


_STANDALONE_CONFIG = {
    "max_load_regressor": {
        "use_filter_values": [True],
        "use_ma_features": True,
        "n_trials": 5,
        "cv_folds": 2,
        "exclude_models": [],
    }
}

if __name__ == "__main__":
    run(_STANDALONE_CONFIG)
