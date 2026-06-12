from __future__ import annotations

import json
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from workalendar.europe.spain import Murcia

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from functions.classification_optimization import (
    optimize_all_models,
    evaluate_optimized_models,
    print_results_table,
    print_evaluation_table
)


def run(config: dict) -> None:
    cfg = config.get("day_classifier", {})
    OPTIMAL_K_VALUES = cfg.get("optimal_k_values", [2, 3])
    CHOSEN_INTERVAL_CONFIGURATION_INDEX_VALUES = cfg.get("chosen_interval_configuration_index_values", [1, 2, 3, 4])
    USE_FILTER_VALUES = cfg.get("use_filter_values", [True])
    n_trials = cfg.get("n_trials", 5)
    cv_folds = cfg.get("cv_folds", 2)
    exclude_models = cfg.get("exclude_models", ["CatBoost"])

    root_dir = config.get("_project_root", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "data")
    clustered_dir = os.path.join(root_dir, "output", "clustered_data")
    output_dir = os.path.join(root_dir, "output", "classification_model")
    optimization_results_dir = os.path.join(output_dir, "optimization_results")
    evaluation_results_dir = os.path.join(output_dir, "evaluation_results")
    os.makedirs(clustered_dir, exist_ok=True)
    os.makedirs(optimization_results_dir, exist_ok=True)
    os.makedirs(evaluation_results_dir, exist_ok=True)

    combinations = list(product(OPTIMAL_K_VALUES, CHOSEN_INTERVAL_CONFIGURATION_INDEX_VALUES, USE_FILTER_VALUES))

    print(f"\n{'='*80}")
    print(f"TOTAL COMBINATIONS TO PROCESS: {len(combinations)}")
    print(f"{'='*80}\n")

    all_combinations_summary = []

    for combo_idx, (OPTIMAL_K, CHOSEN_INTERVAL_CONFIGURATION_INDEX, USE_FILTER) in enumerate(combinations, 1):

        print(f"\n{'#'*80}")
        print(f"# PROCESSING COMBINATION {combo_idx}/{len(combinations)}")
        print(f"# OPTIMAL_K = {OPTIMAL_K}")
        print(f"# CHOSEN_INTERVAL_CONFIGURATION_INDEX = {CHOSEN_INTERVAL_CONFIGURATION_INDEX}")
        print(f"# USE_FILTER = {USE_FILTER}")
        print(f"{'#'*80}\n")

        try:
            df_energy = pd.read_csv(
                os.path.join(data_dir, "data.csv"),
                parse_dates=["Date"], usecols=["Date", "total", "temperature", "radiation"]
            )
            df_energy.set_index("Date", inplace=True)

            df_intervals_clustered = pd.read_csv(
                os.path.join(clustered_dir, f"df_intervals_clustered_filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.csv")
            )

            df_time_clustered = pd.read_csv(
                os.path.join(clustered_dir, f"df_time_clustered_filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.csv")
            )
            if "DateOnly" in df_time_clustered.columns:
                df_time_clustered.set_index("DateOnly", inplace=True)
            else:
                df_time_clustered.set_index(df_time_clustered.columns[0], inplace=True)
            df_time_clustered.index = pd.to_datetime(df_time_clustered.index, errors="coerce").date

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
            df_energy_daily = df_energy_daily.join(df_time_clustered["cluster"], how="left")

            X = df_energy_daily.drop(columns=['cluster', 'load_max'])
            y = df_energy_daily['cluster'].astype(int)

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
            X[['temperature', 'radiation_sum']] = scaler.fit_transform(X[['temperature', 'radiation_sum']])

            print("\n" + "="*60)
            print("STARTING HYPERPARAMETER OPTIMIZATION")
            print("="*60)

            results_df = optimize_all_models(X=X, y=y, n_trials=n_trials, cv=cv_folds, verbose=True, exclude_models=exclude_models)
            print_results_table(results_df)

            evaluation_df = evaluate_optimized_models(X, y, results_df, cv=cv_folds)
            print_evaluation_table(evaluation_df)

            results_output_path = os.path.join(
                optimization_results_dir,
                f"optimization_results_filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.csv"
            )
            evaluation_output_path = os.path.join(
                evaluation_results_dir,
                f"evaluation_results_filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.csv"
            )
            results_to_save = results_df.copy()
            results_to_save['Best Params'] = results_to_save['Best Params'].apply(json.dumps)
            results_to_save.to_csv(results_output_path, index=False)

            eval_to_save = evaluation_df.copy()
            eval_to_save['Best Params'] = eval_to_save['Best Params'].apply(json.dumps)
            eval_to_save.to_csv(evaluation_output_path, index=False)

            print(f"\nResults saved to:\n  - {results_output_path}\n  - {evaluation_output_path}")

            best_model_row = evaluation_df.iloc[0]
            all_combinations_summary.append({
                'OPTIMAL_K': OPTIMAL_K,
                'CONFIG_INDEX': CHOSEN_INTERVAL_CONFIGURATION_INDEX,
                'USE_FILTER': USE_FILTER,
                'Best_Model': best_model_row['Model'],
                'Accuracy': best_model_row['Accuracy'],
                'Precision': best_model_row['Precision'],
                'Recall': best_model_row['Recall'],
                'ROC_AUC': best_model_row['ROC AUC'],
                'Accuracy_Std': best_model_row['Accuracy Std']
            })

            print(f"\n{'='*80}")
            print(f"COMBINATION {combo_idx}/{len(combinations)} COMPLETED SUCCESSFULLY")
            print(f"Best Model: {best_model_row['Model']} (Accuracy: {best_model_row['Accuracy']:.4f})")
            print(f"{'='*80}\n")

        except FileNotFoundError as e:
            print(f"\n{'!'*80}\nSKIPPING COMBINATION {combo_idx}/{len(combinations)}\nFile not found: {e}\n{'!'*80}\n")
        except Exception as e:
            print(f"\n{'!'*80}\nERROR IN COMBINATION {combo_idx}/{len(combinations)}\nError: {e}\n{'!'*80}\n")

    print(f"\n{'#'*80}\n# ALL COMBINATIONS PROCESSED\n{'#'*80}\n")

    if all_combinations_summary:
        summary_df = pd.DataFrame(all_combinations_summary)
        summary_df = summary_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        summary_path = os.path.join(output_dir, "best_models_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}\nSUMMARY OF BEST MODELS\n{'='*80}")
        print(summary_df.to_string(index=False))
        print(f"\n{'='*80}\nSummary saved to: {summary_path}\n{'='*80}\n")
    else:
        print("\nNo successful combinations to summarize.\n")


_STANDALONE_CONFIG = {
    "day_classifier": {
        "optimal_k_values": [2],
        "chosen_interval_configuration_index_values": [1],
        "use_filter_values": [True],
        "n_trials": 5,
        "cv_folds": 2,
        "exclude_models": [],
    }
}

if __name__ == "__main__":
    run(_STANDALONE_CONFIG)
