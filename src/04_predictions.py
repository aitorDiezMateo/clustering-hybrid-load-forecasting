"""Full predictions and evaluation pipeline."""
from __future__ import annotations

import json
import os
import sys
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from workalendar.europe.spain import Murcia

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from functions.final_model_evaluation import evaluate_model

classification_model_classes: dict[str, type] = {
    'CART': DecisionTreeClassifier,
    'XGBoost': xgb.XGBClassifier,
    'LightGBM': lgb.LGBMClassifier,
    'CatBoost': cb.CatBoostClassifier,
    'SVM Linear': SVC,
    'SVM Non-Linear': SVC,
    'Naive Bayes': GaussianNB,
    'Logistic Regression': LogisticRegression,
    'Random Forest': RandomForestClassifier
}

regression_model_classes: dict[str, type | None] = {
    'CART': DecisionTreeRegressor,
    'XGBoost': xgb.XGBRegressor,
    'LightGBM': lgb.LGBMRegressor,
    'CatBoost': cb.CatBoostRegressor,
    'SVR Linear': SVR,
    'SVR Non-Linear': SVR,
    'Linear Regression': None,
    'Random Forest': RandomForestRegressor
}


def run(config: dict) -> None:
    cfg = config.get("predictions", {})
    USE_ALL_MODELS = cfg.get("use_all_models", False)
    SOFT_CLUSTERING = cfg.get("soft_clustering", False)
    USE_MA_FEATURES = cfg.get("use_ma_features", True)
    OPTIMAL_K_VALUES = cfg.get("optimal_k_values", [2, 3])
    CHOSEN_INTERVAL_CONFIGURATION_INDEX_VALUES = cfg.get("chosen_interval_configuration_index_values", [1, 2, 3, 4])
    USE_FILTER_VALUES = cfg.get("use_filter_values", [True])
    TEST_START_DATE = cfg.get("test_start_date", "2021-01-06")
    N_SPLITS_CV = cfg.get("n_splits_cv", 2)

    root_dir = config.get("_project_root", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "data")
    clustered_dir = os.path.join(root_dir, "output", "clustered_data")
    classification_output_dir = os.path.join(root_dir, "output", "classification_model")
    regression_output_dir = os.path.join(
        root_dir, "output",
        "regression_model_with_ma" if USE_MA_FEATURES else "regression_model"
    )
    predictions_output_dir = os.path.join(
        root_dir, "output",
        "full_predictions_with_ma" if USE_MA_FEATURES else "full_predictions"
    )
    evaluation_output_dir = os.path.join(root_dir, "output", "predictions_evaluation")

    combinations = list(product(OPTIMAL_K_VALUES, CHOSEN_INTERVAL_CONFIGURATION_INDEX_VALUES, USE_FILTER_VALUES))

    print(f"\n{'='*80}")
    print(f"PREDICTIONS PIPELINE")
    print(f"MODE: {'ALL MODELS' if USE_ALL_MODELS else 'BEST MODELS ONLY'}")
    print(f"TOTAL COMBINATIONS: {len(combinations)}")
    print(f"{'='*80}\n")

    all_predictions_summary = []

    for combo_idx, (OPTIMAL_K, CHOSEN_INTERVAL_CONFIGURATION_INDEX, USE_FILTER) in enumerate(combinations, 1):

        print(f"\n{'#'*80}")
        print(f"# COMBINATION {combo_idx}/{len(combinations)}: k={OPTIMAL_K}, config={CHOSEN_INTERVAL_CONFIGURATION_INDEX}")
        print(f"{'#'*80}\n")

        combo_name = f"filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}"

        try:
            classification_eval_file = os.path.join(
                classification_output_dir, "evaluation_results",
                f"evaluation_results_filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.csv"
            )
            if not os.path.exists(classification_eval_file):
                print(f"Classification file not found: {classification_eval_file}")
                continue
            classification_eval_df = pd.read_csv(classification_eval_file)

            regression_eval_file = os.path.join(
                regression_output_dir, "evaluation_results",
                f"evaluation_results_filter_{USE_FILTER}.csv"
            )
            if not os.path.exists(regression_eval_file):
                print(f"Regression file not found: {regression_eval_file}")
                continue
            regression_eval_df = pd.read_csv(regression_eval_file)

            if USE_ALL_MODELS:
                classification_models_to_test = classification_eval_df.to_dict('records')
                regression_models_to_test = regression_eval_df.to_dict('records')
            else:
                classification_models_to_test = [classification_eval_df.iloc[0].to_dict()]
                regression_models_to_test = [regression_eval_df.iloc[0].to_dict()]

            df_energy = pd.read_csv(
                os.path.join(data_dir, "data.csv"),
                parse_dates=["Date"], usecols=["Date", "total", "temperature", "radiation", "total_filter"]
            )
            df_energy.set_index("Date", inplace=True)

            if USE_FILTER:
                df_energy["total"] = df_energy["total_filter"]

            if USE_MA_FEATURES:
                df_energy['temp_ma_24h'] = df_energy['temperature'].rolling(window=96, min_periods=1, center=False).mean()
                df_energy['temp_ma_48h'] = df_energy['temperature'].rolling(window=192, min_periods=1, center=False).mean()
                df_energy['temp_ma_72h'] = df_energy['temperature'].rolling(window=288, min_periods=1, center=False).mean()
                df_energy['radiation_sum_24h'] = df_energy['radiation'].rolling(window=96, min_periods=1, center=False).sum()
                df_energy['radiation_sum_48h'] = df_energy['radiation'].rolling(window=192, min_periods=1, center=False).sum()

            df_time_clustered = pd.read_csv(
                os.path.join(clustered_dir, f"df_time_clustered_filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.csv"),
                parse_dates=["DateOnly"]
            )
            df_time_clustered.set_index("DateOnly", inplace=True)
            df_time_clustered.index = df_time_clustered.index.date

            cluster_mean_profiles = pd.read_csv(
                os.path.join(clustered_dir, f"cluster_mean_profiles_filter_{USE_FILTER}_k_{OPTIMAL_K}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.csv"),
                index_col=0
            )

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
            df_energy_daily = df_energy_daily.join(df_time_clustered["cluster"], how="left")

            test_start_date = pd.Timestamp(TEST_START_DATE).date()
            df_energy_daily_train = df_energy_daily[df_energy_daily.index < test_start_date].copy()
            df_energy_daily_test = df_energy_daily[df_energy_daily.index >= test_start_date].copy()

            print(f"Train: {len(df_energy_daily_train)} days, Test: {len(df_energy_daily_test)} days")

            def transform_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
                X = df.drop(columns=['cluster', 'load_max']).copy()
                X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
                X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
                X = X.drop(columns=['month'])
                X['weekday_sin'] = np.sin(2 * np.pi * X['weekday'] / 7)
                X['weekday_cos'] = np.cos(2 * np.pi * X['weekday'] / 7)
                X = X.drop(columns=['weekday'])
                X['dayofyear_sin'] = np.sin(2 * np.pi * X['dayofyear'] / 365)
                X['dayofyear_cos'] = np.cos(2 * np.pi * X['dayofyear'] / 365)
                X = X.drop(columns=['dayofyear'])
                climate_features = (
                    ['temperature', 'radiation_sum', 'temp_ma_24h', 'temp_ma_48h',
                     'temp_ma_72h', 'radiation_sum_24h', 'radiation_sum_48h']
                    if USE_MA_FEATURES else ['temperature', 'radiation_sum']
                )
                return X, climate_features

            X_train_cls, climate_features = transform_features(df_energy_daily_train)
            X_test_cls, _ = transform_features(df_energy_daily_test)
            y_cluster_train = df_energy_daily_train['cluster'].astype(int)
            y_cluster_test = df_energy_daily_test['cluster'].astype(int)

            X_train_reg, _ = transform_features(df_energy_daily_train)
            X_test_reg, _ = transform_features(df_energy_daily_test)
            y_load_max_train = df_energy_daily_train['load_max']
            y_load_max_test = df_energy_daily_test['load_max']

            scaler_cls = StandardScaler()
            X_train_cls[climate_features] = scaler_cls.fit_transform(X_train_cls[climate_features])
            X_test_cls[climate_features] = scaler_cls.transform(X_test_cls[climate_features])

            scaler_reg = StandardScaler()
            X_train_reg[climate_features] = scaler_reg.fit_transform(X_train_reg[climate_features])
            X_test_reg[climate_features] = scaler_reg.transform(X_test_reg[climate_features])

            test_start_datetime = pd.Timestamp(TEST_START_DATE)
            if df_energy.index.tz is not None:
                test_start_datetime = test_start_datetime.tz_localize(df_energy.index.tz)

            df_energy_test = df_energy[df_energy.index >= test_start_datetime].copy()
            df_energy_actual = df_energy_test[['total']].reset_index()
            df_energy_actual.columns = ['datetime', 'actual']
            if df_energy_actual['datetime'].dt.tz is not None:
                df_energy_actual['datetime'] = df_energy_actual['datetime'].dt.tz_localize(None)

            for class_model_info in classification_models_to_test:
                for reg_model_info in regression_models_to_test:

                    classification_model_name = class_model_info['Model']
                    classification_params: dict[str, Any] = json.loads(class_model_info['Best Params'])
                    classification_accuracy = class_model_info['Accuracy']

                    regression_model_name = reg_model_info['Model']
                    regression_params: dict[str, Any] = json.loads(reg_model_info['Best Params'])
                    regression_r2 = reg_model_info['R2']

                    print(f"  Models: {classification_model_name} + {regression_model_name}")

                    if 'SVM' in classification_model_name:
                        classification_params['probability'] = True

                    classification_model = classification_model_classes[classification_model_name](**classification_params)

                    if regression_model_name == 'Linear Regression':
                        regression_params_copy = regression_params.copy()
                        reg_type = regression_params_copy.pop('reg_type', 'none')
                        if reg_type == 'none':
                            regression_model = LinearRegression()
                        elif reg_type == 'ridge':
                            regression_model = Ridge(**regression_params_copy, random_state=42)
                        else:
                            regression_model = Lasso(**regression_params_copy, random_state=42)
                    else:
                        regression_model = regression_model_classes[regression_model_name](**regression_params)

                    classification_model.fit(X_train_cls, y_cluster_train)
                    regression_model.fit(X_train_reg, y_load_max_train)

                    if SOFT_CLUSTERING:
                        cluster_probabilities = classification_model.predict_proba(X_test_cls)
                    else:
                        predicted_clusters = classification_model.predict(X_test_cls)

                    predicted_load_max = regression_model.predict(X_test_reg)

                    df_predictions = pd.DataFrame(index=df_energy_daily_test.index, columns=cluster_mean_profiles.columns)

                    for idx, date in enumerate(df_energy_daily_test.index):
                        predicted_max = predicted_load_max[idx]
                        if SOFT_CLUSTERING:
                            probs = cluster_probabilities[idx]
                            weighted_profile = np.zeros(len(cluster_mean_profiles.columns))
                            for cluster_idx, prob in enumerate(probs):
                                weighted_profile += prob * cluster_mean_profiles.loc[cluster_idx].values
                            df_predictions.loc[date] = weighted_profile * predicted_max
                        else:
                            predicted_cluster = int(predicted_clusters[idx])
                            df_predictions.loc[date] = cluster_mean_profiles.loc[predicted_cluster] * predicted_max

                    df_predictions.index.name = 'date'
                    df_predictions_reset = df_predictions.reset_index()
                    df_long = df_predictions_reset.melt(id_vars='date', var_name='time', value_name='predicted')
                    df_long['datetime'] = pd.to_datetime(
                        df_long['date'].astype(str) + ' ' + df_long['time'], format='%Y-%m-%d %H:%M:%S'
                    )
                    df_predictions_ts = df_long[['datetime', 'predicted']].sort_values('datetime').reset_index(drop=True)

                    if df_predictions_ts['datetime'].dt.tz is not None:
                        df_predictions_ts['datetime'] = df_predictions_ts['datetime'].dt.tz_localize(None)

                    df_results = df_energy_actual.merge(df_predictions_ts, on='datetime', how='inner')

                    actual = df_results['actual'].values
                    predicted = df_results['predicted'].values

                    mae = mean_absolute_error(actual, predicted)
                    mse = mean_squared_error(actual, predicted)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(actual, predicted)
                    wmape = np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

                    print(f"    Metrics: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, WMAPE={wmape:.2f}%")

                    combo_output_dir = os.path.join(predictions_output_dir, combo_name)
                    os.makedirs(combo_output_dir, exist_ok=True)

                    metrics_df = pd.DataFrame([{
                        'USE_FILTER': USE_FILTER,
                        'OPTIMAL_K': OPTIMAL_K,
                        'CONFIG_INDEX': CHOSEN_INTERVAL_CONFIGURATION_INDEX,
                        'Classification_Model': classification_model_name,
                        'Classification_Accuracy': classification_accuracy,
                        'Regression_Model': regression_model_name,
                        'Regression_R2': regression_r2,
                        'Prediction_R2': r2,
                        'Prediction_MAE': mae,
                        'Prediction_MSE': mse,
                        'Prediction_RMSE': rmse,
                        'Prediction_WMAPE': wmape
                    }])
                    metrics_path = os.path.join(combo_output_dir, "prediction_metrics.csv")
                    predictions_path = os.path.join(combo_output_dir, "predictions.csv")
                    metrics_df.to_csv(metrics_path, index=False)
                    df_results.to_csv(predictions_path, index=False)

                    all_predictions_summary.append({
                        'USE_FILTER': USE_FILTER,
                        'OPTIMAL_K': OPTIMAL_K,
                        'CONFIG_INDEX': CHOSEN_INTERVAL_CONFIGURATION_INDEX,
                        'Classification_Model': classification_model_name,
                        'Regression_Model': regression_model_name,
                        'R2': r2,
                        'MAE': mae,
                        'RMSE': rmse,
                        'WMAPE': wmape
                    })

                    print(f"    Generating evaluation plots...")
                    eval_output_folder = os.path.join(evaluation_output_dir, combo_name)
                    os.makedirs(eval_output_folder, exist_ok=True)
                    evaluate_model(
                        df_energy,
                        df_results.rename(columns={'predicted': 'consumption'}),
                        eval_output_folder, OPTIMAL_K, use_filter=USE_FILTER, print_metrics=False
                    )
                    print(f"    Evaluation plots saved to: {eval_output_folder}")

            # =====================================================================
            # TIME SERIES CROSS-VALIDATION OF THE FULL HYBRID PIPELINE
            # =====================================================================
            print(f"\n{'='*60}")
            print(f"TIME SERIES CV ({N_SPLITS_CV} splits)")
            print(f"{'='*60}")

            tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
            cv_splits_metrics: list[dict] = []

            for cv_split_idx, (_, cv_test_index) in enumerate(tscv.split(df_energy_daily), 1):
                cv_all_idx = np.arange(len(df_energy_daily))
                cv_train_mask = ~np.isin(cv_all_idx, cv_test_index)
                df_cv_train = df_energy_daily.iloc[cv_all_idx[cv_train_mask]].copy()
                df_cv_test_fold = df_energy_daily.iloc[cv_test_index].copy()

                print(f"  Split {cv_split_idx}/{N_SPLITS_CV}: "
                      f"train={len(df_cv_train)} days, "
                      f"test={len(df_cv_test_fold)} days "
                      f"({df_cv_test_fold.index.min()} - {df_cv_test_fold.index.max()})")

                X_cv_cls_tr, cv_climate_feats = transform_features(df_cv_train)
                X_cv_cls_te, _ = transform_features(df_cv_test_fold)
                y_cv_cls_tr = df_cv_train['cluster'].astype(int)

                X_cv_reg_tr, _ = transform_features(df_cv_train)
                X_cv_reg_te, _ = transform_features(df_cv_test_fold)
                y_cv_reg_tr = df_cv_train['load_max']

                scaler_cv_cls = StandardScaler()
                X_cv_cls_tr[cv_climate_feats] = scaler_cv_cls.fit_transform(X_cv_cls_tr[cv_climate_feats])
                X_cv_cls_te[cv_climate_feats] = scaler_cv_cls.transform(X_cv_cls_te[cv_climate_feats])

                scaler_cv_reg = StandardScaler()
                X_cv_reg_tr[cv_climate_feats] = scaler_cv_reg.fit_transform(X_cv_reg_tr[cv_climate_feats])
                X_cv_reg_te[cv_climate_feats] = scaler_cv_reg.transform(X_cv_reg_te[cv_climate_feats])

                cv_date_min = df_cv_test_fold.index.min()
                cv_date_max = df_cv_test_fold.index.max()
                df_cv_actual = (
                    df_energy[
                        (df_energy.index.date >= cv_date_min) &
                        (df_energy.index.date <= cv_date_max)
                    ][['total']]
                    .reset_index()
                )
                df_cv_actual.columns = ['datetime', 'actual']
                if df_cv_actual['datetime'].dt.tz is not None:
                    df_cv_actual['datetime'] = df_cv_actual['datetime'].dt.tz_localize(None)

                for cv_cls_info in classification_models_to_test:
                    for cv_reg_info in regression_models_to_test:
                        cv_cls_name = cv_cls_info['Model']
                        cv_cls_params: dict[str, Any] = json.loads(cv_cls_info['Best Params'])
                        cv_cls_acc = cv_cls_info['Accuracy']
                        cv_reg_name = cv_reg_info['Model']
                        cv_reg_params: dict[str, Any] = json.loads(cv_reg_info['Best Params'])
                        cv_reg_r2_val = cv_reg_info['R2']

                        if 'SVM' in cv_cls_name:
                            cv_cls_params['probability'] = True

                        cv_cls_model = classification_model_classes[cv_cls_name](**cv_cls_params)

                        if cv_reg_name == 'Linear Regression':
                            cv_rp = cv_reg_params.copy()
                            cv_rt = cv_rp.pop('reg_type', 'none')
                            if cv_rt == 'none':
                                cv_reg_model: Any = LinearRegression()
                            elif cv_rt == 'ridge':
                                cv_reg_model = Ridge(**cv_rp, random_state=42)
                            else:
                                cv_reg_model = Lasso(**cv_rp, random_state=42)
                        else:
                            cv_reg_model = regression_model_classes[cv_reg_name](**cv_reg_params)

                        cv_cls_model.fit(X_cv_cls_tr, y_cv_cls_tr)
                        cv_reg_model.fit(X_cv_reg_tr, y_cv_reg_tr)

                        if SOFT_CLUSTERING:
                            cv_cluster_probs = cv_cls_model.predict_proba(X_cv_cls_te)
                        else:
                            cv_pred_clusters = cv_cls_model.predict(X_cv_cls_te)
                        cv_pred_load = cv_reg_model.predict(X_cv_reg_te)

                        df_cv_preds = pd.DataFrame(
                            index=df_cv_test_fold.index, columns=cluster_mean_profiles.columns
                        )
                        for cv_i, cv_date in enumerate(df_cv_test_fold.index):
                            cv_pm = cv_pred_load[cv_i]
                            if SOFT_CLUSTERING:
                                cv_probs_i = cv_cluster_probs[cv_i]
                                cv_wp = np.zeros(len(cluster_mean_profiles.columns))
                                for cv_ci, cv_p in enumerate(cv_probs_i):
                                    cv_wp += cv_p * cluster_mean_profiles.loc[cv_ci].values
                                df_cv_preds.loc[cv_date] = cv_wp * cv_pm
                            else:
                                cv_pc = int(cv_pred_clusters[cv_i])
                                df_cv_preds.loc[cv_date] = cluster_mean_profiles.loc[cv_pc] * cv_pm

                        df_cv_preds.index.name = 'date'
                        df_cv_long = df_cv_preds.reset_index().melt(
                            id_vars='date', var_name='time', value_name='predicted'
                        )
                        df_cv_long['datetime'] = pd.to_datetime(
                            df_cv_long['date'].astype(str) + ' ' + df_cv_long['time'],
                            format='%Y-%m-%d %H:%M:%S'
                        )
                        df_cv_ts = (
                            df_cv_long[['datetime', 'predicted']]
                            .sort_values('datetime')
                            .reset_index(drop=True)
                        )
                        if df_cv_ts['datetime'].dt.tz is not None:
                            df_cv_ts['datetime'] = df_cv_ts['datetime'].dt.tz_localize(None)

                        df_cv_res = df_cv_actual.merge(df_cv_ts, on='datetime', how='inner')
                        cv_act = df_cv_res['actual'].values
                        cv_prd = df_cv_res['predicted'].values

                        cv_splits_metrics.append({
                            'Split': cv_split_idx,
                            'Classification_Model': cv_cls_name,
                            'Classification_Accuracy': cv_cls_acc,
                            'Regression_Model': cv_reg_name,
                            'Regression_R2': cv_reg_r2_val,
                            'R2': r2_score(cv_act, cv_prd),
                            'MAE': mean_absolute_error(cv_act, cv_prd),
                            'MSE': mean_squared_error(cv_act, cv_prd),
                            'RMSE': np.sqrt(mean_squared_error(cv_act, cv_prd)),
                            'WMAPE': np.sum(np.abs(cv_act - cv_prd)) / np.sum(cv_act) * 100,
                        })

            if cv_splits_metrics:
                cv_df = pd.DataFrame(cv_splits_metrics)
                cv_agg = cv_df.groupby(['Classification_Model', 'Regression_Model']).agg(
                    Classification_Accuracy=('Classification_Accuracy', 'mean'),
                    Regression_R2=('Regression_R2', 'mean'),
                    R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
                    MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
                    MSE_mean=('MSE', 'mean'), MSE_std=('MSE', 'std'),
                    RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
                    WMAPE_mean=('WMAPE', 'mean'), WMAPE_std=('WMAPE', 'std'),
                ).reset_index()

                cv_combo_dir = os.path.join(predictions_output_dir, combo_name)
                os.makedirs(cv_combo_dir, exist_ok=True)
                cv_avg_path = os.path.join(cv_combo_dir, "prediction_metrics_cv_averaged.csv")
                cv_det_path = os.path.join(cv_combo_dir, "prediction_metrics_cv_detailed.csv")
                cv_agg.to_csv(cv_avg_path, index=False)
                cv_df.to_csv(cv_det_path, index=False)

                best_cv = cv_agg.sort_values('WMAPE_mean').iloc[0]
                print(f"  Best CV: {best_cv['Classification_Model']} + {best_cv['Regression_Model']}")
                print(f"    R2={best_cv['R2_mean']:.4f}±{best_cv['R2_std']:.4f}  "
                      f"WMAPE={best_cv['WMAPE_mean']:.2f}%±{best_cv['WMAPE_std']:.2f}%")
                print(f"  CV results saved to: {cv_combo_dir}")

            print(f"\n--- Combination {combo_idx} Complete ---\n")

        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    if all_predictions_summary:
        summary_df = pd.DataFrame(all_predictions_summary).sort_values('WMAPE', ascending=True).reset_index(drop=True)
        summary_path = os.path.join(predictions_output_dir, "all_predictions_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}\nPREDICTION SUMMARY\n{'='*80}")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}\n{'='*80}\n")
    else:
        print("\nNo predictions generated.\n")

    print("--- Predictions Pipeline Complete ---")


_STANDALONE_CONFIG = {
    "predictions": {
        "use_all_models": False,
        "soft_clustering": False,
        "use_ma_features": True,
        "optimal_k_values": [2],
        "chosen_interval_configuration_index_values": [1],
        "use_filter_values": [True],
        "test_start_date": "2021-01-06",
        "n_splits_cv": 2,
    }
}

if __name__ == "__main__":
    run(_STANDALONE_CONFIG)
