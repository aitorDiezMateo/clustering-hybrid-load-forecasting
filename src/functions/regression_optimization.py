"""Regression hyperparameter optimization using Optuna."""
from __future__ import annotations

import os
from typing import Iterable

import optuna
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

os.environ['CATBOOST_ALLOW_WRITING_FILES'] = 'false'


def optimize_cart_regressor(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error']),
        'random_state': 42
    }
    model = DecisionTreeRegressor(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_xgboost_regressor(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_lightgbm_regressor(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_catboost_regressor(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 2, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42,
        'verbose': False,
        'allow_writing_files': False,
        'train_dir': None
    }
    model = cb.CatBoostRegressor(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_svr_linear(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'C': trial.suggest_float('C', 0.001, 100.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
        'kernel': 'linear'
    }
    model = SVR(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_svr_nonlinear(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 0.001, 100.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
        'kernel': kernel,
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
    }
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    model = SVR(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_linear_regression(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    reg_type = trial.suggest_categorical('reg_type', ['none', 'ridge', 'lasso'])

    if reg_type == 'none':
        model = LinearRegression()
    elif reg_type == 'ridge':
        alpha = trial.suggest_float('alpha', 0.001, 100.0, log=True)
        model = Ridge(alpha=alpha, random_state=42)
    else:
        alpha = trial.suggest_float('alpha', 0.001, 100.0, log=True)
        model = Lasso(alpha=alpha, random_state=42)

    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_random_forest_regressor(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    model = RandomForestRegressor(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
    return score


def optimize_all_regressors(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    cv: int = 3,
    verbose: bool = True,
    exclude_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    if exclude_models is None:
        exclude_models = []

    models = {
        'CART': optimize_cart_regressor,
        'XGBoost': optimize_xgboost_regressor,
        'LightGBM': optimize_lightgbm_regressor,
        'CatBoost': optimize_catboost_regressor,
        'SVR Linear': optimize_svr_linear,
        'SVR Non-Linear': optimize_svr_nonlinear,
        'Linear Regression': optimize_linear_regression,
        'Random Forest': optimize_random_forest_regressor
    }

    models = {name: func for name, func in models.items() if name not in exclude_models}

    results = []

    for model_name, objective_func in models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimizing {model_name}...")
            print(f"{'='*60}")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            lambda trial: objective_func(trial, X, y, cv=cv),
            n_trials=n_trials,
            show_progress_bar=verbose
        )

        try:
            best_value = study.best_value
            best_params = study.best_params
        except (ValueError, AttributeError):
            print(f"\nWARNING: All trials failed for {model_name}, skipping.")
            continue

        results.append({
            'Model': model_name,
            'Best Score': best_value,
            'Best Params': best_params,
            'N Trials': n_trials
        })

        if verbose:
            print(f"\nBest Score (neg_MSE): {best_value:.4f}")
            print(f"Best params: {best_params}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Best Score', ascending=False).reset_index(drop=True)

    return results_df


def evaluate_optimized_regressors(
    X: pd.DataFrame,
    y: pd.Series,
    results_df: pd.DataFrame,
    cv: int = 3,
) -> pd.DataFrame:
    from sklearn.model_selection import cross_validate

    model_classes = {
        'CART': DecisionTreeRegressor,
        'XGBoost': xgb.XGBRegressor,
        'LightGBM': lgb.LGBMRegressor,
        'CatBoost': cb.CatBoostRegressor,
        'SVR Linear': SVR,
        'SVR Non-Linear': SVR,
        'Linear Regression': None,
        'Random Forest': RandomForestRegressor
    }

    scoring = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mape': 'neg_mean_absolute_percentage_error'
    }

    evaluation_results = []

    for _, row in results_df.iterrows():
        model_name = row['Model']
        best_params = row['Best Params']

        if model_name == 'Linear Regression':
            best_params_copy = best_params.copy()
            reg_type = best_params_copy.pop('reg_type', 'none')

            if reg_type == 'none':
                model = LinearRegression()
            elif reg_type == 'ridge':
                model = Ridge(**best_params_copy, random_state=42)
            else:
                model = Lasso(**best_params_copy, random_state=42)
        elif model_name == 'CatBoost':
            best_params_copy = best_params.copy()
            best_params_copy['allow_writing_files'] = False
            best_params_copy['train_dir'] = None
            model_class = model_classes[model_name]
            model = model_class(**best_params_copy)
        else:
            model_class = model_classes[model_name]
            model = model_class(**best_params)

        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

        mae = -cv_results['test_neg_mae'].mean()
        mse = -cv_results['test_neg_mse'].mean()
        rmse = np.sqrt(mse)
        mape = -cv_results['test_neg_mape'].mean() * 100
        r2 = cv_results['test_r2'].mean()
        r2_std = cv_results['test_r2'].std()

        evaluation_results.append({
            'Model': model_name,
            'R2': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2 Std': r2_std,
            'Best Params': best_params
        })

    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df = evaluation_df.sort_values('MSE', ascending=True).reset_index(drop=True)

    return evaluation_df


def print_regression_results_table(results_df: pd.DataFrame) -> None:
    print("\n" + "="*100)
    print("REGRESSION MODEL OPTIMIZATION RESULTS (Optimized by MSE)")
    print("="*100)

    display_df = results_df.drop(columns=['Best Params', 'N Trials']).copy()
    display_df['Best Score'] = display_df['Best Score'].apply(lambda x: f"{x:.4f}")

    print(display_df.to_string(index=False))
    print("="*100)

    print("\n" + "="*100)
    print("BEST PARAMETERS FOR EACH MODEL")
    print("="*100)
    for idx, row in results_df.iterrows():
        print(f"\n{row['Model']}:")
        for param, value in row['Best Params'].items():
            print(f"  {param}: {value}")
    print("="*100 + "\n")


def print_regression_evaluation_table(evaluation_df: pd.DataFrame) -> None:
    print("\n" + "="*130)
    print("DETAILED REGRESSION MODEL EVALUATION (Cross-Validation) - Sorted by MSE")
    print("="*130)

    display_df = evaluation_df.drop(columns=['Best Params']).copy()
    for col in ['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2 Std']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if col != 'MAPE' else f"{x:.2f}%")

    print(display_df.to_string(index=False))
    print("="*130 + "\n")
