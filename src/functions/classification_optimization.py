"""Classification hyperparameter optimization using Optuna."""
from __future__ import annotations

import os
from typing import Iterable

import optuna
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

os.environ['CATBOOST_ALLOW_WRITING_FILES'] = 'false'


def optimize_cart(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42
    }
    model = DecisionTreeClassifier(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_xgboost(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_lightgbm(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
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
    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_catboost(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
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
    model = cb.CatBoostClassifier(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_svm_linear(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'C': trial.suggest_float('C', 0.001, 100.0, log=True),
        'kernel': 'linear',
        'probability': True,
        'random_state': 42
    }
    model = SVC(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_svm_nonlinear(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 0.001, 100.0, log=True),
        'kernel': kernel,
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'probability': True,
        'random_state': 42
    }
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    model = SVC(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_naive_bayes(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'var_smoothing': trial.suggest_float('var_smoothing', 1e-10, 1e-5, log=True)
    }
    model = GaussianNB(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_logistic_regression(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'C': trial.suggest_float('C', 0.001, 100.0, log=True),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'random_state': 42
    }
    model = LogisticRegression(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_random_forest(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score


def optimize_all_models(
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
        'CART': optimize_cart,
        'XGBoost': optimize_xgboost,
        'LightGBM': optimize_lightgbm,
        'CatBoost': optimize_catboost,
        'SVM Linear': optimize_svm_linear,
        'SVM Non-Linear': optimize_svm_nonlinear,
        'Naive Bayes': optimize_naive_bayes,
        'Logistic Regression': optimize_logistic_regression,
        'Random Forest': optimize_random_forest
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
            'Best Accuracy': best_value,
            'Best Params': best_params,
            'N Trials': n_trials
        })

        if verbose:
            print(f"\nBest accuracy: {best_value:.4f}")
            print(f"Best params: {best_params}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Best Accuracy', ascending=False).reset_index(drop=True)

    return results_df


def evaluate_optimized_models(
    X: pd.DataFrame,
    y: pd.Series,
    results_df: pd.DataFrame,
    cv: int = 3,
) -> pd.DataFrame:
    from sklearn.model_selection import cross_validate

    model_classes = {
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

    evaluation_results = []

    for _, row in results_df.iterrows():
        model_name = row['Model']
        best_params = row['Best Params']

        model_class = model_classes[model_name]

        if 'SVM' in model_name:
            best_params = best_params.copy()
            best_params['probability'] = True

        model = model_class(**best_params)

        scoring_to_use = {
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro'
        }

        try:
            if hasattr(model, 'predict_proba') or 'probability' in best_params:
                scoring_to_use['roc_auc'] = 'roc_auc_ovo'
        except:
            pass

        cv_results = cross_validate(model, X, y, cv=TimeSeriesSplit(n_splits=cv), scoring=scoring_to_use)

        result_dict = {
            'Model': model_name,
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'Recall': cv_results['test_recall'].mean(),
            'Accuracy Std': cv_results['test_accuracy'].std(),
            'Best Params': best_params
        }

        if 'test_roc_auc' in cv_results:
            result_dict['ROC AUC'] = cv_results['test_roc_auc'].mean()
        else:
            result_dict['ROC AUC'] = None

        evaluation_results.append(result_dict)

    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df = evaluation_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    return evaluation_df


def print_results_table(results_df: pd.DataFrame) -> None:
    print("\n" + "="*100)
    print("MODEL OPTIMIZATION RESULTS")
    print("="*100)

    display_df = results_df.drop(columns=['Best Params', 'N Trials']).copy()
    display_df['Best Accuracy'] = display_df['Best Accuracy'].apply(lambda x: f"{x:.4f}")

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


def print_evaluation_table(evaluation_df: pd.DataFrame) -> None:
    print("\n" + "="*120)
    print("DETAILED MODEL EVALUATION (Cross-Validation)")
    print("="*120)

    display_df = evaluation_df.drop(columns=['Best Params']).copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'ROC AUC', 'Accuracy Std']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")

    print(display_df.to_string(index=False))
    print("="*120 + "\n")
