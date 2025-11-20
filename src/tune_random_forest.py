"""Hyperparameter tuning for RandomForest using RandomizedSearchCV.
Loads data via src.data_loader and uses src.preprocess.fit_transform to get preprocessed train/test arrays.
Saves best model to artifacts/RandomForest_tuned.joblib and metrics to results_tuned.json
"""
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.data_loader import load_ipl_data
from src.preprocess import fit_transform


def evaluate(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred))
    }


def run(n_iter: int = 40, cv: int = 3):
    print('Loading data...')
    df = load_ipl_data()

    # split inside fit_transform by providing test_df
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    print('Preprocessing (one-hot + K-fold target encoding)')
    X_train, X_test, y_train, y_test, artifacts = fit_transform(train_df, test_df)

    print('Setting up RandomizedSearchCV...')
    param_dist = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42)
    rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                            scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1, random_state=42)
    print('Running RandomizedSearchCV...')
    rs.fit(X_train, y_train)

    best = rs.best_estimator_
    print('Best params:', rs.best_params_)

    print('Evaluating best model on test set...')
    preds = best.predict(X_test)
    metrics = evaluate(y_test, preds)
    print('Test metrics:', metrics)

    # Save model and metrics
    artifacts_dir = Path(__file__).resolve().parents[1] / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    # use stronger compression (LZMA) to reduce artifact size further
    # LZMA gives better compression at the cost of slower save/load.
    try:
        joblib.dump(best, artifacts_dir / 'RandomForest_tuned.joblib', compress=('lzma', 9))
    except Exception:
        # fallback to default joblib compression if LZMA not available
        joblib.dump(best, artifacts_dir / 'RandomForest_tuned.joblib', compress=3)
    with open('results_tuned.json', 'w', encoding='utf-8') as f:
        json.dump({'best_params': rs.best_params_, 'test_metrics': metrics}, f, indent=2)

    print('Saved tuned model to artifacts/RandomForest_tuned.joblib and results to results_tuned.json')


if __name__ == '__main__':
    run()
