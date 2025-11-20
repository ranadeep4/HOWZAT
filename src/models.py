"""Model definitions and helpers for training/evaluation."""
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import joblib
from pathlib import Path
import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """Return a dictionary of model name -> estimator instances to try."""
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        'HistGB': HistGradientBoostingRegressor(random_state=random_state),
        'MLP': MLPRegressor(hidden_layer_sizes=(128,64), max_iter=300, random_state=random_state)
    }
    return models


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    # mean_squared_error supports `squared` argument in newer versions of sklearn.
    try:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        # older sklearn: compute sqrt of MSE
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': float(r2_score(y_true, y_pred))
    }
    return metrics


def save_model(model, name: str):
    path = ARTIFACTS_DIR / f"{name}.joblib"

    # 🔥 compress the model to reduce file size
    joblib.dump(model, path, compress=3)

    return path


def load_model(name: str):
    path = ARTIFACTS_DIR / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    raise FileNotFoundError(path)
