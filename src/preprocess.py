"""Preprocessing utilities: one-hot for teams/venue, K-fold target encoding for high-cardinality features,
scaling, and saving artifacts.

API:
- fit_transform(train_df, test_df=None): fits encoders on train_df and returns X_train, X_test (if provided), y_train, y_test (if provided), artifacts
- transform_with_artifacts(df, artifacts): transform new dataframe using saved artifacts
"""
from typing import Tuple, Dict, Optional
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import KFold


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _kfold_target_encode(train_series: pd.Series, target: pd.Series, n_splits: int = 5, smoothing: float = 10.0, random_state: int = 42) -> Tuple[pd.Series, Dict]:
    """Perform out-of-fold target encoding on a training series.

    Returns encoded_series (aligned with train_series index) and mapping for full-train (for use on test set).
    Smoothing follows: (count * mean + prior * smoothing) / (count + smoothing)
    """
    prior = target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    encoded = pd.Series(index=train_series.index, dtype=float)

    for train_idx, val_idx in kf.split(train_series):
        tr_s = train_series.iloc[train_idx]
        tr_t = target.iloc[train_idx]
        stats = tr_t.groupby(tr_s).agg(['mean', 'count'])
        means = stats['mean']
        counts = stats['count']

        # smoothing per category
        smooth = (counts * means + smoothing * prior) / (counts + smoothing)

        # map to validation
        val_vals = train_series.iloc[val_idx].map(smooth)
        # fill unseen in smooth with prior
        val_vals = val_vals.fillna(prior)
        encoded.iloc[val_idx] = val_vals

    # Build final mapping on full train for test-time transform
    full_stats = target.groupby(train_series).agg(['mean', 'count'])
    counts = full_stats['count']
    means = full_stats['mean']
    final_mapping = ((counts * means + smoothing * prior) / (counts + smoothing)).to_dict()

    return encoded, {'mapping': final_mapping, 'prior': float(prior), 'smoothing': float(smoothing)}


def fit_transform(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None,
                  onehot_cols: Optional[list] = None,
                  target_encode_cols: Optional[list] = None,
                  numeric_cols: Optional[list] = None,
                  n_splits: int = 5, smoothing: float = 10.0) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Dict]:
    """Fit encoders on train_df and transform train and optional test.

    Returns: X_train, X_test (or None), y_train, y_test (or None), artifacts
    """
    if onehot_cols is None:
        onehot_cols = ['bat_team', 'bowl_team', 'venue']
    if target_encode_cols is None:
        target_encode_cols = ['batsman', 'bowler']
    if numeric_cols is None:
        numeric_cols = ['runs', 'wickets', 'overs', 'striker']

    # copy to avoid modifying
    tr = train_df.copy()
    te = test_df.copy() if test_df is not None else None

    # y
    y_train = tr['total'].reset_index(drop=True)
    y_test = te['total'].reset_index(drop=True) if te is not None and 'total' in te.columns else None

    # One-hot encode low-cardinality columns
    # OneHotEncoder changed parameter name in newer scikit-learn versions.
    # Try the older 'sparse' kwarg first, fall back to 'sparse_output' if needed.
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(tr[onehot_cols].astype(str))
    ohe_cols = ohe.get_feature_names_out(onehot_cols)
    X_tr_ohe = pd.DataFrame(ohe.transform(tr[onehot_cols].astype(str)), columns=ohe_cols, index=tr.index)
    X_te_ohe = pd.DataFrame(ohe.transform(te[onehot_cols].astype(str)), columns=ohe_cols, index=te.index) if te is not None else None

    # Target encode high-cardinality columns using K-fold on training data
    te_artifacts = {}
    X_tr_te = pd.DataFrame(index=tr.index)
    X_te_te = pd.DataFrame(index=te.index) if te is not None else None

    for col in target_encode_cols:
        enc_series, meta = _kfold_target_encode(tr[col].astype(str), y_train, n_splits=n_splits, smoothing=smoothing)
        X_tr_te[col + '_te'] = enc_series.values
        te_artifacts[col] = meta

        # transform test set using final mapping
        if te is not None:
            mapping = meta['mapping']
            prior = meta['prior']
            X_te_te[col + '_te'] = te[col].astype(str).map(mapping).fillna(prior).values

    # Numeric columns
    X_tr_num = tr[numeric_cols].reset_index(drop=True)
    X_te_num = te[numeric_cols].reset_index(drop=True) if te is not None else None

    # Concatenate all features
    X_train_df = pd.concat([X_tr_ohe.reset_index(drop=True), X_tr_num, X_tr_te.reset_index(drop=True)], axis=1)
    X_test_df = pd.concat([X_te_ohe.reset_index(drop=True), X_te_num, X_te_te.reset_index(drop=True)], axis=1) if te is not None else None

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values) if X_test_df is not None else None

    # Save artifacts
    artifacts = {
        'onehot': ohe,
        'onehot_cols': list(ohe_cols),
        'target_encode': te_artifacts,
        'numeric_cols': numeric_cols,
        'scaler': scaler
    }
    joblib.dump(artifacts, ARTIFACTS_DIR / 'preprocess_artifacts.joblib')

    return X_train, X_test, y_train.values, (y_test.values if y_test is not None else None), artifacts


def transform_with_artifacts(df: pd.DataFrame, artifacts: Dict) -> np.ndarray:
    """Transform a new dataframe using saved artifacts (onehot, target mapping, scaler).

    Note: target encoding for unseen categories maps to prior.
    """
    df = df.copy()
    ohe = artifacts['onehot']
    ohe_cols = artifacts['onehot_cols']
    numeric_cols = artifacts['numeric_cols']
    te_artifacts = artifacts['target_encode']
    scaler = artifacts['scaler']

    X_ohe = pd.DataFrame(ohe.transform(df[[c for c in ohe.feature_names_in_]].astype(str)), columns=ohe_cols)

    X_num = df[numeric_cols].reset_index(drop=True)

    X_te = pd.DataFrame(index=df.index)
    for col, meta in te_artifacts.items():
        mapping = meta['mapping']
        prior = meta['prior']
        X_te[col + '_te'] = df[col].astype(str).map(mapping).fillna(prior).values

    X_df = pd.concat([X_ohe.reset_index(drop=True), X_num, X_te.reset_index(drop=True)], axis=1)
    X_scaled = scaler.transform(X_df.values)
    return X_scaled


def load_artifacts() -> Dict:
    path = ARTIFACTS_DIR / 'preprocess_artifacts.joblib'
    if path.exists():
        return joblib.load(path)
    return {}
