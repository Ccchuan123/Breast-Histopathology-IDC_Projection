"""Traditional machine-learning baselines."""

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import BEST_ML_MODELS, LR_PARAMS, RF_PARAMS, STAGE1_SCALER, XGB_PARAMS


def create_logistic_regression() -> LogisticRegression:
    return LogisticRegression(**LR_PARAMS)


def create_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


def create_xgboost(y_train: np.ndarray | None = None):
    try:
        import xgboost as xgb
    except ImportError:
        print("[WARN] XGBoost is not installed; skipping it.")
        return None

    params = dict(XGB_PARAMS)
    if y_train is not None:
        neg = max(int((y_train == 0).sum()), 1)
        pos = max(int((y_train == 1).sum()), 1)
        params["scale_pos_weight"] = neg / pos
    return xgb.XGBClassifier(**params)


def train_ml_model(model, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None):
    if model is None:
        return None
    fit_kwargs = {}
    if X_val is not None and y_val is not None and type(model).__name__.startswith("XGB"):
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    print(f"[OK] {type(model).__name__} trained.")
    return model


def predict_ml_model(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model is None:
        return np.array([]), np.array([])
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def save_ml_models(models: dict):
    if models:
        with open(BEST_ML_MODELS, "wb") as f:
            pickle.dump(models, f)
        print(f"[Saved] ML models: {BEST_ML_MODELS}")


def load_ml_models() -> dict:
    if BEST_ML_MODELS.exists():
        with open(BEST_ML_MODELS, "rb") as f:
            return pickle.load(f)
    return {}


def save_scaler(scaler):
    with open(STAGE1_SCALER, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Saved] Stage1 scaler: {STAGE1_SCALER}")


def load_scaler():
    if not STAGE1_SCALER.exists():
        raise FileNotFoundError(
            f"Missing Stage1 scaler: {STAGE1_SCALER}. Run stage1_baseline_ml.py first."
        )
    with open(STAGE1_SCALER, "rb") as f:
        return pickle.load(f)
