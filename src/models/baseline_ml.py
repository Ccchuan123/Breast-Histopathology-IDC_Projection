"""传统机器学习 baseline 模型。

Stage1 先使用预训练 ResNet18 提取图像特征，再把这些特征交给传统机器学习模型：
- Logistic Regression：线性分类器，训练快、可解释性较好。
- Random Forest：树模型集成，能建模一定非线性关系。
- XGBoost：梯度提升树，常用于结构化特征分类任务。

Logistic Regression 需要 StandardScaler。scaler 只能在训练集 fit，
验证集和测试集只能 transform，避免测试集信息泄漏。
"""

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import BEST_ML_MODELS, LR_PARAMS, RF_PARAMS, STAGE1_SCALER, XGB_PARAMS


def create_logistic_regression() -> LogisticRegression:
    """创建 Logistic Regression，并使用 balanced class weight 缓解类别不平衡。"""
    return LogisticRegression(**LR_PARAMS)


def create_random_forest() -> RandomForestClassifier:
    """创建 Random Forest，并使用 balanced class weight 缓解类别不平衡。"""
    return RandomForestClassifier(**RF_PARAMS)


def create_xgboost(y_train: np.ndarray | None = None):
    """创建 XGBoost 分类器。

    XGBoost 不使用 sklearn 的 class_weight 参数，因此根据训练集负/正样本比例
    设置 scale_pos_weight，降低类别不平衡带来的偏差。
    """
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
    """训练传统机器学习模型。"""
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
    """返回预测标签和阳性类别概率。"""
    if model is None:
        return np.array([]), np.array([])
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def save_ml_models(models: dict):
    """保存 Stage1 训练好的传统 ML 模型。"""
    if models:
        with open(BEST_ML_MODELS, "wb") as f:
            pickle.dump(models, f)
        print(f"[Saved] ML models: {BEST_ML_MODELS}")


def load_ml_models() -> dict:
    """加载 Stage1 保存的传统 ML 模型，供 Stage4 比较和集成使用。"""
    if BEST_ML_MODELS.exists():
        with open(BEST_ML_MODELS, "rb") as f:
            return pickle.load(f)
    return {}


def save_scaler(scaler):
    """保存只在训练集 fit 过的 StandardScaler。"""
    with open(STAGE1_SCALER, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Saved] Stage1 scaler: {STAGE1_SCALER}")


def load_scaler():
    """加载 Stage1 保存的 StandardScaler。

    Stage4 必须复用同一个 scaler，不能在测试集重新 fit_transform。
    """
    if not STAGE1_SCALER.exists():
        raise FileNotFoundError(
            f"Missing Stage1 scaler: {STAGE1_SCALER}. Run stage1_baseline_ml.py first."
        )
    with open(STAGE1_SCALER, "rb") as f:
        return pickle.load(f)
