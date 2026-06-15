"""
==============================================================================
传统机器学习基线模型 (Baseline ML Models)
包含：逻辑回归 (Logistic Regression)、随机森林 (Random Forest)、XGBoost

这些模型用于回答：用传统方法做 IDC 分类能达到什么性能？
为后续深度学习模型提供性能参考基线 (baseline)。
==============================================================================
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import LR_PARAMS, RF_PARAMS, XGB_PARAMS, BEST_ML_MODELS, SEED


def create_logistic_regression() -> LogisticRegression:
    """创建逻辑回归分类器。

    逻辑回归是最简单的线性分类器，它学习一个线性决策边界。
    优点：训练快、可解释性强
    缺点：只能学习线性关系
    """
    return LogisticRegression(**LR_PARAMS)


def create_random_forest() -> RandomForestClassifier:
    """创建随机森林分类器。

    随机森林通过集成多棵决策树来提升泛化能力。
    优点：能捕捉非线性关系、对超参数不太敏感、自带特征重要性
    缺点：模型较大、推理较慢
    """
    return RandomForestClassifier(**RF_PARAMS)


def create_xgboost():
    """创建 XGBoost 分类器。

    XGBoost 是目前 Kaggle 竞赛中最常用的模型之一。
    它通过梯度提升 (Gradient Boosting) 逐步优化残差。
    优点：性能强、正则化好、不易过拟合
    缺点：超参数较多需要调优
    """
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(**XGB_PARAMS)
    except ImportError:
        print("[WARN]  XGBoost 未安装，正在跳过。安装方法: pip install xgboost")
        return None


def train_ml_model(model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None):
    """训练机器学习模型。返回训练好的模型。"""
    if model is None:
        return None

    model_name = type(model).__name__
    if hasattr(model, "fit"):
        # 某些模型（如 XGBoost）支持 early_stopping
        if hasattr(model, "early_stopping_rounds"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        print(f"  [OK] {model_name} 训练完成")
    return model


def predict_ml_model(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ML 模型预测。返回 (预测标签, 正类概率)。"""
    if model is None:
        return np.array([]), np.array([])
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def save_ml_models(models: dict):
    """保存训练好的 ML 模型。"""
    if models:
        with open(BEST_ML_MODELS, "wb") as f:
            pickle.dump(models, f)
        print(f"[Saved] ML 模型已保存: {BEST_ML_MODELS}")


def load_ml_models() -> dict:
    """加载已保存的 ML 模型。"""
    if BEST_ML_MODELS.exists():
        with open(BEST_ML_MODELS, "rb") as f:
            return pickle.load(f)
    return {}
