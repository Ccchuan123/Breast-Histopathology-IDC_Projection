"""
阶段一：传统机器学习基线模型 (Stage 1: Baseline ML Models)

实验设计：
    1. 使用预训练的 ResNet18 提取 512 维图像特征
    2. 将特征喂给三个经典 ML 模型：
       - 逻辑回归 (Logistic Regression) -- 最简单
       - 随机森林 (Random Forest) -- 集成学习
       - XGBoost -- 梯度提升树
    3. 在测试集上评估并保存结果

    用ML 基线和后续深度学习的结果进行对比参照和结果分析
"""

import sys
from pathlib import Path

# 将项目根目录加入路径，方便 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import SEED, FEATURE_CACHE, FIGURE_DIR, MODEL_DIR
from src.data_utils import collect_samples, patient_level_split, make_data_loader, print_split_summary
from src.feature_extraction import get_feature_extractor, extract_features
from src.models.baseline_ml import (
    create_logistic_regression,
    create_random_forest,
    create_xgboost,
    train_ml_model,
    predict_ml_model,
    save_ml_models,
)
from src.train_utils import compute_metrics, print_metrics, save_model_results
from src.visualization import plot_confusion_matrix, plot_roc_curve


def main():
    print("\n")
    print("  阶段一：传统机器学习基线模型")

    # 1. 准备数据
    print("\n 加载数据...")
    df = patient_level_split(collect_samples(
        Path(__file__).resolve().parent.parent / "data" / "IDC_regular_ps50_idx5"
    ))
    print_split_summary(df)

    # 2. 提取特征
    feature_extractor = get_feature_extractor(device)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        loaders[split_name] = make_data_loader(split_df, shuffle=False)

    feature_data = {}
    for split_name, loader in loaders.items():
        x, y = extract_features(feature_extractor, loader, device, desc=f"Feature {split_name}")
        feature_data[split_name] = (x.numpy(), y.numpy())

    X_train, y_train = feature_data["train"]
    X_val, y_val = feature_data["val"]
    X_test, y_test = feature_data["test"]

    print(f"\n特征维度: {X_train.shape[1]} (ResNet18 输出)")
    print(f"训练样本: {X_train.shape[0]:,}  验证样本: {X_val.shape[0]:,}  测试样本: {X_test.shape[0]:,}")

    # 3. 特征标准化（逻辑回归需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练三个基线模型
    print("\n 训练传统机器学习模型")

    models = {}
    results = {}  # 格式: {模型名: (metrics, y_pred, y_prob)}

    # 4.1 逻辑回归
    print("\n1  逻辑回归 (Logistic Regression)")
    lr = train_ml_model(create_logistic_regression(), X_train_scaled, y_train)
    y_pred_lr, y_prob_lr = predict_ml_model(lr, X_test_scaled)
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr)
    print_metrics(metrics_lr, "Logistic Regression - Test")
    models["Logistic Regression"] = lr
    results["Logistic Regression"] = (metrics_lr, y_pred_lr, y_prob_lr)

    # 4.2 随机森林
    print("\n2  随机森林 (Random Forest)")
    rf = train_ml_model(create_random_forest(), X_train, y_train)
    y_pred_rf, y_prob_rf = predict_ml_model(rf, X_test)
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf)
    print_metrics(metrics_rf, "Random Forest - Test")
    models["Random Forest"] = rf
    results["Random Forest"] = (metrics_rf, y_pred_rf, y_prob_rf)

    # 4.3 XGBoost
    print("\n3  XGBoost")
    xgb_model = create_xgboost()
    if xgb_model is not None:
        xgb_trained = train_ml_model(xgb_model, X_train, y_train, X_val, y_val)
        y_pred_xgb, y_prob_xgb = predict_ml_model(xgb_trained, X_test)
        metrics_xgb = compute_metrics(y_test, y_pred_xgb, y_prob_xgb)
        print_metrics(metrics_xgb, "XGBoost - Test")
        models["XGBoost"] = xgb_trained
        results["XGBoost"] = (metrics_xgb, y_pred_xgb, y_prob_xgb)

    # 5. 保存模型和结果
    save_ml_models(models)

    for name, (metrics, y_pred_model, y_prob_model) in results.items():
        save_model_results(name, metrics)
        plot_confusion_matrix(y_test, y_pred_model,
                              title=f"Confusion Matrix - {name}")
        if "auc" in metrics:
            plot_roc_curve(y_test, y_prob_model, model_name=name)

    print(" 阶段一完成")

if __name__ == "__main__":
    main()
