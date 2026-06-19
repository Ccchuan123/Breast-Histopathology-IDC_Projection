"""Stage 1：传统机器学习基线模型。

本阶段先使用预训练 ResNet18 提取 512 维特征，再训练 Logistic Regression、
Random Forest 和 XGBoost。它为后续深度学习模型提供 baseline 对照。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import StandardScaler

from tool.data_utils import make_data_loader, print_split_summary
from tool.experiment_utils import ensure_output_dirs, get_device, load_experiment_split, print_experiment_config
from tool.feature_extraction import get_feature_extractor, load_or_extract_features
from tool.models.baseline_ml import (
    create_logistic_regression,
    create_random_forest,
    create_xgboost,
    predict_ml_model,
    save_ml_models,
    save_scaler,
    train_ml_model,
)
from tool.train_utils import compute_metrics, print_metrics, save_model_results
from tool.visualization import plot_confusion_matrix, plot_roc_curve


def main():
    ensure_output_dirs()
    device = get_device()
    print_experiment_config("Stage 1: Traditional ML baselines", device)

    # 加载固定 patient-level split，确保四个阶段使用同一训练/验证/测试划分。
    df = load_experiment_split()
    print_split_summary(df)

    # 构建 ResNet18 特征提取器，并为 train / val / test 构建 DataLoader。
    feature_extractor = get_feature_extractor(device)
    loaders = {
        split: make_data_loader(df[df["split"] == split], shuffle=False)
        for split in ["train", "val", "test"]
    }

    feature_data = load_or_extract_features(feature_extractor, loaders, device)
    X_train, y_train = feature_data["train"][0].numpy(), feature_data["train"][1].numpy()
    X_val, y_val = feature_data["val"][0].numpy(), feature_data["val"][1].numpy()
    X_test, y_test = feature_data["test"][0].numpy(), feature_data["test"][1].numpy()

    # StandardScaler 只能在训练集 fit，验证集和测试集只 transform，避免数据泄漏。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    save_scaler(scaler)

    models = {}
    results = {}

    # 训练三个传统机器学习 baseline，并在同一测试集上计算指标。
    lr = train_ml_model(create_logistic_regression(), X_train_scaled, y_train)
    y_pred, y_prob = predict_ml_model(lr, X_test_scaled)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, "Logistic Regression - Test")
    models["Logistic Regression"] = lr
    results["Logistic Regression"] = (metrics, y_pred, y_prob)

    rf = train_ml_model(create_random_forest(), X_train, y_train)
    y_pred, y_prob = predict_ml_model(rf, X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, "Random Forest - Test")
    models["Random Forest"] = rf
    results["Random Forest"] = (metrics, y_pred, y_prob)

    xgb_model = create_xgboost(y_train)
    if xgb_model is not None:
        xgb_trained = train_ml_model(xgb_model, X_train, y_train, X_val, y_val)
        y_pred, y_prob = predict_ml_model(xgb_trained, X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics, "XGBoost - Test")
        models["XGBoost"] = xgb_trained
        results["XGBoost"] = (metrics, y_pred, y_prob)

    save_ml_models(models)

    # 保存指标、混淆矩阵和单模型 ROC 曲线。
    for name, (metrics, y_pred, y_prob) in results.items():
        save_model_results(name, metrics)
        plot_confusion_matrix(y_test, y_pred, title=name)
        plot_roc_curve(y_test, y_prob, model_name=name)

    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
