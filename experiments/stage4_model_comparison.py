"""
==============================================================================
阶段四：模型对比与集成 (Stage 4: Model Comparison & Ensemble)
==============================================================================

本阶段是项目的"大总结"：
    1. 汇总阶段一到阶段三所有模型的测试结果
    2. 可视化对比各模型的性能
    3. 尝试简单的模型集成（投票法），看看"三个臭皮匠"是否顶一个诸葛亮
    4. 输出最终总结报告

模型集成 (Ensemble) 是什么？
    多个模型对同一张图像各自给出预测，然后投票。如果大多数模型
    认为是 IDC 阳性，最终结果就是阳性。这能减少单个模型的偏差。
==============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
from collections import Counter

from config import (
    SEED, RESULTS_CSV, BEST_MODEL_STAGE2, BEST_MODEL_STAGE3, BEST_ML_MODELS,
    FEATURE_DIM, FIGURE_DIR, OUTPUT_DIR,
)
from src.data_utils import (
    collect_samples, patient_level_split, make_data_loader, print_split_summary,
)
from src.feature_extraction import get_feature_extractor, extract_features, make_feature_loader
from src.models.baseline_ml import load_ml_models, predict_ml_model
from src.models.deep_models import create_feature_classifier, create_resnet18_full
from src.train_utils import compute_metrics, print_metrics, predict_model
from src.visualization import (
    plot_model_comparison, plot_confusion_matrix, plot_roc_curve,
)


def load_stage2_model(device: torch.device) -> torch.nn.Module:
    """加载阶段二训练好的特征分类器。"""
    classifier = create_feature_classifier(feature_dim=FEATURE_DIM, num_classes=2)
    classifier = classifier.to(device).eval()
    return classifier


def load_stage3_model(device: torch.device) -> torch.nn.Module:
    """加载阶段三训练好的微调模型。"""
    model = create_resnet18_full(num_classes=2, freeze_backbone=False)
    model = model.to(device).eval()
    return model


def ensemble_vote(predictions: list[np.ndarray]) -> np.ndarray:
    """
    集成投票：多个模型的预测结果，取多数。

    Args:
        predictions: 每个模型的预测标签数组列表

    Returns:
        多数投票后的预测标签
    """
    stacked = np.column_stack(predictions)  # shape: (n_samples, n_models)
    final = []
    for row in stacked:
        # 统计每个样本的投票
        counts = Counter(row)
        # 取票数最多的类别，平票时偏向正类 (class1)
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            final.append(1)  # tie-breaking: prefer positive
        else:
            final.append(most_common[0][0])
    return np.array(final)


def ensemble_average(probabilities: list[np.ndarray]) -> np.ndarray:
    """
    集成平均：多个模型的概率输出取平均，然后 >0.5 判为正类。
    通常比硬投票更稳定。
    """
    avg_prob = np.mean(probabilities, axis=0)
    return (avg_prob > 0.5).astype(int), avg_prob


def main():
    print("\n")
    print("  阶段四：模型对比与集成")

    # ---- 1. 准备测试数据 ----
    print("\n 加载数据...")
    df = patient_level_split(collect_samples(
        Path(__file__).resolve().parent.parent / "data" / "IDC_regular_ps50_idx5"
    ))
    test_df = df[df["split"] == "test"]

    # 提取特征（供阶段二模型和 ML 模型使用）
    feature_extractor = get_feature_extractor(device)
    test_image_loader = make_data_loader(test_df, shuffle=False)
    X_test, y_test = extract_features(feature_extractor, test_image_loader, device, desc="Extract test features")
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    # 阶段三模型直接用图像
    # (这里我们只做特征级预测，阶段三用全模型预测)
    # 为了全面对比，需要分别处理

    all_predictions = []
    all_probabilities = []
    model_names = []

    # ---- 2. 加载并评估所有模型 ----
    print("\n 评估所有模型...")

    # 2a. 传统 ML 模型
    ml_models = load_ml_models()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_np)

    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name in ml_models and ml_models[name] is not None:
            y_pred, y_prob = predict_ml_model(ml_models[name],
                                              X_test_scaled if name == "Logistic Regression" else X_test_np)
            metrics = compute_metrics(y_test_np, y_pred, y_prob)
            print_metrics(metrics, f"{name} - Test")
            all_predictions.append(y_pred)
            all_probabilities.append(y_prob)
            model_names.append(name)

    # 2b. 阶段二：ResNet18 Frozen
    stage2_model = load_stage2_model(device)
    if BEST_MODEL_STAGE2.exists():
        test_feat_loader = make_feature_loader(X_test, y_test, shuffle=False)
        y_true, y_pred, y_prob = predict_model(stage2_model, test_feat_loader, device)
        metrics_s2 = compute_metrics(y_true, y_pred, y_prob)
        print_metrics(metrics_s2, "ResNet18 Frozen - Test")
        all_predictions.append(y_pred)
        all_probabilities.append(y_prob)
        model_names.append("ResNet18 Frozen")

    # 2c. 阶段三：ResNet18 Finetune
    stage3_model = load_stage3_model(device)
    if BEST_MODEL_STAGE3.exists():
        test_image_loader_eval = make_data_loader(test_df, shuffle=False)
        y_true, y_pred, y_prob = predict_model(stage3_model, test_image_loader_eval, device)
        metrics_s3 = compute_metrics(y_true, y_pred, y_prob)
        print_metrics(metrics_s3, "ResNet18 Finetune - Test")
        all_predictions.append(y_pred)
        all_probabilities.append(y_prob)
        model_names.append("ResNet18 Finetune")

    # ---- 3. 模型集成 ----
    print("\n" + "-" * 50)
    print(" 模型集成实验")
    print("-" * 50)

    ensemble_pred = ensemble_vote(all_predictions)
    metrics_vote = compute_metrics(y_test_np, ensemble_pred)
    print("\n 集成方法：硬投票 (Majority Voting)")
    print(f"   参与模型: {', '.join(model_names)}")
    print_metrics(metrics_vote, "Ensemble (Hard Voting)")

    # 软投票（概率平均）
    ensemble_pred_soft, ensemble_prob = ensemble_average(all_probabilities)
    metrics_soft = compute_metrics(y_test_np, ensemble_pred_soft, ensemble_prob)
    print("\n 集成方法：软投票 (Probability Averaging)")
    print(f"   参与模型: {', '.join(model_names)}")
    print_metrics(metrics_soft, "Ensemble (Soft Voting)")

    # 保存集成结果
    from src.train_utils import save_model_results
    save_model_results("Ensemble Hard Voting", metrics_vote)
    save_model_results("Ensemble Soft Voting", metrics_soft)

    # ---- 4. 模型对比可视化 ----
    print("\n 生成模型对比图...")
    if RESULTS_CSV.exists():
        results_df = pd.read_csv(RESULTS_CSV)
        print("\n")
        print("  所有模型结果汇总")
        print(results_df.to_string(index=False))

        if len(results_df) >= 2:
            plot_model_comparison(results_df)

        # 找出最佳模型
        best_model = results_df.loc[results_df["auc"].idxmax()]
        print(f"\n 最佳模型 (按 AUC): {best_model['model']} "
              f"(AUC={best_model['auc']:.4f}, F1={best_model['f1_score']:.4f})")

    # ---- 5. 最终总结 ----
    print("\n")
    print("  全部实验完成")
    print("")
    print("项目四个阶段概览：")
    print("  阶段一：传统 ML (LR/RF/XGBoost) -- 建立基线")
    print("  阶段二：ResNet18 冻结骨干 + FC -- 快速深度学习")
    print("  阶段三：ResNet18 微调 -- 端到端优化")
    print("  阶段四：模型对比与集成 -- 综合评估")
    print("")
    print("实验结果文件：")
    print(f"  模型对比: {RESULTS_CSV}")
    print(f"  对比图表: {FIGURE_DIR / 'model_comparison.png'}")
    print(f"  各模型图表: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
