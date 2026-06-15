"""
==============================================================================
阶段二：ResNet18 冻结骨干 + FC 分类器 (Stage 2: Frozen Backbone)
==============================================================================

本阶段复现了搭档的基础方法：
    - 使用 ImageNet 预训练的 ResNet18 作为特征提取器
    - 冻结骨干网络（卷积层参数不更新）
    - 只训练最后加的一个全连接层 (512→2)
    - 在特征级上训练，速度很快，适合 CPU

为什么这样做？
    预训练的 ResNet18 已经学会了识别边缘、纹理等通用视觉特征。
    我们只需要教会它"IDC 阳性和阴性的区别是什么" 
    冻结骨干的好处是：
    1. 训练快（只更新几千个参数）
    2. 不容易过拟合
    3. CPU 也能跑

与阶段一的区别：
    阶段一：ResNet18 提特征 → sklearn 分类器
    阶段二：ResNet18 提特征 → PyTorch FC 层（端到端可微，能用交叉熵损失直接优化）
==============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from config import (
    SEED, NUM_EPOCHS_FROZEN, LEARNING_RATE_FC, WEIGHT_DECAY,
    FEATURE_DIM, BEST_MODEL_STAGE2, HISTORY_CSV, FIGURE_DIR,
)
from src.data_utils import (
    collect_samples, patient_level_split, make_data_loader, print_split_summary,
)
from src.feature_extraction import (
    get_feature_extractor, extract_features, make_feature_loader,
)
from src.models.deep_models import create_feature_classifier, count_parameters
from src.train_utils import (
    train_classifier_on_features, predict_model, compute_metrics,
    print_metrics, save_model_results,
)
from src.visualization import (
    plot_class_distribution, plot_training_curves,
    plot_confusion_matrix, plot_roc_curve,
)


def main():
    print("\n")
    print("  阶段二：ResNet18 冻结骨干 + FC 分类器")

    # ---- 1. 数据准备 ----
    print("\n 加载并划分数据...")
    df = patient_level_split(collect_samples(
        Path(__file__).resolve().parent.parent / "data" / "IDC_regular_ps50_idx5"
    ))
    print_split_summary(df)
    plot_class_distribution(df)

    # ---- 2. 特征提取 ----
    print("\n 使用预训练 ResNet18 提取特征（骨干冻结，仅此一次）...")
    feature_extractor = get_feature_extractor(device)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        loaders[split_name] = make_data_loader(split_df, shuffle=False)

    feature_data = {}
    for split_name, loader in loaders.items():
        x, y = extract_features(feature_extractor, loader, device, desc=f"Extract {split_name}")
        feature_data[split_name] = (x, y)

    train_feat_loader = make_feature_loader(*feature_data["train"], shuffle=True)
    val_feat_loader = make_feature_loader(*feature_data["val"], shuffle=False)
    test_feat_loader = make_feature_loader(*feature_data["test"], shuffle=False)

    print(f"特征维度: {feature_data['train'][0].shape[1]}")
    print(f"训练特征数: {len(feature_data['train'][0]):,}")
    print(f"验证特征数: {len(feature_data['val'][0]):,}")
    print(f"测试特征数: {len(feature_data['test'][0]):,}")

    # ---- 3. 构建分类器 ----
    print("\n  构建分类器...")
    classifier = create_feature_classifier(feature_dim=FEATURE_DIM, num_classes=2)
    classifier = classifier.to(device)

    total, trainable = count_parameters(classifier)
    print(f"  分类器参数量: {total:,} ")

    # 类别权重（处理不平衡）
    train_labels = feature_data["train"][1]
    class_counts = train_labels.bincount(minlength=2).float()
    class_weights = len(train_labels) / (2 * class_counts)
    class_weights = class_weights.to(device)
    print(f"  类别权重: IDC- = {class_weights[0]:.2f}, IDC+ = {class_weights[1]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE_FC,
                            weight_decay=WEIGHT_DECAY)

    # ---- 4. 训练 ----
    print(f"\n 开始训练 ({NUM_EPOCHS_FROZEN} epochs)...")
    history = train_classifier_on_features(
        classifier, train_feat_loader, val_feat_loader,
        criterion, optimizer, device,
        num_epochs=NUM_EPOCHS_FROZEN,
        save_path=BEST_MODEL_STAGE2,
    )

    # 保存训练历史
    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

    # 绘制训练曲线
    plot_training_curves(history, stage_name="Stage2 Frozen")

    # ---- 5. 测试集评估 ----
    print("\n 测试集评估...")
    y_true, y_pred, y_prob = predict_model(classifier, test_feat_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Stage2 ResNet18 Frozen - Test")

    save_model_results("ResNet18 Frozen", metrics)
    plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix - ResNet18 Frozen")
    plot_roc_curve(y_true, y_prob, model_name="ResNet18 Frozen")

    # ---- 6. 小结 ----
    print("\n")
    print(" 阶段二完成")
    print(f"测试集 Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
