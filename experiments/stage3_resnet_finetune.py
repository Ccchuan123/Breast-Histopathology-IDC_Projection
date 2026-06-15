"""
==============================================================================
阶段三：ResNet18 微调 (Stage 3: Fine-tuning ResNet18)
==============================================================================

本阶段在阶段二的基础上更进一步：
    - 不再冻结骨干网络，而是解冻全部参数
    - 使用更小的学习率（1e-4 vs 1e-3），防止破坏预训练权重
    - 加入数据增强（翻转、旋转），提升泛化能力
    - 端到端训练整个 ResNet18

为什么叫"微调"（Fine-tuning）而不是"重新训练"？
    预训练的 ResNet18 在 ImageNet 上见过数百万张自然图像，学到了丰富的
    视觉特征。我们要做的是在这些特征的基础上"微调"，让它适应病理切片
    图像----而不是从零开始学。就像一个有绘画基础的人学素描，比完全没学
    过的人上手快得多。

注意：
    本阶段计算量较大。如果在 CPU 上运行，可能需要较长时间。
    可以考虑减少 epoch 数，或使用小批量数据。
==============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter

from config import (
    SEED, NUM_EPOCHS_FINETUNE, LEARNING_RATE_FINETUNE, WEIGHT_DECAY,
    BATCH_SIZE, BEST_MODEL_STAGE3, FIGURE_DIR,
)
from src.data_utils import (
    collect_samples, patient_level_split, make_data_loader, print_split_summary,
)
from src.models.deep_models import create_resnet18_full, count_parameters
from src.train_utils import (
    train_one_epoch, evaluate_model, predict_model, compute_metrics,
    print_metrics, save_model_results,
)
from src.visualization import (
    plot_class_distribution, plot_training_curves,
    plot_confusion_matrix, plot_roc_curve,
)


def main():
    print("\n")
    print("  阶段三：ResNet18 微调 (Fine-tuning)")

    print("\n 加载并划分数据...")
    df = patient_level_split(collect_samples(
        Path(__file__).resolve().parent.parent / "data" / "IDC_regular_ps50_idx5"
    ))
    print_split_summary(df)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # 微调时使用数据增强
    train_loader = make_data_loader(train_df, shuffle=True, augment=True)
    val_loader = make_data_loader(val_df, shuffle=False, augment=False)
    test_loader = make_data_loader(test_df, shuffle=False, augment=False)

    print(f"训练批次数: {len(train_loader)} | 验证批次数: {len(val_loader)}")

    # ---- 2. 构建模型 ----
    print("\n  构建 ResNet18（不冻结骨干）...")
    model = create_resnet18_full(num_classes=2, freeze_backbone=False)
    model = model.to(device)

    total, trainable = count_parameters(model)
    print(f"  总参数量:    {total:,}")
    print(f"  可训练参数:  {trainable:,} ")

    # 类别权重
    train_labels = train_df["label"].tolist()
    class_counts = Counter(train_labels)
    class_weights = torch.tensor([
        len(train_labels) / (2 * class_counts[0]),
        len(train_labels) / (2 * class_counts[1]),
    ], dtype=torch.float32).to(device)
    print(f"  类别权重: IDC- = {class_weights[0]:.2f}, IDC+ = {class_weights[1]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE,
                            weight_decay=WEIGHT_DECAY)

    # 学习率调度：当验证损失不再下降时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ---- 3. 训练 ----
    print(f"\n 开始微调训练 ({NUM_EPOCHS_FINETUNE} epochs)...")
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_model(
            model, val_loader, criterion, device
        )

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_STAGE3)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS_FINETUNE} | "
              f"LR: {current_lr:.1e} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load(BEST_MODEL_STAGE3, map_location=device))
    print(f"[Saved] 最佳模型已保存: {BEST_MODEL_STAGE3} (Val Acc: {best_val_acc:.4f})")

    # 保存训练历史
    pd.DataFrame(history).to_csv(
        Path(__file__).resolve().parent.parent / "outputs" / "training_history_stage3.csv",
        index=False,
    )

    # 绘制训练曲线
    plot_training_curves(history, stage_name="Stage3 Finetune")

    # ---- 4. 测试集评估 ----
    print("\n 测试集评估...")
    y_true, y_pred, y_prob = predict_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Stage3 ResNet18 Finetune - Test")

    save_model_results("ResNet18 Finetune", metrics)
    plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix - ResNet18 Finetune")
    plot_roc_curve(y_true, y_prob, model_name="ResNet18 Finetune")

    # ---- 5. 小结 ----
    print("\n")
    print(" 阶段三完成")
    print("ResNet18 微调完成。现在对比阶段一和阶段二的结果：")
    print("  阶段一 (传统ML):   建立基线")
    print("  阶段二 (冻结骨干): 快速训练，只更新 FC 层")
    print("  阶段三 (微调):     端到端训练，充分释放模型潜力")


if __name__ == "__main__":
    main()
