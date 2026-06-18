"""Stage 3：微调 ResNet18。

本阶段不再冻结 ResNet18 主干，而是使用较小学习率端到端微调整个模型。
它计算量更大，但可以让预训练特征进一步适应乳腺病理图像。
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BEST_MODEL_STAGE3,
    HISTORY_STAGE3_CSV,
    LEARNING_RATE_FINETUNE,
    NUM_EPOCHS_FINETUNE,
    WEIGHT_DECAY,
)
from tool.data_utils import make_data_loader, print_split_summary
from tool.experiment_utils import ensure_output_dirs, get_device, load_experiment_split, print_experiment_config
from tool.models.deep_models import count_parameters, create_resnet18_full
from tool.train_utils import compute_metrics, evaluate_model, predict_model, print_metrics, save_model_results, train_one_epoch
from tool.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_curves


def main():
    ensure_output_dirs()
    device = get_device()
    print_experiment_config("Stage 3: ResNet18 fine-tuning", device)

    # 加载固定 patient-level split，确保与 Stage1/Stage2 使用相同数据划分。
    df = load_experiment_split()
    print_split_summary(df)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # 构建图像 DataLoader。训练集启用轻量数据增强，验证/测试不增强。
    train_loader = make_data_loader(train_df, shuffle=True, augment=True)
    val_loader = make_data_loader(val_df, shuffle=False, augment=False)
    test_loader = make_data_loader(test_df, shuffle=False, augment=False)

    # 构建可训练的 ResNet18，并替换最后分类层为二分类输出。
    model = create_resnet18_full(num_classes=2, freeze_backbone=False).to(device)
    total, trainable = count_parameters(model)
    print(f"Model parameters: {total:,}; trainable: {trainable:,}")

    # 根据训练集类别比例设置 class-weighted loss。
    train_labels = train_df["label"].tolist()
    counts = Counter(train_labels)
    class_weights = torch.tensor(
        [
            len(train_labels) / (2 * max(counts.get(0, 0), 1)),
            len(train_labels) / (2 * max(counts.get(1, 0), 1)),
        ],
        dtype=torch.float32,
        device=device,
    )
    print(f"Class weights: IDC-={class_weights[0]:.3f}, IDC+={class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    # 正式训练循环：每轮训练、验证，并保存验证集表现最好的权重。
    for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_STAGE3)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS_FINETUNE} | LR={optimizer.param_groups[0]['lr']:.1e} | "
            f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}"
        )

    # 加载最佳权重后，在测试集上计算指标并保存图表。
    model.load_state_dict(torch.load(BEST_MODEL_STAGE3, map_location=device))
    pd.DataFrame(history).to_csv(HISTORY_STAGE3_CSV, index=False)
    plot_training_curves(history, stage_name="Stage3", output_name="training_curve_stage3.png")

    y_true, y_pred, y_prob = predict_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "ResNet18 Finetune - Test")
    save_model_results("ResNet18 Finetune", metrics)
    plot_confusion_matrix(y_true, y_pred, title="ResNet18 Finetune")
    plot_roc_curve(y_true, y_prob, model_name="ResNet18 Finetune")

    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
