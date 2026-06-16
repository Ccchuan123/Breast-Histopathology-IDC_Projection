"""Stage 3: fine-tune the full ResNet18 model."""

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
from src.data_utils import make_data_loader, print_split_summary
from src.experiment_utils import ensure_output_dirs, get_device, load_experiment_split, print_experiment_config
from src.models.deep_models import count_parameters, create_resnet18_full
from src.train_utils import compute_metrics, evaluate_model, predict_model, print_metrics, save_model_results, train_one_epoch
from src.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_curves


def main():
    ensure_output_dirs()
    device = get_device()
    print_experiment_config("Stage 3: ResNet18 fine-tuning", device)

    df = load_experiment_split()
    print_split_summary(df)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_loader = make_data_loader(train_df, shuffle=True, augment=True)
    val_loader = make_data_loader(val_df, shuffle=False, augment=False)
    test_loader = make_data_loader(test_df, shuffle=False, augment=False)

    model = create_resnet18_full(num_classes=2, freeze_backbone=False).to(device)
    total, trainable = count_parameters(model)
    print(f"Model parameters: {total:,}; trainable: {trainable:,}")

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
