"""Stage 2: ResNet18 frozen backbone features plus trainable FC classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from config import (
    BEST_MODEL_STAGE2,
    FEATURE_DIM,
    HISTORY_STAGE2_CSV,
    LEARNING_RATE_FC,
    NUM_EPOCHS_FROZEN,
    WEIGHT_DECAY,
)
from src.data_utils import make_data_loader, print_split_summary
from src.experiment_utils import ensure_output_dirs, get_device, load_experiment_split, print_experiment_config
from src.feature_extraction import extract_features, get_feature_extractor, make_feature_loader
from src.models.deep_models import count_parameters, create_feature_classifier
from src.train_utils import compute_metrics, predict_model, print_metrics, save_model_results, train_classifier_on_features
from src.visualization import plot_class_distribution, plot_confusion_matrix, plot_roc_curve, plot_training_curves


def main():
    ensure_output_dirs()
    device = get_device()
    print_experiment_config("Stage 2: ResNet18 frozen backbone", device)

    df = load_experiment_split()
    print_split_summary(df)
    plot_class_distribution(df)

    feature_extractor = get_feature_extractor(device)
    loaders = {
        split: make_data_loader(df[df["split"] == split], shuffle=False)
        for split in ["train", "val", "test"]
    }
    feature_data = {}
    for split, loader in loaders.items():
        feature_data[split] = extract_features(feature_extractor, loader, device, desc=f"Extract {split}")

    train_feat_loader = make_feature_loader(*feature_data["train"], shuffle=True)
    val_feat_loader = make_feature_loader(*feature_data["val"], shuffle=False)
    test_feat_loader = make_feature_loader(*feature_data["test"], shuffle=False)

    classifier = create_feature_classifier(feature_dim=FEATURE_DIM, num_classes=2).to(device)
    total, trainable = count_parameters(classifier)
    print(f"Classifier parameters: {total:,}; trainable: {trainable:,}")

    train_labels = feature_data["train"][1]
    class_counts = train_labels.bincount(minlength=2).float().clamp_min(1)
    class_weights = (len(train_labels) / (2 * class_counts)).to(device)
    print(f"Class weights: IDC-={class_weights[0]:.3f}, IDC+={class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE_FC, weight_decay=WEIGHT_DECAY)

    history = train_classifier_on_features(
        classifier,
        train_feat_loader,
        val_feat_loader,
        criterion,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS_FROZEN,
        save_path=BEST_MODEL_STAGE2,
    )

    pd.DataFrame(history).to_csv(HISTORY_STAGE2_CSV, index=False)
    plot_training_curves(history, stage_name="Stage2", output_name="training_curve_stage2.png")

    y_true, y_pred, y_prob = predict_model(classifier, test_feat_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "ResNet18 Frozen - Test")
    save_model_results("ResNet18 Frozen", metrics)
    plot_confusion_matrix(y_true, y_pred, title="ResNet18 Frozen")
    plot_roc_curve(y_true, y_prob, model_name="ResNet18 Frozen")

    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
