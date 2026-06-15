"""
==============================================================================
训练与评估工具模块 (Training Utilities)
包含：训练循环、评估函数、指标计算、结果汇总
==============================================================================
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BEST_MODEL_STAGE2, BEST_MODEL_STAGE3


# ============================== 指标计算 ==============================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    计算二分类全部评价指标。
    返回: accuracy, precision, recall, f1_score, auc, confusion_matrix
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Metrics"):
    """格式化打印指标。"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  AUC:        {metrics['auc']:.4f}")
    print(f"  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"{'='*50}")


# ============================== PyTorch 训练循环 ==============================

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    """训练一个 epoch。返回 (loss, accuracy)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False, bar_format="{desc}: {n_fmt}/{total_fmt} ({percentage:.0f}%)"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """评估模型。返回 (loss, accuracy)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False, bar_format="{desc}: {n_fmt}/{total_fmt} ({percentage:.0f}%)"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """模型预测，返回 (true_labels, predicted_labels, positive_probs)。"""
    model.eval()
    labels_all, preds_all, probs_all = [], [], []

    for images, labels in tqdm(loader, desc="Predicting", leave=False, bar_format="{desc}: {n_fmt}/{total_fmt} ({percentage:.0f}%)"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]

        labels_all.extend(labels.numpy())
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

    return np.array(labels_all), np.array(preds_all), np.array(probs_all)


# ============================== 特征级训练 (Stage2 用) ==============================

def train_classifier_on_features(
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    save_path: str = None,
) -> dict:
    """
    在预提取的特征上训练 FC 分类器（Stage2）。

    特征已经固定，只训练分类头，适合 CPU 环境。
    返回训练历史记录。
    """
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # 训练
        classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += features.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证
        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = classifier(features)
                loss = criterion(logits, labels)

                val_loss += loss.item() * features.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += features.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # 记录
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in classifier.state_dict().items()}

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 恢复最佳权重
    if best_state is not None:
        classifier.load_state_dict(best_state)
        if save_path:
            torch.save(classifier.state_dict(), save_path)
            print(f"[Saved] 最佳模型已保存: {save_path} (Val Acc: {best_val_acc:.4f})")

    return history


# ============================== 结果汇总 ==============================

def save_model_results(model_name: str, metrics: dict):
    """将单个模型的测试结果追加到汇总 CSV。"""
    row = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "auc": metrics["auc"],
    }
    df = pd.DataFrame([row])

    from config import RESULTS_CSV
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        # 避免重复记录
        existing = existing[existing["model"] != model_name]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f" 结果已追加至: {RESULTS_CSV}")
    return df
