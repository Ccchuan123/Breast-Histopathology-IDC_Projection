"""Training, prediction, metrics, and result-summary utilities."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": cm,
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    metrics["auc"] = metrics["roc_auc"]
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Metrics"):
    print(f"\n{'=' * 58}")
    print(f"  {title}")
    print(f"{'=' * 58}")
    for key in [
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    ]:
        print(f"  {key:18s}: {metrics[key]:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")
    print(f"{'=' * 58}")


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy())
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        probs_all.extend(probs.cpu().numpy())
    return np.array(labels_all), np.array(preds_all), np.array(probs_all)


def train_classifier_on_features(
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    device,
    num_epochs: int = 10,
    save_path=None,
) -> dict:
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
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
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += features.size(0)

        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = classifier(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += features.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

        print(
            f"Epoch {epoch:2d}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    if best_state is not None:
        classifier.load_state_dict(best_state)
        if save_path:
            torch.save(classifier.state_dict(), save_path)
            print(f"[Saved] Best model: {save_path} (Val Acc: {best_val_acc:.4f})")
    return history


def save_model_results(model_name: str, metrics: dict):
    row = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }
    df = pd.DataFrame([row])
    from config import RESULTS_CSV

    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        existing = existing[existing["model"] != model_name]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[Saved] Results summary: {RESULTS_CSV}")
    return df
