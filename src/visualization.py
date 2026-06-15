"""
==============================================================================
可视化模块 (Visualization)
所有绘图函数集中管理：类别分布、训练曲线、混淆矩阵、ROC 曲线
==============================================================================
"""

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from config import FIGURE_DIR

# 设置中文字体（如果可用）和全局样式
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    """绘制各数据集的类别分布图。"""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="split", hue="label", order=["train", "val", "test"],
                  palette=["#3498db", "#e74c3c"])
    plt.title("Class Distribution by Dataset Split", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset Split", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.legend(title="Class", labels=["class0 (IDC-)", "class1 (IDC+)"])
    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=200)
        print(f" 类别分布图已保存: {FIGURE_DIR / 'class_distribution.png'}")
    plt.close()


def plot_training_curves(history: dict, stage_name: str = "", save: bool = True):
    """绘制训练损失和准确率曲线。"""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
    axes[0].set_title(f"Loss Curve {stage_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Validation", linewidth=2)
    axes[1].set_title(f"Accuracy Curve {stage_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    suffix = f"_{stage_name.lower().replace(' ', '_')}" if stage_name else ""
    path = FIGURE_DIR / f"training_curve{suffix}.png"
    if save:
        plt.savefig(path, dpi=200)
        print(f" 训练曲线已保存: {path}")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Confusion Matrix", save: bool = True):
    """绘制混淆矩阵热力图。"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted IDC-", "Predicted IDC+"],
                yticklabels=["Actual IDC-", "Actual IDC+"],
                annot_kws={"size": 14})
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    suffix = f"_{title.lower().replace(' ', '_')}" if title != "Confusion Matrix" else ""
    path = FIGURE_DIR / f"confusion_matrix{suffix}.png"
    if save:
        plt.savefig(path, dpi=200)
        print(f" 混淆矩阵已保存: {path}")
    plt.close()
    return cm


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str = "", save: bool = True):
    """绘制 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - {model_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    suffix = f"_{model_name.lower().replace(' ', '_')}" if model_name else ""
    path = FIGURE_DIR / f"roc_curve{suffix}.png"
    if save:
        plt.savefig(path, dpi=200)
        print(f" ROC 曲线已保存: {path}")
    plt.close()
    return auc


def plot_model_comparison(results_df: pd.DataFrame, save: bool = True):
    """绘制模型对比柱状图。"""
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
    models = results_df["model"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.15

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = results_df[metric].tolist()
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", fontsize=7)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = FIGURE_DIR / "model_comparison.png"
    if save:
        plt.savefig(path, dpi=200)
        print(f" 模型对比图已保存: {path}")
    plt.close(fig)
