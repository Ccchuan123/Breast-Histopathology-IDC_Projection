"""可视化工具函数。

本模块负责保存混淆矩阵、ROC 曲线、PR 曲线、训练曲线、模型对比图，
以及 false positive / false negative 错误样本图。
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from config import ERROR_SAMPLE_DIR, FIGURE_DIR

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    """绘制 train / val / test 中 class0 和 class1 的数量分布。"""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="split", hue="label", order=["train", "val", "test"])
    plt.title("Class Distribution by Dataset Split", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Images")
    plt.legend(title="Class", labels=["class0 (IDC-)", "class1 (IDC+)"])
    plt.tight_layout()
    if save:
        path = FIGURE_DIR / "class_distribution.png"
        plt.savefig(path, dpi=200)
        print(f"[Saved] {path}")
    plt.close()


def plot_training_curves(history: dict, stage_name: str = "", save: bool = True, output_name: str | None = None):
    """绘制训练曲线。

    左图为 loss 曲线，右图为 accuracy 曲线，用于观察是否收敛或过拟合。
    """
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title(f"Loss Curve {stage_name}".strip(), fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Validation", linewidth=2)
    axes[1].set_title(f"Accuracy Curve {stage_name}".strip(), fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()

    path = FIGURE_DIR / (output_name or f"training_curve_{_safe_name(stage_name)}.png")
    if save:
        plt.savefig(path, dpi=200)
        print(f"[Saved] {path}")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix", save: bool = True):
    """绘制混淆矩阵。

    混淆矩阵展示 TN、FP、FN、TP。医学任务中 FN（漏检阳性）通常需要重点关注。
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred IDC-", "Pred IDC+"],
        yticklabels=["True IDC-", "True IDC+"],
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURE_DIR / f"confusion_matrix_{_safe_name(title.replace('Confusion Matrix', ''))}.png"
    if save:
        plt.savefig(path, dpi=200)
        print(f"[Saved] {path}")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "", save: bool = True):
    """绘制单个模型的 ROC 曲线。

    ROC 曲线反映不同阈值下 sensitivity 和 false positive rate 的权衡。
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = FIGURE_DIR / f"roc_curve_{_safe_name(model_name)}.png"
    if save:
        plt.savefig(path, dpi=200)
        print(f"[Saved] {path}")
    plt.close()


def plot_combined_roc_curves(curve_data: dict[str, tuple[np.ndarray, np.ndarray]], save: bool = True):
    """绘制多个模型的 ROC 曲线到同一张图，便于横向比较。"""
    plt.figure(figsize=(7, 6))
    for name, (y_true, y_prob) in curve_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title("ROC Curves", fontweight="bold")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save:
        path = FIGURE_DIR / "roc_curves.png"
        plt.savefig(path, dpi=200)
        print(f"[Saved] {path}")
    plt.close()


def plot_combined_pr_curves(curve_data: dict[str, tuple[np.ndarray, np.ndarray]], save: bool = True):
    """绘制多个模型的 PR 曲线。

    PR 曲线展示 precision 和 recall 的关系，在阳性样本较少时比 ROC 更敏感。
    """
    plt.figure(figsize=(7, 6))
    for name, (y_true, y_prob) in curve_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AUC={auc(recall, precision):.3f})")
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves", fontweight="bold")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    if save:
        path = FIGURE_DIR / "pr_curves.png"
        plt.savefig(path, dpi=200)
        print(f"[Saved] {path}")
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, save: bool = True):
    """根据 results_summary.csv 绘制不同模型的指标对比柱状图。"""
    metrics = ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc", "pr_auc", "balanced_accuracy"]
    available = [m for m in metrics if m in results_df.columns]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(results_df))
    width = 0.8 / max(len(available), 1)
    for i, metric in enumerate(available):
        values = results_df[metric].to_numpy()
        ax.bar(x + i * width, values, width, label=metric)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(results_df["model"].tolist(), rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURE_DIR / "model_comparison.png"
    if save:
        plt.savefig(path, dpi=200)
        print(f"[Saved] {path}")
    plt.close(fig)


def plot_error_samples(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, max_images: int = 16):
    """保存 false positive 和 false negative 图像网格。

    false positive：真实阴性但预测为阳性。
    false negative：真实阳性但预测为阴性，是医学筛查任务中尤其需要关注的错误类型。
    """
    def _grid(indices: np.ndarray, path: Path, title: str):
        n = min(len(indices), max_images)
        if n == 0:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, f"No {title}", ha="center", va="center")
            ax.axis("off")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            return
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, idx in zip(axes, indices[:n]):
            row = test_df.iloc[int(idx)]
            ax.imshow(Image.open(row["path"]).convert("RGB"))
            ax.set_title(f"patient {row['patient_id']}", fontsize=8)
            ax.axis("off")
        fig.suptitle(title, fontweight="bold")
        plt.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    ERROR_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    _grid(fp_idx, ERROR_SAMPLE_DIR / "false_positive_examples.png", "False Positive Examples")
    _grid(fn_idx, ERROR_SAMPLE_DIR / "false_negative_examples.png", "False Negative Examples")
    print(f"[Saved] Error samples: {ERROR_SAMPLE_DIR}")
