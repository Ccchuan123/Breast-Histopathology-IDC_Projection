"""
==============================================================================
数据加载与预处理模块 (Data Utilities)
负责：标签提取、样本收集、病例级划分、PyTorch Dataset 定义
==============================================================================
"""

import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED


# ============================== 标签提取 ==============================

def extract_label_from_filename(image_path: Path) -> int:
    """
    从文件名中提取标签。
    例：.../10253_idx5_class0.png → 0，.../10253_idx5_class1.png → 1
    """
    match = re.search(r"class([01])", image_path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot extract class0/class1 label from filename: {image_path.name}")
    return int(match.group(1))


# ============================== 样本收集 ==============================

def collect_samples(data_dir: Path) -> pd.DataFrame:
    """
    遍历数据目录，收集所有 PNG 图像，构建元数据 DataFrame。
    DataFrame 列: path (图像路径), patient_id (病例编号), label (0/1)
    """
    image_paths = sorted(data_dir.rglob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {data_dir}")

    rows = []
    for image_path in image_paths:
        label = extract_label_from_filename(image_path)
        # 安全检查：如果文件在 0/ 或 1/ 文件夹下，验证标签一致
        if image_path.parent.name in {"0", "1"} and int(image_path.parent.name) != label:
            raise ValueError(f"Folder label and filename label mismatch: {image_path}")
        rows.append({
            "path": str(image_path),
            "patient_id": image_path.parts[-3],  # 病例 ID 在路径倒数第三层
            "label": label,
        })

    df = pd.DataFrame(rows)
    if df["label"].nunique() != 2:
        raise ValueError("Both class0 and class1 images are required for binary classification.")
    return df


# ============================== 病例级划分 ==============================

def patient_level_split(df: pd.DataFrame, test_ratio: float = 0.20,
                        val_ratio: float = 0.10) -> pd.DataFrame:
    """
    按病例（patient_id）划分训练/验证/测试集。
    核心原则：同一病例的所有切片必须属于同一个集合，防止数据泄漏。

    划分逻辑：
    1. 先按病例 ID 分组
    2. 从中划出 test_ratio 的病例作为测试集
    3. 从剩余病例中按比例划出验证集
    """
    patients = np.array(sorted(df["patient_id"].unique()))

    # 第一步：划出测试集病例
    train_val_patients, test_patients = train_test_split(
        patients, test_size=test_ratio, random_state=SEED
    )

    # 第二步：从剩余病例中划出验证集
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_ratio_adjusted, random_state=SEED
    )

    # 确定集合互斥
    patient_sets = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }
    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])

    # 为每行数据打上 split 标签
    df = df.copy()
    df["split"] = df["patient_id"].map(
        lambda pid: "train" if pid in patient_sets["train"]
        else "val" if pid in patient_sets["val"]
        else "test"
    )
    return df


# ============================== PyTorch Dataset ==============================

class IDCDataset(Dataset):
    """
    PyTorch Dataset 适配器。
    从 DataFrame 读取图像路径，加载图像并应用 transform。
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================== DataLoader 工厂 ==============================

def get_image_transform(augment: bool = False) -> transforms.Compose:
    """
    获取图像预处理变换。
    - augment=False: 仅 resize + 归一化（验证/测试用）
    - augment=True:  额外加数据增强（训练用）
    """
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


def make_data_loader(dataframe: pd.DataFrame, shuffle: bool = False,
                     augment: bool = False) -> DataLoader:
    """根据 DataFrame 创建 DataLoader。"""
    return DataLoader(
        IDCDataset(dataframe, get_image_transform(augment=augment)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


# ============================== 数据集划分摘要 ==============================

def print_split_summary(df: pd.DataFrame) -> dict:
    """打印并返回数据集划分统计信息。"""
    info = {
        "total_images": len(df),
        "total_patients": df["patient_id"].nunique(),
        "idc_negative": int((df["label"] == 0).sum()),
        "idc_positive": int((df["label"] == 1).sum()),
    }
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        info[f"{split_name}_images"] = len(split_df)
        info[f"{split_name}_patients"] = split_df["patient_id"].nunique()

    print("数据集划分摘要")
    print(f"总图像数:   {info['total_images']:,}")
    print(f"总病例数:   {info['total_patients']}")
    print(f"IDC 阴性:   {info['idc_negative']:,}  (class0)")
    print(f"IDC 阳性:   {info['idc_positive']:,}  (class1)")
    print(f"  训练集:   {info['train_images']:,} 张图像, "
          f"{info['train_patients']} 个病例")
    print(f"  验证集:   {info['val_images']:,} 张图像, "
          f"{info['val_patients']} 个病例")
    print(f"  测试集:   {info['test_images']:,} 张图像, "
          f"{info['test_patients']} 个病例")
    print(f"测试病例比例: {info['test_patients']/info['total_patients']:.2%}")
    print("\n")
    return info
