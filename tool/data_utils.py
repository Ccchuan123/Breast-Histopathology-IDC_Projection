"""数据收集、patient-level split 与 PyTorch Dataset 工具。

本模块负责把原始病理图像整理成 DataFrame，并保证同一 patient_id 的图像只会出现在
train / val / test 中的一个集合里。这样可以避免同一病人图像泄漏到测试集，导致模型性能虚高。
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    DATA_DIR,
    DEBUG_MODE,
    IMG_SIZE,
    NUM_WORKERS,
    SAMPLE_PER_CLASS,
    SEED,
    SPLIT_CSV,
    TEST_PATIENT_RATIO,
    VAL_PATIENT_RATIO,
)


def extract_label_from_filename(image_path: Path) -> int:
    """从文件名中提取二分类标签。

    数据集文件名通常包含 class0 或 class1：
    - class0：IDC 阴性
    - class1：IDC 阳性
    """
    match = re.search(r"class([01])", image_path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot extract class label from filename: {image_path.name}")
    return int(match.group(1))


def collect_samples(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """遍历数据目录，收集所有 PNG 图像及其 patient_id、label。

    patient_id 从路径层级中提取，label 从文件名的 class0/class1 中提取。
    返回的 DataFrame 是后续 patient-level split 和 Dataset 构建的统一入口。
    """
    image_paths = sorted(data_dir.rglob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {data_dir}")

    rows = []
    for image_path in image_paths:
        label = extract_label_from_filename(image_path)
        # 如果文件夹名本身也是 0/1，则额外检查文件夹标签和文件名标签是否一致。
        if image_path.parent.name in {"0", "1"} and int(image_path.parent.name) != label:
            raise ValueError(f"Folder label and filename label mismatch: {image_path}")
        rows.append(
            {
                "path": str(image_path),
                "patient_id": image_path.parts[-3],
                "label": label,
            }
        )

    df = pd.DataFrame(rows)
    if df["label"].nunique() != 2:
        raise ValueError("Both class0 and class1 images are required.")
    return df


def patient_level_split(
    df: pd.DataFrame,
    test_ratio: float = TEST_PATIENT_RATIO,
    val_ratio: float = VAL_PATIENT_RATIO,
) -> pd.DataFrame:
    """按 patient_id 划分 train / val / test。

    医学图像中同一病人的多个 patch 往往高度相关。如果随机按图片划分，
    同一病人的图像可能同时出现在训练集和测试集，造成数据泄漏。
    因此这里先按 patient_id 划分，再把 split 标签映射回每一张图像。
    """
    patients = np.array(sorted(df["patient_id"].unique()))
    train_val_patients, test_patients = train_test_split(
        patients, test_size=test_ratio, random_state=SEED
    )
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_ratio_adjusted, random_state=SEED
    )

    patient_sets = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }
    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])

    df = df.copy()
    patient_to_split = {pid: s for s, pids in patient_sets.items() for pid in pids}
    df["split"] = df["patient_id"].map(patient_to_split)
    return df


def _apply_debug_sample(df: pd.DataFrame) -> pd.DataFrame:
    """DEBUG 模式下的小样本抽样。

    只在 DEBUG_MODE=True 时生效。每个类别最多保留 SAMPLE_PER_CLASS 张图像，
    用于快速检查代码流程、输出文件和图表是否正常生成。
    """
    if not DEBUG_MODE:
        return df
    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(SAMPLE_PER_CLASS, len(x)), random_state=SEED))
        .sort_values(["split", "patient_id", "path"])
        .reset_index(drop=True)
    )
    print(
        f"[DEBUG_MODE] Using {len(sampled):,} images "
        f"({SAMPLE_PER_CLASS} max per class)."
    )
    return sampled


def load_or_create_patient_split(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """读取或生成当前模式对应的固定 patient-level split。

    DEBUG_MODE=True 时使用 split_patient_level_debug.csv；
    DEBUG_MODE=False 时使用 split_patient_level_full.csv。
    二者分开保存，避免先跑 DEBUG 后正式训练误用小样本 split。
    """
    if SPLIT_CSV.exists():
        df = pd.read_csv(SPLIT_CSV)
        print(f"[OK] Loaded fixed patient-level split: {SPLIT_CSV}")
    else:
        df = patient_level_split(collect_samples(data_dir))
        SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SPLIT_CSV, index=False)
        print(f"[Saved] Fixed patient-level split: {SPLIT_CSV}")
    print(
        "[INFO] Split mode: "
        f"{'DEBUG' if DEBUG_MODE else 'FULL'} "
        f"({SPLIT_CSV.name})"
    )
    return _apply_debug_sample(df)


class IDCDataset(Dataset):
    """IDC 图像数据集适配器。

    从 DataFrame 中读取图像路径和标签，加载图像并应用 torchvision transform。
    """
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = row["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_image_transform(augment: bool = False) -> transforms.Compose:
    """构建图像预处理流程。

    验证/测试只做 resize、ToTensor 和 ImageNet normalization；
    训练时可额外加入轻量数据增强。
    """
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def make_data_loader(dataframe: pd.DataFrame, shuffle: bool = False, augment: bool = False) -> DataLoader:
    """根据 DataFrame 构建 PyTorch DataLoader。"""
    return DataLoader(
        IDCDataset(dataframe, get_image_transform(augment=augment)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


def print_split_summary(df: pd.DataFrame):
    """打印当前 split 的图像数量、病人数和类别分布。"""
    print("\nDataset split summary")
    print(f"Total images:   {len(df):,}")
    print(f"Total patients: {df['patient_id'].nunique():,}")
    print(f"IDC negative:   {(df['label'] == 0).sum():,}")
    print(f"IDC positive:   {(df['label'] == 1).sum():,}")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        print(
            f"{split_name:>5}: {len(split_df):,} images, "
            f"{split_df['patient_id'].nunique():,} patients"
        )
