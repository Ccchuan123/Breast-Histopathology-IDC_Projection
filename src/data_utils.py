"""Data collection, fixed patient-level split, and PyTorch datasets."""

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
    match = re.search(r"class([01])", image_path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot extract class label from filename: {image_path.name}")
    return int(match.group(1))


def collect_samples(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    image_paths = sorted(data_dir.rglob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {data_dir}")

    rows = []
    for image_path in image_paths:
        label = extract_label_from_filename(image_path)
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
    df["split"] = df["patient_id"].map(
        lambda pid: "train"
        if pid in patient_sets["train"]
        else "val"
        if pid in patient_sets["val"]
        else "test"
    )
    return df


def _apply_debug_sample(df: pd.DataFrame) -> pd.DataFrame:
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


def get_image_transform(augment: bool = False) -> transforms.Compose:
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
    return DataLoader(
        IDCDataset(dataframe, get_image_transform(augment=augment)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


def print_split_summary(df: pd.DataFrame) -> dict:
    info = {
        "total_images": len(df),
        "total_patients": df["patient_id"].nunique(),
        "idc_negative": int((df["label"] == 0).sum()),
        "idc_positive": int((df["label"] == 1).sum()),
    }
    print("\nDataset split summary")
    print(f"Total images:   {info['total_images']:,}")
    print(f"Total patients: {info['total_patients']:,}")
    print(f"IDC negative:   {info['idc_negative']:,}")
    print(f"IDC positive:   {info['idc_positive']:,}")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        info[f"{split_name}_images"] = len(split_df)
        info[f"{split_name}_patients"] = split_df["patient_id"].nunique()
        print(
            f"{split_name:>5}: {len(split_df):,} images, "
            f"{split_df['patient_id'].nunique():,} patients"
        )
    return info
