"""
==============================================================================
特征提取模块 (Feature Extraction)
使用预训练的 ResNet18 将图像转为 512 维特征向量
提取一次后可缓存复用，供 Stage1 (ML) 和 Stage2 (FC) 共享
==============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm

from config import FEATURE_CACHE


def get_feature_extractor(device: torch.device) -> nn.Module:
    """
    构建特征提取器：ImageNet 预训练的 ResNet18，去掉最后的 FC 层。
    输出维度：512
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # 移除分类头
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(feature_extractor: nn.Module,
                     loader: DataLoader,
                     device: torch.device,
                     desc: str = "Extracting") -> tuple[torch.Tensor, torch.Tensor]:
    """
    遍历 DataLoader，将每张图像通过特征提取器得到 512 维特征向量。

    返回:
        features: shape (N, 512)
        labels:   shape (N,)
    """
    feature_extractor.eval()
    all_features, all_labels = [], []

    for images, labels in tqdm(loader, desc=desc, leave=False, bar_format="{desc}: {n_fmt}/{total_fmt} ({percentage:.0f}%)"):
        images = images.to(device)
        batch_features = feature_extractor(images).cpu()
        all_features.append(batch_features)
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels).long()


def make_feature_loader(features: torch.Tensor, labels: torch.Tensor,
                        shuffle: bool = True) -> DataLoader:
    """将特征张量包装为 DataLoader，用于 FC 层训练。"""
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=4096,  # 特征向量很小，可以用大批次
        shuffle=shuffle,
    )


def load_or_extract_features(feature_extractor: nn.Module,
                             loaders: dict,
                             device: torch.device,
                             force_recompute: bool = False) -> dict:
    """
    加载缓存的特征，如果不存在则提取并缓存。

    Args:
        feature_extractor: 特征提取模型
        loaders: {"train": loader, "val": loader, "test": loader}
        device: 设备
        force_recompute: 是否强制重新提取

    Returns:
        {"train": (features, labels), "val": (features, labels), "test": ...}
    """
    if FEATURE_CACHE.exists() and not force_recompute:
        print(f"[OK] 从缓存加载特征: {FEATURE_CACHE}")
        return torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)

    print(" 正在提取图像特征（这可能需要一段时间）...")
    feature_data = {}
    for split_name, loader in loaders.items():
        x, y = extract_features(feature_extractor, loader, device, desc=f"Extract {split_name}")
        feature_data[split_name] = (x, y)

    torch.save(feature_data, FEATURE_CACHE)
    print(f"[Saved] 特征已缓存至: {FEATURE_CACHE}")
    return feature_data
