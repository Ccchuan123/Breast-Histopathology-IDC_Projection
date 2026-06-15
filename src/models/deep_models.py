"""
==============================================================================
深度学习模型模块 (Deep Learning Models)
包含：ResNet18 分类器（全模型 + FC-only 两种模式）
==============================================================================
"""

import torch
import torch.nn as nn
from torchvision import models


def create_resnet18_full(num_classes: int = 2,
                         freeze_backbone: bool = False) -> nn.Module:
    """
    创建基于 ResNet18 的二分类模型。

    Args:
        num_classes: 分类数（默认 2，即 IDC 阴性/阳性）
        freeze_backbone: True=冻结骨干只训练 FC 层（Stage2）
                         False=全模型可训练（Stage3 微调）

    返回:
        ResNet18 模型实例
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        # 冻结骨干网络的所有参数
        for param in model.parameters():
            param.requires_grad = False

    # 替换最后的全连接层为二分类器
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def create_feature_classifier(feature_dim: int = 512,
                              num_classes: int = 2) -> nn.Linear:
    """
    创建特征级分类器（用于 Stage2：在预提取特征上训练）。
    这比训练完整 ResNet 快得多，适合 CPU 环境。

    本质就是一个线性层：512 → 2
    等价于 ResNet18 的 FC 层单独训练。
    """
    return nn.Linear(feature_dim, num_classes)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """统计模型的总参数量和可训练参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
