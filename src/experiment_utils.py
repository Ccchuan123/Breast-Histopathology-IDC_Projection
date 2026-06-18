"""四个实验 stage 共享的轻量工具函数。

这里只放入口级别的通用逻辑，例如 device 选择、输出目录创建、split 加载和实验配置打印。
不把 stage 改成复杂类结构，避免过度工程化。
"""

import torch

from config import (
    DEBUG_MODE,
    ERROR_SAMPLE_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    SAMPLE_PER_CLASS,
    SPLIT_CSV,
)
from tool.data_utils import load_or_create_patient_split


def get_device() -> torch.device:
    """选择训练设备。

    如果当前环境可用 CUDA，则使用 GPU；否则回退到 CPU。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs():
    """确保输出目录存在。

    包括总输出目录、图表目录、模型目录和错误样本目录。
    """
    for path in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, ERROR_SAMPLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_experiment_split():
    """加载当前 DEBUG/FULL 模式对应的固定 patient-level split。"""
    return load_or_create_patient_split()


def print_experiment_config(stage_name: str, device: torch.device):
    """在每个 stage 开始时打印关键运行信息。

    这样运行日志中能清楚看到当前 stage、DEBUG/FULL 模式、split 文件、device 和输出目录。
    """
    mode = "DEBUG" if DEBUG_MODE else "FULL"
    print("\n")
    print(stage_name)
    print(f"Mode:       {mode}")
    if DEBUG_MODE:
        print(f"Sample cap: {SAMPLE_PER_CLASS} images per class")
    print(f"Split file: {SPLIT_CSV}")
    print(f"Device:     {device}")
    print(f"Output dir: {OUTPUT_DIR}")
