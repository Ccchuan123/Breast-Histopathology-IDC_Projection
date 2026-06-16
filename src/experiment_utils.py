"""Small shared helpers for experiment entry scripts."""

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
from src.data_utils import load_or_create_patient_split


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs():
    for path in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, ERROR_SAMPLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_experiment_split():
    return load_or_create_patient_split()


def print_experiment_config(stage_name: str, device: torch.device):
    mode = "DEBUG" if DEBUG_MODE else "FULL"
    print("\n" + "=" * 72)
    print(stage_name)
    print("=" * 72)
    print(f"Mode:       {mode}")
    if DEBUG_MODE:
        print(f"Sample cap: {SAMPLE_PER_CLASS} images per class")
    print(f"Split file: {SPLIT_CSV}")
    print(f"Device:     {device}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 72)
