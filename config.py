"""Central project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "IDC_regular_ps50_idx5"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
ERROR_SAMPLE_DIR = OUTPUT_DIR / "error_samples"

for d in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, ERROR_SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Debug mode keeps the full pipeline runnable on CPU/cloud notebooks.
DEBUG_MODE = True
SAMPLE_PER_CLASS = 1000
DEBUG_EPOCHS = 1

TEST_PATIENT_RATIO = 0.20
VAL_PATIENT_RATIO = 0.10
SPLIT_CSV = OUTPUT_DIR / (
    "split_patient_level_debug.csv" if DEBUG_MODE else "split_patient_level_full.csv"
)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
FEATURE_DIM = 512

NUM_EPOCHS_FROZEN = DEBUG_EPOCHS if DEBUG_MODE else 10
NUM_EPOCHS_FINETUNE = DEBUG_EPOCHS if DEBUG_MODE else 5
LEARNING_RATE_FC = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "eval_metric": "logloss",
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

BEST_MODEL_STAGE2 = MODEL_DIR / "best_resnet18_frozen.pth"
BEST_MODEL_STAGE3 = MODEL_DIR / "best_resnet18_finetune.pth"
BEST_ML_MODELS = MODEL_DIR / "best_ml_models.pkl"
STAGE1_SCALER = MODEL_DIR / "stage1_standard_scaler.pkl"

FEATURE_CACHE = OUTPUT_DIR / ("extracted_features_debug.pt" if DEBUG_MODE else "extracted_features.pt")
RESULTS_CSV = OUTPUT_DIR / "results_summary.csv"
HISTORY_STAGE2_CSV = OUTPUT_DIR / "training_history_stage2.csv"
HISTORY_STAGE3_CSV = OUTPUT_DIR / "training_history_stage3.csv"
