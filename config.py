"""项目统一配置文件。

本文件集中管理数据路径、输出路径、DEBUG/正式训练开关、模型训练参数和结果保存路径。
四个 stage 脚本都从这里读取配置，避免在不同文件中散落“魔法数字”。
"""

from pathlib import Path

# PROJECT_ROOT 是项目根目录，即当前 config.py 所在目录。
PROJECT_ROOT = Path(__file__).resolve().parent

# DATA_DIR 是数据集路径。请将 Kaggle 数据解压到该目录下。
DATA_DIR = PROJECT_ROOT / "data" / "IDC_regular_ps50_idx5"

# OUTPUT_DIR 保存所有实验输出；FIGURE_DIR 保存图表；MODEL_DIR 保存模型权重和传统 ML 模型；
# ERROR_SAMPLE_DIR 保存 false positive / false negative 错误样本图。
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
ERROR_SAMPLE_DIR = OUTPUT_DIR / "error_samples"

for d in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, ERROR_SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 固定随机种子，尽量保证数据划分和 DEBUG 抽样可复现。
SEED = 42

# DEBUG_MODE=True 只用于小样本流程测试，不能作为正式实验结果。
# DEBUG_MODE=False 才用于完整数据集的正式训练与结果汇报。
DEBUG_MODE = True

# SAMPLE_PER_CLASS 只在 DEBUG_MODE=True 时生效，每类最多抽取这么多张图像。
SAMPLE_PER_CLASS = 1000

# DEBUG_EPOCHS 控制 DEBUG 模式下 Stage2 / Stage3 的训练轮数，便于 CPU 快速跑通流程。
DEBUG_EPOCHS = 1

# TEST_PATIENT_RATIO 表示按病人划分测试集的比例。
# VAL_PATIENT_RATIO 表示按病人划分验证集的比例。
TEST_PATIENT_RATIO = 0.20
VAL_PATIENT_RATIO = 0.10

# DEBUG 和正式训练使用不同 split 文件，避免 DEBUG 小样本流程影响正式实验。
# split_patient_level_debug.csv：DEBUG_MODE=True 时使用。
# split_patient_level_full.csv：DEBUG_MODE=False 时使用。
SPLIT_CSV = OUTPUT_DIR / (
    "split_patient_level_debug.csv" if DEBUG_MODE else "split_patient_level_full.csv"
)

# 图像输入尺寸、DataLoader batch size 和 worker 数。
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

# ResNet18 去掉最后分类层后的特征维度。
FEATURE_DIM = 512

# Stage2 冻结 ResNet18 主干，只训练分类头；Stage3 微调整个 ResNet18。
# DEBUG 模式下使用 DEBUG_EPOCHS，正式训练时使用后面的默认轮数。
NUM_EPOCHS_FROZEN = DEBUG_EPOCHS if DEBUG_MODE else 10
NUM_EPOCHS_FINETUNE = DEBUG_EPOCHS if DEBUG_MODE else 5

# 学习率和 weight decay。Stage3 微调使用更小学习率，避免破坏预训练权重。
LEARNING_RATE_FC = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4

# 传统机器学习模型参数。LR/RF 使用 class_weight="balanced" 缓解类别不平衡。
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

# 模型权重与传统 ML 模型保存路径。
BEST_MODEL_STAGE2 = MODEL_DIR / "best_resnet18_frozen.pth"
BEST_MODEL_STAGE3 = MODEL_DIR / "best_resnet18_finetune.pth"
BEST_ML_MODELS = MODEL_DIR / "best_ml_models.pkl"

# Stage1 的 StandardScaler 保存路径。Stage4 必须加载同一个 scaler，禁止在测试集重新 fit。
STAGE1_SCALER = MODEL_DIR / "stage1_standard_scaler.pkl"

# ResNet18 特征缓存路径。DEBUG 与正式模式分开，避免混用小样本特征。
FEATURE_CACHE = OUTPUT_DIR / ("extracted_features_debug.pt" if DEBUG_MODE else "extracted_features.pt")

# results_summary.csv 保存所有模型的核心评价指标。
RESULTS_CSV = OUTPUT_DIR / "results_summary.csv"
HISTORY_STAGE2_CSV = OUTPUT_DIR / "training_history_stage2.csv"
HISTORY_STAGE3_CSV = OUTPUT_DIR / "training_history_stage3.csv"
