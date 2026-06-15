"""
==============================================================================
项目统一配置文件
所有超参数和路径集中管理，避免魔法数字散落在各个文件中
==============================================================================
"""

from pathlib import Path

# ============================== 项目路径 ==============================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "IDC_regular_ps50_idx5"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
# 确保输出目录存在
for d in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================== 随机种子 ==============================

SEED = 42

# ============================== 数据集划分 ==============================

TEST_PATIENT_RATIO = 0.20   # 测试集病例比例（不少于 20%）
VAL_PATIENT_RATIO = 0.10    # 验证集病例比例（从剩余 80% 中取 10%，即总体的 8%）

# ============================== 图像参数 ==============================

IMG_SIZE = 224              # ResNet 标准输入尺寸
BATCH_SIZE = 32             # DataLoader 批次大小
NUM_WORKERS = 0             # CPU 环境建议设为 0，避免多进程问题

# ============================== 特征提取参数 ==============================

# 预训练的 ResNet18 输出 512 维特征向量
FEATURE_DIM = 512

# ============================== ResNet 训练参数 ==============================

NUM_EPOCHS_FROZEN = 10      # Stage2: 冻结骨干训练轮数
NUM_EPOCHS_FINETUNE = 5     # Stage3: 微调训练轮数
LEARNING_RATE_FC = 1e-3     # 全连接层学习率
LEARNING_RATE_FINETUNE = 1e-4  # 微调时学习率（更小，避免破坏预训练权重）
WEIGHT_DECAY = 1e-4         # 权重衰减（L2正则化）

# ============================== 传统 ML 参数 ==============================

# XGBoost 参数
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

# Random Forest 参数
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": SEED,
    "n_jobs": -1,
}

# Logistic Regression 参数
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": SEED,
    "n_jobs": -1,
}

# ============================== 保存路径 ==============================

# 模型文件
BEST_MODEL_STAGE2 = MODEL_DIR / "best_resnet18_frozen.pth"
BEST_MODEL_STAGE3 = MODEL_DIR / "best_resnet18_finetune.pth"
BEST_ML_MODELS = MODEL_DIR / "best_ml_models.pkl"

# 特征缓存（避免重复提取）
FEATURE_CACHE = OUTPUT_DIR / "extracted_features.pt"

# 结果文件
RESULTS_CSV = OUTPUT_DIR / "all_model_results.csv"
HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
