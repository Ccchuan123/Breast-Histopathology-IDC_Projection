## 项目结构（重构后）

```
Breast-Histopathology-IDC_Projection-main/
├── README.md                           # 项目说明（你正在读的文件）
├── requirements.txt                    # Python 依赖
├── config.py                           # 统一配置（所有参数集中管理）
│
├── src/                                # 核心代码（高内聚低耦合）
│   ├── data_utils.py                   # 数据加载、划分、Dataset
│   ├── feature_extraction.py           # ResNet18 特征提取
│   ├── train_utils.py                  # 训练循环、评估、指标
│   ├── visualization.py               # 所有绘图函数
│   └── models/
│       ├── baseline_ml.py              # 传统 ML：LR / RF / XGBoost
│       └── deep_models.py              # 深度学习：ResNet18
│
├── experiments/                        # 分阶段实验脚本
│   ├── stage1_baseline_ml.py           # 阶段一：传统 ML 基线
│   ├── stage2_resnet_frozen.py         # 阶段二：ResNet18 冻结骨干
│   ├── stage3_resnet_finetune.py       # 阶段三：ResNet18 全模型微调
│   └── stage4_model_comparison.py      # 阶段四：模型对比与集成
│
├── notebooks/                          # Jupyter Notebook（原始版本保留）
├── outputs/                            # 输出（模型、图表、结果）
│   ├── figures/                        # 所有可视化图表
│   └── models/                         # 训练好的模型文件
├── data/                               # 数据集（需自行下载）
└── docs/                               # 文档
```

## 实验设计：四阶段渐进式

| 阶段 | 方法 | 说明 |
|------|------|------|
| **阶段一** | 传统 ML 基线 | Logistic Regression / Random Forest / XGBoost |
| **阶段二** | ResNet18 冻结骨干 | 预训练特征 + FC 层（搭档原始方法） |
| **阶段三** | ResNet18 微调 | 解冻全部参数，端到端训练 |
| **阶段四** | 模型对比与集成 | 综合评估 + 投票集成 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

从 [Kaggle Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) 下载数据，解压到：

```text
data/IDC_regular_ps50_idx5/
```

### 3. 按顺序运行实验

```bash
# 阶段一：传统 ML 基线（先跑这个，建立性能基准）
python experiments/stage1_baseline_ml.py

# 阶段二：ResNet18 冻结骨干（搭档的原始方法）
python experiments/stage2_resnet_frozen.py

# 阶段三：ResNet18 微调（进一步提升）
python experiments/stage3_resnet_finetune.py

# 阶段四：模型对比与集成（总结）
python experiments/stage4_model_comparison.py
```

## 评价指标

- **Accuracy**：总体准确率
- **Precision**：精确率（预测为阳性的样本中，真正阳性的比例）
- **Recall**：召回率（真实阳性样本中，被正确识别的比例）
- **F1-Score**：精确率与召回率的调和平均
- **AUC**：ROC 曲线下面积（越大越好，>0.9 为优秀）
- **Confusion Matrix**：混淆矩阵

