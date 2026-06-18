# Breast Histopathology IDC Projection

## Project Overview

本项目面向乳腺病理切片图像中的 IDC（Invasive Ductal Carcinoma，浸润性导管癌）二分类任务，目标是在同一套 patient-level split 下，比较传统机器学习方法与基于 ResNet18 的深度学习方法。

项目保留四阶段实验结构：

1. Stage 1：传统机器学习 baseline
2. Stage 2：ResNet18 冻结主干，只训练分类头
3. Stage 3：ResNet18 微调
4. Stage 4：模型比较、简单集成与可视化

本项目强调可复现性、公平比较、类别不平衡处理，以及更适合医学任务的评价指标。

## Dataset

数据集使用 Kaggle Breast Histopathology Images：

[Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

本地数据目录应为：

```text
data/IDC_regular_ps50_idx5/
```

程序从文件名中提取标签，例如 `*_class0.png` 表示 IDC 阴性，`*_class1.png` 表示 IDC 阳性。程序还会从图像路径中提取 `patient_id`，用于 patient-level split，避免同一病人的图像同时出现在训练集和测试集中造成数据泄漏。

## Methods

Stage 1 使用 ImageNet 预训练 ResNet18 提取 512 维图像特征，然后训练传统机器学习模型：

- Logistic Regression，使用 `class_weight="balanced"`
- Random Forest，使用 `class_weight="balanced"`
- XGBoost，根据训练集类别比例设置 `scale_pos_weight`

Stage 2 使用预训练 ResNet18 作为冻结特征提取器，只训练最后的线性分类头，并使用 class-weighted CrossEntropyLoss 处理类别不平衡。

Stage 3 在 Stage 2 的基础上进一步微调整个 ResNet18，使模型能够更充分适应病理图像特征。

Stage 4 加载 Stage 1、Stage 2、Stage 3 的已保存模型，在同一个测试集上比较指标，并进行 hard voting 和 soft voting 集成。

## Experimental Setup

项目使用固定 patient-level split。根据 `DEBUG_MODE` 自动选择不同 split 文件：

```text
outputs/split_patient_level_debug.csv   # DEBUG_MODE = True 时使用
outputs/split_patient_level_full.csv    # DEBUG_MODE = False 时使用
```

这样可以避免先运行 DEBUG 小样本流程后，正式训练误用 DEBUG split。

`config.py` 中的 DEBUG 设置如下：

```python
DEBUG_MODE = True
SAMPLE_PER_CLASS = 1000
DEBUG_EPOCHS = 1
```

当 `DEBUG_MODE=True` 时，每类最多抽取 1000 张图像，Stage 2 和 Stage 3 只训练 1 个 epoch。该模式只用于快速检查流程、文件保存和图表生成，不能作为正式实验结果。

当 `DEBUG_MODE=False` 时，使用完整数据集和正式训练参数，生成正式实验结果。

## Results

正式训练后更新。

当前项目会自动保存主要指标到：

```text
outputs/results_summary.csv
```

评价指标包括：

- accuracy
- precision
- recall / sensitivity
- specificity
- F1
- ROC-AUC
- PR-AUC
- balanced accuracy

医学图像任务不能只看 accuracy。尤其在类别不平衡时，recall/sensitivity、specificity、PR-AUC 和 balanced accuracy 更能反映模型对阳性病例和阴性病例的识别能力。

## Visualization

程序会自动生成以下图表：

```text
outputs/figures/confusion_matrix_*.png
outputs/figures/roc_curves.png
outputs/figures/pr_curves.png
outputs/figures/training_curve_stage2.png
outputs/figures/training_curve_stage3.png
outputs/figures/model_comparison.png
```

这些图表用于展示混淆矩阵、ROC 曲线、PR 曲线、训练过程曲线和不同模型的指标对比。

## Error Analysis

Stage 4 会保存 soft voting 集成模型的错误样本示例：

```text
outputs/error_samples/false_positive_examples.png
outputs/error_samples/false_negative_examples.png
```

False positive 表示真实为 IDC 阴性但模型预测为阳性；false negative 表示真实为 IDC 阳性但模型预测为阴性。医学任务中 false negative 通常需要重点关注，因为漏检阳性病例可能带来更高风险。

## Discussion

patient-level split 是本项目公平比较的关键。病理 patch 来自同一病人时往往具有相似组织形态，如果同一病人的 patch 同时进入训练集和测试集，会导致测试结果虚高。

传统机器学习模型提供了可解释、训练较快的 baseline；Stage 2 验证冻结预训练特征是否足够；Stage 3 通过微调释放模型能力，但也带来更高计算成本和过拟合风险。

## Limitations

- 当前评价是 patch-level 评价，不能直接等同于 patient-level 临床诊断性能。
- DEBUG_MODE 只用于流程测试，不能作为正式结果。
- 数据标签来自文件名，应视为数据集提供的弱标签。
- 如果用于更严谨研究，还需要外部数据集验证。

## How to Run

安装依赖：

```bash
pip install -r requirements.txt
```

### DEBUG 测试运行

在 `config.py` 中保持：

```python
DEBUG_MODE = True
```

然后按顺序运行：

```bash
python experiments/stage1_baseline_ml.py
python experiments/stage2_resnet_frozen.py
python experiments/stage3_resnet_finetune.py
python experiments/stage4_model_comparison.py
```

### 正式训练运行

在 `config.py` 中设置：

```python
DEBUG_MODE = False
```

然后按相同顺序运行四个阶段：

```bash
python experiments/stage1_baseline_ml.py
python experiments/stage2_resnet_frozen.py
python experiments/stage3_resnet_finetune.py
python experiments/stage4_model_comparison.py
```

Stage 4 需要以下文件已经由前面阶段生成：

```text
outputs/models/best_ml_models.pkl
outputs/models/stage1_standard_scaler.pkl
outputs/models/best_resnet18_frozen.pth
outputs/models/best_resnet18_finetune.pth
```

如果缺少这些文件，Stage 4 会给出明确报错提示。

## GitHub 提交说明

可以提交：

- `tool/`
- `experiments/`
- `config.py`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `outputs/results_summary.csv`
- 少量核心结果图，例如 ROC、PR、confusion matrix、training curve

不应提交：

- `data/`
- 大型 `.pth` 模型权重
- 大型 `.pkl` 模型文件或 scaler 文件
- `outputs/*.pt` 特征缓存
- 大量中间输出文件
