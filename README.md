# Breast Histopathology IDC Projection

## Project Overview

This project evaluates invasive ductal carcinoma (IDC) classification on breast histopathology image patches with a four-stage experimental pipeline:

1. Stage 1: traditional machine-learning baselines
2. Stage 2: ResNet18 frozen backbone with a trainable classifier head
3. Stage 3: ResNet18 fine-tuning
4. Stage 4: model comparison and simple ensemble methods

The optimization keeps the original staged structure and focuses on reproducibility, fair comparison, debug-friendly execution, class-imbalance handling, and medical-task metrics.

## Dataset

Use the Kaggle Breast Histopathology Images dataset:

[Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

Expected local layout:

```text
data/IDC_regular_ps50_idx5/
```

The code extracts labels from filenames such as `*_class0.png` and `*_class1.png`. Splits are patient-level to avoid leakage across train, validation, and test sets.

## Methods

Stage 1 extracts 512-dimensional ResNet18 features and trains:

- Logistic Regression with `class_weight="balanced"`
- Random Forest with `class_weight="balanced"`
- XGBoost with `scale_pos_weight` from the training split

Stage 2 uses ImageNet-pretrained ResNet18 as a frozen feature extractor and trains a linear classifier with class-weighted cross-entropy.

Stage 3 fine-tunes the full ResNet18 with class-weighted cross-entropy and light image augmentation.

Stage 4 reloads all saved models and the Stage 1 scaler, evaluates them on the same fixed test split, and builds hard-voting and probability-averaging ensembles.

## Experimental Setup

The first run creates a mode-specific fixed patient split:

```text
outputs/split_patient_level_debug.csv   # used when DEBUG_MODE = True
outputs/split_patient_level_full.csv    # used when DEBUG_MODE = False
```

All four stages read the split file matching the current mode. This keeps training, validation, and test sets consistent across all models while preventing a DEBUG run from being accidentally reused for formal training.

Debug mode is configured in `config.py`:

```python
DEBUG_MODE = True
SAMPLE_PER_CLASS = 1000
DEBUG_EPOCHS = 1
```

When `DEBUG_MODE=True`, each class is sampled down to at most 1000 images and Stage 2/3 train for 1 epoch. The split is stored in `outputs/split_patient_level_debug.csv`.

When `DEBUG_MODE=False`, the full dataset and formal epoch settings are used. The split is stored separately in `outputs/split_patient_level_full.csv`, so formal training will not reuse the DEBUG split file.

## Results

The main metrics summary is saved to:

```text
outputs/results_summary.csv
```

Metrics include:

- accuracy
- precision
- recall / sensitivity
- specificity
- F1
- ROC-AUC
- PR-AUC
- balanced accuracy

## Visualization

Generated figures include:

```text
outputs/figures/confusion_matrix_*.png
outputs/figures/roc_curves.png
outputs/figures/pr_curves.png
outputs/figures/training_curve_stage2.png
outputs/figures/training_curve_stage3.png
outputs/figures/model_comparison.png
```

## Error Analysis

Stage 4 saves representative false-positive and false-negative examples from the soft-voting ensemble:

```text
outputs/error_samples/false_positive_examples.png
outputs/error_samples/false_negative_examples.png
```

These images should be reviewed manually because histopathology errors can reflect ambiguous tissue regions, labeling noise, or model bias toward visually dominant patterns.

## Discussion

The fixed patient-level split is essential for fair comparison because image patches from the same patient can be visually correlated. Traditional ML models provide a useful feature-based baseline, while Stage 2 tests whether a lightweight classifier is enough after transfer learning. Stage 3 has the highest capacity but also the highest compute cost and overfitting risk.

## Limitations

- Patch-level metrics do not directly equal patient-level clinical performance.
- Debug mode is only for pipeline validation and should not be reported as final performance.
- The dataset labels are inherited from file names and should be treated as weak labels unless independently verified.
- External validation on another cohort is needed before making clinical claims.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

### DEBUG Test Run

Keep `DEBUG_MODE = True` in `config.py`. This mode uses `outputs/split_patient_level_debug.csv`, samples at most 1000 images per class, and trains Stage 2/3 for 1 epoch. Use it only to check that the pipeline, saved files, and plots are generated correctly.

```bash
python experiments/stage1_baseline_ml.py
python experiments/stage2_resnet_frozen.py
python experiments/stage3_resnet_finetune.py
python experiments/stage4_model_comparison.py
```

### Formal Training Run

Set `DEBUG_MODE = False` in `config.py`, then confirm the formal values for epochs, batch size, and learning rates. This mode uses `outputs/split_patient_level_full.csv`, not the DEBUG split file.

```bash
python experiments/stage1_baseline_ml.py
python experiments/stage2_resnet_frozen.py
python experiments/stage3_resnet_finetune.py
python experiments/stage4_model_comparison.py
```

Run the stages in order. Stage 4 expects artifacts from Stage 1, Stage 2, and Stage 3:

```text
outputs/models/best_ml_models.pkl
outputs/models/stage1_standard_scaler.pkl
outputs/models/best_resnet18_frozen.pth
outputs/models/best_resnet18_finetune.pth
```

If any are missing, Stage 4 will stop with a clear message telling you which previous stage to run.

### Before Formal Training

You do not need to delete the DEBUG split file before formal training because the two modes use different split files. You may delete old generated outputs if you want a clean report folder, especially:

```text
outputs/results_summary.csv
outputs/figures/
outputs/error_samples/
outputs/models/
```

Do not delete `outputs/split_patient_level_full.csv` once formal experiments have started unless you intentionally want a new patient-level split.

### GitHub Upload Checklist

Safe to upload:

- source code under `src/` and `experiments/`
- `config.py`
- `README.md`
- `requirements.txt`
- `.gitignore`
- compact final files such as `outputs/results_summary.csv`
- selected core figures in `outputs/figures/`

Do not upload:

- `data/`
- large model weights such as `*.pth`
- large serialized models or scalers such as `*.pkl`
- feature caches such as `outputs/*.pt`
- raw full output folders unless your teacher explicitly asks for them
