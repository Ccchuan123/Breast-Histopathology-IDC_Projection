"""Stage 4: model comparison and simple ensembles."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from config import BEST_ML_MODELS, BEST_MODEL_STAGE2, BEST_MODEL_STAGE3, FEATURE_DIM, RESULTS_CSV, STAGE1_SCALER
from src.data_utils import make_data_loader, print_split_summary
from src.experiment_utils import ensure_output_dirs, get_device, load_experiment_split, print_experiment_config
from src.feature_extraction import extract_features, get_feature_extractor, make_feature_loader
from src.models.baseline_ml import load_ml_models, load_scaler, predict_ml_model
from src.models.deep_models import create_feature_classifier, create_resnet18_full
from src.train_utils import compute_metrics, predict_model, print_metrics, save_model_results
from src.visualization import (
    plot_combined_pr_curves,
    plot_combined_roc_curves,
    plot_confusion_matrix,
    plot_error_samples,
    plot_model_comparison,
)


def load_stage2_model(device: torch.device) -> torch.nn.Module:
    classifier = create_feature_classifier(feature_dim=FEATURE_DIM, num_classes=2).to(device)
    if not BEST_MODEL_STAGE2.exists():
        raise FileNotFoundError(f"Missing Stage2 weights: {BEST_MODEL_STAGE2}")
    classifier.load_state_dict(torch.load(BEST_MODEL_STAGE2, map_location=device))
    classifier.eval()
    return classifier


def load_stage3_model(device: torch.device) -> torch.nn.Module:
    model = create_resnet18_full(num_classes=2, freeze_backbone=False).to(device)
    if not BEST_MODEL_STAGE3.exists():
        raise FileNotFoundError(f"Missing Stage3 weights: {BEST_MODEL_STAGE3}")
    model.load_state_dict(torch.load(BEST_MODEL_STAGE3, map_location=device))
    model.eval()
    return model


def ensemble_vote(predictions: list[np.ndarray]) -> np.ndarray:
    stacked = np.column_stack(predictions)
    final = []
    for row in stacked:
        counts = Counter(row)
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            final.append(1)
        else:
            final.append(most_common[0][0])
    return np.array(final)


def ensemble_average(probabilities: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    avg_prob = np.mean(probabilities, axis=0)
    return (avg_prob > 0.5).astype(int), avg_prob


def validate_stage4_artifacts():
    required = {
        "Stage1 ML models": (BEST_ML_MODELS, "Run experiments/stage1_baseline_ml.py first."),
        "Stage1 scaler": (STAGE1_SCALER, "Run experiments/stage1_baseline_ml.py first."),
        "Stage2 weights": (BEST_MODEL_STAGE2, "Run experiments/stage2_resnet_frozen.py first."),
        "Stage3 weights": (BEST_MODEL_STAGE3, "Run experiments/stage3_resnet_finetune.py first."),
    }
    missing = [f"{name}: {path}\n  {hint}" for name, (path, hint) in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 4 requires trained artifacts from Stage 1, Stage 2, and Stage 3.\n"
            + "\n".join(missing)
        )


def main():
    ensure_output_dirs()
    device = get_device()
    print_experiment_config("Stage 4: Model comparison and ensemble", device)
    validate_stage4_artifacts()

    df = load_experiment_split()
    print_split_summary(df)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    feature_extractor = get_feature_extractor(device)
    test_image_loader = make_data_loader(test_df, shuffle=False)
    X_test, y_test = extract_features(feature_extractor, test_image_loader, device, desc="Extract test features")
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    model_names: list[str] = []
    curve_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    ml_models = load_ml_models()
    scaler = load_scaler()
    X_test_scaled = scaler.transform(X_test_np)

    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name in ml_models and ml_models[name] is not None:
            features = X_test_scaled if name == "Logistic Regression" else X_test_np
            y_pred, y_prob = predict_ml_model(ml_models[name], features)
            metrics = compute_metrics(y_test_np, y_pred, y_prob)
            print_metrics(metrics, f"{name} - Test")
            save_model_results(name, metrics)
            plot_confusion_matrix(y_test_np, y_pred, title=name)
            all_predictions.append(y_pred)
            all_probabilities.append(y_prob)
            model_names.append(name)
            curve_data[name] = (y_test_np, y_prob)

    stage2_model = load_stage2_model(device)
    test_feat_loader = make_feature_loader(X_test, y_test, shuffle=False)
    y_true, y_pred, y_prob = predict_model(stage2_model, test_feat_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "ResNet18 Frozen - Test")
    save_model_results("ResNet18 Frozen", metrics)
    plot_confusion_matrix(y_true, y_pred, title="ResNet18 Frozen")
    all_predictions.append(y_pred)
    all_probabilities.append(y_prob)
    model_names.append("ResNet18 Frozen")
    curve_data["ResNet18 Frozen"] = (y_true, y_prob)

    stage3_model = load_stage3_model(device)
    test_image_loader_eval = make_data_loader(test_df, shuffle=False)
    y_true, y_pred, y_prob = predict_model(stage3_model, test_image_loader_eval, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "ResNet18 Finetune - Test")
    save_model_results("ResNet18 Finetune", metrics)
    plot_confusion_matrix(y_true, y_pred, title="ResNet18 Finetune")
    all_predictions.append(y_pred)
    all_probabilities.append(y_prob)
    model_names.append("ResNet18 Finetune")
    curve_data["ResNet18 Finetune"] = (y_true, y_prob)

    if not all_predictions:
        raise RuntimeError("No trained models were available for Stage 4.")

    ensemble_pred = ensemble_vote(all_predictions)
    metrics_vote = compute_metrics(y_test_np, ensemble_pred)
    print_metrics(metrics_vote, f"Ensemble Hard Voting ({', '.join(model_names)})")
    save_model_results("Ensemble Hard Voting", metrics_vote)
    plot_confusion_matrix(y_test_np, ensemble_pred, title="Ensemble Hard Voting")

    ensemble_pred_soft, ensemble_prob = ensemble_average(all_probabilities)
    metrics_soft = compute_metrics(y_test_np, ensemble_pred_soft, ensemble_prob)
    print_metrics(metrics_soft, f"Ensemble Soft Voting ({', '.join(model_names)})")
    save_model_results("Ensemble Soft Voting", metrics_soft)
    plot_confusion_matrix(y_test_np, ensemble_pred_soft, title="Ensemble Soft Voting")
    curve_data["Ensemble Soft Voting"] = (y_test_np, ensemble_prob)

    plot_combined_roc_curves(curve_data)
    plot_combined_pr_curves(curve_data)
    plot_error_samples(test_df, y_test_np, ensemble_pred_soft)

    if RESULTS_CSV.exists():
        results_df = pd.read_csv(RESULTS_CSV)
        plot_model_comparison(results_df)
        metric_for_best = "roc_auc" if "roc_auc" in results_df.columns else "auc"
        best_model = results_df.loc[results_df[metric_for_best].idxmax()]
        print(
            f"Best model by {metric_for_best}: {best_model['model']} "
            f"({metric_for_best}={best_model[metric_for_best]:.4f})"
        )

    print("Stage 4 complete.")


if __name__ == "__main__":
    main()
