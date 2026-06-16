"""Stage 1: traditional machine-learning baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import StandardScaler

from src.data_utils import make_data_loader, print_split_summary
from src.experiment_utils import ensure_output_dirs, get_device, load_experiment_split, print_experiment_config
from src.feature_extraction import extract_features, get_feature_extractor
from src.models.baseline_ml import (
    create_logistic_regression,
    create_random_forest,
    create_xgboost,
    predict_ml_model,
    save_ml_models,
    save_scaler,
    train_ml_model,
)
from src.train_utils import compute_metrics, print_metrics, save_model_results
from src.visualization import plot_confusion_matrix, plot_roc_curve


def main():
    ensure_output_dirs()
    device = get_device()
    print_experiment_config("Stage 1: Traditional ML baselines", device)

    df = load_experiment_split()
    print_split_summary(df)

    feature_extractor = get_feature_extractor(device)
    loaders = {
        split: make_data_loader(df[df["split"] == split], shuffle=False)
        for split in ["train", "val", "test"]
    }

    feature_data = {}
    for split, loader in loaders.items():
        x, y = extract_features(feature_extractor, loader, device, desc=f"Feature {split}")
        feature_data[split] = (x.numpy(), y.numpy())

    X_train, y_train = feature_data["train"]
    X_val, y_val = feature_data["val"]
    X_test, y_test = feature_data["test"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    save_scaler(scaler)

    models = {}
    results = {}

    lr = train_ml_model(create_logistic_regression(), X_train_scaled, y_train)
    y_pred, y_prob = predict_ml_model(lr, X_test_scaled)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, "Logistic Regression - Test")
    models["Logistic Regression"] = lr
    results["Logistic Regression"] = (metrics, y_pred, y_prob)

    rf = train_ml_model(create_random_forest(), X_train, y_train)
    y_pred, y_prob = predict_ml_model(rf, X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, "Random Forest - Test")
    models["Random Forest"] = rf
    results["Random Forest"] = (metrics, y_pred, y_prob)

    xgb_model = create_xgboost(y_train)
    if xgb_model is not None:
        xgb_trained = train_ml_model(xgb_model, X_train, y_train, X_val, y_val)
        y_pred, y_prob = predict_ml_model(xgb_trained, X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics, "XGBoost - Test")
        models["XGBoost"] = xgb_trained
        results["XGBoost"] = (metrics, y_pred, y_prob)

    save_ml_models(models)

    for name, (metrics, y_pred, y_prob) in results.items():
        save_model_results(name, metrics)
        plot_confusion_matrix(y_test, y_pred, title=name)
        plot_roc_curve(y_test, y_prob, model_name=name)

    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
