"""
Train Classical Machine Learning Models
========================================

Dataset:
    Wisconsin Diagnostic Breast Cancer (WDBC)

Task:
    Binary classification
        0 = Benign
        1 = Malignant

Features:
    All 30 original WDBC numerical features

Models:
    1. Logistic Regression
    2. Support Vector Machine (SVM)
    3. Random Forest
    4. XGBoost

Pipeline:
    Raw WDBC data
        -> Remove ID
        -> Encode diagnosis
        -> Stratified train/validation/test split
        -> Standardization
        -> Model training
        -> Validation evaluation
        -> Final test evaluation
        -> Save models and results
"""

from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# train_classical.py
#      |
#      └── classical_models/
#              |
#              └── train_classical.py
#
# parents[0] = classical_models
# parents[1] = src
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_PATH = RAW_DATA_DIR / "wdbc.data"


# Create output directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. WDBC COLUMN NAMES
# ============================================================

COLUMNS = [
    "id",
    "diagnosis",

    # Mean features
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    # Standard error features
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    # Worst features
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


FEATURE_COLUMNS = COLUMNS[2:]


# ============================================================
# 3. RANDOM SEED
# ============================================================

RANDOM_STATE = 42


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data():
    """Load the raw WDBC dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nWDBC dataset not found:\n"
            f"{RAW_DATA_PATH}\n\n"
            f"Place 'wdbc.data' inside:\n"
            f"{RAW_DATA_DIR}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=COLUMNS,
    )

    return df


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):
    """
    Remove ID and encode diagnosis.

    B = 0 = Benign
    M = 1 = Malignant
    """

    # ID is not a predictive medical feature
    X = df.drop(columns=["id", "diagnosis"])

    # Encode target
    y = df["diagnosis"].map({
        "B": 0,
        "M": 1,
    })

    # Safety checks
    if y.isnull().any():
        raise ValueError(
            "Unexpected diagnosis value found in dataset."
        )

    if X.shape[1] != 30:
        raise ValueError(
            f"Expected 30 features, found {X.shape[1]}."
        )

    return X, y


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(X, y):
    """
    Split dataset into:

        60% training
        20% validation
        20% test

    Stratification preserves the class distribution.
    """

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# 7. SCALE FEATURES
# ============================================================

def scale_data(
    X_train,
    X_val,
    X_test,
):
    """
    Standardize features.

    IMPORTANT:
    The scaler is fitted ONLY on the training data.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = scaler.transform(X_test)

    return (
        scaler,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
    )


# ============================================================
# 8. DEFINE MODELS
# ============================================================

def create_models():
    """Create all classical baseline models."""

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y):
    """
    Calculate classification metrics.

    Positive class:
        1 = Malignant

    Therefore:
        Recall/Sensitivity = ability to detect malignant cases
        Specificity = ability to correctly identify benign cases
    """

    y_pred = model.predict(X)

    # Probability of malignant class
    y_prob = model.predict_proba(X)[:, 1]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y,
        y_pred,
    ).ravel()

    # Metrics
    accuracy = accuracy_score(y, y_pred)

    precision = precision_score(
        y,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y,
        y_pred,
        zero_division=0,
    )

    specificity = tn / (tn + fp)

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        y_prob,
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ============================================================
# 10. TRAIN MODELS
# ============================================================

def train_models(
    models,
    X_train,
    y_train,
):
    """Train every classical model."""

    trained_models = {}
    training_times = {}

    for name, model in models.items():

        print(f"\nTraining: {name}")

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        elapsed = time.perf_counter() - start_time

        trained_models[name] = model
        training_times[name] = elapsed

        print(
            f"Completed in {elapsed:.4f} seconds"
        )

    return trained_models, training_times


# ============================================================
# 11. EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    trained_models,
    X,
    y,
    dataset_name,
    training_times=None,
):
    """Evaluate every trained model."""

    results = {}
    rows = []

    for name, model in trained_models.items():

        metrics = evaluate_model(
            model,
            X,
            y,
        )

        results[name] = metrics

        row = {
            "Dataset": dataset_name,
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "F1 Score": metrics["F1 Score"],
            "ROC-AUC": metrics["ROC-AUC"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
        }

        if training_times is not None:
            row["Training Time (sec)"] = training_times[name]

        rows.append(row)

    return results, pd.DataFrame(rows)


# ============================================================
# 12. SAVE MODELS
# ============================================================

def save_models(
    trained_models,
    scaler,
):
    """Save trained models and scaler."""

    scaler_path = MODELS_DIR / "classical_scaler.joblib"

    joblib.dump(
        scaler,
        scaler_path,
    )

    print(f"\nSaved scaler: {scaler_path}")

    for name, model in trained_models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".joblib"
        )

        model_path = MODELS_DIR / filename

        joblib.dump(
            model,
            model_path,
        )

        print(f"Saved model: {model_path}")


# ============================================================
# 13. SAVE CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(
    evaluation_results,
    dataset_name,
):
    """Save confusion matrices for every model."""

    for name, metrics in evaluation_results.items():

        cm = np.array([
            [metrics["TN"], metrics["FP"]],
            [metrics["FN"], metrics["TP"]],
        ])

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        path = (
            RESULTS_DIR
            / f"{dataset_name.lower()}_{safe_name}_confusion_matrix.csv"
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual_Benign", "Actual_Malignant"],
            columns=["Predicted_Benign", "Predicted_Malignant"],
        )

        cm_df.to_csv(path)


# ============================================================
# 14. SAVE ROC DATA
# ============================================================

def save_roc_data(
    evaluation_results,
    y,
    dataset_name,
):
    """Save ROC curve coordinates."""

    roc_rows = []

    for name, metrics in evaluation_results.items():

        fpr, tpr, thresholds = roc_curve(
            y,
            metrics["y_prob"],
        )

        for fpr_value, tpr_value, threshold in zip(
            fpr,
            tpr,
            thresholds,
        ):

            roc_rows.append({
                "Dataset": dataset_name,
                "Model": name,
                "False Positive Rate": fpr_value,
                "True Positive Rate": tpr_value,
                "Threshold": threshold,
            })

    roc_df = pd.DataFrame(roc_rows)

    path = (
        RESULTS_DIR
        / f"{dataset_name.lower()}_roc_data.csv"
    )

    roc_df.to_csv(
        path,
        index=False,
    )

    return roc_df


# ============================================================
# 15. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    trained_models,
):
    """Save Random Forest feature importance."""

    if "Random Forest" not in trained_models:
        return

    model = trained_models["Random Forest"]

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    )

    path = (
        RESULTS_DIR
        / "random_forest_feature_importance.csv"
    )

    importance_df.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved feature importance: {path}"
    )


# ============================================================
# 16. PRINT RESULTS
# ============================================================

def print_results(results_df):
    """Print formatted model results."""

    metric_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1 Score",
        "ROC-AUC",
    ]

    display_df = results_df[
        metric_columns
    ].copy()

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1 Score",
        "ROC-AUC",
    ]

    display_df[numeric_columns] = (
        display_df[numeric_columns]
        .round(4)
    )

    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE")
    print("=" * 90)

    print(
        display_df.to_string(
            index=False
        )
    )

    print("=" * 90)


# ============================================================
# 17. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 90)
    print("WDBC CLASSICAL MACHINE LEARNING PIPELINE")
    print("=" * 90)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\n[1/8] Loading dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    print("\n[2/8] Preparing features and target...")

    X, y = prepare_data(df)

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")

    print("\nClass distribution:")

    print(
        y.value_counts()
        .rename(
            index={
                0: "Benign",
                1: "Malignant",
            }
        )
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print("\n[3/8] Creating train/validation/test split...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Testing:    {X_test.shape}")

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    print("\n[4/8] Standardizing features...")

    (
        scaler,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
    ) = scale_data(
        X_train,
        X_val,
        X_test,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("\n[5/8] Creating classical models...")

    models = create_models()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n[6/8] Training models...")

    (
        trained_models,
        training_times,
    ) = train_models(
        models,
        X_train_scaled,
        y_train,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n[7/8] Evaluating validation set...")

    (
        validation_results,
        validation_df,
    ) = evaluate_all_models(
        trained_models,
        X_val_scaled,
        y_val,
        "Validation",
        training_times,
    )

    print_results(validation_df)

    validation_path = (
        RESULTS_DIR
        / "classical_validation_results.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print("\n[8/8] Evaluating final test set...")

    (
        test_results,
        test_df,
    ) = evaluate_all_models(
        trained_models,
        X_test_scaled,
        y_test,
        "Test",
        training_times,
    )

    print_results(test_df)

    test_path = (
        RESULTS_DIR
        / "classical_test_results.csv"
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    print("\nSaving trained models and artifacts...")

    save_models(
        trained_models,
        scaler,
    )

    save_confusion_matrices(
        test_results,
        "test",
    )

    save_roc_data(
        test_results,
        y_test,
        "test",
    )

    save_feature_importance(
        trained_models,
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CLASSICAL PIPELINE COMPLETE")
    print("=" * 90)

    print(f"\nModels saved to:")
    print(MODELS_DIR)

    print(f"\nResults saved to:")
    print(RESULTS_DIR)

    print("\nFinal test results:")

    print_results(test_df)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()