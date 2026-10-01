import json
import time
import numpy as np
import os
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.primitives import QMLSampler
import joblib

from src.data_loader import load_clinical_data
from src.quantum_circuit import build_feature_map, build_ansatz

def evaluate_model(y_true, y_pred, y_proba=None, labels=["Benign", "Malignant"]):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred)), # recall = sensitivity
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1_score": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": labels
        },
        "n_test": len(y_true)
    }
    return metrics

def main():
    print("=== WDBC INDEPENDENT TEST EVALUATION ===")
    
    # 1. Load Data
    splits, feature_names, target_labels, scaler = load_clinical_data(dataset_key="wdbc", test_size=0.2, seed=42)
    X_train_sc, X_test_sc, y_train, y_test = splits
    print(f"Test Set Size: {len(y_test)} samples")
    
    # 2. Classical Baseline (Logistic Regression on all 30 features)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    lr.fit(X_train_sc, y_train)
    y_pred_c = lr.predict(X_test_sc)
    y_proba_c = lr.predict_proba(X_test_sc)[:, 1]
    
    metrics_c = evaluate_model(y_test, y_pred_c, y_proba_c)
    print("\n[Classical - Logistic Regression]")
    print(f"Accuracy: {metrics_c['accuracy']:.4f}, Sens: {metrics_c['sensitivity']:.4f}, Spec: {metrics_c['specificity']:.4f}")
    
    # 3. Hybrid Quantum Pipeline
    # WDBC VQC model uses PCA to 8 qubits, then tanh scaling
    pca8 = PCA(n_components=8, random_state=42)
    X_train_p8 = pca8.fit_transform(X_train_sc)
    X_test_p8 = pca8.transform(X_test_sc)
    
    X_train_q8 = np.tanh(X_train_p8) * np.pi
    X_test_q8 = np.tanh(X_test_p8) * np.pi
    
    best_weights_path = "results/best_model/best_vqc_weights_8q.npy"
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Could not find {best_weights_path}")
        
    weights = np.load(best_weights_path)
    
    print("\n[VQC] Instantiating 8-qubit VQC...")
    fm8 = build_feature_map(8, reps=1, entanglement='linear')
    ans8 = build_ansatz(8, reps=2, entanglement='linear')
    
    vqc8 = VQC(feature_map=fm8, ansatz=ans8, sampler=QMLSampler())
    # Dummy fit to initialize parameters structure before injecting weights
    vqc8.fit(X_train_q8[:2], y_train[:2])
    vqc8._fit_result.x = weights
    
    print("[VQC] Running test set predictions...")
    # Get raw probabilities
    p_vqc8 = vqc8.predict_proba(X_test_q8)[:, 1]
    
    # Apply optimal threshold identified previously (tau=0.65)
    tau = 0.65
    y_pred_q = (p_vqc8 >= tau).astype(int)
    
    metrics_q = evaluate_model(y_test, y_pred_q, p_vqc8)
    print("\n[Quantum - 8Q VQC]")
    print(f"Accuracy: {metrics_q['accuracy']:.4f}, Sens: {metrics_q['sensitivity']:.4f}, Spec: {metrics_q['specificity']:.4f}")
    
    os.makedirs("results/artifacts", exist_ok=True)
    joblib.dump({"scaler": scaler}, "results/artifacts/wdbc_logistic_regression_preprocessing.joblib")
    joblib.dump({"scaler": scaler, "pca": pca8}, "results/artifacts/wdbc_vqc_preprocessing.joblib")
    
    # Save results
    out_dict = {
        "dataset": "wdbc",
        "n_test": len(y_test),
        "benchmark_artifacts": {
            "radar": "results/figures/wdbc_clinical_radar.png",
            "delta": "results/figures/quantum_vs_classical_delta.png"
        },
        "models": {
            "classical_baseline": {
                "name": "Logistic Regression",
                "preprocessing": {
                    "display": "StandardScaler",
                    "steps": [
                        {"type": "StandardScaler"}
                    ]
                },
                "model_type": "logistic_regression",
                "protocol": {
                    "test_size": len(y_test),
                    "split_strategy": "stratified",
                    "seed": 42
                },
                "artifacts": {
                    "weights": None,
                    "preprocessing": "results/artifacts/wdbc_logistic_regression_preprocessing.joblib",
                    "explainability": None
                },
                "capabilities": {
                    "prediction": False,
                    "interactive_inference_supported": False
                },
                "metrics": metrics_c
            },
            "hybrid_quantum": {
                "name": "8-Qubit VQC",
                "preprocessing": {
                    "display": "StandardScaler -> PCA (8) -> Tanh",
                    "steps": [
                        {"type": "StandardScaler"},
                        {"type": "PCA", "parameters": {"n_components": 8, "random_state": 42}},
                        {"type": "TanhScale", "parameters": {"multiplier": "pi"}}
                    ]
                },
                "quantum_config": {
                    "qubits": 8,
                    "feature_map": {
                        "name": "ZZFeatureMap",
                        "reps": 1,
                        "entanglement": "linear"
                    },
                    "ansatz": {
                        "name": "RealAmplitudes",
                        "reps": 2,
                        "entanglement": "linear"
                    },
                    "backend": "QMLSampler",
                    "tau": 0.65
                },
                "model_type": "vqc",
                "protocol": {
                    "test_size": len(y_test),
                    "split_strategy": "stratified",
                    "seed": 42
                },
                "artifacts": {
                    "weights": "results/best_model/best_vqc_weights_8q.npy",
                    "preprocessing": "results/artifacts/wdbc_vqc_preprocessing.joblib",
                    "explainability": None
                },
                "capabilities": {
                    "prediction": True,
                    "interactive_inference_supported": True
                },
                "metrics": metrics_q
            }
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/wdbc_metrics.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print("\nSaved clean evaluation metrics to results/wdbc_metrics.json")

if __name__ == "__main__":
    main()
