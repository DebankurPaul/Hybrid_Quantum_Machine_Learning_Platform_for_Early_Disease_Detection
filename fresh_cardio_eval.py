import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.data_loader import load_clinical_data
from src.autoencoder import train_autoencoder
from src.quantum_kernel import QuantumKernelClassifier
from qiskit.circuit.library import ZZFeatureMap
import torch
import joblib
import warnings
warnings.filterwarnings("ignore")

def main():
    print("=== CARDIO INDEPENDENT TEST EVALUATION ===")
    os.makedirs("results", exist_ok=True)
    
    # 1. Load Data
    (X_train_sc, X_test_sc, y_train, y_test), feature_names, target_labels, scaler = load_clinical_data(
        dataset_key="cardio", test_size=0.25, seed=42
    )
    
    # Subsampling for speed in this demo environment
    np.random.seed(42)
    train_idx = np.random.choice(len(y_train), 500, replace=False)
    test_idx = np.random.choice(len(y_test), 100, replace=False)
    
    X_train_sc, y_train = X_train_sc[train_idx], y_train[train_idx]
    X_test_sc, y_test = X_test_sc[test_idx], y_test[test_idx]

    train_samples = len(X_train_sc)
    test_samples = len(X_test_sc)
    
    # 2. Classical Baseline (Logistic Regression)
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_sc, y_train)
    y_pred_c = clf.predict(X_test_sc)
    y_prob_c = clf.predict_proba(X_test_sc)[:, 1]
    
    metrics_c = {
        "accuracy": accuracy_score(y_test, y_pred_c),
        "precision": precision_score(y_test, y_pred_c, zero_division=0),
        "sensitivity": recall_score(y_test, y_pred_c),
        "specificity": recall_score(y_test, y_pred_c, pos_label=0),
        "f1": f1_score(y_test, y_pred_c),
        "roc_auc": roc_auc_score(y_test, y_prob_c),
        "confusion_matrix": {
            "matrix": confusion_matrix(y_test, y_pred_c).tolist(),
            "labels": target_labels
        }
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_c).ravel()
    mcc_c = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    metrics_c["mcc"] = mcc_c
    
    # 3. Hybrid Quantum Pipeline
    print("\n[PyTorch Autoencoder - CARDIO] Training bottleneck...")
    ae_model, X_train_q, X_test_q = train_autoencoder(X_train_sc, X_test_sc, dataset_key="cardio", bottleneck_dim=8, epochs=50)
    
    print("[QSVC] Instantiating QuantumKernelClassifier...")
    qkc = QuantumKernelClassifier(n_qubits=8, fm_reps=1, entanglement='full', C=10.0, scale_factor=0.05)
    print("\n[QSVC] Computing Quantum Kernel Matrix for training samples...")
    qkc.fit(X_train_q, y_train)
    y_pred_q = qkc.predict(X_test_q)
    
    metrics_q = {
        "accuracy": accuracy_score(y_test, y_pred_q),
        "precision": precision_score(y_test, y_pred_q, zero_division=0),
        "sensitivity": recall_score(y_test, y_pred_q),
        "specificity": recall_score(y_test, y_pred_q, pos_label=0),
        "f1": f1_score(y_test, y_pred_q),
        "roc_auc": None,
        "confusion_matrix": {
            "matrix": confusion_matrix(y_test, y_pred_q).tolist(),
            "labels": target_labels
        }
    }
    tn_q, fp_q, fn_q, tp_q = confusion_matrix(y_test, y_pred_q).ravel()
    mcc_q = (tp_q * tn_q - fp_q * fn_q) / np.sqrt((tp_q + fp_q) * (tp_q + fn_q) * (tn_q + fp_q) * (tn_q + fn_q)) if (tp_q + fp_q) * (tp_q + fn_q) * (tn_q + fp_q) * (tn_q + fn_q) != 0 else 0
    metrics_q["mcc"] = mcc_q
    
    os.makedirs("results/artifacts", exist_ok=True)
    joblib.dump({"scaler": scaler}, "results/artifacts/cardio_logistic_regression_preprocessing.joblib")
    torch.save({"scaler": scaler, "autoencoder_state": ae_model.state_dict()}, "results/artifacts/cardio_qsvc_preprocessing.pt")

    payload = {
        "dataset": "Cardio",
        "benchmark_artifacts": {
            "radar": "results/figures/cardio_clinical_radar.png",
            "delta": "results/figures/quantum_vs_classical_delta.png"
        },
        "evaluation_protocol": "Stratified Test Set (Subsampled)",
        "models": {
            "classical_baseline": {
                "name": "Logistic Regression",
                "metrics": metrics_c
            },
            "hybrid_quantum": {
                "name": "Quantum QSVC (8-qubit)",
                "metrics": metrics_q
            }
        }
    }
    
    with open("results/cardio_metrics.json", "w") as f:
        json.dump(payload, f, indent=4)
        
    print("\nSaved clean evaluation metrics to results/cardio_metrics.json")
    
if __name__ == "__main__":
    main()
