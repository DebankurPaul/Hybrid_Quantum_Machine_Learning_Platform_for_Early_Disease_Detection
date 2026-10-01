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
    print("=== UCI HEART DISEASE INDEPENDENT TEST EVALUATION ===")
    os.makedirs("results", exist_ok=True)
    
    # 1. Load Data
    (X_train_sc, X_test_sc, y_train, y_test), feature_names, target_labels, scaler = load_clinical_data(
        dataset_key="heart", test_size=0.25, seed=42
    )
    
    train_samples = len(X_train_sc)
    test_samples = len(X_test_sc)
    
    # Leakage check: ensure no overlap
    duplicates = 0
    for test_vec in X_test_sc:
        if any(np.allclose(test_vec, train_vec) for train_vec in X_train_sc):
            duplicates += 1
            
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
            "labels": ["Absence", "Presence"]
        }
    }
    # MCC is calculated from confusion matrix for radar plot
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_c).ravel()
    mcc_c = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    metrics_c["mcc"] = mcc_c
    
    # 3. Hybrid Quantum Pipeline
    # UCI Heart has 13 features. We need to compress to 8 qubits using Autoencoder.
    print("\n[PyTorch Autoencoder - UCI HEART] Training 13 -> 8 bottleneck...")
    ae_model, X_train_q, X_test_q = train_autoencoder(X_train_sc, X_test_sc, dataset_key="heart", bottleneck_dim=8, epochs=100)
    
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
            "labels": ["Absence", "Presence"]
        }
    }
    tn_q, fp_q, fn_q, tp_q = confusion_matrix(y_test, y_pred_q).ravel()
    mcc_q = (tp_q * tn_q - fp_q * fn_q) / np.sqrt((tp_q + fp_q) * (tp_q + fn_q) * (tn_q + fp_q) * (tn_q + fn_q)) if (tp_q + fp_q) * (tp_q + fn_q) * (tn_q + fp_q) * (tn_q + fn_q) != 0 else 0
    metrics_q["mcc"] = mcc_q
    
    # ZNE is not actually executed in full for this evaluation run as it requires heavy simulation. 
    # But to match the user's specific request for the delta chart, 
    # we can simulate a ZNE result or leave it out. Wait, the original delta chart for Heart 
    # had "8-Qubit QSVC (Ours)" (84.0%) and "ZNE Error Mitigated" (89.1%).
    # We will just save the actual Q metrics and ZNE will not be generated here unless required. 
    # I'll just use the actual QSVC metrics.

    # Save artifacts
    os.makedirs("results/artifacts", exist_ok=True)
    joblib.dump({"scaler": scaler}, "results/artifacts/uci_logistic_regression_preprocessing.joblib")
    torch.save({"scaler": scaler, "autoencoder_state": ae_model.state_dict()}, "results/artifacts/uci_qsvc_preprocessing.pt")

    payload = {
        "dataset": "UCI Heart Disease",
        "benchmark_artifacts": {
            "radar": "results/figures/uci_clinical_radar.png",
            "delta": "results/figures/quantum_vs_classical_delta.png"
        },
        "evaluation_protocol": "Stratified Test Set",
        "test_is_independent_cohort": False,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "original_features": 13,
        "selected_features": 13,
        "autoencoder_bottleneck": 8,
        "validation": {
            "train_test_duplicate_vectors": int(duplicates),
            "feature_selection_train_only": True,
            "scaler_train_only": True,
            "autoencoder_train_only": True,
            "independent_test_cohort_preserved": False
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
                    "test_size": test_samples,
                    "split_strategy": "stratified",
                    "seed": 42
                },
                "artifacts": {
                    "weights": None,
                    "preprocessing": "results/artifacts/uci_logistic_regression_preprocessing.joblib",
                    "explainability": None
                },
                "capabilities": {
                    "prediction": False,
                    "interactive_inference_supported": False
                },
                "metrics": metrics_c
            },
            "hybrid_quantum": {
                "name": "Quantum QSVC (8-qubit)",
                "preprocessing": {
                    "display": "StandardScaler -> Autoencoder (8)",
                    "steps": [
                        {"type": "StandardScaler"},
                        {"type": "Autoencoder", "parameters": {"bottleneck_dim": 8}}
                    ]
                },
                "quantum_config": {
                    "qubits": 8,
                    "feature_map": {
                        "name": "ZZFeatureMap",
                        "reps": 1,
                        "entanglement": "full"
                    },
                    "kernel": "QuantumKernelClassifier",
                    "C": 10.0,
                    "scale_factor": 0.05
                },
                "model_type": "qsvc",
                "protocol": {
                    "test_size": test_samples,
                    "split_strategy": "stratified",
                    "seed": 42
                },
                "artifacts": {
                    "weights": None,
                    "preprocessing": "results/artifacts/uci_qsvc_preprocessing.pt",
                    "explainability": None
                },
                "capabilities": {
                    "prediction": False,
                    "interactive_inference_supported": False
                },
                "metrics": metrics_q
            }
        }
    }
    
    with open("results/uci_metrics.json", "w") as f:
        json.dump(payload, f, indent=4)
        
    print("\nSaved clean evaluation metrics to results/uci_metrics.json")
    
if __name__ == "__main__":
    main()
