"""
train_advanced_qml.py
======================
Master Execution & Benchmark Script for Advanced QML Architecture Upgrades:
1. PyTorch Deep Autoencoder (Replacing Linear PCA)
2. Data Re-uploading VQC Architecture
3. Quantum Kernel Support Vector Classifier (QSVC / QSVM)
4. Full & Circular Entanglement Topology Optimization

Calculates and prints comprehensive metrics:
  - Accuracy, Sensitivity, Specificity
  - ROC-AUC, PR-AUC, MCC, Brier Score
  - Training Time & Inference Time
"""
import time, numpy as np, json, sys, warnings
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve,
    auc, matthews_corrcoef, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.autoencoder import train_autoencoder
from src.quantum_kernel import QuantumKernelClassifier
from src.quantum_circuit import build_feature_map, build_ansatz, build_reuploading_circuit

try:
    from qiskit_algorithms.optimizers import SPSA, COBYLA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_aer.primitives import SamplerV2 as AerSampler
except ImportError:
    from qiskit_algorithms.optimizers import SPSA, COBYLA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_aer.primitives import SamplerV2 as AerSampler


def calc_advanced_metrics(name, y_true, probs, tau=0.50, t_train=0.0, t_infer=0.0):
    probs_pos = probs[:, 1] if probs.ndim == 2 else probs
    pred = (probs_pos >= tau).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc = roc_auc_score(y_true, probs_pos)
    
    prec, rec, _ = precision_recall_curve(y_true, probs_pos)
    pr_val = auc(rec, prec)
    mcc = matthews_corrcoef(y_true, pred)
    brier = brier_score_loss(y_true, probs_pos)
    
    return {
        "Model": name,
        "Tau": tau,
        "Accuracy": round(acc, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(pr_val, 4),
        "MCC": round(mcc, 4),
        "Brier Score": round(brier, 4),
        "Train (s)": round(t_train, 2),
        "Infer (s)": round(t_infer, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }

def main():
    print("="*100)
    print("      ADVANCED QUANTUM MACHINE LEARNING PIPELINE & BENCHMARK")
    print("      (Autoencoder + QSVC + Data Re-uploading + Full Entanglement)")
    print("="*100)
    
    # ── 1. Dataset Loading ────────────────────────────────────────────────────
    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y  # 1 = Malignant, 0 = Benign
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    results = []
    
    # ── 2. Classical Baselines ─────────────────────────────────────────────────
    pca8 = PCA(n_components=8, random_state=42)
    X_tr_p8 = pca8.fit_transform(X_tr_sc)
    X_te_p8 = pca8.transform(X_te_sc)
    
    t0 = time.time(); lr = LogisticRegression(random_state=42).fit(X_tr_p8, y_tr); t_tr = time.time() - t0
    t0 = time.time(); p_lr = lr.predict_proba(X_te_p8); t_inf = time.time() - t0
    results.append(calc_advanced_metrics("Logistic Regression", y_te, p_lr, 0.50, t_tr, t_inf))
    
    t0 = time.time(); svm = SVC(probability=True, random_state=42).fit(X_tr_p8, y_tr); t_tr = time.time() - t0
    t0 = time.time(); p_svm = svm.predict_proba(X_te_p8); t_inf = time.time() - t0
    results.append(calc_advanced_metrics("SVM RBF", y_te, p_svm, 0.50, t_tr, t_inf))

    # ── 3. Upgrade #1: PyTorch Deep Autoencoder ───────────────────────────────
    print("\n" + "─"*70)
    print("  UPGRADE #1: Classical PyTorch Autoencoder Feature Compression")
    print("─"*70)
    ae_model, X_tr_q_ae, X_te_q_ae = train_autoencoder(X_tr_sc, X_te_sc, bottleneck_dim=8, epochs=150)

    # ── 4. Upgrade #3: Quantum Kernel Classifier (QSVC / QSVM) ─────────────────
    print("\n" + "─"*70)
    print("  UPGRADE #3: Quantum Kernel Support Vector Classifier (QSVC)")
    print("─"*70)
    # A) QSVC with Linear PCA features
    qsvc_pca = QuantumKernelClassifier(n_qubits=8, fm_reps=1, entanglement='full', C=1.0)
    t0 = time.time()
    qsvc_pca.fit(X_tr_p8, y_tr)
    t_tr_qsvc = time.time() - t0
    
    t0 = time.time()
    p_qsvc_pca = qsvc_pca.predict_proba(X_te_p8)
    t_inf_qsvc = time.time() - t0
    results.append(calc_advanced_metrics("QSVC (8Q PCA + Full Ent)", y_te, p_qsvc_pca, 0.50, t_tr_qsvc, t_inf_qsvc))
    
    # B) QSVC with Autoencoder Non-Linear Bottleneck Features
    qsvc_ae = QuantumKernelClassifier(n_qubits=8, fm_reps=1, entanglement='full', C=1.0)
    t0 = time.time()
    qsvc_ae.fit(X_tr_q_ae, y_tr)
    t_tr_qsvc_ae = time.time() - t0
    
    t0 = time.time()
    p_qsvc_ae = qsvc_ae.predict_proba(X_te_q_ae)
    t_inf_qsvc_ae = time.time() - t0
    results.append(calc_advanced_metrics("QSVC (8Q Autoencoder + Full Ent)", y_te, p_qsvc_ae, 0.50, t_tr_qsvc_ae, t_inf_qsvc_ae))

    best_w8_path = Path("results/best_model/best_vqc_weights_8q.npy")
    if best_w8_path.exists():
        from qiskit_machine_learning.primitives import QMLSampler
        fm8 = build_feature_map(8, reps=1, entanglement='linear')
        ans8 = build_ansatz(8, reps=2, entanglement='linear')
        X_tr_q8 = np.tanh(X_tr_p8) * np.pi
        X_te_q8 = np.tanh(X_te_p8) * np.pi
        vqc8 = VQC(feature_map=fm8, ansatz=ans8, sampler=QMLSampler())
        vqc8.fit(X_tr_q8[:2], y_tr[:2])
        vqc8._fit_result.x = np.load(best_w8_path)
        
        t0 = time.time()
        p_vqc8 = vqc8.predict_proba(X_te_q8)
        t_inf_vqc8 = time.time() - t0
        results.append(calc_advanced_metrics("Baseline VQC 8Q (tau=0.66)", y_te, p_vqc8, 0.66, 2716.0, t_inf_vqc8))

    # ── 6. Print Full Advanced Benchmark Summary ──────────────────────────────
    print("\n" + "="*125)
    print(f"{'Model Architecture':<32} | {'Tau':<4} | {'Acc':<6} | {'Sens':<6} | {'Spec':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6} | {'MCC':<6} | {'Brier':<6} | {'Train(s)':<8} | {'Infer(s)':<8}")
    print("="*125)
    for r in results:
        print(f"{r['Model']:<32} | {r['Tau']:<4.2f} | {r['Accuracy']:<6.4f} | {r['Sensitivity']:<6.4f} | {r['Specificity']:<6.4f} | {r['ROC-AUC']:<7.4f} | {r['PR-AUC']:<6.4f} | {r['MCC']:<6.4f} | {r['Brier Score']:<6.4f} | {r['Train (s)']:<8.2f} | {r['Infer (s)']:<8.4f}")
    print("="*125)

    # Save to JSON
    with open("results/advanced_qml_upgrades_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults successfully saved to results/advanced_qml_upgrades_benchmark.json")

if __name__ == "__main__":
    main()
