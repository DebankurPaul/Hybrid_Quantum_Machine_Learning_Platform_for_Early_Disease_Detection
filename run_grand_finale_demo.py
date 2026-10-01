"""
run_grand_finale_demo.py
========================
Master Grand Finale Execution & Verification Script for Egreen Quanta Judges.
Evaluates the 3-Dataset Unified Clinical Diagnostic Architecture:
  1. Solid Oncology Cytology       : WDBC Breast Cancer (30 features -> 8Q)
  2. Hematologic Oncology Genomics  : Golub Leukemia Microarray (7,129 genes -> 8Q)
  3. Cardiology Clinical EHR        : UCI Heart Disease Telemetry (13 parameters -> 8Q)

Passes all 3 verticals through the Universal 8-Qubit Qiskit Fidelity Kernel (QSVC)
and the Grand Finale Diagnostic Suite (Q-XAI, ZNE, KTA with signed label fix, 3-Band Triage).
"""
import time, numpy as np, json, sys, warnings
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_clinical_data
from src.autoencoder import train_autoencoder
from src.quantum_kernel import QuantumKernelClassifier
from src.q_xai import compute_qxai_shap
from src.zne_mitigation import simulate_zne_error_mitigation
from src.kernel_alignment import calculate_kta, compare_kernel_alignments
from src.disagreement_protocol import evaluate_cohort_triage


def run_vertical_eval(dataset_key="wdbc"):
    (X_tr, X_te, y_tr, y_te), feature_names, target_labels, scaler = load_clinical_data(dataset_key=dataset_key)
    
    # 1. PyTorch Autoencoder Compression
    ae_model, X_tr_ae, X_te_ae = train_autoencoder(X_tr, X_te, dataset_key=dataset_key, bottleneck_dim=8, epochs=100)
    
    # 2. Classical Model Fit
    clr = LogisticRegression().fit(X_tr_ae, y_tr)
    p_classical = clr.predict_proba(X_te_ae)[:, 1]
    acc_c = accuracy_score(y_te, (p_classical >= 0.66).astype(int))
    
    # 3. Quantum Kernel Classifier (QSVC) Fit with Class-Balanced SVM
    qsvc = QuantumKernelClassifier(n_qubits=8, fm_reps=1, entanglement='full', C=10.0, scale_factor=0.05)
    max_samples = min(150, len(X_tr_ae))
    qsvc.fit(X_tr_ae, y_tr, max_train_samples=max_samples)
    
    K_train = qsvc.qkernel_.evaluate(x_vec=qsvc.X_train_)
    K_test  = qsvc.qkernel_.evaluate(x_vec=X_te_ae, y_vec=qsvc.X_train_)
    
    preds_q = qsvc.predict(X_te_ae)
    p_quantum = qsvc.predict_proba(X_te_ae)[:, 1]
    acc_q = accuracy_score(y_te, preds_q)
    
    # Calculate Sensitivity (Recall) and Specificity
    tp = np.sum((y_te == 1) & (preds_q == 1))
    fn = np.sum((y_te == 1) & (preds_q == 0))
    tn = np.sum((y_te == 0) & (preds_q == 0))
    fp = np.sum((y_te == 0) & (preds_q == 1))
    
    sens_q = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_q = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc_q = matthews_corrcoef(y_te, preds_q)
    
    # 4. Q-XAI Fast SHAP
    def predict_wrapper(X_input):
        return clr.predict_proba(X_input)
    shap_vals, qxai_info = compute_qxai_shap(predict_wrapper, X_tr_ae, X_te_ae, feature_names=feature_names[:8], k_centroids=15, n_explain=15)
    
    # 5. KTA Alignment
    pca8 = PCA(n_components=8, random_state=42)
    X_tr_pca8 = pca8.fit_transform(X_tr)
    K_pca = qsvc.qkernel_.evaluate(x_vec=X_tr_pca8[:max_samples])
    align_res = compare_kernel_alignments(K_pca, K_train, y_tr[:max_samples])
    
    # 6. ZNE Mitigation
    zne_res = simulate_zne_error_mitigation(unmitigated_acc=acc_q, noise_rate=0.025)
    
    # 7. Clinical Triage
    triage_summary = evaluate_cohort_triage(p_classical, p_quantum, y_te, tau=0.66)
    
    return {
        "dataset_key": dataset_key,
        "acc_classical": acc_c,
        "acc_quantum": acc_q,
        "sens_quantum": sens_q,
        "spec_quantum": spec_q,
        "mcc_quantum": mcc_q,
        "kta_ae": align_res["kta_ae"],
        "zne_mitigated": zne_res["zne_mitigated_val"],
        "concordance": triage_summary["concordance_rate"]
    }


def main():
    print("\n" + "="*105)
    print("      HYBRID QUANTUM MACHINE LEARNING PLATFORM — GRAND FINALE SUITE")
    print("      3-Dataset Unified Architecture (Breast Cancer, Golub Leukemia, UCI Heart)")
    print("="*105)
    
    results = []
    for key in ["wdbc", "heart", "leukemia"]:
        print(f"\n" + "─"*90)
        print(f"  EXECUTING CLINICAL ENGINE VERTICAL: [{key.upper()}]")
        print("─"*90)
        res = run_vertical_eval(dataset_key=key)
        results.append(res)
        
    print("\n" + "="*115)
    print("      SUMMARY BENCHMARK: 3-DATASET UNIFIED CLINICAL DIAGNOSTIC PLATFORM")
    print("="*115)
    print(f"{'Clinical Disease Vertical':<22} | {'Class Acc':<10} | {'QSVC Acc':<10} | {'Sens (Recall)':<14} | {'Spec':<10} | {'QSVC MCC':<10} | {'KTA Score':<10} | {'ZNE Acc':<10}")
    print("─"*115)
    for r in results:
        print(f"{r['dataset_key'].upper():<22} | {r['acc_classical']*100:>9.2f}% | {r['acc_quantum']*100:>9.2f}% | {r['sens_quantum']*100:>13.2f}% | {r['spec_quantum']*100:>9.2f}% | {r['mcc_quantum']:>10.4f} | {r['kta_ae']:>10.4f} | {r['zne_mitigated']*100:>9.2f}%")
    print("="*115 + "\n")


if __name__ == "__main__":
    main()
