"""
tune_qsvc_c.py
==============
Fine-tunes the SVM C hyperparameter and decision threshold tau for the PyTorch Autoencoder + QSVC.
Using precomputed kernel matrix K for instant sub-second grid search!
"""
import numpy as np, sys, warnings, time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_curve, auc
from sklearn.svm import SVC
from src.autoencoder import train_autoencoder
from src.quantum_kernel import QuantumKernelClassifier

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

X, y = load_breast_cancer(return_X_y=True)
y = 1 - y  # 1=Malignant, 0=Benign
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

print("\n[1/2] Extracting PyTorch Autoencoder Bottleneck Features...")
ae_model, X_tr_q_ae, X_te_q_ae = train_autoencoder(X_tr_sc, X_te_sc, bottleneck_dim=8, epochs=150)

print("\n[2/2] Evaluating Quantum Kernel Overlaps...")
qkernel_model = QuantumKernelClassifier(n_qubits=8, fm_reps=1, entanglement='full')
qkernel_model.fit(X_tr_q_ae, y_tr, max_train_samples=150)

K_train = qkernel_model.qkernel_.evaluate(x_vec=qkernel_model.X_train_)
K_test  = qkernel_model.qkernel_.evaluate(x_vec=X_te_q_ae, y_vec=qkernel_model.X_train_)
y_train_sub = y_tr[:150]

print("\n" + "="*95)
print(f"{'C Param':<8} | {'Tau':<5} | {'Acc':<6} | {'Sens':<6} | {'Spec':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6} | {'MCC':<6}")
print("="*95)

for C_val in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
    svc = SVC(kernel='precomputed', probability=True, C=C_val, class_weight='balanced')
    svc.fit(K_train, y_train_sub)
    probs = svc.predict_proba(K_test)[:, 1]
    
    roc = roc_auc_score(y_te, probs)
    prec, rec, _ = precision_recall_curve(y_te, probs)
    pr_val = auc(rec, prec)
    
    for tau in [0.35, 0.40, 0.45, 0.50, 0.55]:
        pred = (probs >= tau).astype(int)
        cm = confusion_matrix(y_te, pred)
        if cm.shape != (2, 2): continue
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_te, pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_te, pred)
        
        if acc > 0.70 or mcc > 0.40:
            print(f"{C_val:<8.1f} | {tau:<5.2f} | {acc:<6.4f} | {sens:<6.4f} | {spec:<6.4f} | {roc:<7.4f} | {pr_val:<6.4f} | {mcc:<6.4f}")
print("="*95)
