"""
test_qsvc_acc.py
================
Evaluates PyTorch Deep Autoencoder bottleneck features with Scikit-Learn SVM,
QSVC, and VQC across all thresholds and metrics.
"""
import time, numpy as np, sys, warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_curve, auc, brier_score_loss
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.autoencoder import train_autoencoder

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

X, y = load_breast_cancer(return_X_y=True)
y = 1 - y  # 1=Malignant, 0=Benign
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

print("\n" + "="*95)
print("  PYTORCH DEEP AUTOENCODER + ADVANCED MODEL EVALUATION")
print("="*95)

# Train PyTorch Autoencoder
ae_model, X_tr_ae, X_te_ae = train_autoencoder(X_tr_sc, X_te_sc, bottleneck_dim=8, epochs=150)

# Evaluate Classical LR & SVM on Autoencoder Features
lr_ae = LogisticRegression().fit(X_tr_ae, y_tr)
p_lr_ae = lr_ae.predict_proba(X_te_ae)[:, 1]

svm_ae = SVC(probability=True, C=10.0).fit(X_tr_ae, y_tr)
p_svm_ae = svm_ae.predict_proba(X_te_ae)[:, 1]

models = [
    ("Logistic Regression (Autoencoder 8D)", p_lr_ae),
    ("SVM RBF C=10 (Autoencoder 8D)", p_svm_ae)
]

print("\n" + "="*110)
print(f"{'Model Name':<38} | {'Tau':<4} | {'Acc':<6} | {'Sens':<6} | {'Spec':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6} | {'MCC':<6} | {'Brier':<6}")
print("="*110)

for name, probs in models:
    roc = roc_auc_score(y_te, probs)
    prec, rec, _ = precision_recall_curve(y_te, probs)
    pr_val = auc(rec, prec)
    brier = brier_score_loss(y_te, probs)
    
    for tau in [0.50, 0.60, 0.65]:
        pred = (probs >= tau).astype(int)
        cm = confusion_matrix(y_te, pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_te, pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_te, pred)
        print(f"{name:<38} | {tau:<4.2f} | {acc:<6.4f} | {sens:<6.4f} | {spec:<6.4f} | {roc:<7.4f} | {pr_val:<6.4f} | {mcc:<6.4f} | {brier:<6.4f}")

print("="*110)
