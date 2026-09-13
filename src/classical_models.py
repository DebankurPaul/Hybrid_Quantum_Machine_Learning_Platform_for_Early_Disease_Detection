"""
classical_models.py
=====================
Trains three classical ML baselines on the WDBC dataset.
These models use ALL 30 features (no PCA reduction) for maximum fairness.

Models:
  1. Logistic Regression   — linear, fast, interpretable
  2. SVM (RBF kernel)      — non-linear, typically best classical result
  3. Random Forest         — ensemble, handles feature interactions

Usage:
    from src.classical_models import train_classical_baselines
    results = train_classical_baselines(X_train, X_test, y_train, y_test)
"""
import numpy as np
import time

from sklearn.linear_model  import LogisticRegression
from sklearn.svm            import SVC
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
)


CLASSICAL_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42
    ),
    "SVM (RBF Kernel)": SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=42
    ),
}


def train_classical_baselines(
    X_train : np.ndarray,
    X_test  : np.ndarray,
    y_train : np.ndarray,
    y_test  : np.ndarray,
) -> dict:
    """
    Train and evaluate all three classical models.

    Note: Classical models get all 30 raw features (standard-scaled only,
    no PCA) to give them their full advantage vs. the VQC which only sees
    PCA-4 features.

    Args:
        X_train, X_test : raw feature arrays (n_samples, 30)
        y_train, y_test : label arrays (0/1)

    Returns:
        results dict:
            {model_name: {accuracy, roc_auc, f1, sensitivity, specificity,
                          train_time, model, scaler}}
    """
    # Scale features (fit on train only)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    results = {}

    for name, clf in CLASSICAL_MODELS.items():
        print(f"\n── {name} ─────────────────────────────────────────")

        t0 = time.time()
        clf.fit(X_tr_sc, y_train)
        train_time = time.time() - t0

        y_pred  = clf.predict(X_te_sc)
        y_proba = clf.predict_proba(X_te_sc)[:, 1]

        cm_vals = _confusion_vals(y_test, y_pred)
        results[name] = {
            "accuracy"    : accuracy_score(y_test, y_pred),
            "roc_auc"     : roc_auc_score(y_test, y_proba),
            "f1_score"    : f1_score(y_test, y_pred),
            "sensitivity" : recall_score(y_test, y_pred),
            "specificity" : cm_vals['specificity'],
            "precision"   : precision_score(y_test, y_pred, zero_division=0),
            "train_time"  : train_time,
            "model"       : clf,
            "scaler"      : scaler,
        }

        r = results[name]
        print(f"  Accuracy    : {r['accuracy']:.4f}")
        print(f"  AUC-ROC     : {r['roc_auc']:.4f}")
        print(f"  F1 Score    : {r['f1_score']:.4f}")
        print(f"  Sensitivity : {r['sensitivity']:.4f}")
        print(f"  Specificity : {r['specificity']:.4f}")
        print(f"  Train time  : {train_time:.4f}s")
        print(classification_report(y_test, y_pred,
                                     target_names=["Benign", "Malignant"],
                                     zero_division=0))

    return results


def _confusion_vals(y_true, y_pred) -> dict:
    """Extract TP, TN, FP, FN and derived metrics from predictions."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }
