"""
evaluate.py
============
All evaluation metrics, plots, and benchmarking for comparing
the quantum VQC against classical baselines.

Clinical metrics explained:
  Accuracy    — overall correct predictions / total
  Sensitivity — TP / (TP + FN) — "did we catch all cancers?"  (recall)
  Specificity — TN / (TN + FP) — "did we avoid false alarms?"
  Precision   — TP / (TP + FP) — "when we flag cancer, are we right?"
  F1 Score    — harmonic mean of Precision and Sensitivity
  AUC-ROC     — area under ROC curve; 1.0 = perfect, 0.5 = random

For cancer screening, SENSITIVITY is the most critical metric:
  A missed cancer (FN) is far more dangerous than a false alarm (FP).

Usage:
    from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curves, benchmark_table
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import json
import os

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    y_proba     : np.ndarray,
    model_name  : str = "Model"
) -> dict:
    """
    Compute all evaluation metrics for a binary classifier.

    Args:
        y_true     : ground-truth labels (0=Benign, 1=Malignant)
        y_pred     : predicted class labels
        y_proba    : predicted probability of class 1 (Malignant)
        model_name : label for display

    Returns:
        dict with all metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model"       : model_name,
        "accuracy"    : accuracy_score(y_true, y_pred),
        "precision"   : precision_score(y_true, y_pred, zero_division=0),
        "sensitivity" : recall_score(y_true, y_pred),
        "specificity" : tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1_score"    : f1_score(y_true, y_pred),
        "roc_auc"     : roc_auc_score(y_true, y_proba),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0
    ))
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  AUC-ROC     : {metrics['roc_auc']:.4f}")
    print(f"  Confusion   : TP={tp} TN={tn} FP={fp} FN={fn}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    model_name : str,
    save_path  : str = None
):
    """Plot annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
        xticklabels=['Pred Benign', 'Pred Malignant'],
        yticklabels=['True Benign', 'True Malignant'],
        ax=ax
    )
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, pad=12)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(
    model_names : list,
    y_true_list : list,
    y_proba_list: list,
    save_path   : str = None
):
    """
    Overlay ROC curves for all models on one plot.

    Args:
        model_names  : list of model name strings
        y_true_list  : list of y_true arrays (one per model)
        y_proba_list : list of predicted probability arrays
        save_path    : optional save path for figure
    """
    COLORS = ['#2ECC71', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12']

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, y_true, y_proba, color in zip(
        model_names, y_true_list, y_proba_list, COLORS
    ):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name}  (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='grey')
    ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves — All Models vs VQC Baseline', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def benchmark_table(all_metrics: list) -> pd.DataFrame:
    """
    Print and return a formatted comparison table of all models.

    Args:
        all_metrics : list of metric dicts (from compute_metrics())

    Returns:
        pd.DataFrame sorted by AUC-ROC descending
    """
    cols = ['model', 'accuracy', 'sensitivity', 'specificity',
            'precision', 'f1_score', 'roc_auc']
    df = pd.DataFrame(all_metrics)[cols]
    df = df.sort_values('roc_auc', ascending=False).reset_index(drop=True)

    # Format numeric columns
    fmt_cols = [c for c in cols if c != 'model']
    df_display = df.copy()
    for col in fmt_cols:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

    print("\n" + "="*80)
    print("  BENCHMARKING RESULTS — All Models (sorted by AUC-ROC)")
    print("="*80)
    print(df_display.to_string(index=False))
    print("="*80)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def save_results(all_metrics: list, path: str = '../results/metrics.json'):
    """Save all benchmark results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Remove non-serializable objects (model, scaler)
    clean = []
    for m in all_metrics:
        clean.append({k: v for k, v in m.items()
                      if not hasattr(v, 'predict')})
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"Results saved → {path}")
