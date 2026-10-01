"""
q_xai.py
========
Quantum Explainable AI (Q-XAI) using SHAP (SHapley Additive exPlanations).
Uses shap.kmeans to summarize background dataset to 15 centroid prototypes,
enabling fast < 3 second SHAP feature attribution calculation.
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_qxai_shap(predict_fn, X_train_q, X_test_q, feature_names=None, k_centroids=15, n_explain=10):
    """
    Compute SHAP values for Quantum Model using shap.kmeans background summarization.
    
    Args:
        predict_fn    : Callable taking (N, n_qubits) array and returning (N, 2) probabilities
        X_train_q     : Training features (background)
        X_test_q      : Test features to explain
        feature_names : List of 8 feature names
        k_centroids   : Number of kmeans centroids for background data (default=15)
        n_explain     : Number of test samples to explain
        
    Returns:
        shap_values   : Array of SHAP values shape (n_explain, n_features)
        summary_dict  : Dictionary containing feature rankings and importance scores
    """
    if not SHAP_AVAILABLE:
        print("[Q-XAI] SHAP package not installed. Skipping SHAP analysis.")
        return None, {}
        
    if feature_names is None:
        feature_names = [f"Latent_AE_Q{i}" for i in range(X_train_q.shape[1])]
        
    print(f"\n[Q-XAI] Computing Fast SHAP Values using shap.kmeans (k={k_centroids})...")
    
    # 1. Summarize background distribution to k centroids for fast evaluation
    background_summary = shap.kmeans(X_train_q, k_centroids)
    
    # Wrapper function for malignant class probability
    def predict_malignant_prob(X):
        probs = predict_fn(X)
        return probs[:, 1] if probs.ndim == 2 else probs
        
    explainer = shap.KernelExplainer(predict_malignant_prob, background_summary)
    
    X_explain = X_test_q[:n_explain]
    shap_vals = explainer.shap_values(X_explain, nsamples=100)
    
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    rank_indices = np.argsort(mean_abs_shap)[::-1]
    
    ranking = [
        {"feature": feature_names[idx], "importance": round(float(mean_abs_shap[idx]), 4)}
        for idx in rank_indices
    ]
    
    print("  [+] SHAP Feature Importance Ranking:")
    for r in ranking[:5]:
        print(f"      • {r['feature']:<20} : {r['importance']:.4f}")
        
    # Save SHAP Bar Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, mean_abs_shap[rank_indices[::-1]], color='#2980B9', align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in rank_indices[::-1]])
    ax.set_xlabel("Mean |SHAP Value| (Impact on Malignant Prediction)", fontsize=10)
    ax.set_title("Q-XAI Quantum Feature Attribution (SHAP Waterfall)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs("results/figures", exist_ok=True)
    plot_path = "results/figures/qxai_shap_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Q-XAI Plot → {plot_path}")
    
    return shap_vals, {"ranking": ranking, "plot_path": plot_path}
