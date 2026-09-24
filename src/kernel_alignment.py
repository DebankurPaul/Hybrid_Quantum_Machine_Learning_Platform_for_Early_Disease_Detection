"""
kernel_alignment.py
===================
Kernel Target Alignment (KTA) Engine for Quantum Kernel Evaluation.
Calculates alignment score between the Quantum Kernel Gram Matrix K_ij and
target ground-truth label matrix Y_ij = y_i y_j.

Includes mandatory mathematical fix converting binary labels {0, 1} -> {-1, +1}
so benign sample similarity contributes correctly to the alignment score.
"""
import numpy as np
import sys, warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")


def calculate_kta(K: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Kernel Target Alignment (KTA):
    
        KTA(K, y) = <K, Y>_F / (||K||_F * ||Y||_F)
        where Y_ij = y_signed_i * y_signed_j
        
    Args:
        K : Quantum Kernel Gram matrix shape (N, N)
        y : Binary class labels array shape (N,) with values in {0, 1}
        
    Returns:
        kta_score : Alignment score in range [-1.0, 1.0] (higher is better)
    """
    # ── Mandatory Fix: Convert binary labels from {0, 1} to {-1, +1} ──────────
    # If y=0 for benign, y_i * y_j would be 0 and ignore benign cluster alignment.
    # Mapping to {-1, +1} ensures matching labels give +1 and differing give -1.
    y_signed = np.where(y == 0, -1, 1).astype(float).reshape(-1, 1)
    Y = np.dot(y_signed, y_signed.T)
    
    # Frobenius inner product and norms
    inner_product = np.sum(K * Y)
    norm_K = np.linalg.norm(K, 'fro')
    norm_Y = np.linalg.norm(Y, 'fro')
    
    if norm_K == 0 or norm_Y == 0:
        return 0.0
        
    kta_score = float(inner_product / (norm_K * norm_Y))
    return round(kta_score, 4)


def compare_kernel_alignments(K_pca: np.ndarray, K_autoencoder: np.ndarray, y: np.ndarray):
    """
    Compare Kernel Target Alignment for Linear PCA vs PyTorch Autoencoder feature spaces.
    """
    kta_pca = calculate_kta(K_pca, y)
    kta_ae  = calculate_kta(K_autoencoder, y)
    
    print("\n[KTA Engine] Quantum Kernel Target Alignment Results:")
    print(f"  • Linear PCA Quantum Kernel KTA Score        : {kta_pca:.4f}")
    print(f"  • PyTorch Autoencoder Quantum Kernel KTA Score : {kta_ae:.4f}")
    
    if kta_ae > kta_pca:
        print(f"  [+] PyTorch Autoencoder improves Quantum Kernel alignment by +{kta_ae - kta_pca:.4f}!")
    return {"kta_pca": kta_pca, "kta_ae": kta_ae}
