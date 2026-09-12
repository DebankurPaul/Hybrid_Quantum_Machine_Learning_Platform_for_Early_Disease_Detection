"""
explainability.py
==================
Model-agnostic explainability for the quantum VQC and classical baselines.

Techniques used:
  1. SHAP KernelExplainer — works with ANY model (black-box)
  2. Qiskit circuit drawing — visualise the circuit structure
  3. Feature importance bar plot — which PCA components matter most

Why SHAP works with quantum models:
  The VQC exposes a predict() function → SHAP treats it as a black box.
  KernelExplainer perturbs input features and measures output change.
  Result: "PC1 pushes this prediction toward Malignant by 0.12".

Usage:
    from src.explainability import shap_analysis, plot_circuit
    shap_analysis(clf.predict_proba, X_train_q, X_test_q,
                  feature_names=['PC1','PC2','PC3','PC4'])
"""
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[explainability] SHAP not found. Install with: pip install shap")


# ─────────────────────────────────────────────────────────────────────────────
def shap_analysis(
    predict_fn     ,           # function(X) -> proba array shape (n, 2)
    X_train_sample : np.ndarray,
    X_test_sample  : np.ndarray,
    feature_names  : list,
    model_name     : str = "Model",
    save_path      : str = None,
    nsamples       : int = 100,
):
    """
    Run SHAP KernelExplainer on any model.

    KernelExplainer is fully model-agnostic:
      • It treats the model as a black box
      • Works with quantum models, classical models, or anything with predict()
      • Computes Shapley values: the marginal contribution of each feature

    Args:
        predict_fn      : function that takes X (numpy) and returns
                          probability array [[P(0), P(1)], ...] shape (n, 2)
        X_train_sample  : background reference data (50 samples is sufficient)
        X_test_sample   : samples to explain (10-20 is practical)
        feature_names   : list of feature names (e.g. ['PC1','PC2','PC3','PC4'])
        model_name      : for plot titles
        save_path       : optional path to save figure
        nsamples        : SHAP integration samples (100 is good balance)
    """
    if not SHAP_AVAILABLE:
        print("Install SHAP: pip install shap")
        return None

    print(f"\n[SHAP] Analysing {model_name}...")
    print(f"  Background : {len(X_train_sample)} samples")
    print(f"  Explain    : {len(X_test_sample)} samples")
    print(f"  nsamples   : {nsamples}")
    print("  (This may take 2–10 minutes for the quantum model)")

    # Wrapper: KernelExplainer expects f(X) -> scalar or 1D array
    def predict_class1(X):
        return predict_fn(X)[:, 1]   # P(Malignant)

    explainer  = shap.KernelExplainer(predict_class1, X_train_sample[:50])
    shap_vals  = explainer.shap_values(X_test_sample[:20], nsamples=nsamples)

    # ── Summary (beeswarm) plot ────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    shap.summary_plot(
        shap_vals,
        X_test_sample[:20],
        feature_names=feature_names,
        title=f"SHAP Feature Importance — {model_name}",
        show=False
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # ── Bar plot of mean |SHAP| ────────────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_idx = np.argsort(mean_abs)[::-1]
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color='#3498DB'
    )
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(f'Feature Importance — {model_name}', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        bar_path = save_path.replace('.png', '_bar.png')
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.show()

    return shap_vals


# ─────────────────────────────────────────────────────────────────────────────
def plot_circuit(n_qubits=4, fm_reps=1, ans_reps=2, save_path=None):
    """
    Draw the Qiskit VQC circuit (Feature Map + Ansatz).

    Args:
        n_qubits  : number of qubits
        fm_reps   : ZZFeatureMap repetitions
        ans_reps  : RealAmplitudes repetitions
        save_path : optional file path to save figure
    """
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit import QuantumCircuit

    fm  = ZZFeatureMap(feature_dimension=n_qubits, reps=fm_reps, entanglement='linear')
    ans = RealAmplitudes(num_qubits=n_qubits, reps=ans_reps, entanglement='linear')
    qc  = fm.compose(ans)
    qc.name = "VQC (ZZFeatureMap + RealAmplitudes)"

    # Text version
    print("\nVQC Circuit (text):")
    print(qc.decompose().draw('text'))

    # Matplotlib version
    try:
        fig = qc.decompose().draw('mpl', fold=40, style={'name': 'bw'})
        fig.suptitle(
            f"VQC Circuit — {n_qubits} qubits | "
            f"ZZFeatureMap(reps={fm_reps}) + RealAmplitudes(reps={ans_reps})",
            fontsize=11
        )
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Circuit diagram saved → {save_path}")
        plt.show()
    except Exception as e:
        print(f"Matplotlib circuit draw failed: {e}")
        print("Tip: install pylatexenc  →  pip install pylatexenc")


# ─────────────────────────────────────────────────────────────────────────────
def plot_pca_explained_variance(pca, save_path=None):
    """
    Plot PCA explained variance to justify n_qubits choice.

    Args:
        pca       : fitted sklearn PCA object
        save_path : optional save path
    """
    n = len(pca.explained_variance_ratio_)
    cumvar = pca.explained_variance_ratio_.cumsum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Per-component bar
    ax1.bar(range(1, n+1), pca.explained_variance_ratio_ * 100, color='#3498DB')
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Explained Variance (%)', fontsize=11)
    ax1.set_title('Per-Component Explained Variance', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Cumulative line
    ax2.plot(range(1, n+1), cumvar * 100, 'o-', color='#E74C3C', lw=2)
    ax2.axhline(87, color='grey', ls='--', alpha=0.6, label='87% (4 PCs)')
    ax2.axhline(95, color='grey', ls=':',  alpha=0.6, label='95% (8 PCs)')
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax2.set_title('Cumulative Explained Variance', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
