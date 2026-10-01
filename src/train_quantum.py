"""
train_quantum.py  (Qiskit version)
====================================
Training utilities for the QuantumClassifier.

Note on Qiskit VQC training vs PyTorch training:
─────────────────────────────────────────────────
PyTorch (PennyLane approach):
  • Explicit epoch loop
  • Mini-batch gradient descent
  • backprop / parameter-shift for gradients

Qiskit VQC approach:
  • Single call to vqc.fit(X, y)
  • Optimizer (COBYLA/SPSA) handles iteration internally
  • max_iter = number of function evaluations (≈ "epochs")
  • COBYLA: gradient-FREE (no backprop needed) — good for noisy simulators
  • SPSA: stochastic gradient (good for large parameter counts)

Usage:
    from src.train_quantum import run_training, plot_loss_curve
    results = run_training(X_train, y_train, X_val, y_val, max_iter=150)
    plot_loss_curve(results['model'].history_)
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from src.quantum_model import QuantumClassifier
from src.evaluate import compute_metrics


def run_training(
    X_train : np.ndarray,
    y_train : np.ndarray,
    X_val   : np.ndarray,
    y_val   : np.ndarray,
    n_qubits  : int = 4,
    fm_reps   : int = 1,
    ans_reps  : int = 2,
    optimizer : str = 'cobyla',
    max_iter  : int = 150,
    shots     : int = None,
    seed      : int = 42,
) -> dict:
    """
    Full training run: build → fit → evaluate on validation set.

    Args:
        X_train, y_train : training data (quantum-preprocessed)
        X_val,   y_val   : validation data
        n_qubits         : number of qubits (must match PCA n_components)
        fm_reps          : ZZFeatureMap repetitions
        ans_reps         : RealAmplitudes repetitions
        optimizer        : 'cobyla' | 'spsa' | 'lbfgs'
        max_iter         : maximum optimizer iterations
        shots            : circuit shots (None = exact statevector)
        seed             : random seed for reproducibility

    Returns:
        dict with keys:
          'model'     : fitted QuantumClassifier
          'val_acc'   : validation accuracy
          'val_metrics': full metrics dict
          'train_time': elapsed seconds
    """
    clf = QuantumClassifier(
        n_qubits  = n_qubits,
        fm_reps   = fm_reps,
        ans_reps  = ans_reps,
        optimizer = optimizer,
        max_iter  = max_iter,
        shots     = shots,
        seed      = seed,
    )

    clf.fit(X_train, y_train)

    # ── Validation evaluation ─────────────────────────────────────────────
    print("\n── Validation Results ──────────────────────")
    y_pred  = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_proba, model_name="VQC (Qiskit)")

    return {
        'model'      : clf,
        'val_acc'    : metrics['accuracy'],
        'val_metrics': metrics,
    }


def plot_loss_curve(history: list, save_path: str = None):
    """
    Plot the optimizer loss curve recorded during VQC training.

    Args:
        history   : list of loss values per iteration (from clf.history_)
        save_path : optional path to save the figure
    """
    if not history:
        print("No history to plot. Did training complete?")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(history, color='#E74C3C', lw=2)
    plt.fill_between(range(len(history)), history, alpha=0.15, color='#E74C3C')
    plt.xlabel('Optimizer Iteration', fontsize=12)
    plt.ylabel('Objective (Loss)', fontsize=12)
    plt.title('VQC Training Loss Curve (Qiskit COBYLA)', fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curve to {save_path}")
    plt.show()


def optimizer_comparison(
    X_train, y_train, X_val, y_val,
    optimizers=('cobyla', 'spsa'),
    max_iter=100
) -> dict:
    """
    Train multiple VQCs with different optimizers and compare.
    Useful for hyperparameter tuning.

    Returns:
        dict: {optimizer_name: results_dict}
    """
    results = {}
    for opt_name in optimizers:
        print(f"\n{'─'*55}")
        print(f"  Optimizer: {opt_name.upper()}")
        print(f"{'─'*55}")
        results[opt_name] = run_training(
            X_train, y_train, X_val, y_val,
            optimizer=opt_name, max_iter=max_iter
        )

    # Print comparison summary
    print("\n\nOptimizer Comparison")
    print(f"{'Optimizer':<12} {'Val Accuracy':<15}")
    print("─" * 30)
    for name, res in results.items():
        print(f"{name:<12} {res['val_acc']:.4f}")

    return results
