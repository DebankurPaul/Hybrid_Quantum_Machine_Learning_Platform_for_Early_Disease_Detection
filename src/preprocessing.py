"""
preprocessing.py
================
Feature scaling, PCA dimensionality reduction, and angle normalization
for feeding classical data into a quantum circuit.

Pipeline:
    1. StandardScaler  -- zero mean, unit variance
    2. PCA             -- reduce 30 features to n_qubits features
    3. tanh normalize  -- map to [-pi, pi] for quantum angle encoding

Usage:
    from src.preprocessing import preprocess, save_processed
    X_train_q, X_val_q, X_test_q, scaler, pca = preprocess(
        X_train, X_val, X_test, n_qubits=4
    )
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


def preprocess(X_train, X_val, X_test, n_qubits=4):
    """
    Full pre-processing pipeline for quantum encoding.

    Why each step:
    - StandardScaler: ensures all 30 features are on the same scale;
      quantum gates are sensitive to the magnitude of input angles.
    - PCA: quantum simulators handle ~4-20 qubits; PCA extracts the
      most information-dense n_qubits features from the original 30.
    - tanh normalize: quantum RY(theta) gates expect angles in [-pi, pi];
      tanh smoothly squashes any value into (-1,1), then we scale by pi.

    Args:
        X_train, X_val, X_test : raw feature arrays shape (n_samples, 30)
        n_qubits                : number of qubits = PCA output dimensions

    Returns:
        X_train_q, X_val_q, X_test_q : normalized arrays shape (n, n_qubits)
        scaler                         : fitted StandardScaler
        pca                            : fitted PCA object
    """
    # ── Step 1: Standardize ───────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # ── Step 2: PCA dimensionality reduction ──────────────────────────────
    pca = PCA(n_components=n_qubits, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    var_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_qubits} components explain {var_explained:.1%} of variance")

    # ── Step 3: tanh normalization to [-pi, pi] ───────────────────────────
    # tanh(x) maps any real number to (-1, 1)
    # Multiply by pi to get (-pi, pi)
    def angle_normalize(X):
        return np.tanh(X) * np.pi

    X_train_q = angle_normalize(X_train_pca)
    X_val_q   = angle_normalize(X_val_pca)
    X_test_q  = angle_normalize(X_test_pca)

    print(f"Feature range after normalization: "
          f"[{X_train_q.min():.3f}, {X_train_q.max():.3f}]")

    return X_train_q, X_val_q, X_test_q, scaler, pca


def save_processed(arrays, path_prefix='../data/processed/'):
    """
    Save processed numpy arrays to disk.

    Args:
        arrays      : list of 6 arrays [X_train_q, X_val_q, X_test_q,
                                         y_train,   y_val,   y_test]
        path_prefix : directory to save to
    """
    os.makedirs(path_prefix, exist_ok=True)
    names = ['X_train_q', 'X_val_q', 'X_test_q', 'y_train', 'y_val', 'y_test']
    for name, arr in zip(names, arrays):
        np.save(f"{path_prefix}{name}.npy", arr)
    print(f"Saved {len(arrays)} arrays to {path_prefix}")


def load_processed(path_prefix='../data/processed/'):
    """
    Load pre-processed numpy arrays from disk.

    Returns:
        X_train_q, X_val_q, X_test_q, y_train, y_val, y_test
    """
    names = ['X_train_q', 'X_val_q', 'X_test_q', 'y_train', 'y_val', 'y_test']
    arrays = [np.load(f"{path_prefix}{name}.npy") for name in names]
    print(f"Loaded processed data from {path_prefix}")
    return tuple(arrays)
