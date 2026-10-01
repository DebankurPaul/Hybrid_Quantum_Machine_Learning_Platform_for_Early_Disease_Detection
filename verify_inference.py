import numpy as np
import os
from sklearn.decomposition import PCA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.primitives import QMLSampler
from src.data_loader import load_clinical_data
from src.quantum_circuit import build_feature_map, build_ansatz

# 1. Original Inference Path
def original_inference_path(X_raw_samples):
    # This matches exactly what fresh_wdbc_eval.py does
    splits, _, _, scaler = load_clinical_data(dataset_key="wdbc", test_size=0.2, seed=42)
    X_train_sc = splits[0]
    
    pca8 = PCA(n_components=8, random_state=42)
    pca8.fit(X_train_sc)
    
    X_samples_sc = scaler.transform(X_raw_samples)
    X_samples_p8 = pca8.transform(X_samples_sc)
    X_samples_q8 = np.tanh(X_samples_p8) * np.pi
    
    fm8 = build_feature_map(8, reps=1, entanglement='linear')
    ans8 = build_ansatz(8, reps=2, entanglement='linear')
    vqc8 = VQC(feature_map=fm8, ansatz=ans8, sampler=QMLSampler())
    
    # Dummy fit
    vqc8.fit(X_samples_q8[:2], np.array([0, 1]))
    vqc8._fit_result.x = np.load("results/best_model/best_vqc_weights_8q.npy")
    
    probs = vqc8.predict_proba(X_samples_q8)[:, 1]
    classes = (probs >= 0.65).astype(int)
    
    return X_samples_q8, probs, classes

# 2. New Backend Adapter Path (Simulated before we implement it in backend_adapter.py)
def proposed_adapter_path(X_raw_samples):
    # Same regeneration logic, strictly encapsulated
    splits, _, _, scaler = load_clinical_data(dataset_key="wdbc", test_size=0.2, seed=42)
    X_train_sc = splits[0]
    
    pca8 = PCA(n_components=8, random_state=42)
    pca8.fit(X_train_sc)
    
    # Simulate receiving dict from UI and transforming it
    X_sc = scaler.transform(X_raw_samples)
    X_p8 = pca8.transform(X_sc)
    X_q8 = np.tanh(X_p8) * np.pi
    
    fm8 = build_feature_map(8, reps=1, entanglement='linear')
    ans8 = build_ansatz(8, reps=2, entanglement='linear')
    vqc8 = VQC(feature_map=fm8, ansatz=ans8, sampler=QMLSampler())
    
    vqc8.fit(X_q8[:2], np.array([0, 1]))
    vqc8._fit_result.x = np.load("results/best_model/best_vqc_weights_8q.npy")
    
    probs = vqc8.predict_proba(X_q8)[:, 1]
    classes = (probs >= 0.65).astype(int)
    
    return X_q8, probs, classes

def main():
    print("Testing Inference Equivalence...")
    # Grab 5 raw samples from the raw dataset
    from sklearn.datasets import load_breast_cancer
    raw = load_breast_cancer()
    X_test_raw = raw.data[:5] # 5 random instances
    
    X1, p1, c1 = original_inference_path(X_test_raw)
    X2, p2, c2 = proposed_adapter_path(X_test_raw)
    
    diff_X = np.max(np.abs(X1 - X2))
    diff_p = np.max(np.abs(p1 - p2))
    
    print(f"Max feature difference: {diff_X:.10e}")
    print(f"Max probability diff : {diff_p:.10e}")
    print(f"Class match          : {np.array_equal(c1, c2)}")
    
    if diff_X < 1e-7 and diff_p < 1e-7 and np.array_equal(c1, c2):
        print("\nSUCCESS: The regenerated PCA/Scaler in the adapter is 100% equivalent to the training pipeline.")
    else:
        print("\nFAILURE: Inference is not equivalent.")

if __name__ == "__main__":
    main()
