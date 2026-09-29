import os
import numpy as np
from src.data_loader import load_clinical_data
from src.inference import run_inference
import json
from sklearn.decomposition import PCA

def test_inference_equivalence():
    print("=== FULL TEST SET INFERENCE EQUIVALENCE VERIFICATION ===")
    
    # 1. Load actual test data
    splits, feature_names, target_labels, scaler = load_clinical_data("wdbc", test_size=0.2, seed=42)
    X_train_raw = scaler.inverse_transform(splits[0])
    X_test_raw = scaler.inverse_transform(splits[1])
    y_test = splits[3]
    
    print(f"Verifying across all {len(X_test_raw)} test samples...")
    
    # 2. Emulate the original pipeline precisely for ground truth probabilities
    # The original pipeline in fresh_wdbc_eval.py uses the exact same scaler,
    # then fits PCA(8) on the SCALED train set, then applies tanh.
    X_train_sc = scaler.transform(X_train_raw)
    X_test_sc = scaler.transform(X_test_raw)
    
    pca8 = PCA(n_components=8, random_state=42)
    X_train_p8 = pca8.fit_transform(X_train_sc)
    X_test_p8 = pca8.transform(X_test_sc)
    
    X_test_q8 = np.tanh(X_test_p8) * np.pi
    
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.primitives import QMLSampler
    from src.quantum_circuit import build_feature_map, build_ansatz
    
    weights = np.load("results/best_model/best_vqc_weights_8q.npy")
    fm8 = build_feature_map(8, reps=1, entanglement='linear')
    ans8 = build_ansatz(8, reps=2, entanglement='linear')
    vqc8 = VQC(feature_map=fm8, ansatz=ans8, sampler=QMLSampler())
    # Dummy fit
    vqc8.fit(np.tanh(X_train_p8[:2]) * np.pi, splits[2][:2])
    vqc8._fit_result.x = weights
    
    p_vqc8_truth = vqc8.predict_proba(X_test_q8)[:, 1]
    y_pred_truth = (p_vqc8_truth >= 0.65).astype(int)
    
    # 3. Call the dynamic `run_inference` backend endpoint
    p_inference = []
    y_pred_inference = []
    
    for i in range(len(X_test_raw)):
        # Construct input dict exactly as the UI would
        input_dict = {f: X_test_raw[i][j] for j, f in enumerate(feature_names)}
        
        # Run authoritative endpoint
        res = run_inference("wdbc", "hybrid_quantum", input_dict)
        
        if res["status"] != "success":
            print(f"Error on sample {i}: {res.get('message')}")
            return False
            
        p_inference.append(res["probability"])
        y_pred_inference.append(res["predicted_class"])
    
    p_inference = np.array(p_inference)
    y_pred_inference = np.array(y_pred_inference)
    
    # 4. Compare
    abs_diffs = np.abs(p_vqc8_truth - p_inference)
    max_diff = np.max(abs_diffs)
    mean_diff = np.mean(abs_diffs)
    
    matches = (y_pred_truth == y_pred_inference)
    agreement_count = np.sum(matches)
    
    print("\n[VERIFICATION RESULTS]")
    print(f"Total Samples Tested : {len(X_test_raw)}")
    print(f"Class Agreement      : {agreement_count} / {len(X_test_raw)} ({(agreement_count/len(X_test_raw))*100:.2f}%)")
    print(f"Max Absolute Diff    : {max_diff:.12f}")
    print(f"Mean Absolute Diff   : {mean_diff:.12f}")
    
    if agreement_count == len(X_test_raw) and max_diff < 1e-5:
        print("\n✅ Verification PASSED. The new inference architecture is numerically identical to the training path.")
        return True
    else:
        print("\n❌ Verification FAILED. Divergence detected.")
        return False

if __name__ == "__main__":
    test_inference_equivalence()
