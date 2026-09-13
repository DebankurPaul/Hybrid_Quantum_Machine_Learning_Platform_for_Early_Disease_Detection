"""
Backend Adapter Layer for Streamlit Dashboard.
Connects Streamlit UI to the actual ML/QML artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional

def create_response(status: str, data: Optional[Any] = None, message: str = "") -> Dict[str, Any]:
    return {"status": status, "data": data, "message": message}

# Resolve paths absolutely based on this file's location
# __file__ is in Streamlit_dashboard/src/
# Therefore BASE_DIR should be the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BEST_MODEL_DIR = os.path.join(RESULTS_DIR, "best_model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Global state to cache models
_cache = {
    "classical_models": {},
    "classical_scaler": None,
    "quantum_model": None,
    "wdbc_data": None
}

def _load_wdbc_data():
    if _cache["wdbc_data"] is not None:
        return _cache["wdbc_data"]
    
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        X, y = load_breast_cancer(return_X_y=True)
        y = 1 - y # 1=Malignant, 0=Benign
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        _cache["wdbc_data"] = (X_tr, X_te, y_tr, y_te)
        return _cache["wdbc_data"]
    except Exception as e:
        print(f"Error loading WDBC: {e}")
        return None

def _get_quantum_pipeline():
    if _cache["quantum_model"] is not None:
        return _cache["quantum_model"]
        
    wdbc = _load_wdbc_data()
    if not wdbc:
        return None
    X_tr, X_te, y_tr, y_te = wdbc
    
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        pca = PCA(n_components=8, random_state=42)
        pca.fit(X_tr_sc)
        
        # Load weights
        weights_path = os.path.join(BEST_MODEL_DIR, "best_vqc_weights_8q.npy")
        if not os.path.exists(weights_path):
            return None
        weights = np.load(weights_path)
        
        try:
            from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
            fm = ZZFeatureMap(feature_dimension=8, reps=1, entanglement="linear")
            ans = RealAmplitudes(num_qubits=8, reps=2, entanglement="linear")
        except ImportError:
            from qiskit.circuit.library import zz_feature_map, real_amplitudes
            fm = zz_feature_map(feature_dimension=8, reps=1, entanglement="linear")
            ans = real_amplitudes(num_qubits=8, reps=2, entanglement="linear")
            
        from qiskit_machine_learning.primitives import QMLSampler
        from qiskit_machine_learning.algorithms.classifiers import VQC
        from qiskit_algorithms.optimizers import SPSA
        
        vqc = VQC(sampler=QMLSampler(), feature_map=fm, ansatz=ans, optimizer=SPSA(maxiter=1))
        # Dummy fit to initialize shapes
        vqc.fit(np.tanh(pca.transform(X_tr_sc[:2])) * np.pi, y_tr[:2])
        vqc._fit_result.x = weights
        
        _cache["quantum_model"] = (sc, pca, vqc)
        return _cache["quantum_model"]
    except Exception as e:
        print(f"Error loading Quantum Pipeline: {e}")
        return None

def get_system_status() -> Dict[str, Any]:
    return create_response("available", {
        "dataset": True,
        "preprocessor": os.path.exists(os.path.join(MODELS_DIR, "classical_scaler.joblib")),
        "quantum_backend": os.path.exists(os.path.join(BEST_MODEL_DIR, "best_vqc_weights_8q.npy")),
        "model_artifacts": os.path.exists(MODELS_DIR)
    })

def get_dataset_metadata() -> Dict[str, Any]:
    return create_response("available", {
        "name": "Breast Cancer Wisconsin (Diagnostic) - WDBC",
        "samples": 569,
        "original_features": 30,
        "classes": 2,
        "missing_values": 0
    })

def get_available_models() -> Dict[str, Any]:
    available = []
    if os.path.exists(os.path.join(BEST_MODEL_DIR, "best_vqc_weights_8q.npy")):
        available.append("Quantum VQC (8-qubit)")
        
    for m in ["logistic_regression", "svm", "random_forest", "xgboost"]:
        if os.path.exists(os.path.join(MODELS_DIR, f"{m}.joblib")):
            available.append(m.replace("_", " ").title().replace("Svm", "SVM").replace("Xgboost", "XGBoost"))
            
    return create_response("available", {
        "available": available,
        "planned_quantum": [],
        "planned_classical": []
    })

def predict_patient(patient_data: dict, model_name: str, threshold: float) -> Dict[str, Any]:
    try:
        from sklearn.datasets import load_breast_cancer
        feature_names = load_breast_cancer().feature_names
        x_arr = np.array([[patient_data.get(f, 0.0) for f in feature_names]])
        
        if "Quantum" in model_name:
            pipeline = _get_quantum_pipeline()
            if not pipeline:
                return create_response("not_available", message="Quantum model not loaded.")
            sc, pca, vqc = pipeline
            x_sc = sc.transform(x_arr)
            x_pca = pca.transform(x_sc)
            x_q = np.tanh(x_pca) * np.pi
            probs = vqc.predict_proba(x_q)[0]
            p_mal = float(probs[1])
            pred_class = "Malignant" if p_mal >= threshold else "Benign"
            return create_response("available", {"predicted_class": pred_class, "probability": p_mal})
        else:
            if _cache["classical_scaler"] is None:
                scaler_path = os.path.join(MODELS_DIR, "classical_scaler.joblib")
                if not os.path.exists(scaler_path):
                    return create_response("not_available", message="Classical scaler not found.")
                _cache["classical_scaler"] = joblib.load(scaler_path)
            sc = _cache["classical_scaler"]
            
            model_file = model_name.lower().replace(" ", "_").replace("svm", "svm").replace("xgboost", "xgboost")
            if model_file not in _cache["classical_models"]:
                m_path = os.path.join(MODELS_DIR, f"{model_file}.joblib")
                if not os.path.exists(m_path):
                    return create_response("not_available", message=f"{model_name} not found.")
                _cache["classical_models"][model_file] = joblib.load(m_path)
                
            clf = _cache["classical_models"][model_file]
            x_sc = sc.transform(x_arr)
            probs = clf.predict_proba(x_sc)[0]
            p_mal = float(probs[1])
            pred_class = "Malignant" if p_mal >= threshold else "Benign"
            return create_response("available", {"predicted_class": pred_class, "probability": p_mal})
            
    except Exception as e:
        return create_response("error", message=str(e))

def get_optimal_threshold() -> float:
    """Return the optimal threshold (best_tau) for the Quantum model if available, else 0.5."""
    metrics_path = os.path.join(BEST_MODEL_DIR, "best_model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
            return float(data.get("best_tau", 0.5))
        except Exception:
            return 0.5
    return 0.5

def get_quantum_model_info(model_name: str) -> Dict[str, Any]:
    if "Quantum" not in model_name:
        return create_response("not_available", message="Not a quantum model.")
    
    metrics_path = os.path.join(BEST_MODEL_DIR, "best_model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        return create_response("available", {
            "pca_components": cfg.get("pca_components", 8),
            "qubits": cfg.get("n_qubits", 8),
            "trainable_parameters": cfg.get("n_params", 24),
            "feature_map": "ZZFeatureMap (Reps=1, linear ent.)",
            "ansatz": "RealAmplitudes (Reps=2, linear ent.)",
            "backend": "Qiskit VQC"
        })
    return create_response("not_available", message="Quantum model config not found.")

def get_quantum_circuit(model_name: str) -> Dict[str, Any]:
    return create_response("not_available", message="Circuit visualization unavailable.")

def get_evaluation_metrics(model_name: str, threshold: float) -> Dict[str, Any]:
    if "Quantum" in model_name:
        metrics_path = os.path.join(BEST_MODEL_DIR, "best_model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
            best_match = data["metrics"]
            scan = data.get("threshold_scan", [])
            for s in scan:
                if abs(s["tau"] - threshold) < 0.001:
                    best_match = s
                    break
            
            fn = best_match.get("fn", 0)
            fp = best_match.get("fp", 0)
            tp = 42 - fn
            tn = 72 - fp
            
            acc = best_match.get("accuracy", 0)
            sens = best_match.get("sensitivity", 0)
            spec = best_match.get("specificity", 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
            
            return create_response("available", {
                "accuracy": acc,
                "precision": prec,
                "sensitivity": sens,
                "specificity": spec,
                "f1": f1,
                "roc_auc": data["metrics"].get("roc_auc", 0),
                "inference_time": "N/A"
            })
    else:
        res_path = os.path.join(RESULTS_DIR, "classical_test_results.csv")
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            m_map = {"Logistic Regression": "Logistic Regression", "SVM": "SVM", "Random Forest": "Random Forest", "XGBoost": "XGBoost"}
            match = m_map.get(model_name)
            if match:
                row = df[(df["Model"] == match) & (df["Dataset"] == "Test")]
                if len(row) > 0:
                    r = row.iloc[0]
                    return create_response("available", {
                        "accuracy": r.get("Accuracy", 0),
                        "precision": r.get("Precision", 0),
                        "sensitivity": r.get("Sensitivity", 0),
                        "specificity": r.get("Specificity", 0),
                        "f1": r.get("F1 Score", 0),
                        "roc_auc": r.get("ROC-AUC", 0),
                        "inference_time": "N/A"
                    })
    
    return create_response("not_available", message="Evaluation metrics not found.")

def get_confusion_matrix(model_name: str, threshold: float) -> Dict[str, Any]:
    if "Quantum" in model_name:
        metrics_path = os.path.join(BEST_MODEL_DIR, "best_model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
            scan = data.get("threshold_scan", [])
            for s in scan:
                if abs(s["tau"] - threshold) < 0.001:
                    fn = s["fn"]
                    fp = s["fp"]
                    tp = 42 - fn
                    tn = 72 - fp
                    return create_response("available", {"TN": tn, "FP": fp, "FN": fn, "TP": tp})
            m = data["metrics"]
            return create_response("available", {"TN": m.get("tn"), "FP": m.get("fp"), "FN": m.get("fn"), "TP": m.get("tp")})
    else:
        res_path = os.path.join(RESULTS_DIR, "classical_test_results.csv")
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            match = {"Logistic Regression": "Logistic Regression", "SVM": "SVM", "Random Forest": "Random Forest", "XGBoost": "XGBoost"}.get(model_name)
            if match:
                row = df[(df["Model"] == match) & (df["Dataset"] == "Test")]
                if len(row) > 0:
                    r = row.iloc[0]
                    return create_response("available", {"TN": int(r.get("TN",0)), "FP": int(r.get("FP",0)), "FN": int(r.get("FN",0)), "TP": int(r.get("TP",0))})
    return create_response("not_available", message="Confusion matrix not found.")

def get_threshold_analysis(model_name: str) -> Dict[str, Any]:
    if "Quantum" in model_name:
        metrics_path = os.path.join(BEST_MODEL_DIR, "best_model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
            scan = data.get("threshold_scan", [])
            if scan:
                df = pd.DataFrame(scan)
                return create_response("available", {
                    "thresholds": df["tau"].tolist(),
                    "sensitivity": df["sensitivity"].tolist(),
                    "specificity": df["specificity"].tolist(),
                    "precision": [], 
                    "f1": []
                })
    return create_response("not_available", message="Threshold scan data only saved for Quantum model.")

def get_roc_curve(model_name: str) -> Dict[str, Any]:
    return create_response("not_available", message="ROC Curve raw data not exposed in this prototype.")

def get_benchmark_results() -> Dict[str, Any]:
    res_path = os.path.join(RESULTS_DIR, "classical_test_results.csv")
    if os.path.exists(res_path):
        df = pd.read_csv(res_path)
        df = df[df["Dataset"] == "Test"]
        
        metrics_path = os.path.join(BEST_MODEL_DIR, "best_model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
            m = data["metrics"]
            q_row = {
                "Dataset": "Test",
                "Model": "Quantum VQC (8-qubit)",
                "Accuracy": m["accuracy"],
                "Precision": m["tp"]/(m["tp"]+m["fp"]) if m["tp"]+m["fp"]>0 else 0,
                "Sensitivity": m["sensitivity"],
                "Specificity": m["specificity"],
                "F1 Score": 0, 
                "ROC-AUC": m["roc_auc"],
                "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"],
                "Training Time (sec)": 0
            }
            p = q_row["Precision"]
            s = q_row["Sensitivity"]
            q_row["F1 Score"] = 2 * p * s / (p + s) if p + s > 0 else 0
            df = pd.concat([df, pd.DataFrame([q_row])], ignore_index=True)
            
        return create_response("available", df)
    return create_response("not_available", message="Benchmark results not found.")

def get_explainability(model_name: str) -> Dict[str, Any]:
    if "Quantum" in model_name:
        plot_path = os.path.join(RESULTS_DIR, "figures", "qxai_shap_summary.png")
        if os.path.exists(plot_path):
            return create_response("available", {
                "attribution": plot_path
            })
    else:
        csv_path = os.path.join(RESULTS_DIR, "random_forest_feature_importance.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return create_response("available", {
                "pca_loadings": df  
            })
    return create_response("not_available", message="Explainability artifacts not found for this model.")

def evaluate_patient_agreement(patient_data: dict, tau: float) -> Dict[str, Any]:
    """Helper for model agreement analysis."""
    p_q = predict_patient(patient_data, "Quantum VQC (8-qubit)", tau)
    # Get a reliable classical reference, e.g. SVM or Random Forest
    available = get_available_models().get("data", {}).get("available", [])
    ref_model = None
    for c in ["SVM", "Random Forest", "Logistic Regression"]:
        if c in available:
            ref_model = c
            break
            
    if p_q["status"] == "available" and ref_model:
        p_c = predict_patient(patient_data, ref_model, 0.5) 
        if p_c["status"] == "available":
            pred_q = p_q["data"]["predicted_class"]
            pred_c = p_c["data"]["predicted_class"]
            return create_response("available", {
                "quantum_pred": pred_q,
                "quantum_prob": p_q["data"]["probability"],
                "classical_pred": pred_c,
                "classical_prob": p_c["data"]["probability"],
                "classical_model": ref_model,
                "concordant": pred_q == pred_c
            })
    return create_response("not_available", message="Required models not loaded for agreement analysis.")
