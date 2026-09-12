"""
Backend Adapter Layer for Streamlit Dashboard.
Provides a clean interface between the Streamlit UI and the ML/QML backend.
Currently returns structured 'not_available' states as the real backend is pending.
"""

from typing import Dict, Any, Optional

def create_response(status: str, data: Optional[Any] = None, message: str = "") -> Dict[str, Any]:
    """Helper to create standardized responses."""
    return {
        "status": status,
        "data": data,
        "message": message
    }

def get_system_status() -> Dict[str, Any]:
    """Returns the current availability of backend components."""
    return create_response("partial", {
        "dataset": True,
        "preprocessor": False,
        "quantum_backend": False,
        "model_artifacts": False
    })

def get_dataset_metadata() -> Dict[str, Any]:
    """Returns static metadata about the WDBC dataset."""
    return create_response("available", {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": 569,
        "original_features": 30,
        "classes": 2,
        "missing_values": 0
    })

def get_available_models() -> Dict[str, Any]:
    """Returns the lists of currently available and planned models."""
    return create_response("available", {
        "available": [],
        "planned_quantum": ["8-Qubit VQC", "4-Qubit VQC", "QSVC"],
        "planned_classical": ["Logistic Regression", "SVM", "Random Forest"]
    })

def predict_patient(patient_data: dict, model_name: str, threshold: float) -> Dict[str, Any]:
    """
    Submits a patient record for prediction via the backend adapter.
    Expected data structure on 'available':
    {
        "predicted_class": str,
        "probability": float
    }
    """
    return create_response(
        "not_available", 
        message="The trained model artifacts required for inference are not currently connected."
    )

def get_quantum_model_info(model_name: str) -> Dict[str, Any]:
    """
    Returns configuration for the specified quantum model.
    Expected data structure on 'available':
    {
        "pca_components": int,
        "qubits": int,
        "trainable_parameters": int,
        "feature_map": str,
        "ansatz": str,
        "backend": str
    }
    """
    return create_response(
        "not_available",
        message="Awaiting model artifact."
    )

def get_quantum_circuit(model_name: str) -> Dict[str, Any]:
    """
    Returns the Qiskit circuit representation for the UI.
    Expected data structure on 'available':
    String representation or matplotlib figure of the circuit.
    """
    return create_response(
        "not_available",
        message="Quantum circuit unavailable. Connect the trained quantum model to render the actual circuit configuration."
    )

def get_evaluation_metrics(model_name: str, threshold: float) -> Dict[str, Any]:
    """
    Returns evaluation metrics for the given model and threshold.
    Expected data structure on 'available':
    {
        "accuracy": float,
        "precision": float,
        "sensitivity": float,
        "specificity": float,
        "f1": float,
        "roc_auc": float,
        "inference_time": str
    }
    """
    return create_response(
        "not_available",
        message="Awaiting evaluation results."
    )

def get_confusion_matrix(model_name: str, threshold: float) -> Dict[str, Any]:
    """
    Returns confusion matrix data.
    Expected data structure on 'available':
    {
        "TN": int, "FP": int, "FN": int, "TP": int
    }
    or a pandas DataFrame.
    """
    return create_response(
        "not_available",
        message="Awaiting evaluation results."
    )

def get_threshold_analysis(model_name: str) -> Dict[str, Any]:
    """
    Returns threshold-dependent evaluation data.
    Expected data structure on 'available':
    {
        "thresholds": list[float],
        "sensitivity": list[float],
        "specificity": list[float],
        "precision": list[float],
        "f1": list[float]
    }
    """
    return create_response(
        "not_available",
        message="Threshold trade-off curves require evaluation data."
    )

def get_roc_curve(model_name: str) -> Dict[str, Any]:
    """
    Returns ROC curve data.
    Expected data structure on 'available':
    {
        "fpr": list[float],
        "tpr": list[float]
    }
    """
    return create_response(
        "not_available",
        message="Awaiting evaluation results."
    )

def get_benchmark_results() -> Dict[str, Any]:
    """
    Returns comparison metrics for all available models.
    Expected data structure on 'available':
    pandas DataFrame with columns: Model, Accuracy, Precision, Sensitivity, Specificity, F1, ROC-AUC, Inference Time.
    """
    return create_response(
        "not_available",
        message="Evaluation results are not connected yet. The comparison will populate automatically when the current model results are available."
    )

def get_explainability(model_name: str) -> Dict[str, Any]:
    """
    Returns explainability data (e.g., PCA loadings, attribution).
    Expected data structure on 'available':
    {
        "pca_loadings": pandas DataFrame,
        "attribution": str / dict / plot,
        "patient_explanation": str / dict
    }
    """
    return create_response(
        "not_available",
        message="Explainability data requires loaded model and preprocessing artifacts."
    )
