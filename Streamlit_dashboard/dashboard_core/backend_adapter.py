"""
Thin presentation adapter for the Streamlit dashboard.

The adapter contains no dataset/model availability registry. It delegates
backend truth to src.result_registry and dataset services, while preserving the
public function names used by the existing dashboard pages.
"""

from __future__ import annotations

import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import data_service
from src import result_registry
from src.inference import run_inference


# ---------------------------------------------------------------------------
# Phase 1 authoritative registry contract
# ---------------------------------------------------------------------------

def get_datasets() -> dict[str, str]:
    return result_registry.get_datasets()


def get_dataset(dataset_key: str) -> dict[str, Any]:
    return result_registry.get_dataset(dataset_key)


def get_models(dataset_key: str) -> dict[str, str]:
    return result_registry.get_models(dataset_key)


def get_model(dataset_key: str, model_key: str) -> dict[str, Any]:
    return result_registry.get_model(dataset_key, model_key)


def get_capabilities(dataset_key: str, model_key: str) -> dict[str, str]:
    return result_registry.get_capabilities(dataset_key, model_key)


def get_metrics(dataset_key: str, model_key: str) -> dict[str, Any]:
    return result_registry.get_metrics(dataset_key, model_key)


def get_prediction_schema(dataset_key: str, model_key: str) -> dict[str, Any]:
    return result_registry.get_prediction_schema(dataset_key, model_key)


def get_quantum_configuration(dataset_key: str, model_key: str) -> dict[str, Any]:
    return result_registry.get_quantum_configuration(dataset_key, model_key)


def get_explainability(dataset_key: str, model_key: str) -> dict[str, Any]:
    return result_registry.get_explainability(dataset_key, model_key)


# ---------------------------------------------------------------------------
# Dataset service
# ---------------------------------------------------------------------------

def get_dataset_characteristics(dataset_key: str) -> dict[str, Any]:
    return data_service.get_dataset_characteristics(dataset_key)


def get_dataset_metadata(dataset_key: str) -> dict[str, Any]:
    """
    Compatibility wrapper combining the authoritative dataset record with the
    repository-backed schema/characteristics service.
    """
    dataset_result = result_registry.get_dataset(dataset_key)
    if dataset_result["status"] != "available":
        return dataset_result

    characteristics = data_service.get_dataset_characteristics(dataset_key)
    schema = data_service.get_authoritative_schema(dataset_key)

    data = dataset_result["data"]
    char_data = (
        characteristics
        if characteristics.get("status") in {"available", "partial"}
        else {}
    )

    return {
        "status": "available",
        "data": {
            "dataset_key": data["dataset_key"],
            "name": data["name"],
            "domain": char_data.get("domain"),
            "sample_count": char_data.get("total_samples"),
            "feature_count": char_data.get("feature_count"),
            "feature_names": schema.get("feature_names"),
            "target_labels": schema.get("target_labels"),
            "class_count": (
                len(schema["target_labels"])
                if isinstance(schema.get("target_labels"), list)
                else None
            ),
            "classes": char_data.get("classes"),
            "class_ratio": char_data.get("class_ratio"),
            "cohort_source": char_data.get("cohort_source"),
            "schema_status": schema.get("status"),
            "schema_source": schema.get("source"),
        },
    }


def get_data_sample(dataset_key: str, n: int = 6) -> dict[str, Any]:
    return data_service.get_data_sample(dataset_key, n=n)


# ---------------------------------------------------------------------------
# Backward-compatible names used by existing pages/app.py
# ---------------------------------------------------------------------------

def get_dataset_catalog() -> dict[str, str]:
    return get_datasets()


def get_available_models(dataset_key: str) -> dict[str, str]:
    return get_models(dataset_key)


def get_model_metadata(dataset_key: str, model_key: str) -> dict[str, Any]:
    return get_model(dataset_key, model_key)


def get_evaluation_metrics(dataset_key: str, model_key: str) -> dict[str, Any]:
    return get_metrics(dataset_key, model_key)


def get_confusion_matrix(dataset_key: str, model_key: str) -> dict[str, Any]:
    result = get_metrics(dataset_key, model_key)
    if result["status"] != "available":
        return result

    matrix = result["data"].get("confusion_matrix")
    if matrix is None:
        return {
            "status": "unavailable",
            "message": "Confusion matrix is not present in the evaluation record.",
        }

    return {"status": "available", "data": matrix}


def get_benchmark_figures(dataset_key: str) -> dict[str, str]:
    return result_registry.get_benchmark_artifacts(dataset_key)


def get_benchmark_table() -> dict[str, Any]:
    """
    Return a backend-driven benchmark table without manufacturing unavailable
    values. Formatting remains presentation-neutral.
    """
    rows: list[dict[str, Any]] = []

    for dataset_key, dataset_name in result_registry.get_datasets().items():
        for model_key, model_name in result_registry.get_models(dataset_key).items():
            model_result = result_registry.get_model(dataset_key, model_key)
            if model_result["status"] != "available":
                continue

            model = model_result["data"]
            metrics = model.get("metrics", {})
            protocol = model.get("protocol") or {}

            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Type": model.get("type"),
                "Protocol": protocol.get("split_strategy"),
                "Test Size": protocol.get("test_size"),
                "Accuracy": metrics.get("accuracy"),
                "Sensitivity": metrics.get("sensitivity"),
                "Specificity": metrics.get("specificity"),
                "F1 Score": metrics.get("f1_score", metrics.get("f1")),
                "ROC-AUC": metrics.get("roc_auc"),
                "MCC": metrics.get("mcc"),
                "Evaluation": (
                    "Available"
                    if model.get("evaluated") is True
                    else "Unavailable"
                ),
                "Inference": (
                    "Available"
                    if model.get("inference_ready") is True
                    else "Unavailable"
                ),
            })

    if not rows:
        return {
            "status": "unavailable",
            "message": "No evaluated model records were discovered.",
        }

    return {"status": "available", "data": rows}


def get_explainability_artifact(dataset_key: str, model_key: str) -> str | None:
    result = get_explainability(dataset_key, model_key)
    if result["status"] != "available":
        return None

    artifact = result["data"].get("artifact", {})
    return artifact.get("path") if isinstance(artifact, dict) else None


def predict_instance(
    dataset_key: str,
    model_key: str,
    input_data: dict[str, Any],
) -> dict[str, Any]:
    # Inference remains Phase 2 functionality; this adapter only delegates.
    return run_inference(dataset_key, model_key, input_data)


def get_quantum_circuit(dataset_key: str, model_key: str) -> dict[str, Any]:
    """
    Circuit construction remains a backend capability used by later phases.
    The adapter delegates to the existing quantum circuit utilities rather than
    declaring a circuit available merely because configuration exists.
    """
    model_result = get_model(dataset_key, model_key)
    if model_result["status"] != "available":
        return model_result

    model = model_result["data"]
    config = model.get("quantum_config")
    if not config:
        return {
            "status": "unavailable",
            "message": "No quantum configuration is available for this model.",
        }

    try:
        from src.quantum_circuit import build_feature_map, build_ansatz

        qubits = config.get("qubits")
        if not isinstance(qubits, int) or qubits < 1:
            return {
                "status": "unavailable",
                "message": "Quantum configuration does not contain a valid qubit count.",
            }

        fm_cfg = config.get("feature_map") or {}
        fm = build_feature_map(
            qubits,
            reps=fm_cfg.get("reps", 1),
            entanglement=fm_cfg.get("entanglement", "linear"),
        )

        if model.get("model_key") == "vqc":
            ans_cfg = config.get("ansatz") or {}
            ansatz = build_ansatz(
                qubits,
                reps=ans_cfg.get("reps", 2),
                entanglement=ans_cfg.get("entanglement", "linear"),
            )
            circuit = fm.compose(ansatz)
            title = (
                f"Variational Quantum Classifier Circuit "
                f"({qubits} qubits)"
            )
        else:
            circuit = fm
            title = f"Quantum Kernel Feature Map ({qubits} qubits)"

        return {
            "status": "available",
            "title": title,
            "qubits": qubits,
            "parameters": circuit.num_parameters,
            "depth": circuit.depth(),
            "circuit_str": circuit.draw(output="text").single_string(),
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Circuit reconstruction failed: {exc}",
        }
