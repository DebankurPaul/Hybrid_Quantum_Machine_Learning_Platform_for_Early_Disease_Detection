from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src import result_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROC_DATA_PATH = PROJECT_ROOT / "results" / "test_roc_data.csv"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _normalize_column_name(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "ERROR",
        "code": code,
        "message": message,
    }


def _unavailable(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "code": code,
        "message": message,
    }


def get_metrics(dataset_key: str, model_key: str) -> dict[str, Any]:
    """
    Return authoritative evaluation metrics from result_registry.

    This service does not calculate or reconstruct metrics.
    """
    result = result_registry.get_model(dataset_key, model_key)

    if result.get("status") != "available":
        return _unavailable(
            "EVALUATION_NOT_AVAILABLE",
            result.get("message", "Evaluation data is unavailable."),
        )

    metrics = result.get("data", {}).get("metrics")

    if not isinstance(metrics, dict):
        return _unavailable(
            "METRICS_NOT_AVAILABLE",
            "No authoritative metric artifact is available for this model.",
        )

    return {
        "status": "AVAILABLE",
        "dataset_key": dataset_key,
        "model_key": model_key,
        "metrics": metrics,
    }


def get_confusion_matrix(
    dataset_key: str,
    model_key: str,
) -> dict[str, Any]:
    """
    Return the authoritative confusion matrix already stored in the
    evaluation artifact.

    This function never reconstructs a confusion matrix from scalar metrics.
    """
    metrics_result = get_metrics(dataset_key, model_key)

    if metrics_result.get("status") != "AVAILABLE":
        return metrics_result

    metrics = metrics_result["metrics"]
    confusion_matrix = metrics.get("confusion_matrix")

    if confusion_matrix is None:
        return _unavailable(
            "CONFUSION_MATRIX_NOT_AVAILABLE",
            "No authoritative confusion matrix is available for this model.",
        )

    return {
        "status": "AVAILABLE",
        "dataset_key": dataset_key,
        "model_key": model_key,
        "confusion_matrix": confusion_matrix,
    }


def _load_roc_frame() -> pd.DataFrame | None:
    """
    Load the repository ROC artifact without assigning it to a dataset.

    IMPORTANT:
    The current artifact has an internal Dataset column whose values are
    generic 'test' labels. Repository provenance is UNVERIFIED, so this
    dataframe must never be interpreted as belonging to WDBC, UCI, Golub,
    or any other registry dataset.
    """
    if not ROC_DATA_PATH.exists():
        return None

    try:
        frame = pd.read_csv(ROC_DATA_PATH)
    except Exception:
        return None

    if frame.empty:
        return None

    return frame


def _find_roc_columns(
    frame: pd.DataFrame,
) -> dict[str, str] | None:
    """
    Detect ROC columns using normalized column names.

    This function only identifies column names. It does NOT establish
    dataset provenance.
    """
    normalized = {
        _normalize_column_name(column): column
        for column in frame.columns
    }

    aliases = {
        "model": {
            "model",
            "model_name",
            "classifier",
            "estimator",
        },
        "fpr": {
            "false_positive_rate",
            "fpr",
        },
        "tpr": {
            "true_positive_rate",
            "tpr",
            "sensitivity",
            "recall",
        },
        "threshold": {
            "threshold",
            "thresholds",
        },
    }

    resolved: dict[str, str] = {}

    for semantic_name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                resolved[semantic_name] = normalized[candidate]
                break

    required = {"model", "fpr", "tpr"}

    if not required.issubset(resolved):
        return None

    return resolved


def get_roc_curve(
    dataset_key: str,
    model_key: str,
) -> dict[str, Any]:
    """
    Return authoritative ROC coordinates only when dataset provenance is
    established.

    CURRENT PHASE 3 POLICY:
    results/test_roc_data.csv has UNVERIFIED dataset provenance.

    Therefore this function intentionally returns UNAVAILABLE for every
    dataset/model combination.

    It is deliberately NOT sufficient to match a model name because the
    same model name can exist across multiple datasets.
    """
    return _unavailable(
        "ROC_PROVENANCE_UNVERIFIED",
        (
            "ROC coordinate data is unavailable because "
            "results/test_roc_data.csv has no authoritative dataset "
            "provenance. The artifact contains a generic Dataset value "
            "'test' and cannot safely be attributed to a registry dataset."
        ),
    )


def get_evaluation_protocol(
    dataset_key: str,
    model_key: str,
) -> dict[str, Any]:
    """
    Return the evaluation protocol recorded by the authoritative registry
    metadata.
    """
    result = result_registry.get_model(dataset_key, model_key)

    if result.get("status") != "available":
        return _unavailable(
            "EVALUATION_NOT_AVAILABLE",
            result.get("message", "Evaluation data is unavailable."),
        )

    protocol = result.get("data", {}).get("protocol")

    if not isinstance(protocol, dict) or not protocol:
        return _unavailable(
            "EVALUATION_PROTOCOL_NOT_AVAILABLE",
            "No authoritative evaluation protocol is available.",
        )

    return {
        "status": "AVAILABLE",
        "dataset_key": dataset_key,
        "model_key": model_key,
        "protocol": protocol,
    }


def get_evaluation_timing(
    dataset_key: str,
    model_key: str,
) -> dict[str, Any]:
    """
    Return timing fields already recorded in the authoritative metrics
    artifact.

    This function does not benchmark or measure a new runtime.
    """
    metrics_result = get_metrics(dataset_key, model_key)

    if metrics_result.get("status") != "AVAILABLE":
        return metrics_result

    metrics = metrics_result["metrics"]

    timing_keys = (
        "training_time_seconds",
        "training_time",
        "inference_time_seconds",
        "inference_time",
        "evaluation_time_seconds",
        "evaluation_time",
    )

    timing = {
        key: metrics[key]
        for key in timing_keys
        if key in metrics and metrics[key] is not None
    }

    if not timing:
        return _unavailable(
            "EVALUATION_TIMING_NOT_AVAILABLE",
            "No authoritative evaluation timing is recorded for this model.",
        )

    return {
        "status": "AVAILABLE",
        "dataset_key": dataset_key,
        "model_key": model_key,
        "timing": timing,
    }


def get_evaluation_capability(
    dataset_key: str,
    model_key: str,
) -> dict[str, Any]:
    """
    Return the authoritative evaluation capability state for a model.

    ROC coordinate availability is handled separately by get_roc_curve()
    because the current ROC artifact has unverified dataset provenance.
    """
    result = result_registry.get_model(dataset_key, model_key)

    if result.get("status") != "available":
        return _unavailable(
            "EVALUATION_NOT_AVAILABLE",
            result.get("message", "Evaluation data is unavailable."),
        )

    capabilities = result.get("data", {}).get("capabilities", {})
    evaluation_state = capabilities.get("evaluation")

    if not isinstance(evaluation_state, str):
        return _unavailable(
            "EVALUATION_CAPABILITY_NOT_AVAILABLE",
            "Evaluation capability state is not available.",
        )

    return {
        "status": "AVAILABLE",
        "dataset_key": dataset_key,
        "model_key": model_key,
        "evaluation": evaluation_state,
    }


def get_evaluation(
    dataset_key: str,
    model_key: str,
) -> dict[str, Any]:
    """
    Return the unified evaluation payload for one dataset/model pair.

    Each component retains its own evidence state.
    Missing evidence is never converted into a numeric placeholder.
    """
    metrics = get_metrics(dataset_key, model_key)
    confusion_matrix = get_confusion_matrix(dataset_key, model_key)
    roc_curve = get_roc_curve(dataset_key, model_key)
    protocol = get_evaluation_protocol(dataset_key, model_key)
    timing = get_evaluation_timing(dataset_key, model_key)
    capability = get_evaluation_capability(dataset_key, model_key)

    return {
        "status": "AVAILABLE"
        if metrics.get("status") == "AVAILABLE"
        else metrics.get("status", "UNAVAILABLE"),
        "dataset_key": dataset_key,
        "model_key": model_key,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "roc_curve": roc_curve,
        "protocol": protocol,
        "timing": timing,
        "capability": capability,
    }


def get_dataset_evaluations(dataset_key: str) -> dict[str, Any]:
    """
    Return evaluation payloads for all models registered under a dataset.
    """
    dataset = result_registry.get_dataset(dataset_key)

    if dataset.get("status") != "available":
        return _unavailable(
            "DATASET_NOT_AVAILABLE",
            dataset.get("message", "Dataset is unavailable."),
        )

    models = dataset.get("data", {}).get("models", [])

    evaluations: dict[str, Any] = {}

    for model in models:
        model_key = model.get("model_key")

        if not model_key:
            continue

        evaluations[model_key] = get_evaluation(
            dataset_key,
            model_key,
        )

    return {
        "status": "AVAILABLE",
        "dataset_key": dataset_key,
        "evaluations": evaluations,
    }
