"""
Authoritative backend registry for the Hybrid Quantum–Classical Disease Detection
platform.

Phase 1 contract:
- Discover dataset/model records from repository evidence.
- Do not maintain a hard-coded dataset × model Cartesian product.
- Do not manufacture missing metrics, artifacts, schemas, or capabilities.
- Keep evaluation evidence separate from runtime inference readiness.
- Expose one stable API to the presentation adapter.

This module is intentionally independent of Streamlit and does not import the
dashboard layer, data_service, inference engine, or Phase 0 audit code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
INFERENCE_EVIDENCE_PATH = RESULTS_DIR / "inference_capabilities.json"
PHASE0_CAPABILITY_MATRIX_PATH = PROJECT_ROOT / "phase_0_capability_matrix.json"

# These are filesystem-independent aliases used only to normalize keys that are
# explicitly present in repository result metadata. They are not a dataset/model
# availability map.
_NODE_MODEL_HINTS = {
    "logistic_regression": "logistic_regression",
    "logistic": "logistic_regression",
    "svm": "svm",
    "support_vector_machine": "svm",
    "random_forest": "random_forest",
    "xgboost": "xgboost",
    "vqc": "vqc",
    "qsvc": "qsvc",
    "quantum_svc": "qsvc",
    "quantum_kernel_classifier": "qsvc",
}

_MODEL_DISPLAY_FALLBACKS = {
    "logistic_regression": "Logistic Regression",
    "svm": "Support Vector Machine (SVM)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "vqc": "Variational Quantum Classifier (VQC)",
    "qsvc": "Quantum Support Vector Classifier (QSVC)",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_repo_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def _exists_repo_artifact(value: Any) -> bool:
    path = _resolve_repo_path(value)
    return bool(path and path.is_file())


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _model_key_from_record(node_key: str, record: dict[str, Any]) -> str | None:
    explicit = record.get("model_key") or record.get("model_type")
    if isinstance(explicit, str) and explicit.strip():
        normalized = _slug(explicit)
        if normalized in _NODE_MODEL_HINTS:
            return _NODE_MODEL_HINTS[normalized]
        if normalized:
            return normalized

    quantum = record.get("quantum_config")
    name = record.get("name") or record.get("model_name") or ""
    normalized_node = _slug(node_key)
    normalized_name = _slug(name)

    if normalized_node in _NODE_MODEL_HINTS:
        return _NODE_MODEL_HINTS[normalized_node]

    if quantum:
        if "kernel" in quantum or "qsvc" in normalized_name or "quantum_support" in normalized_name:
            return "qsvc"
        if "ansatz" in quantum or "vqc" in normalized_name or "variational" in normalized_name:
            return "vqc"

    for hint, model_key in _NODE_MODEL_HINTS.items():
        if hint in normalized_name:
            return model_key

    return normalized_node or None


def _display_name(model_key: str, record: dict[str, Any]) -> str:
    value = record.get("name") or record.get("model_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _MODEL_DISPLAY_FALLBACKS.get(
        model_key,
        model_key.replace("_", " ").title(),
    )


def _dataset_display_name(dataset_key: str, record: dict[str, Any]) -> str:
    for key in ("dataset_name", "name", "dataset", "display_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # This is a neutral key-derived fallback, not a scientific claim.
    return dataset_key.replace("_", " ").title()


def _is_quantum_model(model_key: str, record: dict[str, Any]) -> bool:
    if model_key in {"vqc", "qsvc"}:
        return True
    return bool(record.get("quantum_config"))


def _load_phase0_artifact_bindings() -> dict[str, dict[str, dict[str, str]]]:
    """Load artifact bindings from the existing Phase 0 capability matrix.

    The matrix is used only to enrich artifact provenance for models that have
    already been discovered from authoritative evaluation evidence. It never
    creates a dataset/model record by itself and therefore is not a model
    availability inventory.
    """
    payload = _load_json(PHASE0_CAPABILITY_MATRIX_PATH)
    if not isinstance(payload, dict):
        return {}

    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        return {}

    bindings: dict[str, dict[str, dict[str, str]]] = {}
    for dataset_key, dataset_record in datasets.items():
        if not isinstance(dataset_record, dict):
            continue
        models = dataset_record.get("models")
        if not isinstance(models, dict):
            continue

        dataset_bindings: dict[str, dict[str, str]] = {}
        for model_key, model_record in models.items():
            if not isinstance(model_record, dict):
                continue

            binding: dict[str, str] = {}
            artifact = model_record.get("artifact")
            if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                binding["artifact"] = artifact["path"]

            preprocessing = model_record.get("preprocessing")
            if isinstance(preprocessing, dict) and isinstance(preprocessing.get("path"), str):
                binding["preprocessing"] = preprocessing["path"]

            if binding:
                dataset_bindings[_slug(model_key)] = binding

        if dataset_bindings:
            bindings[_slug(dataset_key)] = dataset_bindings

    return bindings


def _merge_phase0_artifact_bindings(
    dataset_key: str,
    model_key: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Merge Phase 0 artifact provenance into an already-discovered model record.

    The evaluation metadata remains authoritative for model existence and
    metrics. Phase 0 contributes only explicit artifact paths that it declares.
    No filesystem filename is guessed here.
    """
    enriched = deepcopy(record)
    bindings = _load_phase0_artifact_bindings()
    binding = bindings.get(_slug(dataset_key), {}).get(_slug(model_key), {})
    if not binding:
        return enriched

    artifacts = enriched.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    else:
        artifacts = deepcopy(artifacts)

    artifact_path = binding.get("artifact")
    if isinstance(artifact_path, str) and artifact_path.strip():
        # Preserve an explicit evaluation artifact. Otherwise, use the Phase 0
        # declared artifact under the semantic key expected by the registry.
        if _is_quantum_model(model_key, enriched):
            existing_weights = artifacts.get("weights")
            if existing_weights in (None, ""):
                artifacts["weights"] = artifact_path
        else:
            existing_model = artifacts.get("model")
            existing_weights = artifacts.get("weights")
            if existing_model in (None, "") and existing_weights in (None, ""):
                artifacts["model"] = artifact_path

    preprocessing_path = binding.get("preprocessing")
    if isinstance(preprocessing_path, str) and preprocessing_path.strip():
        if artifacts.get("preprocessing") in (None, ""):
            artifacts["preprocessing"] = preprocessing_path

    enriched["artifacts"] = artifacts
    enriched["phase0_artifact_source"] = _relative(PHASE0_CAPABILITY_MATRIX_PATH)
    return enriched


def _normalize_artifacts(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("artifacts")
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            normalized[key] = {
                "path": value,
                "exists": _exists_repo_artifact(value),
            }
        elif value is None:
            normalized[key] = {
                "path": None,
                "exists": False,
            }
        else:
            normalized[key] = deepcopy(value)
    return normalized


def _load_dataset_result(dataset_key: str) -> dict[str, Any] | None:
    return _load_json(RESULTS_DIR / f"{dataset_key}_metrics.json")


def _discover_metric_models(dataset_key: str, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, dict):
        return {}

    discovered: dict[str, dict[str, Any]] = {}
    for node_key, raw_record in models.items():
        if not isinstance(raw_record, dict):
            continue

        model_key = _model_key_from_record(str(node_key), raw_record)
        if not model_key:
            continue

        record = deepcopy(raw_record)
        record["_node_key"] = str(node_key)
        discovered[model_key] = record

    return discovered


def _discover_wdbc_classical_models() -> dict[str, dict[str, Any]]:
    """
    Discover additional WDBC classical evaluation records from the repository
    comparison CSV. Model identity comes from the CSV's own Model column.
    """
    csv_path = RESULTS_DIR / "classical_test_results.csv"
    if not csv_path.is_file():
        return {}

    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return {}

    if "Model" not in frame.columns:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        name = str(row.get("Model", "")).strip()
        if not name:
            continue

        key = _model_key_from_record(_slug(name), {"name": name})
        if not key or key in {"vqc", "qsvc"}:
            continue

        def numeric(column: str) -> float | None:
            value = row.get(column)
            if pd.isna(value):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        tn = numeric("TN")
        fp = numeric("FP")
        fn = numeric("FN")
        tp = numeric("TP")

        confusion = None
        if all(v is not None for v in (tn, fp, fn, tp)):
            confusion = {
                "matrix": [
                    [int(tn), int(fp)],
                    [int(fn), int(tp)],
                ],
                "labels": ["Benign", "Malignant"],
            }

        result[key] = {
            "_node_key": "classical_test_results.csv",
            "name": name,
            "model_type": key,
            "metrics": {
                "accuracy": numeric("Accuracy"),
                "precision": numeric("Precision"),
                "sensitivity": numeric("Sensitivity"),
                "specificity": numeric("Specificity"),
                "f1_score": numeric("F1 Score"),
                "roc_auc": numeric("ROC-AUC"),
                "training_time_sec": numeric("Training Time (sec)"),
                "confusion_matrix": confusion,
                "n_test": (
                    int(tn + fp + fn + tp)
                    if all(v is not None for v in (tn, fp, fn, tp))
                    else None
                ),
            },
            "protocol": {
                "test_size": (
                    int(tn + fp + fn + tp)
                    if all(v is not None for v in (tn, fp, fn, tp))
                    else None
                )
            },
            "source": _relative(csv_path),
        }

    return result


def _discover_datasets() -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}

    if RESULTS_DIR.is_dir():
        for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
            dataset_key = path.name[: -len("_metrics.json")].strip().lower()
            if not dataset_key:
                continue

            payload = _load_json(path)
            if payload is None:
                continue

            datasets[dataset_key] = payload

    return datasets


def _dataset_schema_from_raw_files(dataset_key: str) -> dict[str, Any]:
    """
    Read schema evidence without invoking src.data_loader.

    The function is intentionally conservative. If the raw files do not expose
    trustworthy semantic feature names, the schema is reported as unavailable
    instead of inventing Feature_1 / Feature_2 style names.
    """
    key = dataset_key.lower().strip()

    if key == "wdbc":
        candidates = [
            DATA_DIR / "raw" / "wdbc" / "WDBC_raw.csv",
            DATA_DIR / "raw" / "wdbc" / "wdbc.data",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if frame.shape[1] >= 3:
                names = [str(c) for c in frame.columns[2:]]
                return {
                    "status": "available",
                    "source": _relative(path),
                    "feature_names": names,
                    "target_labels": ["Benign", "Malignant"],
                    "n_features": len(names),
                }

    if key == "uci":
        path = DATA_DIR / "raw" / "UCI" / "processed.cleveland.data"
        if path.is_file():
            try:
                frame = pd.read_csv(path, header=None)
                if frame.shape[1] == 14:
                    # The raw Cleveland file has no semantic header. Do not
                    # manufacture names here.
                    return {
                        "status": "available",
                        "source": _relative(path),
                        "feature_names": None,
                        "target_labels": ["Absence", "Presence"],
                        "n_features": 13,
                        "feature_names_status": "semantic_names_not_exposed_by_raw_file",
                    }
            except Exception:
                pass

    if key == "golub":
        train_path = DATA_DIR / "raw" / "golub" / "data_set_ALL_AML_train.csv"
        if train_path.is_file():
            try:
                frame = pd.read_csv(train_path)
                gene_columns = [
                    str(c)
                    for c in frame.columns
                    if str(c).isdigit()
                ]
                if gene_columns:
                    return {
                        "status": "available",
                        "source": _relative(train_path),
                        "feature_names": None,
                        "target_labels": ["ALL", "AML"],
                        "n_features": len(frame),
                        "feature_names_status": "gene_feature_names_not_exposed_as_column_schema",
                    }
            except Exception:
                pass

    return {
        "status": "unavailable",
        "source": None,
        "feature_names": None,
        "target_labels": None,
        "n_features": None,
    }


def _sha256(path: Path) -> str | None:
    """Return the SHA-256 fingerprint of a repository artifact, if readable."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _load_inference_evidence() -> dict[str, Any]:
    value = _load_json(INFERENCE_EVIDENCE_PATH)
    if not isinstance(value, dict):
        return {}
    return value


def _artifact_evidence_matches(item: Any) -> bool:
    """Validate one runtime-evidence artifact against the current repository."""
    if not isinstance(item, dict):
        return False
    path_value = item.get("path")
    expected_hash = item.get("sha256")
    path = _resolve_repo_path(path_value)
    if path is None or not path.is_file():
        return False
    if not isinstance(expected_hash, str) or not expected_hash:
        return False
    actual_hash = _sha256(path)
    return actual_hash is not None and actual_hash == expected_hash


def _inference_evidence_matches(
    dataset_key: str,
    model_key: str,
    record: dict[str, Any],
) -> bool:
    """Return True only for current artifacts with independently recorded runtime verification."""
    evidence = _load_inference_evidence()
    dataset_evidence = evidence.get(dataset_key)
    if not isinstance(dataset_evidence, dict):
        return False
    model_evidence = dataset_evidence.get(model_key)
    if not isinstance(model_evidence, dict):
        return False
    if model_evidence.get("status") != "VERIFIED":
        return False

    # Runtime evidence must identify the exact artifact(s) that were executed.
    recorded_artifacts = model_evidence.get("artifacts")
    if not isinstance(recorded_artifacts, dict) or not recorded_artifacts:
        return False

    current_artifacts = _normalize_artifacts(record)
    for artifact_key, evidence_item in recorded_artifacts.items():
        if not _artifact_evidence_matches(evidence_item):
            return False
        current_item = current_artifacts.get(artifact_key)
        if not isinstance(current_item, dict):
            return False
        if current_item.get("path") != evidence_item.get("path"):
            return False
        if current_item.get("exists") is not True:
            return False

    return True


def _capability_state(meta: dict[str, Any], capability: str) -> str:
    """
    Return an evidence-aware capability state.

    The registry never treats absence of evidence as verified false.
    """
    if capability == "evaluation":
        return "AVAILABLE" if meta.get("evaluated") is True else "UNAVAILABLE"

    if capability == "prediction":
        return "AVAILABLE" if meta.get("inference_ready") is True else "UNAVAILABLE"

    if capability == "quantum_configuration":
        return "AVAILABLE" if meta.get("quantum_config") else "NOT_APPLICABLE"

    if capability == "explainability":
        artifacts = meta.get("artifacts", {})
        item = artifacts.get("explainability")
        if isinstance(item, dict) and item.get("exists") is True:
            return "AVAILABLE"
        return "UNAVAILABLE"

    if capability == "benchmark":
        benchmark_artifacts = meta.get("benchmark_artifacts")
        comparison = meta.get("comparison")
        if isinstance(benchmark_artifacts, dict) and benchmark_artifacts:
            return "AVAILABLE"
        if isinstance(comparison, dict) and comparison:
            return "AVAILABLE"
        return "UNAVAILABLE"

    if capability == "circuit":
        # Configuration alone does not prove an executable circuit.
        return "UNVERIFIED"

    return "UNAVAILABLE"


def _build_model_metadata(
    dataset_key: str,
    dataset_payload: dict[str, Any],
    model_key: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _normalize_artifacts(record)
    quantum_config = record.get("quantum_config")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    protocol = record.get("protocol")
    if not isinstance(protocol, dict):
        protocol = {}

    preprocessing = record.get("preprocessing")
    if not isinstance(preprocessing, dict):
        preprocessing = None

    # Explicit metadata capability is evidence, but it is never allowed to
    # manufacture an artifact or runtime state that is absent from disk.
    declared_prediction = record.get("capabilities", {}).get("prediction") if isinstance(record.get("capabilities"), dict) else None

    model = {
        "dataset_key": dataset_key,
        "model_key": model_key,
        "name": _display_name(model_key, record),
        "type": "Quantum" if _is_quantum_model(model_key, record) else "Classical",
        "model_type": model_key,
        "node_key": record.get("_node_key"),
        "evaluated": bool(metrics),
        "metrics": deepcopy(metrics),
        "protocol": deepcopy(protocol),
        "preprocessing": deepcopy(preprocessing),
        "quantum_config": deepcopy(quantum_config) if isinstance(quantum_config, dict) else None,
        "artifacts": artifacts,
        "benchmark_artifacts": deepcopy(record.get("benchmark_artifacts", {}))
        if isinstance(record.get("benchmark_artifacts"), dict)
        else {},
        "comparison": deepcopy(record.get("comparison", {}))
        if isinstance(record.get("comparison"), dict)
        else {},
        "explainability_method": record.get("explainability_method"),
        "inference_ready": False,
        "trained": False,
        "offline_reason": None,
        "capabilities": {},
        "evidence": {
            "source": record.get("source"),
            "declared_prediction": declared_prediction,
        },
    }

    # Only an explicit model artifact that actually exists can establish a
    # trained artifact at registry level. For Phase 1 we deliberately do not
    # infer dataset ownership from generic filenames such as models/*.joblib.
    model_artifact = artifacts.get("model") or artifacts.get("weights")
    if isinstance(model_artifact, dict) and model_artifact.get("exists") is True:
        model["trained"] = True

    # Phase 2 runtime readiness is established only by persistent verification
    # evidence whose artifact fingerprints still match the current repository.
    if _inference_evidence_matches(dataset_key, model_key, record):
        model["inference_ready"] = True
        model["evidence"]["inference_runtime"] = "VERIFIED"
        model["evidence"]["inference_evidence_source"] = _relative(INFERENCE_EVIDENCE_PATH)
    else:
        model["evidence"]["inference_runtime"] = "UNVERIFIED"
        model["evidence"]["inference_evidence_source"] = _relative(INFERENCE_EVIDENCE_PATH) if INFERENCE_EVIDENCE_PATH.is_file() else None

    # A positive runtime declaration is not promoted to inference-ready merely
    # because metadata says so. Runtime readiness belongs to Phase 2.
    if declared_prediction is False:
        model["offline_reason"] = (
            "Interactive inference is not declared for this evaluated model."
        )
    elif declared_prediction is True and not model["trained"]:
        model["offline_reason"] = (
            "Inference is declared in evaluation metadata, but a verified "
            "dataset-specific model artifact is not established by this registry."
        )

    for capability in (
        "evaluation",
        "prediction",
        "quantum_configuration",
        "circuit",
        "explainability",
        "benchmark",
    ):
        model["capabilities"][capability] = _capability_state(model, capability)

    # Preserve a conservative distinction for trained quantum state evidence.
    if model["type"] == "Quantum":
        weights = artifacts.get("weights")
        if isinstance(weights, dict) and weights.get("exists") is True:
            model["trained_quantum_state_available"] = "VERIFIED_TRUE"
        else:
            model["trained_quantum_state_available"] = "VERIFIED_FALSE"

    return model


def _build_registry() -> dict[str, dict[str, Any]]:
    datasets = _discover_datasets()

    registry: dict[str, dict[str, Any]] = {}

    for dataset_key, payload in datasets.items():
        dataset_record = {
            "dataset_key": dataset_key,
            "name": _dataset_display_name(dataset_key, payload),
            "metadata_source": _relative(RESULTS_DIR / f"{dataset_key}_metrics.json"),
            "metadata": deepcopy(payload),
            "models": {},
            "schema": _dataset_schema_from_raw_files(dataset_key),
        }

        discovered_models = _discover_metric_models(dataset_key, payload)

        # WDBC's classical comparison CSV is a separate genuine evaluation
        # source and may contain models not represented as nodes in the metrics
        # JSON. Merge only identities actually present in that CSV.
        if dataset_key == "wdbc":
            for key, record in _discover_wdbc_classical_models().items():
                discovered_models.setdefault(key, record)

        for model_key, record in discovered_models.items():
            enriched_record = _merge_phase0_artifact_bindings(
                dataset_key,
                model_key,
                record,
            )
            dataset_record["models"][model_key] = _build_model_metadata(
                dataset_key,
                payload,
                model_key,
                enriched_record,
            )

        registry[dataset_key] = dataset_record

    return registry


# Registry is rebuilt on import so the dashboard reflects the repository state
# at application start. No UI state or model training occurs here.
_REGISTRY = _build_registry()


def refresh_registry() -> None:
    """Rebuild the in-memory registry from repository evidence."""
    global _REGISTRY
    _REGISTRY = _build_registry()


def get_datasets() -> dict[str, str]:
    """Return discovered dataset keys and display names."""
    return {
        key: value["name"]
        for key, value in _REGISTRY.items()
    }


def get_dataset(dataset_key: str) -> dict[str, Any]:
    """Return one discovered dataset record."""
    key = str(dataset_key).lower().strip()
    record = _REGISTRY.get(key)
    if record is None:
        return {
            "status": "error",
            "message": f"Dataset '{key}' was not discovered in repository evidence.",
        }
    return {"status": "available", "data": deepcopy(record)}


def get_models(dataset_key: str) -> dict[str, str]:
    """Return only models actually discovered for the selected dataset."""
    key = str(dataset_key).lower().strip()
    dataset = _REGISTRY.get(key)
    if dataset is None:
        return {}
    return {
        model_key: model["name"]
        for model_key, model in dataset["models"].items()
    }


def get_model(dataset_key: str, model_key: str) -> dict[str, Any]:
    """Return one discovered dataset/model record."""
    dkey = str(dataset_key).lower().strip()
    mkey = str(model_key).lower().strip()

    dataset = _REGISTRY.get(dkey)
    if dataset is None:
        return {
            "status": "error",
            "message": f"Dataset '{dkey}' was not discovered.",
        }

    model = dataset["models"].get(mkey)
    if model is None:
        return {
            "status": "error",
            "message": f"Model '{mkey}' was not discovered for dataset '{dkey}'.",
        }

    return {"status": "available", "data": deepcopy(model)}


def get_capabilities(dataset_key: str, model_key: str) -> dict[str, str]:
    """Return evidence-aware capabilities for a dataset/model."""
    result = get_model(dataset_key, model_key)
    if result["status"] != "available":
        return {}
    return deepcopy(result["data"]["capabilities"])


def get_metrics(dataset_key: str, model_key: str) -> dict[str, Any]:
    """Return only metrics present in the authoritative evaluation record."""
    result = get_model(dataset_key, model_key)
    if result["status"] != "available":
        return {
            "status": "error",
            "message": result["message"],
        }

    metrics = result["data"].get("metrics", {})
    return {"status": "available", "data": deepcopy(metrics)}


def get_prediction_schema(dataset_key: str, model_key: str) -> dict[str, Any]:
    """
    Return the dataset schema plus model-specific preprocessing context.

    Semantic feature names are returned only when exposed by repository data.
    """
    dkey = str(dataset_key).lower().strip()
    result = get_model(dkey, model_key)
    if result["status"] != "available":
        return result

    dataset = _REGISTRY[dkey]
    schema = deepcopy(dataset["schema"])
    schema["model_key"] = str(model_key).lower().strip()
    schema["preprocessing"] = deepcopy(result["data"].get("preprocessing"))
    schema["prediction_capability"] = result["data"]["capabilities"].get("prediction")
    return {"status": "available", "data": schema}


def get_quantum_configuration(dataset_key: str, model_key: str) -> dict[str, Any]:
    """Return quantum configuration when the selected model has one."""
    result = get_model(dataset_key, model_key)
    if result["status"] != "available":
        return result

    config = result["data"].get("quantum_config")
    if not config:
        return {
            "status": "unavailable",
            "message": "No quantum configuration is present for this model.",
        }

    return {"status": "available", "data": deepcopy(config)}


def get_explainability(dataset_key: str, model_key: str) -> dict[str, Any]:
    """Return explainability metadata only when an actual artifact is present."""
    result = get_model(dataset_key, model_key)
    if result["status"] != "available":
        return result

    artifacts = result["data"].get("artifacts", {})
    item = artifacts.get("explainability")

    if not isinstance(item, dict) or item.get("exists") is not True:
        return {
            "status": "unavailable",
            "message": "No verified explainability artifact is available for this model.",
        }

    return {
        "status": "available",
        "data": {
            "artifact": deepcopy(item),
            "method": result["data"].get("explainability_method"),
        },
    }


# ---------------------------------------------------------------------------
# Backward-compatible names used by the current dashboard adapter.
# ---------------------------------------------------------------------------

def get_dataset_catalog() -> dict[str, str]:
    return get_datasets()


def get_available_models(dataset_key: str) -> dict[str, str]:
    return get_models(dataset_key)


def get_model_metadata(dataset_key: str, model_key: str) -> dict[str, Any]:
    return get_model(dataset_key, model_key)


def get_benchmark_artifacts(dataset_key: str) -> dict[str, str]:
    result = get_dataset(dataset_key)
    if result["status"] != "available":
        return {}

    dataset = result["data"]
    merged: dict[str, str] = {}

    raw = dataset.get("metadata", {}).get("benchmark_artifacts", {})
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, str) and _exists_repo_artifact(value):
                merged[key] = value

    return merged
