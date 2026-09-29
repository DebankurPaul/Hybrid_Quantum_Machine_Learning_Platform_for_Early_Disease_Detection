import os
import re
import glob
import json
import hashlib
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


VERIFIED_TRUE = "VERIFIED_TRUE"
VERIFIED_FALSE = "VERIFIED_FALSE"
UNVERIFIED = "UNVERIFIED"
NOT_APPLICABLE = "NOT_APPLICABLE"


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def relpath(path):
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
            return digest.hexdigest()
    except Exception:
        return None


def normalize_key(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def discover_metric_files():
    if not os.path.isdir(RESULTS_DIR):
        return {}

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_metrics.json")))
    return {
        os.path.basename(path).replace("_metrics.json", ""): path
        for path in files
    }


def discover_raw_files():
    if not os.path.isdir(RAW_DIR):
        return []

    result = []
    for root, _, files in os.walk(RAW_DIR):
        for name in files:
            result.append(os.path.join(root, name))
    return sorted(result)


def inspect_tabular_file(path):
    evidence = {
        "path": relpath(path),
        "rows": None,
        "columns": None,
        "header": None,
        "sha256": sha256(path),
    }

    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
            evidence["rows"] = int(df.shape[0])
            evidence["columns"] = int(df.shape[1])
            evidence["header"] = [str(column) for column in df.columns]
        else:
            df = pd.read_csv(path, header=None, na_values=["?"])
            evidence["rows"] = int(df.shape[0])
            evidence["columns"] = int(df.shape[1])
        return evidence
    except Exception as exc:
        evidence["error"] = str(exc)
        return evidence


def discover_raw_schema_evidence():
    evidence = []

    for path in discover_raw_files():
        lower = path.lower()
        if lower.endswith(".csv") or lower.endswith(".data"):
            evidence.append(inspect_tabular_file(path))

    return evidence


def discover_classical_artifacts():
    """
    Discover serialized classical estimators physically present in models/.

    The audit never assigns a generic model artifact to a dataset merely
    because the model filename matches a model key. Dataset association is
    verified only by explicit dataset-specific metadata/path evidence.
    """
    artifacts = []

    if not os.path.isdir(MODELS_DIR):
        return artifacts

    for name in sorted(os.listdir(MODELS_DIR)):
        if not name.lower().endswith(".joblib"):
            continue

        path = os.path.join(MODELS_DIR, name)
        record = {
            "file": relpath(path),
            "model_key_candidate": normalize_key(name[:-7]),
            "loadable": False,
            "n_features_in": None,
            "class": None,
            "error": None,
        }

        try:
            obj = joblib.load(path)
            record["loadable"] = True
            record["class"] = type(obj).__name__

            n_features = getattr(obj, "n_features_in_", None)
            if n_features is not None:
                record["n_features_in"] = int(n_features)
        except Exception as exc:
            record["error"] = str(exc)

        artifacts.append(record)

    return artifacts


def discover_preprocessing_artifacts():
    result = []

    if not os.path.isdir(RESULTS_DIR):
        return result

    for root, _, files in os.walk(RESULTS_DIR):
        for name in files:
            lower = name.lower()
            if not (
                "preprocess" in lower
                or "scaler" in lower
                or "pca" in lower
                or "selector" in lower
                or lower.endswith(".pt")
            ):
                continue

            path = os.path.join(root, name)
            result.append({
                "file": relpath(path),
                "size": os.path.getsize(path),
                "sha256": sha256(path),
            })

    return sorted(result, key=lambda item: item["file"])


def discover_numpy_artifacts():
    result = []

    if not os.path.isdir(RESULTS_DIR):
        return result

    for root, _, files in os.walk(RESULTS_DIR):
        for name in files:
            if not name.lower().endswith(".npy"):
                continue

            path = os.path.join(root, name)
            record = {
                "file": relpath(path),
                "shape": None,
                "dtype": None,
                "sha256": sha256(path),
            }

            try:
                array = np.load(path, allow_pickle=False)
                record["shape"] = list(array.shape)
                record["dtype"] = str(array.dtype)
            except Exception as exc:
                record["error"] = str(exc)

            result.append(record)

    return sorted(result, key=lambda item: item["file"])


def discover_csv_models():
    path = os.path.join(RESULTS_DIR, "classical_test_results.csv")
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if "Model" not in df.columns:
        return []

    return [
        {
            "model_key": normalize_key(name),
            "model_name": str(name),
            "source": "classical_test_results.csv",
        }
        for name in df["Model"].dropna().unique()
    ]


def extract_metric_models(dataset_key, data):
    """
    Model identity comes only from the model node itself.

    Generic node names such as 'classical_baseline' are retained as-is;
    the audit never converts them into a guessed algorithm.
    """
    models = []
    raw_models = data.get("models", {})

    if not isinstance(raw_models, dict):
        return models

    for node_key, node in raw_models.items():
        if not isinstance(node, dict):
            continue

        model_key = (
            node.get("model_key")
            or node.get("model_type")
            or node.get("name")
            or node_key
        )
        model_name = (
            node.get("name")
            or node.get("model_name")
            or node.get("display_name")
            or node_key
        )

        models.append({
            "dataset_key": dataset_key,
            "model_key": normalize_key(model_key),
            "model_name": str(model_name),
            "node_key": str(node_key),
            "metadata": node,
            "source": "metrics_json",
        })

    return models


def quantum_configuration_evidence(metadata):
    config = metadata.get("quantum_config")
    if not isinstance(config, dict):
        return {
            "present": False,
            "configuration": None,
        }

    return {
        "present": True,
        "configuration": config,
    }


def evaluate_quantum_reconstruction(metadata):
    """
    This audit does not claim executable circuit reconstruction from the
    mere presence of a quantum_config object.

    A configuration can document a circuit without proving that the current
    environment can instantiate it. Therefore this function verifies only
    whether the metadata contains the minimum structural information needed
    to attempt reconstruction. The resulting capability remains UNVERIFIED
    unless actual reconstruction is performed.
    """
    config = metadata.get("quantum_config")
    if not isinstance(config, dict):
        return {
            "state": NOT_APPLICABLE,
            "reason": "No quantum configuration is present.",
        }

    qubits = config.get("qubits", config.get("n_qubits"))
    feature_map = config.get("feature_map")
    ansatz = config.get("ansatz")

    if qubits is None or feature_map is None or ansatz is None:
        return {
            "state": UNVERIFIED,
            "reason": (
                "Quantum configuration is present, but it does not contain "
                "enough explicit structure to independently validate circuit "
                "construction."
            ),
        }

    try:
        if int(qubits) <= 0:
            return {
                "state": VERIFIED_FALSE,
                "reason": "Quantum configuration contains a non-positive qubit count.",
            }
    except (TypeError, ValueError):
        return {
            "state": VERIFIED_FALSE,
            "reason": "Quantum configuration contains an invalid qubit count.",
        }

    return {
        "state": UNVERIFIED,
        "reason": (
            "Quantum configuration contains qubit, feature-map, and ansatz "
            "metadata, but executable circuit reconstruction was not "
            "independently validated by this audit."
        ),
    }


def find_quantum_state_candidates():
    candidates = []

    for artifact in discover_numpy_artifacts():
        shape = artifact.get("shape")
        if isinstance(shape, list) and len(shape) == 1 and shape[0] > 0:
            candidates.append({
                "file": artifact["file"],
                "shape": shape,
                "size": int(shape[0]),
                "dtype": artifact.get("dtype"),
                "sha256": artifact.get("sha256"),
            })

    return candidates


def explicit_artifact_link(metadata):
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    weights = artifacts.get("weights")
    if not isinstance(weights, str) or not weights.strip():
        return None

    return os.path.normpath(weights).replace("\\", "/").lower()


def same_artifact_path(left, right):
    return os.path.basename(left).lower() == os.path.basename(right).lower()


def artifact_has_dataset_token(path, dataset_key):
    filename = os.path.basename(path).lower()
    token = normalize_key(dataset_key)
    return bool(token) and token in normalize_key(filename)


def build_audit():
    metric_files = discover_metric_files()
    raw_evidence = discover_raw_schema_evidence()
    classical_artifacts = discover_classical_artifacts()
    preprocessing = discover_preprocessing_artifacts()
    numpy_artifacts = discover_numpy_artifacts()
    csv_models = discover_csv_models()
    quantum_candidates = find_quantum_state_candidates()

    audit = {
        "audit_version": "4.0",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "independence": {
            "uses_result_registry": False,
            "uses_data_service": False,
            "uses_inference_engine_for_discovery": False,
            "uses_raw_filesystem": True,
        },
        "datasets": {},
        "models": [],
        "artifacts": {
            "classical": classical_artifacts,
            "preprocessing": preprocessing,
            "numpy": numpy_artifacts,
        },
        "csv_model_evidence": csv_models,
        "raw_schema_evidence": raw_evidence,
        "issues": [],
        "figures": [],
    }

    for dataset_key, path in metric_files.items():
        data = load_json(path)

        if data is None:
            audit["issues"].append(
                f"Unable to parse metric file: {relpath(path)}"
            )
            continue

        protocol = data.get("protocol")
        sample_count_evidence = {
            key: data[key]
            for key in ("n_train", "n_validation", "n_test", "n_samples")
            if key in data
        }

        audit["datasets"][dataset_key] = {
            "metric_file": relpath(path),
            "metric_file_sha256": sha256(path),
            "protocol_present": isinstance(protocol, dict),
            "protocol": protocol if isinstance(protocol, dict) else None,
            "sample_count_evidence": sample_count_evidence,
        }

        for discovered in extract_metric_models(dataset_key, data):
            metadata = discovered["metadata"]
            q_evidence = quantum_configuration_evidence(metadata)
            reconstruction = evaluate_quantum_reconstruction(metadata)

            evaluated = (
                VERIFIED_TRUE
                if isinstance(metadata.get("metrics"), dict)
                or "metrics" in metadata
                else UNVERIFIED
            )

            audit["models"].append({
                "dataset_key": dataset_key,
                "model_key": discovered["model_key"],
                "model_name": discovered["model_name"],
                "node_key": discovered["node_key"],
                "source": discovered["source"],
                "implemented": VERIFIED_TRUE,
                "evaluated": evaluated,
                "quantum_configuration": (
                    VERIFIED_TRUE
                    if q_evidence["present"]
                    else NOT_APPLICABLE
                ),
                "quantum_configuration_evidence": q_evidence["configuration"],
                "circuit_reconstructible": reconstruction["state"],
                "circuit_reconstruction_reason": reconstruction["reason"],
                "metadata": metadata,
                "artifact_evidence": [],
                "preprocessing_evidence": [],
                "trained_state_candidates": [],
                "trained_artifact_exists": (
                    UNVERIFIED
                    if q_evidence["present"]
                    else UNVERIFIED
                ),
                "preprocessing_artifact_exists": UNVERIFIED,
                "runtime_loadable": UNVERIFIED,
                "inference_executable": UNVERIFIED,
                "trained_quantum_state_available": (
                    UNVERIFIED
                    if q_evidence["present"]
                    else NOT_APPLICABLE
                ),
                "explainability": UNVERIFIED,
                "benchmark": UNVERIFIED,
                "deployment_status": UNVERIFIED,
                "issues": [],
            })

    # Classical artifact association.
    # A generic model filename is evidence that the artifact exists, but not
    # evidence that it belongs to a particular dataset.
    for model in audit["models"]:
        if model["quantum_configuration"] != NOT_APPLICABLE:
            continue

        linked = explicit_artifact_link(model["metadata"])

        for artifact in classical_artifacts:
            if artifact["model_key_candidate"] != model["model_key"]:
                continue

            filename = artifact["file"]
            explicit_link = False

            if linked:
                explicit_link = (
                    filename.lower() == linked
                    or same_artifact_path(filename, linked)
                )

            dataset_specific = artifact_has_dataset_token(
                filename,
                model["dataset_key"],
            )

            evidence = dict(artifact)
            evidence["explicit_link"] = bool(explicit_link)
            evidence["dataset_specific_filename"] = bool(dataset_specific)
            evidence["association"] = (
                "verified"
                if explicit_link or dataset_specific
                else "unverified_candidate"
            )
            model["artifact_evidence"].append(evidence)

        verified = [
            item
            for item in model["artifact_evidence"]
            if item["association"] == "verified"
        ]

        if verified:
            model["trained_artifact_exists"] = VERIFIED_TRUE
            model["runtime_loadable"] = (
                VERIFIED_TRUE
                if any(item.get("loadable") for item in verified)
                else VERIFIED_FALSE
            )

    # Dataset/model-specific preprocessing filenames are retained as evidence,
    # but they do not prove that the complete runtime pipeline is executable.
    for model in audit["models"]:
        dataset_token = normalize_key(model["dataset_key"])
        model_token = normalize_key(model["model_key"])

        for artifact in preprocessing:
            filename_token = normalize_key(os.path.basename(artifact["file"]))
            if dataset_token in filename_token and model_token in filename_token:
                model["preprocessing_evidence"].append(artifact)

        if model["preprocessing_evidence"]:
            model["preprocessing_artifact_exists"] = VERIFIED_TRUE

    # Quantum state association.
    for model in audit["models"]:
        if model["quantum_configuration"] == NOT_APPLICABLE:
            continue

        linked = explicit_artifact_link(model["metadata"])
        model_token = normalize_key(model["model_key"])
        dataset_token = normalize_key(model["dataset_key"])

        for candidate in quantum_candidates:
            filename = candidate["file"]
            filename_token = normalize_key(os.path.basename(filename))

            explicit_link = bool(
                linked
                and (
                    filename.lower() == linked
                    or same_artifact_path(filename, linked)
                )
            )

            dataset_specific = dataset_token in filename_token
            model_specific = model_token in filename_token

            # A VQC weight file is not treated as QSVC trained state.
            if model_token == "qsvc" and "vqc" in filename_token and not explicit_link:
                continue

            if explicit_link:
                association = "verified"
            elif dataset_specific and model_specific:
                association = "verified"
            else:
                association = "unverified_candidate"

            model["trained_state_candidates"].append({
                **candidate,
                "association": association,
            })

    # Quantum state availability is independent of runtime loadability.
    for model in audit["models"]:
        if model["quantum_configuration"] == NOT_APPLICABLE:
            continue

        verified_states = [
            item
            for item in model["trained_state_candidates"]
            if item["association"] == "verified"
        ]
        candidate_states = [
            item
            for item in model["trained_state_candidates"]
            if item["association"] == "unverified_candidate"
        ]

        if verified_states:
            model["trained_quantum_state_available"] = VERIFIED_TRUE
            model["trained_artifact_exists"] = VERIFIED_TRUE
        elif candidate_states:
            model["trained_quantum_state_available"] = UNVERIFIED
            model["trained_artifact_exists"] = UNVERIFIED
        else:
            model["trained_quantum_state_available"] = VERIFIED_FALSE
            model["trained_artifact_exists"] = VERIFIED_FALSE
            model["runtime_loadable"] = VERIFIED_FALSE
            model["inference_executable"] = VERIFIED_FALSE
            model["deployment_status"] = VERIFIED_FALSE

    # Figures are physical evidence only; without machine-readable provenance
    # they cannot be assigned to a particular dataset/model benchmark.
    if os.path.isdir(FIGURES_DIR):
        for root, _, files in os.walk(FIGURES_DIR):
            for name in sorted(files):
                path = os.path.join(root, name)
                audit["figures"].append({
                    "file": relpath(path),
                    "sha256": sha256(path),
                    "provenance": "unverified",
                    "reason": (
                        "Figure exists physically, but this audit found no "
                        "machine-readable provenance linking it to a specific "
                        "dataset/model evaluation."
                    ),
                })

    return audit


if __name__ == "__main__":
    output = build_audit()
    output_path = os.path.join(
        PROJECT_ROOT,
        "phase_0b_independent_audit.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Independent Phase 0B audit written to:", output_path)
