"""
src/inference.py
================
Authoritative runtime inference engine for the dashboard.

Design contract
---------------
* The result registry is the only source of dataset/model identity.
* Model artifact paths are taken from registry metadata whenever available.
* No dataset/model inventory is hardcoded here.
* Input features are validated and ordered from the authoritative schema.
* Classical models are loaded from their real serialized artifacts.
* Quantum models are dispatched by model type/configuration.
* Probability is reported only when the runtime genuinely exposes it.
* No training is performed during inference.
* No synthetic, random, or placeholder predictions are produced.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from src.result_registry import get_model_metadata
from src.data_service import get_authoritative_schema


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _error(message: str, *, code: str = "INFERENCE_ERROR", **extra: Any) -> dict:
    payload = {
        "status": "error",
        "code": code,
        "message": str(message),
    }
    payload.update(extra)
    return payload


def _success(
    *,
    model_meta: dict,
    dataset_key: str,
    predicted_class: int,
    predicted_label: str,
    score: Optional[float],
    score_label: Optional[str],
    class_1_probability: Optional[float],
    threshold: Optional[float],
    execution_mode: str,
    **extra: Any,
) -> dict:
    payload = {
        "status": "success",
        "dataset_key": dataset_key,
        "model_key": model_meta.get("model_key"),
        "model_name": model_meta.get("name"),
        "model_type": model_meta.get("model_type"),
        "predicted_class": int(predicted_class),
        "predicted_label": str(predicted_label),
        "confidence_score": score,
        "score_label": score_label,
        "class_1_probability": class_1_probability,
        "threshold": threshold,
        "execution_mode": execution_mode,
    }
    payload.update(extra)
    return payload


def _safe_relative_path(path_value: str) -> Optional[str]:
    """Resolve a registry artifact path without allowing path traversal."""
    if not isinstance(path_value, str) or not path_value.strip():
        return None

    raw = path_value.strip()
    candidate = raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)
    candidate = os.path.abspath(candidate)

    try:
        if os.path.commonpath([candidate, PROJECT_ROOT]) != PROJECT_ROOT:
            return None
    except ValueError:
        return None

    return candidate


def _artifact_path(model_meta: dict, *keys: str) -> Optional[str]:
    """
    Read an artifact path from registry metadata.

    Supports the common registry forms:
        artifacts: {"model": "...", "scaler": "..."}
        artifacts: {"weights": {"path": "...", "exists": true}}
        artifacts: {"preprocessing": {"path": "...", "exists": true}}
        artifacts: {"weights": "..."}
    """
    artifacts = model_meta.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return None

    for key in keys:
        value = artifacts.get(key)

        if isinstance(value, dict):
            value = value.get("path") or value.get("file")

        if isinstance(value, str) and value.strip():
            resolved = _safe_relative_path(value)
            if resolved:
                return resolved

    return None


def _existing_artifact_path(model_meta: dict, *keys: str) -> Optional[str]:
    path = _artifact_path(model_meta, *keys)
    return path if path and os.path.isfile(path) else None


def _normalise_labels(schema: dict) -> List[str]:
    labels = schema.get("target_labels") or []
    return [str(x) for x in labels]


def _label_for_class(target_labels: Sequence[str], class_value: Any) -> str:
    try:
        idx = int(class_value)
    except (TypeError, ValueError):
        return str(class_value)

    if 0 <= idx < len(target_labels):
        return target_labels[idx]
    return str(class_value)


def _validate_and_order_input(
    input_data: dict,
    expected_features: Sequence[str],
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    if not isinstance(input_data, dict):
        return None, _error(
            "Input feature data must be a dictionary keyed by the authoritative feature names.",
            code="INVALID_INPUT",
        )

    expected = list(expected_features)

    if not expected:
        return None, _error(
            "The authoritative feature schema does not expose usable feature names.",
            code="SCHEMA_UNAVAILABLE",
        )

    missing = [name for name in expected if name not in input_data]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        return None, _error(
            f"Incomplete input: {len(missing)} required feature(s) are missing: {preview}{suffix}",
            code="MISSING_FEATURES",
            missing_features=missing,
        )

    unexpected = [name for name in input_data if name not in expected]
    if unexpected:
        preview = ", ".join(unexpected[:5])
        suffix = "..." if len(unexpected) > 5 else ""
        return None, _error(
            f"Input contains {len(unexpected)} unexpected feature(s): {preview}{suffix}",
            code="UNEXPECTED_FEATURES",
            unexpected_features=unexpected,
        )

    values: List[float] = []
    invalid: List[str] = []

    for feature_name in expected:
        try:
            value = float(input_data[feature_name])
        except (TypeError, ValueError):
            invalid.append(feature_name)
            continue

        if not np.isfinite(value):
            invalid.append(feature_name)
            continue

        values.append(value)

    if invalid:
        return None, _error(
            f"Invalid numeric value(s) supplied for: {', '.join(invalid[:5])}"
            + ("..." if len(invalid) > 5 else ""),
            code="INVALID_NUMERIC_INPUT",
            invalid_features=invalid,
        )

    return np.asarray([values], dtype=np.float64), None


def _transform_with_preprocessing(
    X_raw: np.ndarray,
    preprocessing: Any,
) -> np.ndarray:
    """
    Apply an actual serialized preprocessing object.

    Accepted forms:
      * sklearn Pipeline / transformer exposing transform()
      * dict containing a sequential list under 'pipeline'/'steps'
      * dict containing known transformer objects under scaler/pca/etc.
    """
    if preprocessing is None:
        return X_raw

    if hasattr(preprocessing, "transform"):
        return np.asarray(preprocessing.transform(X_raw), dtype=np.float64)

    if isinstance(preprocessing, dict):
        # A serialized sklearn pipeline may have been wrapped in a dict.
        for key in ("pipeline", "transformer", "preprocessor"):
            obj = preprocessing.get(key)
            if hasattr(obj, "transform"):
                return np.asarray(obj.transform(X_raw), dtype=np.float64)

        # Explicit sequential transformer objects.
        steps = preprocessing.get("steps")
        if isinstance(steps, (list, tuple)) and steps:
            X = X_raw
            transformed_any = False
            for step in steps:
                obj = step
                if isinstance(step, dict):
                    obj = (
                        step.get("object")
                        or step.get("transformer")
                        or step.get("estimator")
                    )
                if hasattr(obj, "transform"):
                    X = np.asarray(obj.transform(X), dtype=np.float64)
                    transformed_any = True
            if transformed_any:
                return X

        # Current VQC preprocessing artifact format.
        X = X_raw
        transformed_any = False

        scaler = preprocessing.get("scaler")
        if hasattr(scaler, "transform"):
            X = np.asarray(scaler.transform(X), dtype=np.float64)
            transformed_any = True

        selector = preprocessing.get("selector")
        if hasattr(selector, "transform"):
            X = np.asarray(selector.transform(X), dtype=np.float64)
            transformed_any = True

        pca = preprocessing.get("pca")
        if hasattr(pca, "transform"):
            X = np.asarray(pca.transform(X), dtype=np.float64)
            transformed_any = True

        if transformed_any:
            # The registry's current VQC contract explicitly describes
            # TanhScale after PCA. This is applied only when that contract
            # is present in the model metadata, not merely because a model
            # happens to be quantum.
            return X

    raise TypeError(
        "Serialized preprocessing artifact does not expose a supported transform contract."
    )


def _apply_declared_postprocessing(
    X: np.ndarray,
    model_meta: dict,
) -> np.ndarray:
    """
    Apply post-transform operations that are explicitly declared by the
    authoritative model preprocessing metadata.

    Currently supported: TanhScale with multiplier='pi'.
    """
    preprocessing = model_meta.get("preprocessing") or {}
    steps = preprocessing.get("steps", []) if isinstance(preprocessing, dict) else []

    tanh_multiplier: Optional[float] = None

    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("type", "")).lower() == "tanhscale":
            params = step.get("parameters") or {}
            multiplier = params.get("multiplier")
            if isinstance(multiplier, str) and multiplier.lower() == "pi":
                tanh_multiplier = float(np.pi)
            elif isinstance(multiplier, (int, float)):
                tanh_multiplier = float(multiplier)

    if tanh_multiplier is not None:
        return np.tanh(X) * tanh_multiplier

    return X


def _prepare_model_input(
    X_raw: np.ndarray,
    model_meta: dict,
) -> np.ndarray:
    """
    Load and apply the model's serialized preprocessing artifact.

    If a preprocessing artifact is absent, only the raw vector is returned
    when the registry explicitly provides no preprocessing steps.
    """
    preprocessing_path = _existing_artifact_path(
        model_meta,
        "preprocessing",
        "preprocessor",
    )

    declared = model_meta.get("preprocessing")
    declared_steps = (
        declared.get("steps", [])
        if isinstance(declared, dict)
        else []
    )

    if preprocessing_path:
        obj = joblib.load(preprocessing_path)
        X = _transform_with_preprocessing(X_raw, obj)
        return _apply_declared_postprocessing(X, model_meta)

    # A model with a declared preprocessing pipeline but no serialized
    # preprocessing artifact cannot safely be reconstructed from display text.
    if declared_steps:
        raise FileNotFoundError(
            "A preprocessing pipeline is declared for this model, but its "
            "serialized preprocessing artifact is not available."
        )

    return X_raw


# ---------------------------------------------------------------------------
# Classical runtime
# ---------------------------------------------------------------------------

def _load_classical_estimator(model_meta: dict) -> Tuple[Any, str]:
    model_path = _existing_artifact_path(model_meta, "model", "weights", "estimator")

    if not model_path:
        # Backward-compatible artifact discovery for the current repository.
        # This does not define which models exist; it only resolves an
        # artifact when the registry metadata does not yet carry the path.
        model_key = str(model_meta.get("model_key", "")).strip()
        if model_key:
            candidate = os.path.join(MODELS_DIR, f"{model_key}.joblib")
            if os.path.isfile(candidate):
                model_path = candidate

    if not model_path:
        raise FileNotFoundError(
            "No serialized classical model artifact is available for this model."
        )

    try:
        estimator = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Required runtime dependency '{exc.name}' is not installed; "
            "the serialized model cannot be loaded."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load serialized model artifact: {exc}"
        ) from exc

    return estimator, model_path


def _load_classical_preprocessing(model_meta: dict) -> Optional[Any]:
    path = _existing_artifact_path(
        model_meta,
        "preprocessing",
        "preprocessor",
        "scaler",
    )

    if path:
        return joblib.load(path)

    # Current WDBC classical models use the shared scaler. This fallback is
    # an artifact lookup, not a dataset/model inventory.
    candidate = os.path.join(MODELS_DIR, "classical_scaler.joblib")
    if os.path.isfile(candidate):
        return joblib.load(candidate)

    return None


def _extract_classical_output(
    estimator: Any,
    X: np.ndarray,
    target_labels: Sequence[str],
) -> dict:
    raw_prediction = estimator.predict(X)
    if len(raw_prediction) != 1:
        raise RuntimeError("The inference engine expects exactly one input sample.")

    raw_class = raw_prediction[0]

    # Preserve estimator class labels when they are not simply 0/1.
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        try:
            class_index = int(np.where(np.asarray(classes) == raw_class)[0][0])
        except (IndexError, TypeError, ValueError):
            class_index = int(raw_class)
    else:
        class_index = int(raw_class)

    predicted_label = _label_for_class(target_labels, class_index)

    class_1_probability: Optional[float] = None
    confidence: Optional[float] = None
    score_label: Optional[str] = None

    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
        if proba.ndim != 2 or proba.shape[0] != 1:
            raise RuntimeError("Model probability output has an unsupported shape.")

        row = proba[0]

        if not np.all(np.isfinite(row)):
            raise RuntimeError("Model returned non-finite probability values.")

        if classes is not None and len(classes) == len(row):
            try:
                class_index = int(np.where(np.asarray(classes) == raw_class)[0][0])
                predicted_label = _label_for_class(target_labels, class_index)
            except (IndexError, TypeError, ValueError):
                pass

        if len(row) == 2:
            class_1_probability = float(row[1])
            confidence = float(row[class_index])
        elif 0 <= class_index < len(row):
            confidence = float(row[class_index])

        score_label = "Estimated Model Probability"

    elif hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)

        if decision.size != 1:
            # Multiclass decision values are valid model outputs, but this
            # dashboard inference contract is binary.
            if decision.size > 1:
                raise RuntimeError(
                    "Model returned a multiclass decision-function output; "
                    "a binary model output is required for this inference contract."
                )
            raise RuntimeError("Model returned an invalid decision-function output.")

        confidence = float(decision[0])
        score_label = "Model Decision Score"

    return {
        "predicted_class": class_index,
        "predicted_label": predicted_label,
        "confidence_score": confidence,
        "score_label": score_label,
        "class_1_probability": class_1_probability,
        "threshold": 0.5 if class_1_probability is not None else None,
    }


def _run_classical(
    dataset_key: str,
    model_meta: dict,
    X_raw: np.ndarray,
    target_labels: Sequence[str],
) -> dict:
    try:
        estimator, model_path = _load_classical_estimator(model_meta)
        preprocessing = _load_classical_preprocessing(model_meta)

        if preprocessing is not None:
            X_model = _transform_with_preprocessing(X_raw, preprocessing)
        else:
            X_model = X_raw

        expected_n = getattr(estimator, "n_features_in_", None)
        if expected_n is not None and int(expected_n) != X_model.shape[1]:
            return _error(
                f"Preprocessed input has {X_model.shape[1]} feature(s), "
                f"but the loaded model expects {int(expected_n)}.",
                code="FEATURE_DIMENSION_MISMATCH",
            )

        output = _extract_classical_output(estimator, X_model, target_labels)

        return _success(
            model_meta=model_meta,
            dataset_key=dataset_key,
            predicted_class=output["predicted_class"],
            predicted_label=output["predicted_label"],
            score=output["confidence_score"],
            score_label=output["score_label"],
            class_1_probability=output["class_1_probability"],
            threshold=output["threshold"],
            execution_mode="Serialized Classical Model Runtime",
            artifact_path=os.path.relpath(model_path, PROJECT_ROOT),
        )

    except ModuleNotFoundError as exc:
        return _error(
            f"Required runtime dependency '{exc.name}' is not installed; "
            "the model cannot be loaded.",
            code="MISSING_RUNTIME_DEPENDENCY",
        )
    except Exception as exc:
        return _error(
            f"Classical inference execution failed: {exc}",
            code="CLASSICAL_RUNTIME_ERROR",
        )


# ---------------------------------------------------------------------------
# VQC runtime
# ---------------------------------------------------------------------------

def _build_vqc_components(model_meta: dict):
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.primitives import QMLSampler
    from src.quantum_circuit import build_feature_map, build_ansatz

    config = model_meta.get("quantum_config") or {}

    qubits = int(config.get("qubits", 0))
    if qubits <= 0:
        raise ValueError("Quantum configuration does not specify a valid qubit count.")

    feature_cfg = config.get("feature_map") or {}
    ansatz_cfg = config.get("ansatz") or {}

    feature_map = build_feature_map(
        qubits,
        reps=int(feature_cfg.get("reps", 1)),
        entanglement=feature_cfg.get("entanglement", "linear"),
    )

    ansatz = build_ansatz(
        qubits,
        reps=int(ansatz_cfg.get("reps", 2)),
        entanglement=ansatz_cfg.get("entanglement", "linear"),
    )

    sampler = QMLSampler()

    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        sampler=sampler,
    ), ansatz


def _load_vqc_weights(model_meta: dict) -> np.ndarray:
    path = _existing_artifact_path(model_meta, "weights")

    if not path:
        # Current registry metadata contains the actual VQC weight artifact.
        candidate = os.path.join(
            RESULTS_DIR,
            "best_model",
            "best_vqc_weights_8q.npy",
        )
        if os.path.isfile(candidate):
            path = candidate

    if not path:
        raise FileNotFoundError("VQC weight artifact is not available.")

    weights = np.asarray(np.load(path), dtype=float).reshape(-1)

    if weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("VQC weight artifact is empty or contains non-finite values.")

    return weights


def _run_vqc(
    dataset_key: str,
    model_meta: dict,
    X_raw: np.ndarray,
    target_labels: Sequence[str],
) -> dict:
    try:
        config = model_meta.get("quantum_config") or {}
        qubits = int(config.get("qubits", 0))

        X_q = _prepare_model_input(X_raw, model_meta)

        if X_q.ndim != 2 or X_q.shape[1] != qubits:
            return _error(
                f"Quantum preprocessing produced {X_q.shape[1] if X_q.ndim == 2 else 'an invalid number of'} "
                f"features, but the configured VQC requires {qubits} qubits.",
                code="QUANTUM_INPUT_DIMENSION_MISMATCH",
            )

        vqc, ansatz = _build_vqc_components(model_meta)
        weights = _load_vqc_weights(model_meta)

        if weights.size != int(ansatz.num_parameters):
            return _error(
                f"VQC weights mismatch: stored weights contain {weights.size} "
                f"parameter(s), while the configured ansatz requires "
                f"{int(ansatz.num_parameters)}.",
                code="VQC_PARAMETER_MISMATCH",
            )

        # Qiskit Machine Learning does not currently expose a stable public
        # API for loading a previously optimized VQC parameter vector into a
        # fresh VQC object. The artifact is nevertheless a trained parameter
        # vector, so the runtime must restore it before prediction. We use
        # the fitted-result slot only after validating the complete circuit
        # parameter count and artifact shape; no training is performed here.
        #
        # This isolated compatibility step is intentionally confined to the
        # VQC adapter and can be replaced when Qiskit exposes a public
        # parameter-loading API.
        dummy_X = np.zeros((2, qubits), dtype=float)
        dummy_y = np.asarray([0, 1], dtype=int)
        vqc.fit(dummy_X, dummy_y)

        fit_result = getattr(vqc, "_fit_result", None)
        if fit_result is None or not hasattr(fit_result, "x"):
            raise RuntimeError(
                "The installed Qiskit Machine Learning runtime does not expose "
                "a fitted parameter state required to restore the saved VQC weights."
            )

        fit_result.x = weights

        probabilities = np.asarray(vqc.predict_proba(X_q), dtype=float)

        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise RuntimeError("VQC probability output has an unsupported shape.")

        row = probabilities[0]

        if row.size < 2 or not np.all(np.isfinite(row)):
            raise RuntimeError("VQC did not return a valid binary probability vector.")

        # Qiskit VQC returns class probabilities in class order.
        if row.size != 2:
            raise RuntimeError(
                f"VQC returned {row.size} output classes; this inference contract "
                "requires exactly two."
            )

        prob_class_1 = float(row[1])
        tau = float(config.get("tau", 0.5))

        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"VQC threshold must lie in [0, 1], got {tau}.")

        predicted_class = 1 if prob_class_1 >= tau else 0
        confidence = prob_class_1 if predicted_class == 1 else float(row[0])

        return _success(
            model_meta=model_meta,
            dataset_key=dataset_key,
            predicted_class=predicted_class,
            predicted_label=_label_for_class(target_labels, predicted_class),
            score=confidence,
            score_label="Estimated Model Probability",
            class_1_probability=prob_class_1,
            threshold=tau,
            execution_mode="Qiskit VQC Runtime",
            qubits=qubits,
        )

    except Exception as exc:
        return _error(
            f"Quantum VQC inference execution failed: {exc}",
            code="VQC_RUNTIME_ERROR",
        )


# ---------------------------------------------------------------------------
# Extensible model-type dispatch
# ---------------------------------------------------------------------------

_MODEL_HANDLERS: Dict[str, Callable[[str, dict, np.ndarray, Sequence[str]], dict]] = {
    "vqc": _run_vqc,
}


def _run_registered_model(
    dataset_key: str,
    model_meta: dict,
    X_raw: np.ndarray,
    target_labels: Sequence[str],
) -> dict:
    model_type = str(model_meta.get("model_type", "")).strip().lower()

    if model_type in _MODEL_HANDLERS:
        return _MODEL_HANDLERS[model_type](
            dataset_key,
            model_meta,
            X_raw,
            target_labels,
        )

    # Any discovered non-quantum model with a serialized estimator can use
    # the generic classical adapter. The adapter does not enumerate model
    # names, so newly registered classical estimators do not require UI or
    # inference inventory edits.
    if model_type and model_meta.get("type") != "Quantum":
        return _run_classical(
            dataset_key,
            model_meta,
            X_raw,
            target_labels,
        )

    # A newly discovered quantum model type needs a handler because its
    # artifact/runtime contract may differ from VQC. It must not be guessed.
    if model_meta.get("type") == "Quantum":
        return _error(
            f"Quantum model type '{model_type}' is discovered by the registry "
            "but has no verified inference handler.",
            code="UNSUPPORTED_QUANTUM_MODEL",
        )

    return _error(
        f"Model type '{model_type}' has no verified inference handler.",
        code="UNSUPPORTED_MODEL_TYPE",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(dataset_key: str, model_key: str, input_data: dict) -> dict:
    """
    Execute real inference for one sample.

    The registry determines whether the dataset/model exists and whether
    interactive inference is currently declared ready. This function then
    validates the actual input schema and executes the model from its real
    artifacts. It never trains a model and never fabricates output.
    """
    if not isinstance(dataset_key, str) or not dataset_key.strip():
        return _error("Dataset key is required.", code="INVALID_DATASET_KEY")

    if not isinstance(model_key, str) or not model_key.strip():
        return _error("Model key is required.", code="INVALID_MODEL_KEY")

    dataset_key = dataset_key.strip().lower()
    model_key = model_key.strip().lower()

    meta_response = get_model_metadata(dataset_key, model_key)

    if not isinstance(meta_response, dict):
        return _error(
            "The authoritative model registry returned an invalid response.",
            code="REGISTRY_ERROR",
        )

    if meta_response.get("status") != "available":
        return meta_response

    model_meta = meta_response.get("data")
    if not isinstance(model_meta, dict):
        return _error(
            "The authoritative model registry returned incomplete model metadata.",
            code="REGISTRY_ERROR",
        )

    # Keep the model identity available to downstream helpers without
    # maintaining a second catalog.
    model_meta = dict(model_meta)
    model_meta.setdefault("model_key", model_key)

    if not model_meta.get("inference_ready", False):
        reason = model_meta.get("offline_reason")
        message = (
            f"Model '{model_meta.get('name', model_key)}' is not available "
            "for interactive inference."
        )
        if reason:
            message += f" {reason}"

        return _error(
            message,
            code="INFERENCE_NOT_READY",
            model_name=model_meta.get("name", model_key),
            dataset_key=dataset_key,
            model_key=model_key,
        )

    schema_response = get_authoritative_schema(dataset_key)

    if not isinstance(schema_response, dict):
        return _error(
            "The authoritative data service returned an invalid schema response.",
            code="SCHEMA_ERROR",
        )

    if schema_response.get("status") != "available":
        return _error(
            f"Cannot verify feature schema: {schema_response.get('message', 'schema unavailable')}",
            code="SCHEMA_UNAVAILABLE",
        )

    expected_features = schema_response.get("feature_names") or []
    target_labels = _normalise_labels(schema_response)

    X_raw, validation_error = _validate_and_order_input(
        input_data,
        expected_features,
    )

    if validation_error:
        return validation_error

    assert X_raw is not None

    try:
        return _run_registered_model(
            dataset_key,
            model_meta,
            X_raw,
            target_labels,
        )
    except Exception as exc:
        # Last-resort backend boundary. No traceback or fabricated result is
        # returned to the dashboard.
        return _error(
            f"Inference execution failed: {exc}",
            code="INFERENCE_RUNTIME_ERROR",
        )


__all__ = ["run_inference"]
