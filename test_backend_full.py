import os
import sys
import json
import glob

import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def discover_joblibs():
    if not os.path.isdir(MODELS_DIR):
        return []

    return sorted(
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".joblib")
    )


def make_runtime_vector(model):
    n_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    if n_features is None:
        return None

    return np.zeros(
        (1, int(n_features)),
        dtype=np.float64,
    )


def main():

    print("=== INDEPENDENT BACKEND RUNTIME SMOKE TEST ===")

    results = []

    for path in discover_joblibs():

        record = {
            "artifact": os.path.relpath(
                path,
                PROJECT_ROOT,
            ),
            "loadable": False,
            "predict_executed": False,
            "error": None,
        }

        try:
            model = joblib.load(path)
            record["loadable"] = True
            record["model_class"] = type(model).__name__

        except Exception as exc:
            record["error"] = (
                f"load_error: {exc}"
            )

            results.append(record)
            continue

        if hasattr(model, "transform") and not hasattr(model, "predict"):
            record["predict_test"] = "not_applicable"
            record["artifact_type"] = "preprocessing_transformer"
            results.append(record)
            continue

        X = make_runtime_vector(model)

        if X is None:
            record["error"] = (
                "No n_features_in_; runtime smoke test "
                "not applicable to this serialized object."
            )

            results.append(record)
            continue

        try:
            prediction = model.predict(X)

            record["predict_executed"] = True
            record["prediction_shape"] = list(
                np.asarray(prediction).shape
            )

        except Exception as exc:
            record["error"] = (
                f"predict_error: {exc}"
            )

        results.append(record)

    print(
        json.dumps(
            results,
            indent=2,
            default=str,
        )
    )

    with open(
        "phase_0b_runtime_smoke_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            default=str,
        )

    # A smoke test failure is a finding, not a Python test infrastructure
    # failure. Therefore exit 0 after recording all evidence.
    print("\nSTATUS: COMPLETE")
    print(
        "Runtime results are evidence only; they do not prove "
        "scientific validity or model quality."
    )


if __name__ == "__main__":
    main()
