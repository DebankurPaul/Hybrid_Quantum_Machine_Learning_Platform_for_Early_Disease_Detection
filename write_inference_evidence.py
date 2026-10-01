"""Create persistent, artifact-bound Phase 2 runtime verification evidence.

This script does not train models and does not fabricate predictions. It executes
models through the Phase 2 runtime handler using a real sample from the raw WDBC
dataset, then records only successful runtime verification plus fingerprints of
the artifacts used by the registered model.

The model inventory is discovered dynamically from result_registry; no dataset x
model list is hard-coded.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import inference
from src import result_registry


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
EVIDENCE_PATH = RESULTS_DIR / "inference_capabilities.json"


WDBC_RAW_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "wdbc" / "WDBC_raw.csv",
    PROJECT_ROOT / "data" / "raw" / "wdbc" / "wdbc.data",
)


def _sha256(path: Path) -> str | None:
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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def _load_existing() -> dict[str, Any]:
    if not EVIDENCE_PATH.is_file():
        return {}
    try:
        with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _load_wdbc_sample() -> tuple[dict[str, float], list[str]]:
    schema_result = result_registry.get_dataset("wdbc")
    if schema_result.get("status") != "available":
        raise RuntimeError("WDBC is not discovered by the authoritative registry.")

    schema = schema_result["data"].get("schema", {})
    feature_names = schema.get("feature_names")
    target_labels = schema.get("target_labels")
    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("WDBC semantic feature schema is unavailable.")
    if not isinstance(target_labels, list) or not target_labels:
        raise RuntimeError("WDBC target-label schema is unavailable.")

    raw_path = next((p for p in WDBC_RAW_CANDIDATES if p.is_file()), None)
    if raw_path is None:
        raise RuntimeError("WDBC raw dataset file was not found.")

    frame = pd.read_csv(raw_path)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise RuntimeError(f"WDBC raw sample is missing schema features: {missing}")

    row = frame.iloc[0]
    sample: dict[str, float] = {}
    for name in feature_names:
        value = row[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"WDBC sample feature '{name}' is not numeric.") from exc
        sample[name] = numeric

    return sample, [str(label) for label in target_labels]


def _artifact_fingerprints(model_meta: dict[str, Any]) -> dict[str, dict[str, str]]:
    artifacts = model_meta.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return {}

    fingerprints: dict[str, dict[str, str]] = {}
    # These are the artifacts that can affect inference execution. Explainability
    # and benchmark artifacts are deliberately excluded from runtime readiness.
    for key in ("model", "weights", "preprocessing"):
        item = artifacts.get(key)
        if not isinstance(item, dict):
            continue
        path = _resolve(item.get("path"))
        if path is None or not path.is_file():
            continue
        digest = _sha256(path)
        if digest is None:
            continue
        fingerprints[key] = {
            "path": _relative(path),
            "sha256": digest,
        }
    return fingerprints


def _verify_model(
    dataset_key: str,
    model_key: str,
    input_data: dict[str, float],
    target_labels: list[str],
) -> tuple[bool, dict[str, Any]]:
    model_result = result_registry.get_model(dataset_key, model_key)
    if model_result.get("status") != "available":
        return False, {"status": "FAILED", "reason": model_result.get("message")}

    model_meta = model_result["data"]
    expected_features = result_registry.get_dataset(dataset_key)["data"].get("schema", {}).get("feature_names")
    if not isinstance(expected_features, list) or not expected_features:
        return False, {
            "status": "FAILED",
            "reason": "Authoritative feature schema is unavailable for runtime validation.",
        }

    X_raw, validation_error = inference._validate_and_order_input(
        input_data,
        expected_features,
    )
    if validation_error is not None:
        return False, {
            "status": "FAILED",
            "reason": validation_error,
        }

    runtime_result = inference._run_registered_model(
        dataset_key,
        model_meta,
        X_raw,
        target_labels,
    )
    if not isinstance(runtime_result, dict) or runtime_result.get("status") != "success":
        return False, {
            "status": "FAILED",
            "reason": runtime_result.get("message") if isinstance(runtime_result, dict) else str(runtime_result),
            "runtime_code": runtime_result.get("code") if isinstance(runtime_result, dict) else None,
        }

    artifacts = _artifact_fingerprints(model_meta)
    if not artifacts:
        return False, {
            "status": "FAILED",
            "reason": "Runtime succeeded but no inference artifact fingerprint could be recorded.",
        }

    return True, {
        "status": "VERIFIED",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "execution_mode": runtime_result.get("execution_mode"),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_registry.refresh_registry()
    datasets = result_registry.get_datasets()
    if "wdbc" not in datasets:
        raise RuntimeError("WDBC was not discovered; no inference evidence was written.")

    sample, target_labels = _load_wdbc_sample()
    models = result_registry.get_models("wdbc")
    if not models:
        raise RuntimeError("No WDBC models were discovered; no inference evidence was written.")

    evidence = _load_existing()
    wdbc_evidence = evidence.setdefault("wdbc", {})

    print("Phase 2 runtime evidence verification")
    print(f"Dataset: wdbc")
    print(f"Discovered models: {list(models)}")
    print()

    for model_key in models:
        ok, record = _verify_model("wdbc", model_key, sample, target_labels)
        if ok:
            wdbc_evidence[model_key] = record
            print(f"[VERIFIED] {model_key}")
        else:
            # Preserve no stale positive claim for a model that failed this run.
            existing = wdbc_evidence.get(model_key)
            if isinstance(existing, dict) and existing.get("status") == "VERIFIED":
                wdbc_evidence.pop(model_key, None)
            print(f"[NOT VERIFIED] {model_key}: {record.get('reason')}")

    # Remove evidence for models that are no longer discovered. This keeps the
    # evidence file aligned with the repository inventory.
    discovered_keys = set(models)
    for key in list(wdbc_evidence):
        if key not in discovered_keys:
            wdbc_evidence.pop(key, None)

    with EVIDENCE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print()
    print(f"Evidence written: {_relative(EVIDENCE_PATH)}")

    # Rebuild the registry after writing evidence so this process can also prove
    # that the public capability state now reflects the verified runtime.
    result_registry.refresh_registry()

    print("\nRegistry prediction capabilities after evidence reconciliation:")
    all_ok = True
    for model_key in models:
        capabilities = result_registry.get_capabilities("wdbc", model_key)
        state = capabilities.get("prediction")
        print(f"  {model_key}: prediction={state}")
        if state != "AVAILABLE":
            all_ok = False

    if not all_ok:
        raise SystemExit("One or more WDBC models did not obtain VERIFIED runtime evidence.")

    print("\nPASS: all currently discovered WDBC models with successful runtime execution are inference-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
