import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import audit_backend
from Streamlit_dashboard.dashboard_core import backend_adapter


def registry_value(value):
    """
    Normalize registry capability values without inventing support for fields
    that the registry does not expose.
    """
    if value is None:
        return "unavailable"

    if isinstance(value, bool):
        return "VERIFIED_TRUE" if value else "VERIFIED_FALSE"

    return str(value)


def compare_capability(audit_value, registry_value_raw):
    """
    Return exactly one semantic relationship.

    'unavailable' means the registry has no corresponding capability field.
    It is a coverage gap, not an association problem and not a contradiction.
    """
    registry_state = registry_value(registry_value_raw)

    if registry_state == "unavailable":
        return "REGISTRY_COVERAGE_GAP"

    if audit_value == registry_state:
        return "MATCH"

    if audit_value == "UNVERIFIED" and registry_state in {
        "VERIFIED_TRUE",
        "VERIFIED_FALSE",
    }:
        return "UNVERIFIED_REGISTRY_CLAIM"

    if {
        audit_value,
        registry_state,
    } == {
        "VERIFIED_TRUE",
        "VERIFIED_FALSE",
    }:
        return "CONTRADICTION"

    if audit_value == "NOT_APPLICABLE" and registry_state == "VERIFIED_FALSE":
        return "MATCH"

    return "OTHER_FINDING"


def has_unverified_artifact_association(independent, dataset_key, model_key):
    """
    Determine whether the independent audit has actual artifact evidence for
    this dataset/model pair but cannot prove its provenance.

    This is deliberately separate from registry coverage.
    """
    for model in independent.get("models", []):
        if (
            model.get("dataset_key") == dataset_key
            and model.get("model_key") == model_key
        ):
            for artifact in model.get("artifact_evidence", []):
                if artifact.get("association") == "unverified_candidate":
                    return True

            for state in model.get("trained_state_candidates", []):
                if state.get("association") == "unverified_candidate":
                    return True

    return False


def discover_registry_models():
    registry_datasets_raw = backend_adapter.get_dataset_catalog()
    registry_datasets = set(registry_datasets_raw.keys())
    registry_models = set()

    for dataset_key in sorted(registry_datasets):
        try:
            models = backend_adapter.get_available_models(dataset_key)
        except Exception as exc:
            print(
                f"Registry model discovery failed for {dataset_key}: {exc}"
            )
            continue

        if isinstance(models, dict):
            model_keys = models.keys()
        else:
            model_keys = models

        for model_key in model_keys:
            registry_models.add(
                (dataset_key, str(model_key))
            )

    return registry_datasets, registry_models


def main():
    print("=== PHASE 0B BIDIRECTIONAL REGISTRY TEST ===")

    independent = audit_backend.build_audit()

    audit_datasets = set(independent["datasets"].keys())
    audit_models = {
        (model["dataset_key"], model["model_key"])
        for model in independent["models"]
    }

    print("\n1. Independent filesystem discovery")
    print("Audit datasets:", sorted(audit_datasets))
    print("Audit model combinations:")
    for item in sorted(audit_models):
        print(" ", item)

    print("\n2. Application registry discovery")
    registry_datasets, registry_models = discover_registry_models()

    print("Registry datasets:", sorted(registry_datasets))
    print("Registry model combinations:")
    for item in sorted(registry_models):
        print(" ", item)

    discrepancies = {
        "registry_only_datasets": sorted(
            registry_datasets - audit_datasets
        ),
        "audit_only_datasets": sorted(
            audit_datasets - registry_datasets
        ),
        "registry_only_models": [],
        "audit_only_models": sorted(
            audit_models - registry_models
        ),
        "registry_claim_with_unverified_independent_association": [],
        "true_contradictions": [],
        "registry_coverage_gaps": [],
        "unverified_registry_claims": [],
        "unverified_associations": [],
        "other_findings": [],
    }

    # A registry-only combination is not automatically an error. If the
    # independent audit found model/artifact evidence but could not establish
    # its dataset association, record that as an unverified association.
    for dataset_key, model_key in sorted(registry_models - audit_models):
        if has_unverified_artifact_association(
            independent,
            dataset_key,
            model_key,
        ):
            discrepancies[
                "registry_claim_with_unverified_independent_association"
            ].append({
                "dataset_key": dataset_key,
                "model_key": model_key,
                "reason": (
                    "Registry exposes the combination, while independent "
                    "filesystem evidence exists without a verified "
                    "dataset/model association."
                ),
            })
        else:
            discrepancies["registry_only_models"].append(
                (dataset_key, model_key)
            )

    print("\n3. Dataset/model set differences")
    for item in discrepancies["registry_only_datasets"]:
        print("REGISTRY_ONLY_DATASET:", item)
    for item in discrepancies["audit_only_datasets"]:
        print("AUDIT_ONLY_DATASET:", item)
    for item in discrepancies["registry_only_models"]:
        print("REGISTRY_ONLY_MODEL:", item)
    for item in discrepancies["audit_only_models"]:
        print("AUDIT_ONLY_MODEL:", item)

    audit_by_key = {
        (model["dataset_key"], model["model_key"]): model
        for model in independent["models"]
    }

    print("\n4. Capability comparison")

    # Registry capability mapping. 'None' means that the registry does not
    # expose that capability and therefore creates a coverage gap.
    for dataset_key, model_key in sorted(
        audit_models & registry_models
    ):
        audit_model = audit_by_key[(dataset_key, model_key)]

        try:
            capabilities = backend_adapter.get_capabilities(
                dataset_key,
                model_key,
            )
        except Exception as exc:
            discrepancies["other_findings"].append({
                "dataset_key": dataset_key,
                "model_key": model_key,
                "capability": "registry_runtime",
                "relationship": "OTHER_FINDING",
                "detail": str(exc),
            })
            continue

        if not isinstance(capabilities, dict):
            discrepancies["other_findings"].append({
                "dataset_key": dataset_key,
                "model_key": model_key,
                "capability": "capability_object",
                "relationship": "OTHER_FINDING",
                "detail": "Registry returned a non-dictionary capability object.",
            })
            continue

        capability_pairs = {
            "implemented": (
                audit_model["implemented"],
                None,
            ),
            "evaluated": (
                audit_model["evaluated"],
                capabilities.get("evaluation"),
            ),
            "trained_artifact_exists": (
                audit_model["trained_artifact_exists"],
                None,
            ),
            "preprocessing_artifact_exists": (
                audit_model["preprocessing_artifact_exists"],
                None,
            ),
            "runtime_loadable": (
                audit_model["runtime_loadable"],
                None,
            ),
            "inference_executable": (
                audit_model["inference_executable"],
                capabilities.get("prediction"),
            ),
            "quantum_configuration": (
                audit_model["quantum_configuration"],
                capabilities.get("quantum_configuration"),
            ),
            "circuit_reconstructible": (
                audit_model["circuit_reconstructible"],
                capabilities.get("circuit"),
            ),
            "trained_quantum_state_available": (
                audit_model["trained_quantum_state_available"],
                None,
            ),
            "benchmark": (
                audit_model["benchmark"],
                capabilities.get("benchmark"),
            ),
            "explainability": (
                audit_model["explainability"],
                capabilities.get("explainability"),
            ),
            "deployment_status": (
                audit_model["deployment_status"],
                None,
            ),
        }

        for capability, (audit_value, registry_raw) in capability_pairs.items():
            relationship = compare_capability(
                audit_value,
                registry_raw,
            )

            record = {
                "dataset_key": dataset_key,
                "model_key": model_key,
                "capability": capability,
                "independent_audit": audit_value,
                "registry": registry_value(registry_raw),
                "relationship": relationship,
            }

            if relationship == "CONTRADICTION":
                discrepancies["true_contradictions"].append(record)
            elif relationship == "REGISTRY_COVERAGE_GAP":
                discrepancies["registry_coverage_gaps"].append(record)
            elif relationship == "UNVERIFIED_REGISTRY_CLAIM":
                discrepancies["unverified_registry_claims"].append(record)
            elif relationship == "OTHER_FINDING":
                discrepancies["other_findings"].append(record)

    # Independent unverified artifact/state associations are reported
    # separately from registry comparisons.
    for model in independent.get("models", []):
        dataset_key = model["dataset_key"]
        model_key = model["model_key"]

        for artifact in model.get("artifact_evidence", []):
            if artifact.get("association") == "unverified_candidate":
                discrepancies["unverified_associations"].append({
                    "dataset_key": dataset_key,
                    "model_key": model_key,
                    "artifact": artifact.get("file"),
                    "type": "classical_artifact",
                    "reason": (
                        "Artifact matches the model key but lacks explicit "
                        "dataset provenance."
                    ),
                })

        for state in model.get("trained_state_candidates", []):
            if state.get("association") == "unverified_candidate":
                discrepancies["unverified_associations"].append({
                    "dataset_key": dataset_key,
                    "model_key": model_key,
                    "artifact": state.get("file"),
                    "type": "quantum_state_candidate",
                    "reason": (
                        "Quantum state candidate exists but lacks explicit "
                        "dataset/model provenance."
                    ),
                })

    # Remove duplicates while preserving deterministic output.
    for key in [
        "registry_only_datasets",
        "audit_only_datasets",
        "registry_only_models",
        "audit_only_models",
    ]:
        discrepancies[key] = sorted(
            set(discrepancies[key])
        )

    for key in [
        "registry_claim_with_unverified_independent_association",
        "true_contradictions",
        "registry_coverage_gaps",
        "unverified_registry_claims",
        "unverified_associations",
        "other_findings",
    ]:
        seen = set()
        unique = []
        for item in discrepancies[key]:
            marker = json.dumps(
                item,
                sort_keys=True,
                default=str,
            )
            if marker not in seen:
                seen.add(marker)
                unique.append(item)
        discrepancies[key] = unique

    # Dataset/model set differences are potentially significant, but they are
    # not automatically capability contradictions. Only explicit capability
    # contradictions are fatal in this validation.
    fatal = bool(discrepancies["true_contradictions"])

    print("\n5. RESULT")
    print(
        "True contradictions:",
        len(discrepancies["true_contradictions"]),
    )
    print(
        "Registry coverage gaps:",
        len(discrepancies["registry_coverage_gaps"]),
    )
    print(
        "Unverified registry claims:",
        len(discrepancies["unverified_registry_claims"]),
    )
    print(
        "Unverified associations:",
        len(discrepancies["unverified_associations"]),
    )
    print(
        "Other findings:",
        len(discrepancies["other_findings"]),
    )

    output_path = os.path.join(
        os.path.dirname(__file__),
        "phase_0b_registry_discrepancies.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            discrepancies,
            f,
            indent=2,
            default=str,
        )

    if fatal:
        print("STATUS: FAIL_WITH_CONTRADICTIONS")
        sys.exit(1)

    print("STATUS: PASS")
    print("No true bidirectional registry contradictions detected.")


if __name__ == "__main__":
    main()
