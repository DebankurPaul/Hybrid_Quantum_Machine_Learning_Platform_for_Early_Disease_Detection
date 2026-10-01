"""
Repository-backed dataset and schema service for the Hybrid QML platform.

Phase 1 rules:
- Do not maintain a second dataset/model registry.
- Do not invoke the training data loader for dashboard metadata.
- Do not silently fall back to synthetic data.
- Derive dataset characteristics from repository files and result metadata.
- Return unavailable when authoritative semantic information is not exposed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src import result_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _read_wdbc() -> pd.DataFrame | None:
    path = DATA_DIR / "raw" / "wdbc" / "WDBC_raw.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_uci() -> pd.DataFrame | None:
    path = DATA_DIR / "raw" / "UCI" / "processed.cleveland.data"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, header=None)
    except Exception:
        return None


def _read_golub_labels() -> pd.DataFrame | None:
    path = DATA_DIR / "raw" / "golub" / "actual.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_authoritative_schema(dataset_key: str) -> dict[str, Any]:
    """
    Return schema information that can be established from repository evidence.

    Semantic feature names are never invented. When a raw source does not expose
    semantic names, the response explicitly says so.
    """
    key = str(dataset_key).lower().strip()
    dataset = result_registry.get_dataset(key)

    if dataset["status"] != "available":
        return dataset

    schema = dataset["data"].get("schema", {})
    if not isinstance(schema, dict):
        return {
            "status": "unavailable",
            "message": "No repository-backed schema record is available.",
        }

    return {
        "status": schema.get("status", "unavailable"),
        "feature_names": schema.get("feature_names"),
        "target_labels": schema.get("target_labels"),
        "n_features": schema.get("n_features"),
        "source": schema.get("source"),
        "feature_names_status": schema.get("feature_names_status"),
    }


def get_dataset_characteristics(dataset_key: str) -> dict[str, Any]:
    """Return dataset characteristics derived from raw files/result metadata."""
    key = str(dataset_key).lower().strip()
    dataset_result = result_registry.get_dataset(key)

    if dataset_result["status"] != "available":
        return dataset_result

    dataset = dataset_result["data"]
    raw_metadata = dataset.get("metadata", {})
    characteristics: dict[str, Any] = {
        "status": "available",
        "dataset_key": key,
        "dataset_name": dataset.get("name", key),
        "domain": raw_metadata.get("domain"),
        "total_samples": None,
        "feature_count": dataset.get("schema", {}).get("n_features"),
        "target_labels": dataset.get("schema", {}).get("target_labels"),
        "classes": None,
        "class_ratio": None,
        "cohort_source": raw_metadata.get("cohort_source"),
    }

    # Prefer explicit non-numeric descriptive metadata only when it is actually
    # present in the result artifact. Numeric counts are independently derived
    # below from the raw files when possible.
    if key == "wdbc":
        frame = _read_wdbc()
        if frame is None or frame.shape[1] < 3:
            return {
                **characteristics,
                "status": "partial",
                "message": "WDBC raw dataset file could not be read.",
            }

        characteristics["total_samples"] = int(len(frame))
        characteristics["feature_count"] = int(frame.shape[1] - 2)

        if "diagnosis" in frame.columns:
            counts = frame["diagnosis"].value_counts(dropna=True)
            classes = {}
            if "B" in counts:
                classes["Benign"] = int(counts["B"])
            if "M" in counts:
                classes["Malignant"] = int(counts["M"])
            if classes:
                characteristics["classes"] = classes
                total = sum(classes.values())
                characteristics["class_ratio"] = {
                    label: count / total for label, count in classes.items()
                }

    elif key == "uci":
        frame = _read_uci()
        if frame is None or frame.shape[1] < 2:
            return {
                **characteristics,
                "status": "partial",
                "message": "UCI Cleveland raw dataset file could not be read.",
            }

        characteristics["total_samples"] = int(len(frame))
        characteristics["feature_count"] = int(frame.shape[1] - 1)

        target = pd.to_numeric(frame.iloc[:, -1], errors="coerce").dropna()
        if not target.empty:
            presence = int((target > 0).sum())
            absence = int((target == 0).sum())
            characteristics["classes"] = {
                "Absence": absence,
                "Presence": presence,
            }
            total = absence + presence
            if total:
                characteristics["class_ratio"] = {
                    "Absence": absence / total,
                    "Presence": presence / total,
                }

    elif key == "golub":
        labels = _read_golub_labels()
        train_path = DATA_DIR / "raw" / "golub" / "data_set_ALL_AML_train.csv"

        if labels is None or not train_path.is_file():
            return {
                **characteristics,
                "status": "partial",
                "message": "Golub raw data or label file could not be read.",
            }

        try:
            train_frame = pd.read_csv(train_path)
        except Exception:
            return {
                **characteristics,
                "status": "partial",
                "message": "Golub training file could not be read.",
            }

        gene_rows = int(len(train_frame))
        characteristics["feature_count"] = gene_rows
        characteristics["total_samples"] = int(len(labels))

        if "cancer" in labels.columns:
            counts = labels["cancer"].value_counts(dropna=True)
            classes = {
                str(label): int(count)
                for label, count in counts.items()
            }
            characteristics["classes"] = classes
            total = sum(classes.values())
            if total:
                characteristics["class_ratio"] = {
                    label: count / total for label, count in classes.items()
                }

    return characteristics


def get_data_sample(dataset_key: str, n: int = 6) -> dict[str, Any]:
    """
    Return a small raw-data preview without invoking synthetic fallbacks.

    The orientation is deliberately kept close to the repository source:
    - WDBC: sample rows with raw column names.
    - UCI: raw Cleveland rows with integer column positions.
    - Golub: sample-level metadata from the label file.
    """
    key = str(dataset_key).lower().strip()
    limit = max(1, int(n))

    if key == "wdbc":
        frame = _read_wdbc()
        if frame is None:
            return {"status": "error", "message": "WDBC raw dataset is unavailable."}
        return {"status": "available", "data": frame.head(limit)}

    if key == "uci":
        frame = _read_uci()
        if frame is None:
            return {"status": "error", "message": "UCI Cleveland raw dataset is unavailable."}
        return {
            "status": "available",
            "data": frame.head(limit),
            "schema_note": "Raw Cleveland file does not expose semantic column names.",
        }

    if key == "golub":
        labels = _read_golub_labels()
        if labels is None:
            return {"status": "error", "message": "Golub label file is unavailable."}
        return {
            "status": "available",
            "data": labels.head(limit),
            "schema_note": "This preview shows sample metadata; gene expression columns are not expanded.",
        }

    return {
        "status": "error",
        "message": f"Dataset '{key}' was not discovered.",
    }
