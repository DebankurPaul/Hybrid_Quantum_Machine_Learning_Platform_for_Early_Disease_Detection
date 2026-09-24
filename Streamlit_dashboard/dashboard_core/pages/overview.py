"""
overview.py
===========

Research Platform Overview Dashboard.

Phase 6:
Provides a backend-driven overview of the active dataset, available model
landscape, measured model performance, and concise processing context.

The dashboard is presentation-only. Backend registry and service data remain
the source of truth.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import dashboard_core.backend_adapter as backend


def _format_metric(value: Any, percent: bool = True) -> str | None:
    """Format an authoritative numeric metric without inventing a fallback."""
    if not isinstance(value, (int, float)):
        return None

    if percent:
        return f"{value * 100:.2f}%"

    return f"{value:.4f}"


def _available_metrics(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """Return only authoritative numeric metrics present in the record."""
    definitions = [
        ("Accuracy", "accuracy", True),
        ("Precision", "precision", True),
        ("Sensitivity", "sensitivity", True),
        ("Specificity", "specificity", True),
        ("F1 Score", "f1_score", True),
        ("ROC-AUC", "roc_auc", True),
        ("MCC", "mcc", False),
    ]

    result: list[tuple[str, str]] = []

    for label, key, percent in definitions:
        value = metrics.get(key)

        if value is None and key == "f1_score":
            value = metrics.get("f1")

        formatted = _format_metric(value, percent=percent)

        if formatted is not None:
            result.append((label, formatted))

    return result


def _model_headline_metric(metrics: dict[str, Any]) -> tuple[str, str] | None:
    """Return the first available headline metric from authoritative evidence."""
    definitions = [
        ("Accuracy", "accuracy", True),
        ("F1", "f1_score", True),
        ("F1", "f1", True),
        ("ROC-AUC", "roc_auc", True),
        ("Sensitivity", "sensitivity", True),
        ("Specificity", "specificity", True),
    ]

    for label, key, percent in definitions:
        value = metrics.get(key)

        if value is None:
            continue

        formatted = _format_metric(value, percent=percent)

        if formatted is not None:
            return label, formatted

    return None


def _render_dataset_summary(
    dataset_info: dict[str, Any],
    available_models: dict[str, str],
    dataset_key: str,
) -> None:
    """Render concise dataset-level evidence."""
    n_models = len(available_models)

    evaluated = 0
    inference_ready = 0

    for model_key in available_models:
        capabilities = backend.get_capabilities(dataset_key, model_key)

        if capabilities.get("evaluation") in {
            True,
            "available",
            "AVAILABLE",
            "evaluation",
            "EVALUATION_ONLY",
        }:
            evaluated += 1

        if capabilities.get("prediction") in {
            True,
            "available",
            "AVAILABLE",
            "inference_ready",
            "INFERENCE_READY",
        }:
            inference_ready += 1

    sample_count = dataset_info.get("sample_count")
    feature_count = dataset_info.get("feature_count")
    class_count = dataset_info.get("class_count")

    metrics = [
        (
            "Samples",
            f"{sample_count:,}"
            if isinstance(sample_count, (int, float))
            else None,
        ),
        (
            "Features",
            str(feature_count)
            if isinstance(feature_count, (int, float))
            else None,
        ),
        (
            "Classes",
            str(class_count)
            if isinstance(class_count, (int, float))
            else None,
        ),
        ("Models", str(n_models)),
        ("Evaluated", str(evaluated)),
        ("Inference Ready", str(inference_ready)),
    ]

    valid_metrics = [(label, value) for label, value in metrics if value is not None]

    if not valid_metrics:
        return

    columns = st.columns(min(len(valid_metrics), 3))

    for index, (label, value) in enumerate(valid_metrics):
        with columns[index % len(columns)]:
            st.metric(label.upper(), value)


def _render_class_distribution(dataset_info: dict[str, Any]) -> None:
    """Render authoritative class distribution when it is exposed."""
    classes = dataset_info.get("classes")
    class_ratio = dataset_info.get("class_ratio")

    if not isinstance(classes, list) or not classes:
        return

    if not isinstance(class_ratio, dict) or not class_ratio:
        return

    rows: list[tuple[str, str]] = []

    for class_name in classes:
        value = class_ratio.get(class_name)

        if not isinstance(value, (int, float)):
            continue

        rows.append((str(class_name), f"{value * 100:.1f}%"))

    if not rows:
        return

    st.markdown("#### Class Distribution")

    distribution_columns = st.columns(len(rows))

    for index, (label, value) in enumerate(rows):
        with distribution_columns[index]:
            st.metric(label.upper(), value)


def _capability_state(capabilities: dict[str, Any]) -> str:
    """Resolve the current model capability state without inventing capability."""
    prediction = capabilities.get("prediction")
    evaluation = capabilities.get("evaluation")

    if prediction in {
        True,
        "available",
        "AVAILABLE",
        "inference_ready",
        "INFERENCE_READY",
    }:
        return "Inference Ready"

    if evaluation in {
        True,
        "available",
        "AVAILABLE",
        "evaluation",
        "EVALUATION_ONLY",
    }:
        return "Evaluation Only"

    return "Unavailable"


def _render_model_card(
    dataset_key: str,
    model_key: str,
    model_name: str,
    selected_model: str,
) -> None:
    """Render one backend-driven model card."""
    model_result = backend.get_model_metadata(dataset_key, model_key)

    if model_result.get("status") != "available":
        return

    model = model_result.get("data") or {}
    capabilities = backend.get_capabilities(dataset_key, model_key)

    model_type = model.get("type")
    metrics = model.get("metrics") or {}

    is_active = model_key == selected_model
    state = _capability_state(capabilities)
    headline = _model_headline_metric(metrics)

    with st.container(border=True):
        type_label = (
            "Quantum"
            if model_type == "Quantum"
            else "Classical"
            if model_type == "Classical"
            else None
        )

        if type_label:
            st.caption(type_label)

        st.markdown(f"**{model_name}**")

        st.caption(state)

        if headline:
            metric_label, metric_value = headline
            st.metric(metric_label, metric_value)

        if is_active:
            st.button(
                "Active",
                key=f"ov_active_{model_key}",
                disabled=True,
                type="primary",
                use_container_width=True,
            )
        else:
            if st.button(
                "Set Active",
                key=f"ov_set_{model_key}",
                use_container_width=True,
            ):
                st.session_state["selected_model"] = model_key
                st.rerun()


def _render_model_landscape(
    dataset_key: str,
    available_models: dict[str, str],
    selected_model: str,
) -> None:
    """Render the complete dynamically discovered model landscape."""
    if not available_models:
        st.info("No model records are currently available for this dataset.")
        return

    st.markdown("### Model Landscape")
    st.caption("Models discovered from the authoritative backend registry.")

    items = list(available_models.items())

    # Two cards per row keeps model names and controls readable at the
    # required 1280x720 viewport while remaining balanced on larger screens.
    for start in range(0, len(items), 2):
        row_items = items[start : start + 2]
        columns = st.columns(len(row_items))

        for column, (model_key, model_name) in zip(columns, row_items):
            with column:
                _render_model_card(
                    dataset_key,
                    model_key,
                    model_name,
                    selected_model,
                )


def _render_selected_model(
    dataset_key: str,
    selected_model: str,
    available_models: dict[str, str],
) -> None:
    """Render measured performance for the currently selected model."""
    model_result = backend.get_model_metadata(dataset_key, selected_model)

    if model_result.get("status") != "available":
        st.info("Selected model metadata is unavailable.")
        return

    model = model_result.get("data") or {}
    model_name = available_models.get(selected_model, selected_model)
    metrics = model.get("metrics") or {}

    st.markdown("### Selected Model")
    st.caption(f"Measured evaluation evidence for **{model_name}**.")

    metric_items = _available_metrics(metrics)

    if metric_items:
        # Keep the selected-model evidence readable without forcing too many
        # narrow native metric widgets into a single row.
        for start in range(0, len(metric_items), 3):
            row_items = metric_items[start : start + 3]
            columns = st.columns(len(row_items))

            for column, (label, value) in zip(columns, row_items):
                with column:
                    with st.container():
                        st.caption(label.upper())
                        st.markdown(
                            f"<div class='ov-selected-value'>{value}</div>",
                            unsafe_allow_html=True,
                        )
    else:
        st.info("No numerical evaluation metrics are recorded for this model.")

    protocol = model.get("protocol")

    if isinstance(protocol, dict):
        protocol_items: list[str] = []

        split_strategy = protocol.get("split_strategy")
        test_size = protocol.get("test_size")

        if split_strategy is not None:
            protocol_items.append(f"Protocol: {split_strategy}")

        if test_size is not None:
            protocol_items.append(f"Test Size: {test_size}")

        if protocol_items:
            st.caption(" · ".join(protocol_items))


def _render_processing_context(
    dataset_info: dict[str, Any],
    selected_model: str,
    dataset_key: str,
) -> None:
    """Render concise processing information actually exposed by the backend."""
    model_result = backend.get_model_metadata(dataset_key, selected_model)

    if model_result.get("status") != "available":
        return

    model = model_result.get("data") or {}
    preprocessing = model.get("preprocessing")

    has_preprocessing = isinstance(preprocessing, dict) and bool(preprocessing)

    quantum_configuration = None

    if model.get("type") == "Quantum":
        quantum_result = backend.get_quantum_configuration(
            dataset_key,
            selected_model,
        )

        if quantum_result.get("status") == "available":
            quantum_configuration = quantum_result.get("data")

    if not has_preprocessing and not quantum_configuration:
        return

    st.markdown("### Processing Context")

    columns = []

    if has_preprocessing:
        columns.append(("Preprocessing", preprocessing))

    if quantum_configuration:
        columns.append(("Quantum Configuration", quantum_configuration))

    layout = st.columns(min(len(columns), 2))

    for index, (title, payload) in enumerate(columns):
        with layout[index % len(layout)]:
            with st.container():
                st.markdown(f"**{title}**")

                if title == "Preprocessing":
                    display_value = (
                        payload.get("display")
                        if isinstance(payload, dict)
                        else None
                    )

                    if display_value:
                        st.caption(str(display_value))
                    else:
                        visible_items = []

                        if isinstance(payload, dict):
                            for key, value in payload.items():
                                if isinstance(value, (str, int, float, bool)):
                                    visible_items.append(
                                        f"{key.replace('_', ' ').title()}: {value}"
                                    )

                        if visible_items:
                            for item in visible_items:
                                st.caption(item)

                elif title == "Quantum Configuration":
                    if isinstance(payload, dict):
                        visible_items = []

                        qubits = payload.get("qubits")
                        if isinstance(qubits, int):
                            visible_items.append(f"Qubits: {qubits}")

                        feature_map = payload.get("feature_map")
                        if isinstance(feature_map, dict):
                            feature_map_name = feature_map.get("name")
                            if feature_map_name:
                                visible_items.append(
                                    f"Feature Map: {feature_map_name}"
                                )

                            reps = feature_map.get("reps")
                            if reps is not None:
                                visible_items.append(
                                    f"Feature Map Repetitions: {reps}"
                                )

                        ansatz = payload.get("ansatz")
                        if isinstance(ansatz, dict):
                            ansatz_name = ansatz.get("name")
                            if ansatz_name:
                                visible_items.append(
                                    f"Ansatz: {ansatz_name}"
                                )

                            reps = ansatz.get("reps")
                            if reps is not None:
                                visible_items.append(
                                    f"Ansatz Repetitions: {reps}"
                                )

                        if visible_items:
                            for item in visible_items:
                                st.caption(item)


def _render_explainability_status(
    dataset_key: str,
    selected_model: str,
) -> None:
    """Render a concise explainability state when authoritative evidence exists."""
    result = backend.get_explainability(dataset_key, selected_model)

    if result.get("status") == "available":
        st.caption("Explainability: Available")


def render() -> None:
    """Render the Phase 6 Overview workspace."""
    dataset_key = st.session_state.get("active_dataset")
    selected_model = st.session_state.get("selected_model")

    if not dataset_key:
        st.info("Select a dataset from the navigation context to begin.")
        return

    dataset_result = backend.get_dataset_metadata(dataset_key)

    if dataset_result.get("status") != "available":
        st.error("Dataset metadata is currently unavailable.")
        return

    dataset_info = dataset_result.get("data") or {}
    available_models = backend.get_available_models(dataset_key)

    if not available_models:
        st.warning("No model records are currently available for this dataset.")
        return

    if selected_model not in available_models:
        selected_model = next(iter(available_models))
        st.session_state["selected_model"] = selected_model

    dataset_name = dataset_info.get("name") or dataset_key

    st.markdown(f"## {dataset_name}")
    st.caption(
        "Dataset overview and measured model evidence from the research backend."
    )

    st.divider()

    _render_dataset_summary(
        dataset_info,
        available_models,
        dataset_key,
    )

    st.markdown("")

    _render_class_distribution(dataset_info)

    st.divider()

    # Keep the model landscape full-width so cards retain enough horizontal
    # space for names, capability states, metrics, and controls.
    _render_model_landscape(
        dataset_key,
        available_models,
        selected_model,
    )

    st.divider()

    _render_selected_model(
        dataset_key,
        selected_model,
        available_models,
    )

    _render_explainability_status(
        dataset_key,
        selected_model,
    )

    st.divider()

    _render_processing_context(
        dataset_info,
        selected_model,
        dataset_key,
    )

    st.caption(
        "Research prototype for comparative quantum-classical machine-learning "
        "analysis. Outputs are model results, not clinical diagnoses."
    )
