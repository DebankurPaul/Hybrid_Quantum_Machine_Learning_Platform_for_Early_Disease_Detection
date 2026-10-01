"""
benchmark.py
============
Measured Model Comparison Workspace.

This page presents authoritative evaluation metadata exposed by the backend
for the currently selected dataset. It does not create benchmark artifacts,
calculate replacement metrics, rank models, or infer unavailable scientific
values.
"""

import streamlit as st
import pandas as pd

import dashboard_core.backend_adapter as backend


METRIC_DEFINITIONS = {
    "Accuracy": {"field": "accuracy", "format": "percent"},
    "Precision": {"field": "precision", "format": "percent"},
    "Sensitivity": {"field": "sensitivity", "format": "percent"},
    "Specificity": {"field": "specificity", "format": "percent"},
    "F1 Score": {
        "field": "f1_score",
        "fallback_field": "f1",
        "format": "percent",
    },
    "ROC-AUC": {"field": "roc_auc", "format": "percent"},
    "MCC": {"field": "mcc", "format": "decimal"},
}


def _unavailable_value():
    return "Unavailable"


def _metric_value(metrics, metric_name):
    definition = METRIC_DEFINITIONS[metric_name]
    value = metrics.get(definition["field"])

    if value is None and definition.get("fallback_field"):
        value = metrics.get(definition["fallback_field"])

    return value


def _format_metric(value, metric_name):
    if value is None:
        return _unavailable_value()

    fmt = METRIC_DEFINITIONS[metric_name]["format"]

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return _unavailable_value()

    if fmt == "percent":
        return f"{numeric_value * 100:.2f}%"

    return f"{numeric_value:.4f}"


def _get_model_records(dataset_key):
    records = []
    available_models = backend.get_available_models(dataset_key)

    if not isinstance(available_models, dict):
        return records

    for model_key, model_name in available_models.items():
        result = backend.get_model_metadata(dataset_key, model_key)

        if not isinstance(result, dict):
            continue

        if result.get("status") != "available":
            continue

        model_data = result.get("data")

        if not isinstance(model_data, dict):
            continue

        metrics = model_data.get("metrics")

        if not isinstance(metrics, dict):
            metrics = {}

        records.append(
            {
                "key": model_key,
                "name": model_name,
                "type": model_data.get("type"),
                "metadata": model_data,
                "metrics": metrics,
            }
        )

    return records


def _display_test_size(records):
    values = []

    for record in records:
        metrics = record["metrics"]
        model_data = record["metadata"]

        value = metrics.get("test_size")

        if value is None:
            value = model_data.get("test_size")

        if value is not None:
            values.append(str(value))

    if not values:
        return _unavailable_value()

    unique_values = list(dict.fromkeys(values))

    if len(unique_values) == 1:
        return unique_values[0]

    return "Model-specific"


def _display_protocol(records):
    values = []

    for record in records:
        model_data = record["metadata"]
        metrics = record["metrics"]

        protocol = model_data.get("evaluation_protocol")

        if protocol is None:
            protocol = model_data.get("protocol")

        if protocol is None:
            protocol = metrics.get("evaluation_protocol")

        if protocol is None:
            protocol = metrics.get("protocol")

        if protocol is not None:
            if isinstance(protocol, dict) and protocol:
                proto_str = " · ".join(f"{str(k).replace('_', ' ').title()}: {v}" for k, v in protocol.items())
                values.append(proto_str)
            else:
                values.append(str(protocol))

    if not values:
        return _unavailable_value()

    unique_values = list(dict.fromkeys(values))

    if len(unique_values) == 1:
        return unique_values[0]

    return "Model-specific"


def _display_preprocessing(model_data):
    preprocessing = model_data.get("preprocessing")

    if isinstance(preprocessing, dict):
        display_value = preprocessing.get("display")

        if display_value is not None:
            return str(display_value)

    if isinstance(preprocessing, str) and preprocessing.strip():
        return preprocessing

    return _unavailable_value()


def _display_qubits(model_data):
    quantum_config = model_data.get("quantum_config")

    if not isinstance(quantum_config, dict):
        return _unavailable_value()

    qubits = quantum_config.get("qubits")

    if qubits is None:
        return _unavailable_value()

    return str(qubits)


def _display_transformed_dimension(model_data):
    preprocessing = model_data.get("preprocessing")

    if not isinstance(preprocessing, dict):
        return _unavailable_value()

    candidate_keys = (
        "transformed_dimension",
        "transformed_dim",
        "output_dimension",
        "dimension",
    )

    for key in candidate_keys:
        value = preprocessing.get(key)

        if value is not None:
            return str(value)

    return _unavailable_value()


def _build_comparison_table(records):
    rows = []

    for record in records:
        metrics = record["metrics"]

        rows.append(
            {
                "Model": record["name"],
                "Type": record["type"] or "Unspecified",
                "Accuracy": _format_metric(
                    _metric_value(metrics, "Accuracy"), "Accuracy"
                ),
                "Precision": _format_metric(
                    _metric_value(metrics, "Precision"), "Precision"
                ),
                "Sensitivity": _format_metric(
                    _metric_value(metrics, "Sensitivity"), "Sensitivity"
                ),
                "Specificity": _format_metric(
                    _metric_value(metrics, "Specificity"), "Specificity"
                ),
                "F1 Score": _format_metric(
                    _metric_value(metrics, "F1 Score"), "F1 Score"
                ),
                "ROC-AUC": _format_metric(
                    _metric_value(metrics, "ROC-AUC"), "ROC-AUC"
                ),
                "MCC": _format_metric(
                    _metric_value(metrics, "MCC"), "MCC"
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_metric_chart(records, selected_models, selected_metric):
    rows = []

    for record in records:
        if record["key"] not in selected_models:
            continue

        value = _metric_value(record["metrics"], selected_metric)

        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "Model": record["name"],
                selected_metric: numeric_value,
            }
        )

    if not rows:
        return pd.DataFrame()

    chart_df = pd.DataFrame(rows).set_index("Model")

    if METRIC_DEFINITIONS[selected_metric]["format"] == "percent":
        chart_df[selected_metric] = chart_df[selected_metric] * 100

    return chart_df


def _render_evidence_state(records):
    st.markdown("### Evidence State")

    available_metric_names = []

    for metric_name in METRIC_DEFINITIONS:
        if any(
            _metric_value(record["metrics"], metric_name) is not None
            for record in records
        ):
            available_metric_names.append(metric_name)

    metric_state = (
        ", ".join(available_metric_names)
        if available_metric_names
        else "Unavailable"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Evaluation Metrics**")
        st.write(metric_state)

    with col2:
        st.markdown("**ROC Coordinates**")
        st.write("Unavailable")

    with col3:
        st.markdown("**Benchmark Artifact**")
        st.write("Unavailable")


def _render_experimental_context(records):
    st.markdown("### Experimental Context")

    context_rows = []

    for record in records:
        model_data = record["metadata"]

        context_rows.append(
            {
                "Model": record["name"],
                "Type": record["type"] or "Unspecified",
                "Preprocessing": _display_preprocessing(model_data),
                "Transformed Dimension": _display_transformed_dimension(
                    model_data
                ),
                "Qubits": _display_qubits(model_data),
            }
        )

    if context_rows:
        st.dataframe(
            pd.DataFrame(context_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No experimental-context metadata is available.")


def render():
    dataset_key = st.session_state.get("active_dataset")

    if not dataset_key:
        st.warning("Please select a dataset from the navigation rail.")
        return

    dataset_result = backend.get_dataset_metadata(dataset_key)

    if not isinstance(dataset_result, dict):
        st.error("Dataset metadata is unavailable.")
        return

    dataset_data = dataset_result.get("data")

    if not isinstance(dataset_data, dict):
        st.error("Dataset metadata is unavailable.")
        return

    dataset_name = dataset_data.get("name", dataset_key)

    st.markdown(f"## Model Benchmark: {dataset_name}")
    st.caption(
        "Measured evaluation results for models on the selected dataset. "
        "Values are presented from authoritative backend evaluation metadata "
        "without ranking or derived benchmark scores."
    )

    records = _get_model_records(dataset_key)

    if not records:
        st.info(
            "No authoritative evaluation results are available for this dataset."
        )
        return

    st.markdown("### Dataset Context")

    test_size = _display_test_size(records)
    protocol = _display_protocol(records)

    context_col_1, context_col_2 = st.columns(2)

    with context_col_1:
        st.markdown("**Test Size**")
        st.write(test_size)

    with context_col_2:
        st.markdown("**Evaluation Protocol**")
        st.write(protocol)

    st.markdown("### Measured Model Comparison")

    comparison_df = _build_comparison_table(records)

    if comparison_df.empty:
        st.info("No authoritative metric values are available.")
    else:
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Metric Selection")

    metric_options = list(METRIC_DEFINITIONS.keys())

    selected_metric = st.selectbox(
        "Metric",
        options=metric_options,
        index=0,
        key=f"benchmark_metric_{dataset_key}",
    )

    model_options = [record["key"] for record in records]
    model_labels = {
        record["key"]: record["name"]
        for record in records
    }

    selected_models = st.multiselect(
        "Models",
        options=model_options,
        default=model_options,
        format_func=lambda key: model_labels.get(key, key),
        key=f"benchmark_models_{dataset_key}",
    )

    st.markdown("### Selected Metric Visualization")

    chart_df = _build_metric_chart(
        records,
        selected_models,
        selected_metric,
    )

    if chart_df.empty:
        st.info(
            f"No authoritative {selected_metric} values are available "
            "for the selected models."
        )
    else:
        if METRIC_DEFINITIONS[selected_metric]["format"] == "percent":
            st.caption(
                f"{selected_metric} values shown as percentages from "
                "authoritative backend evaluation metadata."
            )
        else:
            st.caption(
                f"{selected_metric} values shown directly from "
                "authoritative backend evaluation metadata."
            )

        st.bar_chart(
            chart_df,
            use_container_width=True,
        )

    _render_experimental_context(records)
    _render_evidence_state(records)

    st.caption(
        "This research workspace presents measured evaluation evidence. "
        "It does not establish clinical validity, model superiority, or "
        "quantum advantage."
    )
