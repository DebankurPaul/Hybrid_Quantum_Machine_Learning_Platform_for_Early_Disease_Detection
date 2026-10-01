"""
data_preview.py
===============
Data Explorer workspace.

Presents only dataset information that is exposed through the authoritative
dashboard backend contract. Raw sample data is explicitly distinguished from
model preprocessing configuration.
"""

import pandas as pd
import streamlit as st

import dashboard_core.backend_adapter as backend
from dashboard_core import ui_components


def _format_ratio(class_ratio):
    """Format an authoritative class-ratio mapping for compact display."""
    if not isinstance(class_ratio, dict) or not class_ratio:
        return "Unavailable"

    parts = []
    for label, value in class_ratio.items():
        if isinstance(value, (int, float)):
            parts.append(f"{label}: {value * 100:.1f}%")
        else:
            parts.append(f"{label}: {value}")
    return " · ".join(parts)


def _render_dataset_overview(d_info):
    """Render dataset-level metrics exposed by the backend."""
    ui_components.section_header(
        "Dataset Overview",
        "Authoritative characteristics reported by the dataset service.",
    )

    sample_count = d_info.get("sample_count")
    feature_count = d_info.get("feature_count")
    class_count = d_info.get("class_count")
    class_ratio = d_info.get("class_ratio")

    metrics = [
        (
            "Samples",
            f"{sample_count:,}" if isinstance(sample_count, int) else "Unavailable",
        ),
        (
            "Input Features",
            str(feature_count) if feature_count is not None else "Unavailable",
        ),
        (
            "Target Classes",
            str(class_count) if class_count is not None else "Unavailable",
        ),
        ("Class Balance", _format_ratio(class_ratio)),
    ]

    columns = st.columns(4)
    for column, (label, value) in zip(columns, metrics):
        with column:
            ui_components.metric_display(label, value)


def _render_class_distribution(d_info):
    """Render the backend-derived target-class distribution."""
    ui_components.section_header(
        "Class Distribution",
        "Sample counts and proportions reported by the dataset service.",
    )

    classes = d_info.get("classes")
    class_ratio = d_info.get("class_ratio")

    if not isinstance(classes, dict) or not classes:
        ui_components.empty_state(
            "Class distribution unavailable",
            "The current backend does not expose class counts for this dataset.",
        )
        return

    class_df = pd.DataFrame(
        {
            "Class Label": list(classes.keys()),
            "Sample Count": list(classes.values()),
        }
    )

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
    st.bar_chart(class_df.set_index("Class Label"))

    if isinstance(class_ratio, dict) and class_ratio:
        ratio_df = pd.DataFrame(
            [
                {
                    "Class Label": label,
                    "Proportion": value,
                }
                for label, value in class_ratio.items()
                if isinstance(value, (int, float))
            ]
        )
        if not ratio_df.empty:
            st.dataframe(
                ratio_df,
                use_container_width=True,
                hide_index=True,
            )


def _render_raw_sample(dataset_key, d_info):
    """Render the raw sample payload exposed by the backend."""
    ui_components.section_header(
        "Raw Dataset Sample",
        "A small preview of raw records returned by the dataset service. "
        "This is not a transformed or normalized feature representation.",
    )

    sample_res = backend.get_data_sample(dataset_key, n=6)

    if sample_res.get("status") != "available":
        ui_components.empty_state(
            "Raw sample unavailable",
            "The current data-service interface did not return a sample for this dataset.",
        )
        return

    sample = sample_res.get("data")

    if not isinstance(sample, pd.DataFrame):
        ui_components.empty_state(
            "Raw sample unavailable",
            "The backend did not return the expected tabular sample payload.",
        )
        return

    schema_note = sample_res.get("schema_note")
    if schema_note:
        ui_components.capability_state(
            "Sample schema note",
            "AVAILABLE",
            str(schema_note),
        )

    # Golub's current sample interface intentionally exposes labels/metadata
    # rather than the 7,129-dimensional expression matrix.
    if dataset_key == "golub":
        ui_components.capability_state(
            "Gene-expression feature values",
            "UNAVAILABLE",
            "The current backend sample interface exposes sample labels/metadata "
            "but does not expose the 7,129-dimensional expression matrix.",
        )

    st.dataframe(
        sample,
        use_container_width=True,
        hide_index=True,
    )


def _render_feature_schema(d_info):
    """Render only authoritative semantic feature names."""
    ui_components.section_header(
        "Feature Schema",
        "Semantic feature names are shown only when exposed by the backend schema.",
    )

    feature_names = d_info.get("feature_names")

    if isinstance(feature_names, list) and feature_names:
        ui_components.capability_state(
            "Semantic feature names",
            "AVAILABLE",
            f"{len(feature_names)} authoritative input feature names are exposed.",
        )

        schema_df = pd.DataFrame(
            {
                "Index": range(1, len(feature_names) + 1),
                "Feature Name": feature_names,
            }
        )

        st.dataframe(
            schema_df,
            use_container_width=True,
            hide_index=True,
        )
        return

    ui_components.capability_state(
        "Semantic feature names",
        "UNAVAILABLE",
        "The current backend schema does not expose semantic feature names "
        "for this dataset. No names are inferred or synthesized.",
    )


def _render_preprocessing(dataset_key):
    """Render model preprocessing configurations exposed by the registry."""
    ui_components.section_header(
        "Model Preprocessing Configuration",
        "Backend-defined preprocessing configurations associated with the "
        "available models. These are configurations, not transformed sample values.",
    )

    available_models = backend.get_available_models(dataset_key)

    if not isinstance(available_models, dict) or not available_models:
        ui_components.empty_state(
            "Preprocessing configuration unavailable",
            "No model preprocessing configurations are currently exposed "
            "for this dataset.",
        )
        return

    rendered = False

    for model_key, model_name in available_models.items():
        model_meta = backend.get_model_metadata(dataset_key, model_key)

        if not isinstance(model_meta, dict):
            continue

        if model_meta.get("status") != "available":
            continue

        model_data = model_meta.get("data", {})
        preprocessing = model_data.get("preprocessing")

        if not isinstance(preprocessing, dict):
            continue

        display_text = preprocessing.get("display")
        steps = preprocessing.get("steps")

        if not display_text and not steps:
            continue

        rendered = True

        model_type = model_data.get("type")
        state = "AVAILABLE"

        st.markdown(
            f"**{model_name}**"
            + (f" · {model_type}" if model_type else "")
        )

        if display_text:
            st.caption(str(display_text))

        if isinstance(steps, list) and steps:
            with st.expander("Configuration details"):
                st.json(steps)

        st.divider()

    if not rendered:
        ui_components.empty_state(
            "Preprocessing configuration unavailable",
            "The current model registry does not expose a usable preprocessing "
            "configuration for the available models.",
        )


def _render_provenance(d_info):
    """Render provenance fields exposed by the dataset metadata contract."""
    ui_components.section_header(
        "Dataset Provenance",
        "Source information exposed by the authoritative dataset registry.",
    )

    source = d_info.get("schema_source")
    cohort_source = d_info.get("cohort_source")

    if not source and not cohort_source:
        ui_components.empty_state(
            "Provenance unavailable",
            "The current backend does not expose dataset source information.",
        )
        return

    if source:
        st.markdown("**Dataset source**")
        st.code(str(source), language="text")

    if cohort_source:
        st.markdown("**Source description**")
        st.caption(str(cohort_source))


def render():
    dataset_key = st.session_state.get("active_dataset")

    if not dataset_key:
        ui_components.empty_state(
            "No dataset selected",
            "Select a dataset from the navigation rail to open the Data Explorer.",
        )
        return

    ds_meta = backend.get_dataset_metadata(dataset_key)

    if not isinstance(ds_meta, dict) or ds_meta.get("status") != "available":
        ui_components.empty_state(
            "Dataset metadata unavailable",
            "The current backend did not return authoritative metadata for the selected dataset.",
        )
        return

    d_info = ds_meta.get("data")

    if not isinstance(d_info, dict):
        ui_components.empty_state(
            "Dataset metadata unavailable",
            "The backend returned an invalid dataset metadata payload.",
        )
        return

    dataset_name = d_info.get("name") or dataset_key

    ui_components.page_header(
        "Data Explorer",
        f"Authoritative dataset characteristics and raw sample access for {dataset_name}.",
    )

    _render_dataset_overview(d_info)

    overview_left, overview_right = st.columns([1, 1])

    with overview_left:
        _render_class_distribution(d_info)

    with overview_right:
        _render_provenance(d_info)

    _render_raw_sample(dataset_key, d_info)
    _render_feature_schema(d_info)
    _render_preprocessing(dataset_key)
