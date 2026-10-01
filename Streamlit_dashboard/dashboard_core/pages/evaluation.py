"""
evaluation.py
=============
Analytical Evaluation Workspace.

Displays measured offline evaluation metrics, authoritative confusion
matrices, scalar discrimination statistics, and verified evaluation
protocol information.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dashboard_core.backend_adapter as backend


def _display_value(value, suffix=""):
    """Return a safe display value without inventing missing evidence."""
    if value is None:
        return "Unavailable"

    if isinstance(value, str):
        value = value.strip()
        return value if value else "Unavailable"

    return f"{value}{suffix}"


def _format_metric(value, decimals=4, percentage=False):
    """Format an authoritative metric value for display."""
    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if percentage:
        return f"{numeric_value * 100:.2f}%"

    return f"{numeric_value:.{decimals}f}"


def render():
    dataset_key = st.session_state.get("active_dataset")
    selected_model = st.session_state.get("selected_model")

    if not dataset_key or not selected_model:
        st.warning(
            "Please select an active dataset and model from the navigation rail."
        )
        return

    ds_meta = backend.get_dataset_metadata(dataset_key)
    mod_meta = backend.get_model_metadata(dataset_key, selected_model)

    if not mod_meta or mod_meta.get("status") != "available":
        st.error("Model evaluation metadata is unavailable.")
        return

    m_info = mod_meta.get("data") or {}
    caps = backend.get_capabilities(dataset_key, selected_model) or {}

    evaluation_state = caps.get("evaluation")

    if evaluation_state != "AVAILABLE":
        if evaluation_state:
            st.info(
                f"Evaluation evidence is currently marked as "
                f"**{evaluation_state}** for this model."
            )
        else:
            st.info("Evaluation metrics are not available for this model.")
        return

    mets = m_info.get("metrics") or {}
    proto = m_info.get("protocol") or {}

    if not isinstance(mets, dict):
        mets = {}

    if not isinstance(proto, dict):
        proto = {}

    preprocessing = m_info.get("preprocessing")

    if isinstance(preprocessing, dict):
        preprocessing_display = preprocessing.get("display")
    else:
        preprocessing_display = None

    dataset_data = {}
    if isinstance(ds_meta, dict) and ds_meta.get("status") == "available":
        dataset_data = ds_meta.get("data") or {}

    dataset_name = dataset_data.get("name") or dataset_key
    model_name = m_info.get("name") or selected_model

    protocol_name = proto.get("split_strategy")

    # ------------------------------------------------------------------
    # 1. HEADER
    # ------------------------------------------------------------------
    st.markdown(f"## Evaluation: {model_name}")

    cohort_text = f"**Cohort:** {dataset_name}"

    if protocol_name:
        cohort_text += f" | **Protocol:** {protocol_name}"

    st.caption(cohort_text)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2. METRICS KPI STRIP
    # ------------------------------------------------------------------
    accuracy = mets.get("accuracy")
    precision = mets.get("precision")
    sensitivity = mets.get("sensitivity")
    specificity = mets.get("specificity")
    f1_score = mets.get("f1_score", mets.get("f1"))
    roc_auc = mets.get("roc_auc")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "ACCURACY",
            _format_metric(accuracy, percentage=True),
        )

    with c2:
        st.metric(
            "PRECISION",
            _format_metric(precision, percentage=True),
        )

    with c3:
        st.metric(
            "SENSITIVITY",
            _format_metric(sensitivity, percentage=True),
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        st.metric(
            "SPECIFICITY",
            _format_metric(specificity, percentage=True),
        )

    with c5:
        st.metric(
            "F1 SCORE",
            _format_metric(f1_score, percentage=True),
        )

    with c6:
        st.metric(
            "ROC-AUC",
            _format_metric(roc_auc, percentage=True),
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. CONFUSION MATRIX + SCALAR DISCRIMINATION METRICS
    # ------------------------------------------------------------------
    col_cm, col_summary = st.columns(2)

    with col_cm:
        st.markdown("### Confusion Matrix")
        st.caption(
            "Measured classification counts from the authoritative "
            "offline evaluation result."
        )

        cm_data = mets.get("confusion_matrix")

        if isinstance(cm_data, dict):
            matrix = cm_data.get("matrix")
            labels = cm_data.get("labels")

            if matrix is not None and labels is not None:
                try:
                    cm_array = np.asarray(matrix)

                    if cm_array.ndim != 2 or cm_array.size == 0:
                        raise ValueError("Invalid confusion matrix dimensions.")

                    if len(labels) != cm_array.shape[0]:
                        raise ValueError(
                            "Confusion matrix labels do not match matrix dimensions."
                        )

                    fig, ax = plt.subplots(figsize=(4.5, 3.8))
                    fig.patch.set_facecolor("#FFFFFF")
                    ax.set_facecolor("#FFFFFF")

                    sns.heatmap(
                        cm_array,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=False,
                        ax=ax,
                        xticklabels=labels,
                        yticklabels=labels,
                        annot_kws={
                            "size": 14,
                            "weight": "bold",
                        },
                    )

                    ax.set_xlabel(
                        "Predicted Cohort Class",
                        fontsize=11,
                        fontweight="bold",
                        color="#1E293B",
                    )
                    ax.set_ylabel(
                        "True Cohort Label",
                        fontsize=11,
                        fontweight="bold",
                        color="#1E293B",
                    )
                    ax.tick_params(
                        colors="#475569",
                        labelsize=10,
                    )

                    plt.tight_layout()

                    with st.container(border=True):
                        st.pyplot(fig)
                        plt.close(fig)

                    test_size = proto.get("test_size")

                    if test_size is not None:
                        st.caption(
                            f"Test samples: **{_display_value(test_size)}**"
                        )

                except (TypeError, ValueError):
                    st.info("Confusion matrix data is unavailable or invalid.")
            else:
                st.info("Confusion matrix data unavailable.")
        else:
            st.info("Confusion matrix data unavailable.")

    with col_summary:
        st.markdown("### Discrimination Metrics")
        st.caption(
            "Additional scalar evaluation evidence available for this model."
        )

        mcc = mets.get("mcc")
        training_time = mets.get("training_time_sec")

        with st.container(border=True):
            metric_col_1, metric_col_2 = st.columns(2)

            with metric_col_1:
                st.metric(
                    "ROC-AUC",
                    _format_metric(roc_auc, percentage=True),
                )

            with metric_col_2:
                st.metric(
                    "MCC",
                    _format_metric(mcc),
                )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            st.markdown("**ROC Coordinate Evidence**")
            st.caption(
                "Dataset-specific ROC coordinate data is not currently "
                "available from the authoritative backend."
            )

            st.markdown("**Threshold Analysis**")
            st.caption(
                "Authoritative threshold-analysis data is not currently "
                "available from the evaluation backend."
            )

            if training_time is not None:
                st.markdown("**Recorded Training Time**")
                st.caption(
                    f"{_format_metric(training_time, decimals=4)} seconds"
                )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 4. EVALUATION PROTOCOL & PROVENANCE
    # ------------------------------------------------------------------
    st.markdown("### Evaluation Protocol & Provenance")

    with st.container(border=True):
        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            split_strategy = proto.get("split_strategy")
            test_size = proto.get("test_size")

            st.markdown(
                f"**Split Strategy:** "
                f"`{_display_value(split_strategy)}`"
            )
            st.markdown(
                f"**Test Set Size:** "
                f"`{_display_value(test_size)}"
            )

        with col_p2:
            seed = proto.get("seed")

            st.markdown(
                f"**Reproducibility Seed:** "
                f"`{_display_value(seed)}`"
            )

            if training_time is not None:
                training_time_text = (
                    f"{_format_metric(training_time, decimals=4)}s"
                )
            else:
                training_time_text = "Unavailable"

            st.markdown(
                f"**Training Time:** `{training_time_text}`"
            )

        with col_p3:
            target_labels = dataset_data.get("target_labels")

            if isinstance(target_labels, (list, tuple)) and target_labels:
                target_text = ", ".join(str(label) for label in target_labels)
            else:
                target_text = "Unavailable"

            st.markdown(
                f"**Target Classes:** `{target_text}`"
            )

            st.markdown(
                f"**Preprocessing:** "
                f"`{_display_value(preprocessing_display)}`"
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.caption(
        "Evaluation metrics are empirical measurements from the "
        "authoritative offline evaluation results. No synthetic data, "
        "replacement metrics, or inferred evaluation values are generated "
        "by this workspace."
    )