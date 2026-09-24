"""
prediction.py
=============
Interactive Prediction Workspace.
Supports real-time inference on inference-ready models.
Provides capability-aware evaluation states for models without live inference.
"""

import streamlit as st
import dashboard_core.backend_adapter as backend
from dashboard_core import ui_components


def render():
    dataset_key = st.session_state.get("active_dataset")
    selected_model = st.session_state.get("selected_model")

    if not dataset_key or not selected_model:
        st.warning("Please select an active dataset and model from the navigation rail.")
        return

    ds_meta = backend.get_dataset_metadata(dataset_key)
    mod_meta = backend.get_model_metadata(dataset_key, selected_model)

    if mod_meta.get("status") != "available":
        st.error("Model metadata unavailable.")
        return

    m_info = mod_meta.get("data") or {}
    caps = backend.get_capabilities(dataset_key, selected_model)

    preprocessing = m_info.get("preprocessing")
    if preprocessing is None:
        pipeline_display = "Unspecified"
    elif isinstance(preprocessing, dict):
        pipeline_display = preprocessing.get("display") or "Unspecified"
    else:
        pipeline_display = "Unspecified"

    model_name = m_info.get("name") or selected_model
    dataset_name = ds_meta.get("data", {}).get("name", dataset_key)

    st.markdown(f"## Model Prediction: {model_name}")
    st.caption(
        f"**Cohort:** {dataset_name} | "
        f"**Execution Pipeline:** {pipeline_display}"
    )

    # ------------------------------------------------------------------
    # CASE 1: LIVE INFERENCE NOT AVAILABLE
    # ------------------------------------------------------------------
    if caps.get("prediction") != "AVAILABLE":
        with st.container(border=True):
            st.markdown("### Live Inference Unavailable")

            offline_reason = m_info.get("offline_reason")
            if offline_reason:
                st.markdown(f"**Backend Status:** {offline_reason}")
            else:
                st.markdown(
                    "The current backend capability registry does not authorize "
                    "interactive prediction for this model."
                )

            st.divider()
            st.markdown("#### Evaluation Evidence")

            mets = m_info.get("metrics") or {}

            if mets:
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    acc = mets.get("accuracy")
                    st.metric(
                        "MEASURED ACCURACY",
                        f"{acc * 100:.2f}%" if acc is not None else "N/A",
                    )

                with c2:
                    sens = mets.get("sensitivity")
                    st.metric(
                        "SENSITIVITY",
                        f"{sens * 100:.2f}%" if sens is not None else "N/A",
                    )

                with c3:
                    spec = mets.get("specificity")
                    st.metric(
                        "SPECIFICITY",
                        f"{spec * 100:.2f}%" if spec is not None else "N/A",
                    )

                with c4:
                    f1 = mets.get("f1_score", mets.get("f1"))
                    st.metric(
                        "F1 SCORE",
                        f"{f1 * 100:.2f}%" if f1 is not None else "N/A",
                    )
            else:
                st.info("No recorded evaluation metrics are available.")

            artifacts = m_info.get("artifacts") or {}

            if artifacts:
                st.markdown("#### Stored Backend Artifacts")
                for artifact_name, artifact_value in artifacts.items():
                    display_name = artifact_name.replace("_", " ").title()
                    st.markdown(f"**{display_name}**")

                    if isinstance(artifact_value, dict):
                        if artifact_value:
                            for field_name, field_value in artifact_value.items():
                                field_label = str(field_name).replace("_", " ").title()
                                if field_value is None:
                                    display_value = "Unavailable"
                                elif isinstance(field_value, (dict, list, tuple)):
                                    st.markdown(f"**{field_label}**")
                                    st.json(field_value)
                                    continue
                                else:
                                    display_value = str(field_value)

                                ui_components.information_row(
                                    field_label,
                                    display_value,
                                )
                        else:
                            st.caption("No artifact metadata fields are available.")
                    elif artifact_value is None:
                        st.caption("Unavailable")
                    else:
                        st.caption(str(artifact_value))

        st.caption(
            "Research prototype: evaluation evidence is separate from live "
            "interactive inference."
        )
        return

    # ------------------------------------------------------------------
    # CASE 2: LIVE INFERENCE
    # ------------------------------------------------------------------
    schema_res = backend.get_prediction_schema(dataset_key, selected_model)

    if schema_res.get("status") != "available":
        st.error(
            schema_res.get(
                "message",
                "Prediction schema is unavailable. Cannot render the prediction form.",
            )
        )
        return

    schema_data = schema_res.get("data", schema_res)
    features = schema_data.get("feature_names") or []

    if not features:
        st.error("Prediction schema is empty. Cannot render the prediction form.")
        return

    # Model/dataset-specific state prevents widget-state collisions when
    # switching between datasets or models.
    preset_key = f"prediction_preset_{dataset_key}_{selected_model}"
    version_key = f"prediction_input_version_{dataset_key}_{selected_model}"

    if preset_key not in st.session_state:
        st.session_state[preset_key] = {}

    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    input_version = st.session_state[version_key]

    # ------------------------------------------------------------------
    # TWO-PANEL PREDICTION WORKSPACE
    # ------------------------------------------------------------------
    col_input, col_result = st.columns([3, 2])

    with col_input:
        st.markdown("### Input Feature Vector")
        st.caption(
            "Enter values for the authoritative model feature schema or "
            "load a representative sample from the selected cohort."
        )

        col_sample, col_reset = st.columns(2)

        with col_sample:
            load_sample = st.button(
                "Load Sample (From Cohort)",
                use_container_width=True,
            )

        with col_reset:
            reset_inputs = st.button(
                "Reset Inputs",
                use_container_width=True,
            )

        # --------------------------------------------------------------
        # SAMPLE LOADING
        # --------------------------------------------------------------
        if load_sample:
            sample_res = backend.get_data_sample(dataset_key, n=1)

            if (
                sample_res.get("status") == "available"
                and "data" in sample_res
            ):
                sample_df = sample_res["data"]

                if sample_df is None or sample_df.empty:
                    st.warning("The selected cohort sample is empty.")
                else:
                    row = sample_df.iloc[0]

                    missing_features = [
                        feature
                        for feature in features
                        if feature not in row.index
                    ]

                    if missing_features:
                        st.warning(
                            "The selected cohort sample cannot populate the "
                            "complete prediction schema. Existing input values "
                            "were left unchanged."
                        )
                    else:
                        st.session_state[preset_key] = {
                            feature: float(row[feature])
                            for feature in features
                        }

                        # Force a new widget generation so the newly loaded
                        # preset values are reflected by Streamlit widgets.
                        st.session_state[version_key] += 1
                        st.rerun()
            else:
                st.warning(
                    sample_res.get(
                        "message",
                        "The selected cohort sample is unavailable.",
                    )
                )

        # --------------------------------------------------------------
        # RESET
        # --------------------------------------------------------------
        if reset_inputs:
            st.session_state[preset_key] = {}
            st.session_state[version_key] += 1
            st.rerun()

        input_version = st.session_state[version_key]
        presets = st.session_state.get(preset_key, {})

        input_data = {}

        with st.form(
            f"interactive_prediction_form_{dataset_key}_{selected_model}_{input_version}"
        ):
            form_cols = st.columns(4)

            for index, feature_name in enumerate(features):
                with form_cols[index % 4]:
                    widget_key = (
                        f"prediction_input_"
                        f"{dataset_key}_"
                        f"{selected_model}_"
                        f"{input_version}_"
                        f"{feature_name}"
                    )

                    default_value = presets.get(feature_name, 0.0)

                    input_data[feature_name] = st.number_input(
                        feature_name,
                        value=float(default_value),
                        format="%.4f",
                        key=widget_key,
                    )

            submit_btn = st.form_submit_button(
                "Run Model Prediction",
                type="primary",
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # RESULT PANEL
    # ------------------------------------------------------------------
    with col_result:
        st.markdown("### Model Output")
        st.caption(
            "Prediction is executed through the authoritative backend inference service."
        )

        if submit_btn:
            with st.spinner("Executing inference pipeline..."):
                res = backend.predict_instance(
                    dataset_key,
                    selected_model,
                    input_data,
                )

            if res.get("status") == "success":
                with st.container(border=True):
                    predicted_label = res.get("predicted_label")

                    if predicted_label is not None:
                        st.markdown("#### Predicted Label")
                        st.subheader(str(predicted_label))
                    else:
                        st.info("The backend returned no predicted label.")

                    confidence_score = res.get("confidence_score")
                    score_label = res.get("score_label")

                    if confidence_score is not None:
                        display_score_label = score_label or "Confidence Score"
                        st.metric(
                            display_score_label,
                            f"{float(confidence_score):.4f}",
                        )

                    class_1_probability = res.get("class_1_probability")

                    if class_1_probability is not None:
                        st.markdown(
                            f"**Class 1 Probability:** "
                            f"`{float(class_1_probability):.4f}`"
                        )

                    threshold = res.get("threshold")

                    if threshold is not None:
                        st.markdown(
                            f"**Classification Threshold:** `{threshold}`"
                        )

                    execution_mode = res.get("execution_mode")

                    if execution_mode is not None:
                        st.markdown(
                            f"**Execution Mode:** `{execution_mode}`"
                        )

                    predicted_class = res.get("predicted_class")

                    if predicted_class is not None:
                        st.markdown(
                            f"**Predicted Class:** `{predicted_class}`"
                        )

            else:
                error_message = res.get("message")

                if error_message:
                    st.error(f"Inference execution failed: {error_message}")
                else:
                    st.error(
                        "Inference execution failed without an authoritative backend message."
                    )

        else:
            with st.container(border=True):
                st.markdown("### Ready for Inference")
                st.caption(
                    "Load a cohort sample or enter the required feature values, "
                    "then run the model prediction."
                )

    st.caption(
        "Research Prototype: model predictions are experimental outputs and "
        "do not constitute clinical diagnoses."
    )