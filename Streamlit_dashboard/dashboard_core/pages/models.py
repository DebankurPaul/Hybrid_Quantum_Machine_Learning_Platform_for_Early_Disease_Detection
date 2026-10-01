"""
models.py
=========
Models Landscape Workspace.
Presents genuine classical and quantum model architectures available for the active dataset.
Separates implementation, trained artifact, evaluation, and inference-readiness states.
"""

import streamlit as st
import dashboard_core.backend_adapter as backend
import dashboard_core.ui_components as ui_components

def render():
    dataset_key = st.session_state.get("active_dataset")
    if not dataset_key:
        ui_components.empty_state("No Dataset Selected", "Please select a dataset from the navigation rail to view models.")
        return

    ds_meta = backend.get_dataset_metadata(dataset_key)
    dataset_name = ds_meta.get('data', {}).get('name', dataset_key)
    available_models = backend.get_available_models(dataset_key)
    selected_model = st.session_state.get("selected_model")

    # If the selected model doesn't belong to the active dataset, clear it
    if selected_model and selected_model not in available_models:
        selected_model = None
        st.session_state["selected_model"] = None

    ui_components.page_header(f"Model Landscape: {dataset_name}", "Inspect and select active models. Only genuine backend models for this dataset are displayed.")

    # Separate into Classical and Quantum
    classical_models = {}
    quantum_models = {}

    for mk, mname in available_models.items():
        m_meta_res = backend.get_model_metadata(dataset_key, mk)
        if m_meta_res.get("status") != "available":
            continue
        m_meta = m_meta_res.get("data", {})
        if m_meta.get("type") == "Quantum":
            quantum_models[mk] = (mname, m_meta)
        else:
            classical_models[mk] = (mname, m_meta)

    # Helper to render a card
    def render_model_card(mk, mname, m_meta, is_quantum=False):
        m_caps = backend.get_capabilities(dataset_key, mk)
        is_active = (mk == selected_model)

        with st.container(border=not bool(selected_model)):
            if is_active:
                ui_components.status_indicator("active", "Active Selection")
            else:
                ui_components.status_indicator("quantum" if is_quantum else "classical", "Quantum" if is_quantum else "Classical")

            st.markdown(f"#### {mname}")

            # Key Performance Metric
            mets = m_meta.get("metrics", {})
            acc = mets.get("accuracy")
            f1 = mets.get("f1_score", mets.get("f1"))

            if acc is not None:
                ui_components.metric_display("TEST ACCURACY", f"{acc*100:.2f}%")
            elif f1 is not None:
                ui_components.metric_display("F1 SCORE", f"{f1*100:.2f}%")
            else:
                st.caption("No offline metrics")

            st.markdown("---")

            # Capability checklist - STRICT string checking
            ui_components.capability_state("Evaluation", m_caps.get("evaluation", "UNAVAILABLE"))
            ui_components.capability_state("Inference", m_caps.get("prediction", "UNAVAILABLE"))

            if is_quantum:
                ui_components.capability_state("Quantum Config", m_caps.get("quantum_configuration", "UNAVAILABLE"))

            if is_active:
                st.button("Currently Inspecting", key=f"btn_act_{mk}", disabled=True, type="primary", use_container_width=True)
            else:
                if st.button("Inspect Model", key=f"btn_sel_{mk}", use_container_width=True):
                    st.session_state["selected_model"] = mk
                    st.rerun()

    # RENDER LANDSCAPE
    # If a model is selected, we place the landscape in an expander so it doesn't crowd the detail view.
    if selected_model:
        landscape_container = st.expander("Show Available Models", expanded=False)
    else:
        landscape_container = st.container()

    with landscape_container:
        if classical_models:
            ui_components.section_header("Classical Baseline Models", "Standard machine learning architectures.")
            # Create a 3-column layout for the cards, wrapping automatically
            cols = st.columns(3)
            for idx, (mk, (mname, m_meta)) in enumerate(classical_models.items()):
                with cols[idx % 3]:
                    render_model_card(mk, mname, m_meta, is_quantum=False)

        if quantum_models:
            ui_components.section_header("Hybrid Quantum Architectures", "Variational and kernel-based quantum models.")
            cols = st.columns(3)
            for idx, (mk, (mname, m_meta)) in enumerate(quantum_models.items()):
                with cols[idx % 3]:
                    render_model_card(mk, mname, m_meta, is_quantum=True)


    # RENDER DETAIL VIEW
    if selected_model:
        m_name = available_models.get(selected_model)
        m_meta_res = backend.get_model_metadata(dataset_key, selected_model)
        m_meta = m_meta_res.get("data", {})
        m_caps = backend.get_capabilities(dataset_key, selected_model)
        is_quantum = m_meta.get("type") == "Quantum"

        ui_components.section_header(f"Detailed Inspection: {m_name}")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            # 1. MODEL IDENTITY
            with st.expander("Model Identity", expanded=True):
                ui_components.information_row("Dataset", dataset_name)
                ui_components.information_row("Model Name", m_name)
                ui_components.information_row("Model Key", selected_model)
                if "type" in m_meta:
                    ui_components.information_row("Model Type", m_meta["type"])
                if "protocol" in m_meta:
                    proto = m_meta["protocol"]
                    if isinstance(proto, dict) and proto:
                        proto_str = " · ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in proto.items())
                        ui_components.information_row("Protocol", proto_str)
                    elif proto:
                        ui_components.information_row("Protocol", str(proto))

            # 2. EVALUATION
            with st.expander("Evaluation & Metrics", expanded=True):
                if m_caps.get("evaluation", "UNAVAILABLE") == "AVAILABLE":
                    mets = m_meta.get("metrics", {})

                    # Group metrics nicely
                    metric_cols = st.columns(2)
                    idx = 0
                    for key, val in mets.items():
                        if key == "confusion_matrix" or key == "n_test":
                            continue
                        with metric_cols[idx % 2]:
                            if isinstance(val, float):
                                if key.lower() in ["accuracy", "precision", "sensitivity", "specificity", "f1_score", "f1", "roc_auc"]:
                                    ui_components.metric_display(key.upper(), f"{val*100:.2f}%")
                                else:
                                    ui_components.metric_display(key.upper(), f"{val:.4f}")
                            else:
                                ui_components.metric_display(key.upper(), str(val))
                        idx += 1

                    if "confusion_matrix" in mets:
                        st.markdown("**Confusion Matrix**")
                        cm = mets["confusion_matrix"]
                        if isinstance(cm, dict) and "matrix" in cm and "labels" in cm:
                            import pandas as pd
                            df_cm = pd.DataFrame(cm["matrix"], index=[f"True {l}" for l in cm["labels"]], columns=[f"Pred {l}" for l in cm["labels"]])
                            st.dataframe(df_cm, use_container_width=True)
                        else:
                            st.write(cm)
                else:
                    ui_components.empty_state("Evaluation Unavailable", "This model has not been evaluated offline.")

            # 3. PREPROCESSING
            with st.expander("Preprocessing Details", expanded=False):
                prep = m_meta.get("preprocessing")
                if prep:
                    if "display" in prep:
                        ui_components.information_row("Pipeline", prep["display"])
                    steps = prep.get("steps", [])
                    for i, step in enumerate(steps):
                        if "type" in step:
                            st.markdown(f"**Step {i+1}: {step['type']}**")
                        params = step.get("parameters", {})
                        for pk, pv in params.items():
                            ui_components.information_row(f" - {pk}", str(pv))
                else:
                    ui_components.empty_state("Preprocessing Unavailable", "No preprocessing configuration is exposed for this model.")

        with col_right:
            # 4. QUANTUM CONFIGURATION & CIRCUIT
            if is_quantum:
                with st.expander("Quantum Configuration", expanded=True):
                    q_config = m_meta.get("quantum_config")
                    if q_config:
                        for qk, qv in q_config.items():
                            if isinstance(qv, dict):
                                st.markdown(f"**{qk.title()}**")
                                for sub_k, sub_v in qv.items():
                                    ui_components.information_row(f" - {sub_k}", str(sub_v))
                            else:
                                ui_components.information_row(qk.title(), str(qv))
                    else:
                        ui_components.empty_state("Configuration Unavailable", "Quantum configuration details are not exposed.")

                with st.expander("Quantum Circuit", expanded=False):
                    q_circ_res = backend.get_quantum_circuit(dataset_key, selected_model)
                    if q_circ_res.get("status") == "available":
                        ui_components.information_row("Circuit Title", q_circ_res.get("title", ""))
                        ui_components.information_row("Qubits", str(q_circ_res.get("qubits", "")))
                        ui_components.information_row("Parameters", str(q_circ_res.get("parameters", "")))
                        ui_components.information_row("Depth", str(q_circ_res.get("depth", "")))
                        st.markdown("**Circuit Representation**")
                        st.code(q_circ_res.get("circuit_str", ""), language="text")
                    else:
                        ui_components.empty_state("Circuit Unavailable", q_circ_res.get("message", "Quantum circuit could not be generated."))

            # 5. EXPLAINABILITY
            with st.expander("Explainability", expanded=is_quantum):
                if m_caps.get("explainability", "UNAVAILABLE") == "AVAILABLE":
                    artifacts = m_meta.get("artifacts", {})
                    xai_path = artifacts.get("explainability")
                    if xai_path:
                        ui_components.information_row("Artifact Path", xai_path)
                        if str(xai_path).endswith(('.png', '.jpg', '.jpeg')):
                            import os
                            if os.path.exists(xai_path):
                                st.image(xai_path, caption="Explainability Summary", use_container_width=True)
                            else:
                                st.warning("Explainability artifact image file not found on disk.")
                    else:
                        ui_components.empty_state("Explainability Missing", "Backend claims availability but no artifact path was provided.")
                else:
                    ui_components.empty_state("Explainability Unavailable", "This model does not expose explainability artifacts.")

            # 6. PROVENANCE
            with st.expander("Provenance & Artifacts", expanded=False):
                artifacts = m_meta.get("artifacts", {})
                if artifacts:
                    for ak, av in artifacts.items():
                        if av:
                            ui_components.information_row(ak.title(), str(av))
                        else:
                            ui_components.information_row(ak.title(), "Not present")
                else:
                    ui_components.empty_state("Provenance Unavailable", "No artifact paths exposed.")
