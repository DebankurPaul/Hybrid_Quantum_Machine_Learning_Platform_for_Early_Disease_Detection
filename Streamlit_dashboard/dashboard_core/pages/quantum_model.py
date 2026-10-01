"""
quantum_model.py
================
Quantum Architecture Workspace.

Displays authoritative, backend-driven quantum configuration and
backend-generated circuit evidence without fabricating architecture values.
"""

import streamlit as st

import dashboard_core.backend_adapter as backend
import dashboard_core.ui_components as ui_components


def render():
    dataset_key = st.session_state.get("active_dataset")
    selected_model = st.session_state.get("selected_model")

    if not dataset_key or not selected_model:
        ui_components.empty_state(
            "No Model Selected",
            "Please select a dataset and a model from the navigation rail to view architecture details.",
        )
        return

    ds_meta_res = backend.get_dataset_metadata(dataset_key)
    mod_meta_res = backend.get_model_metadata(dataset_key, selected_model)

    if not isinstance(mod_meta_res, dict) or mod_meta_res.get("status") != "available":
        ui_components.error_state(
            "Metadata Unavailable",
            "Could not retrieve authoritative model metadata.",
        )
        return

    ds_meta = ds_meta_res.get("data")
    m_info = mod_meta_res.get("data")

    if not isinstance(m_info, dict):
        ui_components.error_state(
            "Metadata Unavailable",
            "The authoritative model metadata payload is unavailable or malformed.",
        )
        return

    if not isinstance(ds_meta, dict):
        ui_components.render_unavailable_state(
            "Dataset Context",
            "Authoritative dataset metadata is unavailable for the active dataset.",
        )
        return

    ds_name = ds_meta.get("name")
    m_name = m_info.get("name")

    # The selected dataset/model keys are authoritative application state.
    # They are used only as identifiers when optional display names are absent.
    dataset_label = ds_name if ds_name else str(dataset_key)
    model_label = m_name if m_name else str(selected_model)

    is_quantum = m_info.get("type") == "Quantum"

    ui_components.page_header(
        "Quantum Architecture Workspace",
        f"Verifiable quantum configuration and circuit reconstruction for "
        f"**{model_label}** ({dataset_label}).",
        status="active",
    )

    if not is_quantum:
        ui_components.empty_state(
            "Quantum Architecture Not Applicable",
            f"The selected model ({model_label}) is a classical machine learning "
            "model and does not expose a quantum architecture.",
        )
        return

    # -------------------------------------------------------------------------
    # AUTHORITATIVE STORED CONFIGURATION
    # -------------------------------------------------------------------------
    q_cfg = m_info.get("quantum_config")

    if not isinstance(q_cfg, dict) or not q_cfg:
        ui_components.render_unavailable_state(
            "Quantum Configuration",
            "No authoritative quantum configuration was exposed for this model.",
        )
        return

    prep_info = m_info.get("preprocessing")
    if isinstance(prep_info, dict):
        prep_steps = prep_info.get("steps")
        if not isinstance(prep_steps, list):
            prep_steps = []
    else:
        prep_steps = []

    qubits = q_cfg.get("qubits")

    feature_map_cfg = q_cfg.get("feature_map")
    if not isinstance(feature_map_cfg, dict):
        feature_map_cfg = {}

    fm_name = feature_map_cfg.get("name")
    fm_reps = feature_map_cfg.get("reps")
    fm_entanglement = feature_map_cfg.get("entanglement")

    ansatz_cfg = q_cfg.get("ansatz")
    has_ansatz = isinstance(ansatz_cfg, dict) and bool(ansatz_cfg)
    if not has_ansatz:
        ansatz_cfg = {}

    ans_name = ansatz_cfg.get("name")
    ans_reps = ansatz_cfg.get("reps")

    backend_metadata = q_cfg.get("backend")

    # -------------------------------------------------------------------------
    # DERIVED CIRCUIT EVIDENCE
    # -------------------------------------------------------------------------
    circ_res = backend.get_quantum_circuit(dataset_key, selected_model)
    if not isinstance(circ_res, dict):
        circ_res = {
            "status": "error",
            "message": "The backend returned an invalid circuit payload.",
        }

    has_circuit = circ_res.get("status") == "available"
    circuit_string = circ_res.get("circuit_str") if has_circuit else None
    has_circuit_text = isinstance(circuit_string, str) and bool(circuit_string.strip())

    if has_circuit and not has_circuit_text:
        has_circuit = False

    derived_depth = circ_res.get("depth") if has_circuit else None
    derived_params = circ_res.get("parameters") if has_circuit else None

    # -------------------------------------------------------------------------
    # 1. ARCHITECTURE SUMMARY
    # -------------------------------------------------------------------------
    ui_components.section_header("Architecture Summary")

    col_cfg, col_derived = st.columns(2)

    with col_cfg:
        with st.container(border=True):
            st.markdown("**Authoritative Stored Configuration**")

            if qubits is not None:
                ui_components.information_row("Qubits", str(qubits))

            if fm_name is not None:
                ui_components.information_row("Feature Map", str(fm_name))

            if fm_reps is not None:
                ui_components.information_row(
                    "Feature Map Repetitions",
                    str(fm_reps),
                )

            if fm_entanglement is not None:
                ui_components.information_row(
                    "Feature Map Entanglement",
                    str(fm_entanglement),
                )

            if has_ansatz:
                if ans_name is not None:
                    ui_components.information_row("Ansatz", str(ans_name))
                if ans_reps is not None:
                    ui_components.information_row(
                        "Ansatz Repetitions",
                        str(ans_reps),
                    )

            if backend_metadata is not None:
                ui_components.information_row(
                    "Backend Metadata",
                    str(backend_metadata),
                )

            if not any(
                value is not None
                for value in (
                    qubits,
                    fm_name,
                    fm_reps,
                    fm_entanglement,
                    ans_name if has_ansatz else None,
                    ans_reps if has_ansatz else None,
                    backend_metadata,
                )
            ):
                ui_components.render_unavailable_state(
                    "Stored Configuration",
                    "No displayable configuration fields were exposed by the backend.",
                )

    with col_derived:
        with st.container(border=True):
            st.markdown("**Derived Circuit Evidence**")

            if has_circuit:
                if derived_depth is not None:
                    ui_components.information_row(
                        "Circuit Depth",
                        str(derived_depth),
                    )

                if derived_params is not None:
                    if has_ansatz:
                        parameter_label = (
                            f"{derived_params} "
                            "(Data-encoding + Trainable Ansatz)"
                        )
                    else:
                        parameter_label = (
                            f"{derived_params} "
                            "(Data-encoding only)"
                        )

                    ui_components.information_row(
                        "Circuit Parameters",
                        parameter_label,
                    )

                if derived_depth is None and derived_params is None:
                    ui_components.render_unavailable_state(
                        "Derived Circuit Evidence",
                        "The backend reported a circuit but did not expose derived depth or parameter-count evidence.",
                    )
            else:
                ui_components.render_unavailable_state(
                    "Derived Circuit Evidence",
                    circ_res.get(
                        "message",
                        "Circuit reconstruction is unavailable.",
                    ),
                )

    # -------------------------------------------------------------------------
    # 2. QUANTUM PREPROCESSING
    # -------------------------------------------------------------------------
    ui_components.section_header(
        "Quantum Preprocessing",
        "Authoritative preprocessing steps exposed for the selected model.",
    )

    if prep_steps:
        with st.container(border=True):
            for index, step in enumerate(prep_steps, start=1):
                if not isinstance(step, dict):
                    continue

                step_type = step.get("type")
                if step_type is None:
                    continue

                st.markdown(f"**Step {index}: {step_type}**")

                params = step.get("parameters")
                if isinstance(params, dict):
                    for param_key, param_value in params.items():
                        ui_components.information_row(
                            f"{param_key}",
                            str(param_value),
                        )
    else:
        ui_components.render_unavailable_state(
            "Quantum Preprocessing",
            "No preprocessing steps were explicitly exposed by the backend.",
        )

    # -------------------------------------------------------------------------
    # 3. ARCHITECTURE FLOW
    # -------------------------------------------------------------------------
    ui_components.section_header(
        "Architecture Flow",
        "Directly supported sequence derived from authoritative metadata and circuit evidence.",
    )

    flow_steps = []

    if prep_steps:
        flow_steps.append("Preprocessing Sequence")

    if fm_name is not None:
        flow_steps.append(str(fm_name))

    if has_ansatz and ans_name is not None:
        flow_steps.append(str(ans_name))

    if has_circuit:
        flow_steps.append(
            "Constructed Quantum Circuit"
            if has_ansatz
            else "Constructed Kernel Circuit"
        )

    if flow_steps:
        with st.container(border=True):
            st.markdown("**Flow**")
            st.write(" → ".join(flow_steps))
    else:
        ui_components.render_unavailable_state(
            "Architecture Flow",
            "Insufficient authoritative evidence is available to render the architecture sequence.",
        )

    # -------------------------------------------------------------------------
    # 4. DETAILED QUANTUM CONFIGURATION
    # -------------------------------------------------------------------------
    ui_components.section_header("Detailed Quantum Configuration")

    with st.expander(
        "Expand Authoritative Configuration Metadata",
        expanded=False,
    ):
        for key, value in q_cfg.items():
            label = str(key).replace("_", " ").title()

            if isinstance(value, dict):
                st.markdown(f"**{label}**")
                for sub_key, sub_value in value.items():
                    ui_components.information_row(
                        str(sub_key).replace("_", " ").title(),
                        str(sub_value),
                    )
            else:
                ui_components.information_row(label, str(value))

    # -------------------------------------------------------------------------
    # 5. CIRCUIT VISUALIZATION
    # -------------------------------------------------------------------------
    ui_components.section_header(
        "Circuit Visualization",
        "Actual backend-generated ASCII circuit representation.",
    )

    if has_circuit:
        with st.container(border=True):
            circuit_title = circ_res.get("title")
            if circuit_title is not None:
                st.markdown(f"**{circuit_title}**")

            st.code(circuit_string, language="text")
    else:
        ui_components.render_unavailable_state(
            "Circuit Visualization",
            circ_res.get(
                "message",
                "No circuit text could be generated from the backend.",
            ),
        )
