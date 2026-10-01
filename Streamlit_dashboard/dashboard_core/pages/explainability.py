"""
Explainability Workspace
========================
Backend-driven explainability workspace.

The dashboard renders explainability only when the authoritative backend
exposes the capability and a verified artifact through its adapter API.
The UI does not discover, calculate, reconstruct, or infer explainability
artifacts on its own.
"""

import os
import streamlit as st
import dashboard_core.backend_adapter as backend


def _dataset_name(dataset_key: str, dataset_metadata: dict) -> str:
    data = dataset_metadata.get("data") or {}
    return data.get("name") or dataset_key


def _model_name(model_key: str, model_metadata: dict) -> str:
    data = model_metadata.get("data") or {}
    return data.get("name") or model_key


def _explainability_context(model_metadata: dict) -> dict:
    data = model_metadata.get("data") or {}
    context = {}

    method = data.get("explainability_method")
    if method:
        context["method"] = method

    artifacts = data.get("artifacts")
    if isinstance(artifacts, dict):
        registered = artifacts.get("explainability")
        if registered:
            context["registered_artifact"] = registered

    return context


def _render_unavailable(model_name: str, dataset_name: str) -> None:
    with st.container(border=True):
        st.markdown("### Explainability unavailable")
        st.write(
            f"The backend does not currently provide a validated explainability "
            f"artifact for **{model_name}** on the **{dataset_name}** dataset."
        )
        st.caption(
            "Only explainability artifacts exposed and validated by the backend "
            "are displayed. No synthetic, reconstructed, or dashboard-generated "
            "attributions are created."
        )


def _render_artifact(
    artifact_path: str,
    model_name: str,
    dataset_name: str,
    context: dict,
) -> None:
    if not artifact_path:
        _render_unavailable(model_name, dataset_name)
        return

    # The backend has already resolved and validated the artifact path.
    # The dashboard does not search the results directory or construct paths.
    try:
        st.markdown("### Validated Explainability Artifact")

        if context.get("method"):
            st.caption(f"**Method:** {context['method']}")

        with st.container(border=True):
            st.image(artifact_path, use_container_width=True)

            registered_artifact = context.get("registered_artifact")
            if registered_artifact:
                st.caption(
                    f"Backend-registered artifact: `{registered_artifact}`"
                )
            else:
                st.caption(
                    "Artifact path supplied by the authoritative backend adapter."
                )

    except Exception:
        st.error(
            "The backend reported an explainability artifact, but the dashboard "
            "could not render it."
        )


def render():
    dataset_key = st.session_state.get("active_dataset")
    selected_model = st.session_state.get("selected_model")

    if not dataset_key or not selected_model:
        st.warning(
            "Please select an active dataset and model from the navigation rail."
        )
        return

    dataset_metadata = backend.get_dataset_metadata(dataset_key)
    model_metadata = backend.get_model_metadata(dataset_key, selected_model)

    if model_metadata.get("status") != "available":
        st.error("Model metadata unavailable.")
        return

    dataset_name = _dataset_name(dataset_key, dataset_metadata)
    model_name = _model_name(selected_model, model_metadata)

    st.markdown(f"## Explainability: {model_name}")
    st.caption(
        f"**Dataset:** {dataset_name} | "
        "**Interpretability:** Backend-validated post-hoc explanation artifacts"
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    capabilities = backend.get_capabilities(dataset_key, selected_model)

    # The backend capability flag is the sole scientific gatekeeper.
    if capabilities.get("explainability") != "AVAILABLE":
        _render_unavailable(model_name, dataset_name)
        return

    # Retrieve the authoritative artifact only through the backend adapter.
    artifact_path = backend.get_explainability_artifact(
        dataset_key, selected_model
    )

    if not artifact_path:
        _render_unavailable(model_name, dataset_name)
        return

    context = _explainability_context(model_metadata)

    _render_artifact(
        artifact_path=artifact_path,
        model_name=model_name,
        dataset_name=dataset_name,
        context=context,
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.caption(
        "Research notice: explainability artifacts describe model behavior "
        "and attribution within the representation used by the validated "
        "backend pipeline. They do not establish biological causality."
    )