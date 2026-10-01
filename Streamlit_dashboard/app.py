import streamlit as st
import os
import base64

# Configure page before any other Streamlit calls
st.set_page_config(
    page_title="Hybrid Quantum–Classical Disease Detection Platform",
    page_icon="⚛️", # Minor limitation: left as emoji as requested if SVG is complex
    layout="wide",
    initial_sidebar_state="expanded"
)

import dashboard_core.backend_adapter as backend
import dashboard_core.ui_components as ui
from dashboard_core.pages import (
    overview,
    data_preview,
    models,
    quantum_model,
    prediction,
    evaluation,
    benchmark,
    explainability
)

# Load global CSS tokens
ui.load_css()

# Base64 encode SVGs for CSS injection to replace radio emojis with professional icons
def get_svg_data_uri(filename):
    svg_path = os.path.join(os.path.dirname(__file__), "assets", "svg", "icons", filename)
    if os.path.exists(svg_path):
        with open(svg_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"url('data:image/svg+xml;base64,{encoded}')"
    return "none"

# Note on UI Dependency: The navigation icon implementation relies on Streamlit's
# specific DOM structure (role="radiogroup", label, aria-checked) for CSS injection.
# This avoids a complex React/custom component rewrite but assumes stable DOM.
st.markdown(f"""
<style>
    /* Navigation Icons */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {{
        padding-left: 36px;
        background-repeat: no-repeat;
        background-position: 10px center;
        background-size: 16px 16px;
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) {{ background-image: {get_svg_data_uri('overview.svg')}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(2) {{ background-image: {get_svg_data_uri('data.svg')}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) {{ background-image: {get_svg_data_uri('models.svg')}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(4) {{ background-image: {get_svg_data_uri('quantum.svg')}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5) {{ background-image: {get_svg_data_uri('prediction.svg')}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(6) {{ background-image: {get_svg_data_uri('evaluation.svg')}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7) {{ background-image: {get_svg_data_uri('benchmark.svg')}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(8) {{ background-image: {get_svg_data_uri('explainability.svg')}; }}
</style>
""", unsafe_allow_html=True)

def main():
    # ── INIT SESSION STATE ────────────────────────────────────────────────
    datasets = backend.get_dataset_catalog()
    dataset_keys = list(datasets.keys())

    if "active_dataset" not in st.session_state or st.session_state["active_dataset"] not in dataset_keys:
        st.session_state["active_dataset"] = dataset_keys[0]

    models_dict = backend.get_available_models(st.session_state["active_dataset"])
    model_keys = list(models_dict.keys())

    if "selected_model" not in st.session_state or st.session_state["selected_model"] not in model_keys:
        st.session_state["selected_model"] = model_keys[0] if model_keys else None

    # Update active capabilities state
    if st.session_state.get("selected_model"):
        st.session_state["capabilities"] = backend.get_capabilities(
            st.session_state["active_dataset"],
            st.session_state["selected_model"]
        )
    else:
        st.session_state["capabilities"] = {}

    # ── SIDEBAR (APPLICATION IDENTITY & NAVIGATION) ──────────────────────
    with st.sidebar:
        # Platform Branding
        st.markdown("""
        <div style='padding-bottom: 12px; border-bottom: 1px solid var(--border); margin-bottom: 24px;'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='background: var(--surface-subtle); border: 1px solid var(--border-subtle); padding: 4px; border-radius: var(--radius-sm);'>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                    </svg>
                </div>
                <span style='font-size: 15px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em;'>HYBRID QML PLATFORM</span>
            </div>
            <div style='font-size: 12px; color: var(--text-secondary); margin-top: 6px; font-weight: 500;'>
                Research Prototype
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Workspace Navigation
        st.markdown("<div style='font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>Research Workspaces</div>", unsafe_allow_html=True)

        PAGES = {
            "Overview": overview.render,
            "Data Explorer": data_preview.render,
            "Model Landscape": models.render,
            "Quantum Architecture": quantum_model.render,
            "Prediction": prediction.render,
            "Evaluation": evaluation.render,
            "Benchmark": benchmark.render,
            "Explainability": explainability.render
        }

        page_choice = st.radio(
            "Workspace",
            list(PAGES.keys()),
            label_visibility="collapsed"
        )

    # ── TOP CONTEXT BAR ──────────────────────────────────────────────────
    # Reads the local SVG for subtle, non-interactive header decoration.
    bg_svg_path = os.path.join(os.path.dirname(__file__), "assets", "svg", "decorations", "header_bg.svg")
    bg_style = ""
    if os.path.exists(bg_svg_path):
        with open(bg_svg_path, "rb") as f:
            encoded_bg = base64.b64encode(f.read()).decode("utf-8")
            bg_style = f"url('data:image/svg+xml;base64,{encoded_bg}')"

    # Injecting a marker that wraps the bordered Streamlit container.
    # The CSS variable lets the local SVG be applied without introducing
    # an external asset dependency or altering the container structure.
    st.markdown(
        f'''<div class='top-context-wrapper' style="--context-bg-image: {bg_style or 'none'};">''',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        # We apply the background image directly if Streamlit allows, but border=True gives a clean container
        # Since we can't easily add the background directly to the Streamlit container via kwargs,
        # we'll use CSS, but the priority is a unified look.
        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            st.markdown(f"<div style='padding-top: 10px; font-weight: 600; color: var(--text-primary); font-size: 14px;'>Current Context</div>", unsafe_allow_html=True)

        with col2:
            selected_ds = st.selectbox(
                "Biomedical Dataset",
                options=dataset_keys,
                format_func=lambda k: datasets[k],
                index=dataset_keys.index(st.session_state["active_dataset"])
            )
            if selected_ds != st.session_state["active_dataset"]:
                st.session_state["active_dataset"] = selected_ds
                st.session_state.pop("selected_model", None)
                st.rerun()

        with col3:
            if not model_keys:
                st.selectbox("Active Model", ["No models available"], disabled=True)
            else:
                selected_m = st.selectbox(
                    "Active Model",
                    options=model_keys,
                    format_func=lambda k: models_dict[k],
                    index=model_keys.index(st.session_state["selected_model"]) if st.session_state.get("selected_model") in model_keys else 0
                )
                if selected_m != st.session_state.get("selected_model"):
                    st.session_state["selected_model"] = selected_m
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-bottom: 1px solid var(--border); margin-top: 8px; margin-bottom: 24px;'>", unsafe_allow_html=True)

    # ── RENDER ACTIVE WORKSPACE ──────────────────────────────────────────
    PAGES[page_choice]()

if __name__ == "__main__":
    main()
