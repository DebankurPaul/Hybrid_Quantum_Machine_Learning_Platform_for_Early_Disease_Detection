import streamlit as st
import src.backend_adapter as backend
from src.ui_components import display_disclaimer
import importlib

st.set_page_config(
    page_title="Hybrid-QML Disease Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_sidebar():
    st.sidebar.title("HYBRID-QML")
    st.sidebar.caption("Quantum-Classical Intelligence for Early Disease Detection")
    st.sidebar.divider()
    
    st.sidebar.markdown("**DATASET**\n\nWDBC")
    st.sidebar.markdown("**TASK**\n\nDisease Detection")
    st.sidebar.markdown("**QUANTUM BACKEND**\n\nQiskit Aer (Planned)")
    
    st.sidebar.divider()
    
    # Model Availability
    models = backend.get_available_models()
    if models["status"] == "available":
        available_models = models["data"]["available"]
        planned_quantum = models["data"]["planned_quantum"]
        planned_classical = models["data"]["planned_classical"]
        
        st.sidebar.markdown("**MODEL**")
        if available_models:
            selected_model = st.sidebar.selectbox("Select Model", available_models, label_visibility="collapsed")
            threshold_disabled = False
        else:
            selected_model = None
            st.sidebar.selectbox("Select Model", ["No models currently available"], disabled=True, label_visibility="collapsed")
            threshold_disabled = True
        
        with st.sidebar.expander("Planned Capabilities", expanded=(not available_models)):
            st.markdown("**Quantum**")
            for mq in planned_quantum:
                st.markdown(f"- {mq} `[Unavailable]`")
            st.markdown("**Classical**")
            for mc in planned_classical:
                st.markdown(f"- {mc} `[Unavailable]`")
    else:
        selected_model = None
        threshold_disabled = True
    
    threshold = st.sidebar.slider(
        "DECISION THRESHOLD (τ)", 
        min_value=0.10, max_value=0.90, value=0.65, step=0.01, 
        disabled=threshold_disabled,
        help="Active when a model is loaded." if threshold_disabled else None
    )
    
    st.session_state["selected_model"] = selected_model
    st.session_state["threshold"] = threshold
    
    st.sidebar.divider()
    
    st.sidebar.markdown("**SYSTEM STATUS**")
    status = backend.get_system_status()
    if status["status"] in ["available", "partial"]:
        data = status["data"]
        dataset_status = "Ready" if data.get("dataset") else "Not available"
        prep_status = "Ready" if data.get("preprocessor") else "Not available"
        q_status = "Ready" if data.get("quantum_backend") else "Not available"
        model_status = "Ready" if data.get("model_artifacts") else "Not available"
        
        st.sidebar.markdown(f"""
        Dataset: `{dataset_status}`  
        Preprocessor: `{prep_status}`  
        Quantum backend: `{q_status}`  
        Model artifacts: `{model_status}`
        """)

def main():
    render_sidebar()
    
    pages = {
        "Overview": "src.pages.overview",
        "Prediction": "src.pages.prediction",
        "Quantum Model": "src.pages.quantum_model",
        "Evaluation": "src.pages.evaluation",
        "Benchmark": "src.pages.benchmark",
        "Explainability": "src.pages.explainability"
    }
    
    st.sidebar.divider()
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    st.sidebar.divider()
    display_disclaimer()
    
    page_module = importlib.import_module(pages[selection])
    page_module.render()

if __name__ == "__main__":
    main()
