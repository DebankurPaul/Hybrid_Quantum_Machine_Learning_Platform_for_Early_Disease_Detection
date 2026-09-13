import streamlit as st
import src.backend_adapter as backend
from src.ui_components import render_unavailable_state

def render():
    st.header("Quantum Model Architecture")
    
    selected_model = st.session_state.get("selected_model")
    if selected_model is None:
        st.info("No model selected. No trained model artifacts are currently available.")
        return
        
    st.subheader(f"Current Model: {selected_model}")
    
    # 1. Pipeline Specification
    st.markdown("### Pipeline Specification")
    st.markdown("This represents the architecture mapping classical data to quantum states.")
    st.markdown(
        """
        1. **StandardScaler** (Classical Preprocessing)
        2. **PCA (8 Components)** (Dimensionality Reduction)
        3. **tanh(x) * π** (Angle Encoding Normalization)
        4. **ZZFeatureMap** (Quantum State Preparation)
        5. **RealAmplitudes** (Variational Ansatz)
        6. **Measurement** (Qiskit VQC backend)
        """
    )
    st.divider()
    
    # 2. Model Configuration
    st.markdown("### Loaded Model Configuration")
    config = backend.get_quantum_model_info(selected_model)
    if config["status"] == "available":
        data = config["data"]
        col1, col2, col3 = st.columns(3)
        col1.metric("PCA Components", data.get("pca_components", "—"))
        col2.metric("Qubits", data.get("qubits", "—"))
        col3.metric("Trainable Parameters", data.get("trainable_parameters", "—"))
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Feature Map", data.get("feature_map", "—"))
        col5.metric("Ansatz", data.get("ansatz", "—"))
        col6.metric("Backend Simulator", data.get("backend", "—"))
    else:
        render_unavailable_state("Configuration Not Loaded", config.get("message", "Model configuration not available pending real artifacts."))
        
    st.divider()
    
    # 3. Quantum Circuit
    st.markdown("### Actual Quantum Circuit")
    circuit = backend.get_quantum_circuit(selected_model)
    if circuit["status"] == "available":
        st.code(circuit.get("data", ""), language="text")
    else:
        render_unavailable_state("Circuit Not Rendered", circuit.get("message", "Connect the trained quantum model to render the actual circuit configuration."))
