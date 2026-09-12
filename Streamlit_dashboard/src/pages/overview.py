import streamlit as st
import src.backend_adapter as backend
from src.ui_components import render_unavailable_state

def render():
    st.title("HYBRID-QML")
    st.subheader("Quantum-Classical Intelligence for Early Disease Detection")
    st.markdown("WDBC · Breast Cancer Wisconsin (Diagnostic)")
    
    st.divider()
    
    # Metadata Overview
    st.markdown("### Currently Loaded System Status")
    meta_status = backend.get_dataset_metadata()
    if meta_status["status"] == "available":
        meta = meta_status["data"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", meta.get("samples", "—"))
        col2.metric("Original Features", meta.get("original_features", "—"))
        col3.metric("Classes", meta.get("classes", "—"))
        
        # Check if we have trained models available
        models = backend.get_available_models()
        num_models = len(models["data"]["available"]) if models["status"] == "available" else "—"
        col4.metric("Trained Models", num_models if num_models else "—")
    else:
        render_unavailable_state("Dataset Context", "Waiting for dataset metadata.")
        
    st.divider()
    
    st.markdown("### Pipeline Design (Target Architecture)")
    st.markdown("Compare classical and quantum approaches using the same evaluation framework.")
    
    st.markdown(
        """
        **30 Original Features**  
        &nbsp;&nbsp;&nbsp;&nbsp;↓  
        **StandardScaler** *(Classical Preprocessing)*  
        &nbsp;&nbsp;&nbsp;&nbsp;↓  
        **PCA** *(Dimensionality Reduction)*  
        &nbsp;&nbsp;&nbsp;&nbsp;↓  
        **4 / 8 Components**  
        &nbsp;&nbsp;&nbsp;&nbsp;↓  
        **MinMaxScaler [0, π]** *(Quantum Encoding)*  
        &nbsp;&nbsp;&nbsp;&nbsp;↓  
        **VQC / QSVC** *(Quantum Classifier)*  
        &nbsp;&nbsp;&nbsp;&nbsp;↓  
        **Prediction**
        """
    )
