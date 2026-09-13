import streamlit as st
import src.backend_adapter as backend
from src.ui_components import render_unavailable_state

def render():
    st.title("Hybrid Quantum–Classical Disease Detection")
    st.subheader("A research platform for evaluating classical and quantum machine learning approaches")
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
    
    st.markdown("### Pipeline Architecture")
    st.markdown("Compare classical baselines with a Variational Quantum Classifier (VQC).")
    
    st.markdown(
        """
        **30 Original Features (WDBC)**  
        &nbsp;&nbsp;&nbsp;&nbsp;↓  
        **StandardScaler**  
        &nbsp;&nbsp;&nbsp;&nbsp;↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘  
        **(Classical Branch)**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**(Quantum Branch)**  
        *All 30 Features*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*PCA (8 Components)*  
        &nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
        *Logistic, SVM, RF, XGB*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*tanh(x) × π Normalization*  
        &nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
        **Prediction**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Qiskit VQC (8-qubit)*  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Prediction**
        """
    )
