import streamlit as st
import src.backend_adapter as backend
from src.ui_components import render_unavailable_state

def render():
    st.header("Explainability & Insights")
    
    selected_model = st.session_state.get("selected_model")
    
    if selected_model is None:
        st.info("No trained model artifacts are currently available for explainability analysis.")
        return
        
    st.subheader(f"Model: {selected_model}")
    
    st.divider()
    
    # 1. PCA Loadings
    st.markdown("### PCA Component Loadings")
    st.markdown("Interpretation of how original features contribute to PCA components. (Note: This is dimensionality reduction interpretation, not predictive feature importance.)")
    explain = backend.get_explainability(selected_model)
    if explain["status"] == "available" and "pca_loadings" in explain.get("data", {}):
        st.dataframe(explain["data"]["pca_loadings"])
    else:
        render_unavailable_state(
            "Loadings Not Available", 
            "PCA component loadings require loaded preprocessing artifacts.",
            "Connect the trained preprocessing pipeline to view."
        )
        
    st.divider()
    
    # 2. Predictive Feature Attribution
    st.markdown("### Predictive Feature Attribution")
    st.markdown("Feature-level explanation produced by an appropriate model-specific attribution method.")
    if explain["status"] == "available" and "attribution" in explain.get("data", {}):
        st.write(explain["data"]["attribution"])
    else:
        render_unavailable_state(
            "Attribution Not Available", 
            "Global feature attribution requires a supported post-hoc interpretation method and trained model.",
            "Connect model artifacts to view."
        )
        
    st.divider()
    
    # 3. Patient-Level Explanation
    st.markdown("### Patient-Level Explanation")
    st.markdown("Explanation associated with the currently selected prediction.")
    if explain["status"] == "available" and "patient_explanation" in explain.get("data", {}):
        st.write(explain["data"]["patient_explanation"])
    else:
        render_unavailable_state(
            "Explanation Not Available", 
            "Patient-level explanations require a valid prediction result.",
            "Submit a patient record for prediction on the Prediction page."
        )
