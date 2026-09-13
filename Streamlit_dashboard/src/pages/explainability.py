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
    
    explain = backend.get_explainability(selected_model)
    
    if "Quantum" in selected_model:
        st.markdown("### Q-XAI: Quantum Feature Attribution (SHAP)")
        st.markdown("Feature-level explanation produced by SHAP on the Quantum Variational Circuit. **Note: These features represent PCA components (latent space), not the original 30 WDBC features.**")
        
        if explain["status"] == "available" and "attribution" in explain.get("data", {}):
            plot_path = explain["data"]["attribution"]
            st.image(plot_path, caption="Q-XAI SHAP Summary")
        else:
            render_unavailable_state(
                "Q-XAI Not Available", 
                "Global feature attribution requires a supported post-hoc interpretation method and trained model.",
                "Connect model artifacts to view."
            )
    else:
        st.markdown("### Classical Feature Importance")
        st.markdown("Global feature importance based on classical ensemble methods (e.g., Random Forest). These importance values map directly to the original 30 WDBC features.")
        
        if explain["status"] == "available" and "pca_loadings" in explain.get("data", {}):
            st.dataframe(explain["data"]["pca_loadings"])
        else:
            render_unavailable_state(
                "Feature Importance Not Available", 
                "Classical feature importance requires a loaded artifact (e.g., random_forest_feature_importance.csv)."
            )
