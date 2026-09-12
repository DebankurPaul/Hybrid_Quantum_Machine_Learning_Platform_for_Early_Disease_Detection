import streamlit as st
import pandas as pd
import src.backend_adapter as backend
from src.ui_components import render_unavailable_state, render_error_state

FEATURE_GROUPS = {
    "Mean Measurements": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
    ],
    "Standard Error Measurements": [
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se"
    ],
    "Worst Measurements": [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
}

def render_result(result, threshold):
    if result["status"] == "available":
        st.success("Prediction complete")
        st.markdown("### MODEL PREDICTION")
        data = result["data"]
        st.metric("Predicted Class", data.get("predicted_class", "—"))
        st.metric("Estimated Model Probability", f"{data.get('probability', 0.0):.4f}" if "probability" in data else "—")
        st.metric("Decision Threshold", f"τ = {threshold:.3f}")
        st.caption("Research prototype: This result is a machine-learning prediction for demonstration/research purposes and is not a medical diagnosis or medical advice.")
    else:
        render_unavailable_state("Prediction unavailable", result.get("message", "The trained model artifacts required for inference are not currently connected."))

def render():
    st.header("Patient Prediction")
    st.markdown("Enter 30 original biomedical measurements to generate a model prediction.")
    
    selected_model = st.session_state.get("selected_model")
    threshold = st.session_state.get("threshold", 0.65)
    
    if selected_model is None:
        st.info("No trained model artifacts are currently available to perform predictions.")
        return
    
    tab1, tab2 = st.tabs(["Manual Entry", "Single Patient CSV"])
    
    with tab1:
        with st.form("patient_input_form"):
            patient_data = {}
            
            for group_name, features in FEATURE_GROUPS.items():
                st.subheader(group_name)
                cols = st.columns(5)
                for idx, feature in enumerate(features):
                    col = cols[idx % 5]
                    # value=None forces user to input or it remains None
                    val = col.number_input(feature.replace('_', ' ').title(), value=None, format="%.4f", step=0.1, key=f"manual_{feature}")
                    patient_data[feature] = val
                    
            submitted = st.form_submit_button("Run Model Prediction")
            if submitted:
                # Check for incomplete data
                missing = [k for k, v in patient_data.items() if v is None]
                if missing:
                    render_error_state("Please fill in all 30 measurements before running the model prediction.")
                else:
                    with st.spinner("Preparing patient record..."):
                        # Submit the validated patient record through the backend adapter.
                        result = backend.predict_patient(patient_data, selected_model, threshold)
                    st.divider()
                    render_result(result, threshold)
                
    with tab2:
        st.markdown("Upload a CSV containing exactly one patient row for inference.")
        uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                # Validation
                all_features = [f for group in FEATURE_GROUPS.values() for f in group]
                missing_cols = [f for f in all_features if f not in df.columns]
                
                if missing_cols:
                    render_error_state(f"Missing required columns: {', '.join(missing_cols)}")
                elif len(df) != 1:
                    render_error_state("This prediction workflow currently accepts one patient record at a time. Please upload a CSV containing exactly one patient row.")
                elif df.isnull().values.any():
                    render_error_state("CSV contains missing values. Please provide complete data.")
                else:
                    if st.button("Run Prediction"):
                        with st.spinner("Calculating predictions..."):
                            # Submit the validated patient record through the backend adapter.
                            result = backend.predict_patient(df.iloc[0].to_dict(), selected_model, threshold)
                        st.divider()
                        render_result(result, threshold)
                        
            except Exception as e:
                render_error_state(f"Failed to parse CSV: {str(e)}")
