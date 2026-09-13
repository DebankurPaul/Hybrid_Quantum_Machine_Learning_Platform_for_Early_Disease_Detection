import streamlit as st
import src.backend_adapter as backend
from src.ui_components import render_unavailable_state

def render_metrics(selected_model, threshold):
    st.markdown("### Model Performance")
    metrics = backend.get_evaluation_metrics(selected_model, threshold)
    if metrics["status"] == "available":
        data = metrics["data"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{data.get('accuracy', 0):.4f}")
        col2.metric("Precision", f"{data.get('precision', 0):.4f}")
        col3.metric("Sensitivity", f"{data.get('sensitivity', 0):.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Specificity", f"{data.get('specificity', 0):.4f}")
        col5.metric("F1 Score", f"{data.get('f1', 0):.4f}")
        col6.metric("ROC-AUC", f"{data.get('roc_auc', 0):.4f}")
    else:
        render_unavailable_state("Metrics Not Available", metrics.get("message", "Evaluation metrics require loaded evaluation artifacts."))

def render_confusion_matrix(selected_model, threshold):
    st.markdown("### Confusion Matrix")
    cm = backend.get_confusion_matrix(selected_model, threshold)
    if cm["status"] == "available":
        d = cm["data"]
        # Format as markdown table
        st.markdown(f"""
|                | Predicted Benign | Predicted Malignant |
|----------------|------------------|---------------------|
| **True Benign**    | {d.get('TN', 0)}               | {d.get('FP', 0)}                  |
| **True Malignant** | {d.get('FN', 0)}               | {d.get('TP', 0)}                  |
""")
    else:
        render_unavailable_state("Matrix Not Available", cm.get("message", "Awaiting evaluation results."))

def render_roc_curve(selected_model):
    st.markdown("### ROC Curve")
    roc = backend.get_roc_curve(selected_model)
    if roc["status"] == "available":
        st.line_chart(roc["data"])
    else:
        render_unavailable_state("Curve Not Available", roc.get("message", "Awaiting evaluation results."))

def render_threshold_analysis(selected_model):
    st.markdown("### Threshold Analysis")
    analysis = backend.get_threshold_analysis(selected_model)
    if analysis["status"] == "available":
        st.line_chart(analysis["data"]) 
    else:
        render_unavailable_state("Analysis Not Available", analysis.get("message", "Threshold trade-off curves require evaluation data."))

def render_inference_time(selected_model, threshold):
    st.markdown("### Inference Time")
    metrics = backend.get_evaluation_metrics(selected_model, threshold)
    if metrics["status"] == "available":
        st.metric("Avg Inference Time", metrics["data"].get("inference_time", "—"))
    else:
        render_unavailable_state("Timing Not Available", "Timing metrics require loaded artifacts.")

def render():
    st.header("Evaluation Metrics")
    
    selected_model = st.session_state.get("selected_model")
    threshold = st.session_state.get("threshold", backend.get_default_threshold())
    
    if selected_model is None:
        st.info("No trained model artifacts are currently available to evaluate.")
        return
        
    st.subheader(f"Model: {selected_model} (τ = {threshold:.2f})")
    st.markdown("Performance measured against the validation/test set.")
    
    st.divider()
    
    # 1. Key Metrics
    render_metrics(selected_model, threshold)
        
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    # 2. Confusion Matrix
    with col_left:
        render_confusion_matrix(selected_model, threshold)
            
    # 3. ROC Curve
    with col_right:
        render_roc_curve(selected_model)
            
    st.divider()
    
    col_thresh, col_time = st.columns(2)
    
    # 4. Threshold Analysis
    with col_thresh:
        render_threshold_analysis(selected_model)
    
    # 5. Inference Time
    with col_time:
        render_inference_time(selected_model, threshold)
