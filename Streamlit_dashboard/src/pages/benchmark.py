import streamlit as st
import pandas as pd
import src.backend_adapter as backend
from src.ui_components import render_unavailable_state

def render_benchmark_table(benchmark):
    st.markdown("### Model Comparison")
    if benchmark["status"] == "available":
        st.dataframe(benchmark["data"], use_container_width=True)
    else:
        render_unavailable_state(
            "Comparison Data Not Available", 
            benchmark.get("message", "Evaluation results are not connected yet."), 
            "The comparison table will populate automatically when the current model results are available."
        )

def render_performance_chart(benchmark):
    st.markdown("### Performance Overview")
    if benchmark["status"] == "available":
        st.bar_chart(benchmark["data"], x="Model", y=["Accuracy", "F1 Score"])
    else:
        render_unavailable_state("Chart Not Available", "Awaiting evaluation results for plotting.")

def render_roc_comparison(benchmark):
    st.markdown("### ROC Comparison")
    render_unavailable_state("Chart Not Available", "ROC curves are generated directly from predicted probabilities which are not all saved in the static CSV.")

def render_inference_time_chart(benchmark):
    st.markdown("### Inference Time Comparison")
    if benchmark["status"] == "available" and "Training Time (sec)" in benchmark["data"].columns:
        st.bar_chart(benchmark["data"], x="Model", y=["Training Time (sec)"])
    else:
        render_unavailable_state("Timing Not Available", "Awaiting evaluation results for plotting.")

def render():
    st.header("Classical vs. Quantum Benchmark")
    st.markdown("Compare classical and quantum approaches using the same evaluation framework.")
    
    st.divider()
    
    benchmark = backend.get_benchmark_results()
    
    # Render Table
    render_benchmark_table(benchmark)
        
    st.divider()
    
    col_perf, col_roc = st.columns(2)
    
    with col_perf:
        render_performance_chart(benchmark)
            
    with col_roc:
        render_roc_comparison(benchmark)
            
    st.divider()
    
    render_inference_time_chart(benchmark)
