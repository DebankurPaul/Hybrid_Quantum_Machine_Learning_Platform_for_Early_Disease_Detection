import streamlit as st
import os

def load_css():
    """Loads the global design system CSS."""
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "css", "design_system.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def page_header(title: str, description: str = None, status: str = None):
    """Renders a standardized page header."""
    st.markdown(f"<div class='page-header'><h2>{title}</h2></div>", unsafe_allow_html=True)
    if description or status:
        cols = st.columns([3, 1])
        with cols[0]:
            if description:
                st.markdown(f"<div class='page-header-desc'>{description}</div>", unsafe_allow_html=True)
        with cols[1]:
            if status:
                st.markdown(f"<div class='page-header-status'><span class='status-badge status-{status.lower()}'>{status}</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

def section_header(title: str, description: str = None):
    """Renders a standardized section header."""
    description_html = (
        f"<div class='section-header-desc'>{description}</div>"
        if description
        else ""
    )
    st.markdown(
        f"""
        <div class='section-header'>
            <h4>{title}</h4>
            {description_html}
            <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )

def status_indicator(state: str, label: str = None):
    """Renders a semantic status badge."""
    display_text = label if label else state
    st.markdown(f"<span class='status-badge status-{state.lower().replace('_', '-')}'>{display_text}</span>", unsafe_allow_html=True)

def metric_display(label: str, value: str):
    """Renders a restrained typography-focused metric card."""
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
    </div>
    """, unsafe_allow_html=True)

def information_row(label: str, value: str):
    """Renders a compact technical metadata row."""
    st.markdown(f"""
    <div class='info-row'>
        <div class='info-label'>{label}</div>
        <div class='info-value'>{value}</div>
    </div>
    """, unsafe_allow_html=True)

def empty_state(title: str, description: str):
    """Renders a deliberate empty state without placeholder fabrications."""
    st.markdown(f"""
    <div class='empty-state'>
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def loading_state(message: str = "Processing..."):
    """Renders a subtle loading state."""
    st.markdown(f"""
    <div class='loading-state'>
        <div class='status-badge status-loading'>●</div>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)

def error_state(message: str, details: str = None):
    """Renders a structured technical error boundary."""
    st.error(f"**System Error:** {message}")
    if details:
        with st.expander("Technical Details"):
            st.code(details)

def unverified_state(title: str, description: str):
    """Renders an unverified state to prevent misinterpretation of unproven data."""
    st.markdown(f"""
    <div class='capability-alert unverified'>
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 4px;'>
            <strong>UNVERIFIED ARTIFACT</strong>
        </div>
        <div style='font-size: 14px;'><strong>{title}:</strong> {description}</div>
    </div>
    """, unsafe_allow_html=True)

def capability_state(label: str, state: str, description: str = None):
    """Renders a capability-aware UI pattern for dynamic availability."""
    st.markdown(f"""
    <div class='capability-row'>
        <div class='capability-row-header'>
            <span class='label'>{label}</span>
            <span class='status-badge status-{state.lower().replace("_", "-")}'>{state}</span>
        </div>
        {f"<div class='capability-row-desc'>{description}</div>" if description else ""}
    </div>
    """, unsafe_allow_html=True)

def render_unavailable_state(context: str, explanation: str, next_step: str = None):
    """Preserved for backward compatibility with existing pages."""
    with st.container():
        st.markdown(f"#### {context}")
        st.markdown("**Status:** Not available")
        st.markdown(explanation)
        if next_step:
            st.caption(f"**Next Step:** {next_step}")
        st.markdown("<hr>", unsafe_allow_html=True)

def display_disclaimer():
    """Displays the standard research/prototype disclaimer."""
    st.caption(
        "**Research prototype:** This application is a research prototype for evaluating hybrid "
        "quantum-classical machine learning approaches. Model outputs are computational predictions "
        "and are not a medical diagnosis or a substitute for professional clinical evaluation."
    )
