import streamlit as st

def render_unavailable_state(context: str, explanation: str, next_step: str = None):
    """
    Renders a standardized, professional unavailable state for missing backend components.
    Avoids giant warning banners in favor of a restrained, scientific visual treatment.
    """
    with st.container():
        st.markdown(f"#### {context}")
        st.markdown("**Status:** Not available")
        st.markdown(explanation)
        if next_step:
            st.caption(f"**Next Step:** {next_step}")
        st.divider()

def render_error_state(message: str):
    """Renders an error state."""
    st.error(f"**⚠️ Error**\n\n{message}")

def display_disclaimer():
    """Displays the standard research/prototype disclaimer."""
    st.caption(
        "**Research prototype:** This application is a research prototype for evaluating hybrid "
        "quantum-classical machine learning approaches. Model outputs are computational predictions "
        "and are not a medical diagnosis or a substitute for professional clinical evaluation."
    )
