"""
Data Preparation Pipeline - Main Streamlit Application

A comprehensive, GDPR-compliant data preparation tool for cleaning, validating,
and transforming datasets before correlation analysis.
"""

import streamlit as st
import pandas as pd
import atexit
import tempfile
from pathlib import Path

from utilities.state_manager import StateManager
from utilities.config import PipelineConfig
from utilities.distribution_inspector import render_distribution_inspector
from components.rollback_interface import render_rollback_interface

from phases import (
    phase_1_ingestion,
    phase_1a_duplicates,
    phase_2_scope,
    phase_3_sanitation,
    phase_4_outliers,
    phase_4a_multicollinearity,
    phase_5_feature_eng
)


# Cleanup function for temp files
def cleanup_temp_files():
    """Remove all temporary files on session end."""
    temp_dir = Path(tempfile.gettempdir()) / "data_pipeline"
    if temp_dir.exists():
        for file in temp_dir.glob("*.parquet"):
            try:
                file.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


# Register cleanup
atexit.register(cleanup_temp_files)


# Page configuration
st.set_page_config(
    page_title="Data Preparation Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Initialize session state
if 'state_manager' not in st.session_state:
    st.session_state.state_manager = StateManager()

if 'pipeline_config' not in st.session_state:
    st.session_state.pipeline_config = PipelineConfig()

if 'current_phase' not in st.session_state:
    st.session_state.current_phase = 0

if 'gdpr_consent' not in st.session_state:
    st.session_state.gdpr_consent = False

if 'show_inspector' not in st.session_state:
    st.session_state.show_inspector = False

if 'show_rollback' not in st.session_state:
    st.session_state.show_rollback = False

if 'show_privacy' not in st.session_state:
    st.session_state.show_privacy = False


# GDPR Consent Gate
if not st.session_state.gdpr_consent:
    st.title("Privacy & Data Processing Consent")

    st.markdown("""
    ### How Your Data Is Handled

    **What We Process:**
    - The file(s) you upload for analysis
    - Configuration choices you make during the pipeline

    **What We Do NOT Do:**
    - Store your data beyond your browser session
    - Share your data with third parties
    - Log or track your data content
    - Use your data to train models

    **Your Data Rights:**
    - Processed in-memory only (not stored permanently)
    - Deleted automatically when you close this tab
    - Never shared with third parties
    - You can delete all data anytime via the footer control
    """)

    consent = st.checkbox("I understand and consent to data processing")

    if st.button("Continue to Application", disabled=not consent):
        st.session_state.gdpr_consent = True
        st.rerun()

    st.stop()


# Main Application
st.title("Data Preparation Pipeline")

# Scroll to top - inject CSS and JS
st.markdown(
    """
    <style>
        .stApp > header + div {
            scroll-behavior: auto !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Force scroll using component
import streamlit.components.v1 as components
components.html(
    """
    <script>
        const streamlitDoc = window.parent.document;

        // Method 1: Find and scroll the main container
        const containers = [
            streamlitDoc.querySelector('.main'),
            streamlitDoc.querySelector('.stApp'),
            streamlitDoc.querySelector('[data-testid="stAppViewContainer"]'),
            streamlitDoc.querySelector('section.main'),
            streamlitDoc.body
        ];

        containers.forEach(function(container) {
            if (container) {
                container.scrollTo(0, 0);
                container.scrollTop = 0;
            }
        });

        // Method 2: Find all scrollable elements and reset them
        const allElements = streamlitDoc.querySelectorAll('*');
        allElements.forEach(function(el) {
            if (el.scrollTop > 0) {
                el.scrollTop = 0;
            }
        });
    </script>
    """,
    height=0,
    scrolling=False
)

# Sidebar navigation
with st.sidebar:
    st.header("Pipeline Phases")

    phase_names = [
        "Phase I: Ingestion & Schema",
        "Phase I-A: Duplicate Detection",
        "Phase II: Scope Definition",
        "Phase III: Data Sanitation",
        "Phase IV: Outlier Handling",
        "Phase IV-A: Multicollinearity",
        "Phase V: Feature Engineering"
    ]

    # Phase navigation
    selected_phase = st.radio(
        "Navigate to:",
        range(len(phase_names)),
        format_func=lambda x: phase_names[x],
        index=st.session_state.current_phase
    )

    st.session_state.current_phase = selected_phase

    st.divider()

    # Tools section
    st.subheader("Tools")

    # Distribution Inspector
    if st.button("Distribution Inspector", use_container_width=True):
        st.session_state.show_inspector = not st.session_state.show_inspector
        st.session_state.show_rollback = False

    # Rollback interface
    if st.button("Pipeline History", use_container_width=True):
        st.session_state.show_rollback = not st.session_state.show_rollback
        st.session_state.show_inspector = False

    st.divider()

    # Current state info
    st.subheader("Current State")

    state_manager = st.session_state.state_manager
    current_state = state_manager.get_current_state_id()

    if current_state:
        st.info(f"State: `{current_state}`")
        st.caption(f"Branch: {state_manager.active_branch}")
    else:
        st.caption("No state committed yet")

    # Data preview
    if 'df_schema' in st.session_state:
        current_df = None
        if 'df_final' in st.session_state:
            current_df = st.session_state['df_final']
        elif 'df_collinear_resolved' in st.session_state:
            current_df = st.session_state['df_collinear_resolved']
        elif 'df_outlier_handled' in st.session_state:
            current_df = st.session_state['df_outlier_handled']
        elif 'df_clean' in st.session_state:
            current_df = st.session_state['df_clean']
        elif 'df_scoped' in st.session_state:
            current_df = st.session_state['df_scoped']
        elif 'df_deduplicated' in st.session_state:
            current_df = st.session_state['df_deduplicated']
        else:
            current_df = st.session_state['df_schema']

        if current_df is not None:
            st.caption(f"{len(current_df):,} rows, {len(current_df.columns)} cols")


# Main content area
if st.session_state.show_inspector:
    # Distribution Inspector modal
    st.markdown("---")

    # Get current DataFrame
    current_df = None
    state_name = "current"

    if 'df_final' in st.session_state:
        current_df = st.session_state['df_final']
        state_name = "df_final"
    elif 'df_collinear_resolved' in st.session_state:
        current_df = st.session_state['df_collinear_resolved']
        state_name = "df_collinear_resolved"
    elif 'df_outlier_handled' in st.session_state:
        current_df = st.session_state['df_outlier_handled']
        state_name = "df_outlier_handled"
    elif 'df_clean' in st.session_state:
        current_df = st.session_state['df_clean']
        state_name = "df_clean"
    elif 'df_scoped' in st.session_state:
        current_df = st.session_state['df_scoped']
        state_name = "df_scoped"
    elif 'df_deduplicated' in st.session_state:
        current_df = st.session_state['df_deduplicated']
        state_name = "df_deduplicated"
    elif 'df_schema' in st.session_state:
        current_df = st.session_state['df_schema']
        state_name = "df_schema"

    if current_df is not None:
        render_distribution_inspector(current_df, state_name)
    else:
        st.warning("No data available. Please upload a file first.")

    if st.button("Close Inspector"):
        st.session_state.show_inspector = False
        st.rerun()

elif st.session_state.show_rollback:
    # Rollback interface
    st.markdown("---")
    render_rollback_interface()

    if st.button("Close History"):
        st.session_state.show_rollback = False
        st.rerun()

else:
    # Phase routing
    if selected_phase == 0:
        phase_1_ingestion.render()
    elif selected_phase == 1:
        phase_1a_duplicates.render()
    elif selected_phase == 2:
        phase_2_scope.render()
    elif selected_phase == 3:
        phase_3_sanitation.render()
    elif selected_phase == 4:
        phase_4_outliers.render()
    elif selected_phase == 5:
        phase_4a_multicollinearity.render()
    elif selected_phase == 6:
        phase_5_feature_eng.render()


# Footer
st.markdown("---")

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown("**Data processed in-memory only.** No storage, no tracking.")

with col2:
    if st.button("Privacy Policy"):
        st.session_state.show_privacy = not st.session_state.show_privacy

with col3:
    if st.button("Delete All Data"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Clear temp files
        cleanup_temp_files()

        st.success("All data deleted. Redirecting...")
        st.rerun()


# Privacy policy modal
if st.session_state.show_privacy:
    st.markdown("---")
    st.subheader("Privacy Policy")

    st.markdown("""
    ### Data Processing

    **What We Process:**
    - The file(s) you upload for analysis
    - Configuration choices you make during the pipeline

    ### Data Storage

    **In-Memory Only:** All data processing occurs exclusively in your browser's memory (RAM). No data is:
    - Saved to servers
    - Logged to files
    - Transmitted to third parties

    **Session-Based:** When you close the browser tab, all data is deleted.

    ### Your Rights

    Under GDPR, you have the right to:
    - **Access:** View all data being processed (visible in the UI)
    - **Rectification:** Modify data during the pipeline
    - **Erasure:** Delete all data via "Delete All Data" button
    - **Portability:** Export your processed data at any phase

    ### Contact

    For privacy-related questions, please refer to the repository documentation.
    """)

    if st.button("Close Privacy Policy"):
        st.session_state.show_privacy = False
        st.rerun()
