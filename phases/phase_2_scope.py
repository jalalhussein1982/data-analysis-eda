"""Phase II: Scope Definition (Dimensionality Reduction)."""

import streamlit as st
import pandas as pd
import numpy as np


def render():
    """Render Phase II: Scope Definition."""
    st.header("Phase II: Scope Definition (Dimensionality Reduction)")
    st.markdown("""
    **Objective:** Remove irrelevant features to improve processing speed and reduce noise.
    """)

    # Check if previous phase is complete
    if 'df_deduplicated' not in st.session_state:
        st.warning("Please complete Phase I-A first.")
        if st.button("Go to Phase I-A"):
            st.session_state.current_phase = 1
            st.rerun()
        return

    df = st.session_state['df_deduplicated'].copy()

    # Column Analysis
    st.subheader("1. Column Analysis")

    # Build column metadata
    column_data = []
    recommendations = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df) if len(df) > 0 else 0
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0

        # Determine flags
        flags = []

        # Zero variance
        if unique_count == 1:
            flags.append("Zero Variance")
            recommendations.append(col)

        # Near-zero variance (>95% single value)
        if unique_count > 1:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.95:
                flags.append("Near-Zero Variance")

        # High cardinality / potential ID
        if unique_ratio > 0.9 and dtype == 'object':
            flags.append("High Cardinality")
        elif unique_count == len(df) and dtype in ['int64', 'int32']:
            flags.append("Potential ID")
            recommendations.append(col)

        column_data.append({
            'Column': col,
            'Type': dtype,
            'Unique': unique_count,
            'Missing %': f"{missing_pct:.1f}%",
            'Flags': ', '.join(flags) if flags else '-'
        })

    column_df = pd.DataFrame(column_data)
    st.dataframe(column_df, hide_index=True, use_container_width=True)

    # Recommendations
    if recommendations:
        st.info(f"**Recommended for removal:** {', '.join(recommendations)}")

    # Column Selection
    st.subheader("2. Select Columns for Analysis")

    # Use a simple multiselect instead of checkboxes - much more reliable
    all_columns = df.columns.tolist()

    # Initialize default selection
    if 'phase2_default_selection' not in st.session_state:
        st.session_state.phase2_default_selection = [col for col in all_columns if col not in recommendations]

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Select All"):
            st.session_state.phase2_default_selection = all_columns.copy()
            st.rerun()

    with col2:
        if st.button("Deselect All"):
            st.session_state.phase2_default_selection = []
            st.rerun()

    with col3:
        if st.button("Apply Recommendations"):
            st.session_state.phase2_default_selection = [col for col in all_columns if col not in recommendations]
            st.rerun()

    # Multiselect for column selection
    selected_cols = st.multiselect(
        "Select columns to keep:",
        options=all_columns,
        default=st.session_state.phase2_default_selection,
        help="Select the columns you want to include in the analysis"
    )

    # Update the default for next time
    st.session_state.phase2_default_selection = selected_cols

    # Preview
    st.subheader("3. Preview Selection")

    dropped_cols = [col for col in all_columns if col not in selected_cols]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Selected Columns", len(selected_cols))

    with col2:
        st.metric("Dropped Columns", len(dropped_cols))

    if selected_cols:
        st.markdown(f"**Columns to keep:** {', '.join(selected_cols)}")

    if dropped_cols:
        st.markdown(f"**Columns to drop:** {', '.join(dropped_cols)}")

    if not selected_cols:
        st.error("Please select at least one column to continue.")
        return

    # Apply and proceed
    if st.button("Apply Selection & Proceed", type="primary"):
        from utilities.config import ScopeConfig

        # Create scoped DataFrame
        df_scoped = df[selected_cols].copy()

        # Update config
        st.session_state.pipeline_config.scope = ScopeConfig(
            selected_columns=selected_cols,
            dropped_columns=dropped_cols
        )

        # Save result
        st.session_state['df_scoped'] = df_scoped

        # Commit state
        st.session_state.state_manager.commit_state(
            state_id="scoped",
            df=df_scoped,
            config_snapshot=st.session_state.pipeline_config.get_snapshot(),
            delta_summary=f"Dropped {len(dropped_cols)} columns, kept {len(selected_cols)}"
        )

        st.success(f"Scope defined! Dropped {len(dropped_cols)} columns. Proceed to Phase III: Data Sanitation")
        st.session_state.current_phase = 3
        st.rerun()
