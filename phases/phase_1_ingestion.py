"""Phase I: Ingestion & Schema Enforcement."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any


def render():
    """Render Phase I: Ingestion & Schema Enforcement."""
    st.header("Phase I: Ingestion & Schema Enforcement")
    st.markdown("""
    **Objective:** Correctly interpret raw data into a structured DataFrame with semantically correct data types.
    """)

    # File Upload
    st.subheader("1. Upload Data File")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'parquet', 'json'],
        help="Supported formats: CSV, Excel (.xlsx, .xls), Parquet, JSON"
    )

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        if df is not None:
            st.session_state['df_raw'] = df
            st.success(f"File loaded successfully: {uploaded_file.name}")

            # Initial Profiling
            render_initial_profiling(df)

            # Schema Inference & Validation
            st.subheader("2. Schema Inference & Validation")
            df_schema = render_schema_editor(df)

            if df_schema is not None:
                st.session_state['df_schema'] = df_schema

                # Commit state
                if st.button("Confirm Schema & Proceed", type="primary"):
                    from utilities.config import SchemaConfig

                    # Get type overrides
                    type_overrides = {}
                    for col in df_schema.columns:
                        original_type = str(df[col].dtype)
                        new_type = str(df_schema[col].dtype)
                        if original_type != new_type:
                            type_overrides[col] = new_type

                    # Update config
                    st.session_state.pipeline_config.schema = SchemaConfig(
                        type_overrides=type_overrides
                    )

                    # Commit state
                    st.session_state.state_manager.commit_state(
                        state_id="schema",
                        df=df_schema,
                        config_snapshot=st.session_state.pipeline_config.get_snapshot(),
                        delta_summary=f"Schema enforced with {len(type_overrides)} type overrides"
                    )

                    st.success("Schema confirmed! Proceed to Phase I-A: Duplicate Detection")
                    st.session_state.current_phase = 1
                    st.rerun()


def load_data(uploaded_file) -> pd.DataFrame:
    """Load data from uploaded file."""
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext == 'csv':
            # Try to detect encoding and delimiter
            df = pd.read_csv(uploaded_file)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'parquet':
            df = pd.read_parquet(uploaded_file)
        elif file_ext == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def render_initial_profiling(df: pd.DataFrame):
    """Display initial profiling dashboard."""
    st.subheader("Initial Profiling")

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")

    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")

    # Type distribution
    st.markdown("**Type Distribution**")
    type_counts = df.dtypes.value_counts()
    type_df = pd.DataFrame({
        'Type': type_counts.index.astype(str),
        'Count': type_counts.values
    })
    st.dataframe(type_df, hide_index=True, use_container_width=True)

    # Sample rows
    st.markdown("**Sample Data (First 5 Rows)**")
    st.dataframe(df.head(), use_container_width=True)


def render_schema_editor(df: pd.DataFrame) -> pd.DataFrame:
    """Render editable schema table and return modified DataFrame."""
    st.markdown("""
    **Review and adjust data types:**
    - Categorical Integers (e.g., Zip Codes, IDs) should be cast to String/Category
    - DateTime columns should be properly parsed
    - Boolean-like columns (0/1, Y/N) should be cast appropriately
    """)

    # Build schema information
    schema_data = []
    for col in df.columns:
        inferred_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"

        schema_data.append({
            'Column': col,
            'Inferred Type': inferred_type,
            'Unique Values': unique_count,
            'Sample': str(sample_val)[:50]
        })

    schema_df = pd.DataFrame(schema_data)

    # Display current schema
    st.dataframe(schema_df, hide_index=True, use_container_width=True)

    # Type override interface
    st.markdown("**Override Column Types**")

    type_options = ['(keep current)', 'int64', 'float64', 'object', 'category', 'datetime64', 'bool']

    df_modified = df.copy()
    type_overrides = {}

    # Create columns for type selection
    cols_per_row = 3
    columns = list(df.columns)

    for i in range(0, len(columns), cols_per_row):
        row_cols = st.columns(cols_per_row)

        for j, col_widget in enumerate(row_cols):
            col_idx = i + j
            if col_idx < len(columns):
                col_name = columns[col_idx]

                with col_widget:
                    new_type = st.selectbox(
                        f"{col_name}",
                        type_options,
                        key=f"type_{col_name}",
                        help=f"Current: {df[col_name].dtype}"
                    )

                    if new_type != '(keep current)':
                        type_overrides[col_name] = new_type

    # Apply type overrides
    if type_overrides:
        st.markdown("**Pending Type Changes:**")
        for col, new_type in type_overrides.items():
            try:
                if new_type == 'datetime64':
                    df_modified[col] = pd.to_datetime(df_modified[col], errors='coerce')
                elif new_type == 'category':
                    df_modified[col] = df_modified[col].astype('category')
                elif new_type == 'bool':
                    df_modified[col] = df_modified[col].astype(bool)
                else:
                    df_modified[col] = df_modified[col].astype(new_type)
                st.success(f"`{col}`: {df[col].dtype} -> {new_type}")
            except Exception as e:
                st.error(f"`{col}`: Failed to convert - {str(e)}")

    return df_modified


def get_type_recommendations(df: pd.DataFrame) -> Dict[str, str]:
    """Generate type recommendations based on data analysis."""
    recommendations = {}

    for col in df.columns:
        # Check for potential categorical integers (IDs, codes)
        if df[col].dtype in ['int64', 'int32']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:  # High uniqueness - likely ID
                recommendations[col] = "Consider: category or string (likely ID)"

        # Check for potential datetime strings
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[:100] if len(df[col].dropna()) >= 100 else df[col].dropna()
            try:
                pd.to_datetime(sample)
                recommendations[col] = "Consider: datetime64"
            except:
                pass

        # Check for potential booleans
        if df[col].dtype in ['int64', 'int32', 'float64']:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
                recommendations[col] = "Consider: bool"

    return recommendations
