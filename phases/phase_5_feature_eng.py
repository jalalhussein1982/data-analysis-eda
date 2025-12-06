"""Phase V: The Pre-Correlation Bridge (Feature Engineering)."""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


def render():
    """Render Phase V: Feature Engineering."""
    st.header("Phase V: The 'Pre-Correlation' Bridge (Feature Engineering)")
    st.markdown("""
    **Objective:** Transform categorical variables and scale numerical variables to prepare for correlation analysis.
    """)

    # Check if previous phase is complete
    if 'df_collinear_resolved' not in st.session_state:
        st.warning("Please complete Phase IV-A first.")
        if st.button("Go to Phase IV-A"):
            st.session_state.current_phase = 5
            st.rerun()
        return

    df = st.session_state['df_collinear_resolved'].copy()

    # Tabs for encoding and scaling
    tab1, tab2 = st.tabs(["Step A: Categorical Encoding", "Step B: Numerical Scaling"])

    encoding_configs = []
    scaling_config = None

    with tab1:
        df, encoding_configs = render_categorical_encoding(df)

    with tab2:
        df, scaling_config = render_numerical_scaling(df)

    # Summary
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Final Rows", f"{len(df):,}")

    with col2:
        st.metric("Final Columns", len(df.columns))

    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")

    # Data types breakdown
    st.markdown("**Final Column Types:**")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        st.markdown(f"- {dtype}: {count}")

    # Finalize and Export
    st.subheader("Finalize Pipeline")

    if st.button("Finalize & Create df_final", type="primary"):
        from utilities.config import EncodingConfig, ScalingConfig

        # Update config
        st.session_state.pipeline_config.encoding = encoding_configs
        st.session_state.pipeline_config.scaling = scaling_config

        # Save result
        st.session_state['df_final'] = df

        # Commit state
        st.session_state.state_manager.commit_state(
            state_id="final",
            df=df,
            config_snapshot=st.session_state.pipeline_config.get_snapshot(),
            delta_summary=f"Feature engineering complete: {len(df.columns)} columns"
        )

        st.success("Pipeline complete! df_final is ready for correlation analysis.")

        # Show recommendation
        st.info("""
        **Recommended Next Step:** Review variable distributions to:
        - Verify normality assumptions for Pearson correlation
        - Identify variables requiring Spearman's rho instead
        - Detect potential data quality issues before analysis

        Use the **Distribution Inspector** in the sidebar to review.
        """)

        # Export options
        st.subheader("Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="df_final.csv",
                mime="text/csv"
            )

        with col2:
            import json
            config_json = json.dumps(
                st.session_state.pipeline_config.to_dict(),
                indent=2
            )
            st.download_button(
                label="Download Config (JSON)",
                data=config_json,
                file_name="pipeline_config.json",
                mime="application/json"
            )

        with col3:
            if st.button("View Distribution Inspector"):
                st.session_state.show_inspector = True
                st.rerun()


def render_categorical_encoding(df: pd.DataFrame) -> tuple:
    """Render categorical encoding interface."""
    st.subheader("Step A: Categorical Encoding")
    st.markdown("""
    Correlation requires numerical inputs. Categorical variables must be encoded.

    | Method | Output | Use Case |
    |--------|--------|----------|
    | Label Encoding | Single column | Ordinal categories |
    | One-Hot Encoding | Multiple columns | Nominal categories |
    | Frequency Encoding | Single column | High-cardinality |
    """)

    encoding_configs = []

    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        st.success("No categorical columns found - no encoding needed.")
        return df, encoding_configs

    # Display categorical columns
    st.markdown("**Categorical Columns Found:**")

    cat_info = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        cat_info.append({
            'Column': col,
            'Unique Values': unique_count,
            'Sample Values': ', '.join(map(str, df[col].dropna().unique()[:5]))
        })

    cat_df = pd.DataFrame(cat_info)
    st.dataframe(cat_df, hide_index=True, use_container_width=True)

    # Encoding configuration
    st.markdown("**Configure Encoding:**")

    for col in categorical_cols:
        unique_count = df[col].nunique()

        with st.expander(f"{col} ({unique_count} unique values)"):
            encoding_method = st.selectbox(
                f"Encoding method for {col}:",
                ["One-Hot Encoding (Recommended)", "Label Encoding", "Frequency Encoding", "Skip (Drop Column)"],
                key=f"encode_{col}"
            )

            if st.button(f"Apply Encoding to {col}", key=f"apply_encode_{col}"):
                from utilities.config import EncodingConfig

                if encoding_method == "One-Hot Encoding (Recommended)":
                    # One-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    encoding_configs.append(EncodingConfig(column=col, method="onehot"))
                    st.success(f"One-hot encoded {col} into {len(dummies.columns)} columns")

                elif encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoding_configs.append(EncodingConfig(column=col, method="label"))
                    st.success(f"Label encoded {col}")

                elif encoding_method == "Frequency Encoding":
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map)
                    encoding_configs.append(EncodingConfig(column=col, method="frequency"))
                    st.success(f"Frequency encoded {col}")

                elif encoding_method == "Skip (Drop Column)":
                    df = df.drop(columns=[col])
                    st.success(f"Dropped {col}")

                st.rerun()

    # Quick encode all
    st.markdown("---")

    if st.button("Quick Encode: One-Hot All Categorical"):
        from utilities.config import EncodingConfig

        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoding_configs.append(EncodingConfig(column=col, method="onehot"))

        st.success(f"One-hot encoded {len(categorical_cols)} categorical columns")
        st.rerun()

    return df, encoding_configs


def render_numerical_scaling(df: pd.DataFrame) -> tuple:
    """Render numerical scaling interface."""
    st.subheader("Step B: Numerical Scaling")
    st.markdown("""
    Scaling ensures variables with different scales are comparable.

    | Method | Formula | Use Case |
    |--------|---------|----------|
    | Standardization | z = (x - mean) / std | Default - preserves distribution |
    | Min-Max | (x - min) / (max - min) | Need bounded [0,1] range |
    | Robust | (x - median) / IQR | Resistant to outliers |

    **Note:** Standardization does NOT change Pearson correlation coefficients.
    """)

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        st.info("No numerical columns found for scaling.")
        return df, None

    # Scaling method selection
    scaling_method = st.radio(
        "Scaling Method:",
        ["Standardization (Z-score)", "Min-Max Normalization", "Robust Scaling", "No Scaling"],
        help="Standardization is recommended for correlation analysis"
    )

    if scaling_method == "No Scaling":
        from utilities.config import ScalingConfig
        return df, ScalingConfig(method="none", columns=[])

    # Column selection
    cols_to_scale = st.multiselect(
        "Select columns to scale:",
        numerical_cols,
        default=numerical_cols,
        key="scale_cols"
    )

    if not cols_to_scale:
        st.info("Select columns to scale or choose 'No Scaling'.")
        return df, None

    # Preview
    st.markdown("**Scaling Preview:**")

    preview_data = []
    for col in cols_to_scale:
        preview_data.append({
            'Column': col,
            'Original Min': f"{df[col].min():,.2f}",
            'Original Max': f"{df[col].max():,.2f}",
            'Original Mean': f"{df[col].mean():,.2f}"
        })

    preview_df = pd.DataFrame(preview_data)
    st.dataframe(preview_df, hide_index=True, use_container_width=True)

    # Apply scaling
    if st.button("Apply Scaling"):
        from utilities.config import ScalingConfig

        if scaling_method == "Standardization (Z-score)":
            scaler = StandardScaler()
            method = "standard"
        elif scaling_method == "Min-Max Normalization":
            scaler = MinMaxScaler()
            method = "minmax"
        else:
            scaler = RobustScaler()
            method = "robust"

        # Apply scaler
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        st.success(f"Applied {scaling_method} to {len(cols_to_scale)} columns")

        # Show result preview
        st.markdown("**After Scaling:**")
        result_data = []
        for col in cols_to_scale:
            result_data.append({
                'Column': col,
                'New Min': f"{df[col].min():,.4f}",
                'New Max': f"{df[col].max():,.4f}",
                'New Mean': f"{df[col].mean():,.4f}"
            })

        result_df = pd.DataFrame(result_data)
        st.dataframe(result_df, hide_index=True, use_container_width=True)

        return df, ScalingConfig(method=method, columns=cols_to_scale)

    return df, None
