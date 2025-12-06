"""Phase IV-A: Multicollinearity Pre-Screening."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def render():
    """Render Phase IV-A: Multicollinearity Screening."""
    st.header("Phase IV-A: Multicollinearity Pre-Screening")
    st.markdown("""
    **Objective:** Identify and resolve redundant features before correlation analysis.

    Highly correlated features:
    - Inflate model complexity unnecessarily
    - Make correlation matrices difficult to interpret
    - Violate regression assumptions
    """)

    # Check if previous phase is complete
    if 'df_outlier_handled' not in st.session_state:
        st.warning("Please complete Phase IV first.")
        if st.button("Go to Phase IV"):
            st.session_state.current_phase = 4
            st.rerun()
        return

    df = st.session_state['df_outlier_handled'].copy()

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        st.info("Need at least 2 numerical columns for multicollinearity analysis.")
        if st.button("Skip to Phase V"):
            st.session_state['df_collinear_resolved'] = df
            st.session_state.state_manager.commit_state(
                state_id="collinear_resolved",
                df=df,
                config_snapshot=st.session_state.pipeline_config.get_snapshot(),
                delta_summary="Insufficient columns - skipped multicollinearity screening"
            )
            st.session_state.current_phase = 6
            st.rerun()
        return

    # Tabs for VIF and Correlation Matrix
    tab1, tab2 = st.tabs(["VIF Analysis", "Correlation Matrix"])

    with tab1:
        render_vif_analysis(df, numerical_cols)

    with tab2:
        render_correlation_matrix(df, numerical_cols)

    # Resolution
    st.subheader("Resolution")

    removed_cols = st.multiselect(
        "Select columns to remove (high VIF or high correlation):",
        numerical_cols,
        key="remove_collinear_cols"
    )

    if removed_cols:
        st.warning(f"Will remove {len(removed_cols)} columns: {', '.join(removed_cols)}")

    # Apply and proceed
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Apply Resolution & Proceed", type="primary"):
            from utilities.config import MulticollinearityConfig

            result_df = df.drop(columns=removed_cols) if removed_cols else df

            # Update config
            st.session_state.pipeline_config.multicollinearity = MulticollinearityConfig(
                vif_threshold=10.0,
                auto_remove=False,
                removed_columns=removed_cols
            )

            # Save result
            st.session_state['df_collinear_resolved'] = result_df

            # Commit state
            st.session_state.state_manager.commit_state(
                state_id="collinear_resolved",
                df=result_df,
                config_snapshot=st.session_state.pipeline_config.get_snapshot(),
                delta_summary=f"Removed {len(removed_cols)} collinear columns" if removed_cols else "No columns removed"
            )

            st.success("Multicollinearity handled! Proceed to Phase V: Feature Engineering")
            st.session_state.current_phase = 6
            st.rerun()

    with col2:
        if st.button("Skip (Keep All Columns)"):
            st.session_state['df_collinear_resolved'] = df

            st.session_state.state_manager.commit_state(
                state_id="collinear_resolved",
                df=df,
                config_snapshot=st.session_state.pipeline_config.get_snapshot(),
                delta_summary="Kept all columns - no multicollinearity resolution"
            )

            st.session_state.current_phase = 6
            st.rerun()


def render_vif_analysis(df: pd.DataFrame, numerical_cols: list):
    """Render VIF analysis section."""
    st.subheader("Variance Inflation Factor (VIF)")
    st.markdown("""
    VIF measures how much a variable's variance is inflated by correlation with other variables.

    | VIF Value | Interpretation |
    |-----------|----------------|
    | 1 | No correlation |
    | 1-5 | Moderate correlation |
    | 5-10 | High correlation |
    | > 10 | **Severe multicollinearity** |
    """)

    # VIF threshold selection
    vif_threshold = st.slider(
        "VIF Threshold:",
        min_value=5.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
        help="Variables with VIF above this threshold are flagged"
    )

    # Calculate VIF
    try:
        # Prepare data - drop rows with any NaN
        df_clean = df[numerical_cols].dropna()

        if len(df_clean) < len(numerical_cols) + 1:
            st.error("Insufficient data for VIF calculation after removing missing values.")
            return

        # Add constant for VIF calculation
        from statsmodels.tools.tools import add_constant
        X = add_constant(df_clean)

        vif_data = []
        for i, col in enumerate(numerical_cols):
            try:
                vif = variance_inflation_factor(X.values, i + 1)  # +1 because of constant
                status = "No issue" if vif <= 5 else "Acceptable" if vif <= vif_threshold else "High VIF" if vif <= 20 else "Severe VIF"

                vif_data.append({
                    'Variable': col,
                    'VIF': f"{vif:.2f}",
                    'Status': status
                })
            except Exception as e:
                vif_data.append({
                    'Variable': col,
                    'VIF': "Error",
                    'Status': str(e)[:30]
                })

        vif_df = pd.DataFrame(vif_data)

        # Color coding
        def highlight_vif(row):
            if row['Status'] == "No issue":
                return ['background-color: lightgreen'] * len(row)
            elif row['Status'] == "Acceptable":
                return ['background-color: lightyellow'] * len(row)
            elif row['Status'] == "High VIF":
                return ['background-color: orange'] * len(row)
            elif row['Status'] == "Severe VIF":
                return ['background-color: salmon'] * len(row)
            return [''] * len(row)

        st.dataframe(
            vif_df.style.apply(highlight_vif, axis=1),
            hide_index=True,
            use_container_width=True
        )

        # Recommendations
        high_vif_cols = [row['Variable'] for _, row in vif_df.iterrows()
                        if row['VIF'] != "Error" and float(row['VIF']) > vif_threshold]

        if high_vif_cols:
            st.warning(f"**Recommendation:** Consider removing: {', '.join(high_vif_cols)}")

    except Exception as e:
        st.error(f"Error calculating VIF: {str(e)}")


def render_correlation_matrix(df: pd.DataFrame, numerical_cols: list):
    """Render correlation matrix heatmap."""
    st.subheader("Correlation Matrix Heatmap")

    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()

    # Threshold for flagging
    corr_threshold = st.slider(
        "High Correlation Threshold:",
        min_value=0.5,
        max_value=0.99,
        value=0.85,
        step=0.05,
        help="Pairs with |r| above this threshold are flagged"
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        annot_kws={'size': 8}
    )

    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Find high correlations
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > corr_threshold:
                high_corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': f"{corr_val:.3f}"
                })

    if high_corr_pairs:
        st.markdown(f"**Highly Correlated Pairs (|r| > {corr_threshold}):**")
        high_corr_df = pd.DataFrame(high_corr_pairs)
        st.dataframe(high_corr_df, hide_index=True, use_container_width=True)
    else:
        st.success(f"No variable pairs with |r| > {corr_threshold}")
