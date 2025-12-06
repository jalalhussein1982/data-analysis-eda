"""Phase IV: Distribution Analysis & Outlier Handling."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def render():
    """Render Phase IV: Outlier Handling."""
    st.header("Phase IV: Distribution Analysis & Outlier Handling")
    st.markdown("""
    **Objective:** Detect and resolve extreme values that distort statistical measures.
    """)

    # Check if previous phase is complete
    if 'df_clean' not in st.session_state:
        st.warning("Please complete Phase III first.")
        if st.button("Go to Phase III"):
            st.session_state.current_phase = 3
            st.rerun()
        return

    df = st.session_state['df_clean'].copy()

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        st.warning("No numerical columns found for outlier analysis.")
        if st.button("Skip to Phase IV-A"):
            st.session_state['df_outlier_handled'] = df
            st.session_state.state_manager.commit_state(
                state_id="outlier_handled",
                df=df,
                config_snapshot=st.session_state.pipeline_config.get_snapshot(),
                delta_summary="No numerical columns - skipped outlier handling"
            )
            st.session_state.current_phase = 5
            st.rerun()
        return

    # Outlier configurations
    outlier_configs = []

    # Detection method selection
    st.subheader("1. Select Detection Method")

    detection_method = st.radio(
        "Detection Method:",
        ["IQR (Interquartile Range)", "Z-Score", "Percentile"],
        help="IQR is recommended for skewed distributions"
    )

    # Method-specific parameters
    if detection_method == "IQR (Interquartile Range)":
        iqr_multiplier = st.slider(
            "IQR Multiplier:",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Values beyond Q1 - k*IQR or Q3 + k*IQR are outliers. Default is 1.5"
        )
    elif detection_method == "Z-Score":
        z_threshold = st.slider(
            "Z-Score Threshold:",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Values with |z| > threshold are outliers. Default is 3"
        )
    else:  # Percentile
        col1, col2 = st.columns(2)
        with col1:
            lower_pct = st.number_input("Lower Percentile:", value=1.0, min_value=0.0, max_value=50.0)
        with col2:
            upper_pct = st.number_input("Upper Percentile:", value=99.0, min_value=50.0, max_value=100.0)

    # Column selection
    st.subheader("2. Select Columns for Outlier Detection")

    selected_cols = st.multiselect(
        "Select numerical columns:",
        numerical_cols,
        default=numerical_cols,
        key="outlier_cols"
    )

    if not selected_cols:
        st.info("Please select at least one column.")
        return

    # Detect and display outliers
    st.subheader("3. Outlier Detection Results")

    outlier_summary = []

    for col in selected_cols:
        data = df[col].dropna()

        if detection_method == "IQR (Interquartile Range)":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif detection_method == "Z-Score":
            z_scores = np.abs(stats.zscore(data))
            outliers = pd.Series(np.abs(stats.zscore(df[col].fillna(df[col].median()))) > z_threshold, index=df.index)
            lower_bound = data.mean() - z_threshold * data.std()
            upper_bound = data.mean() + z_threshold * data.std()

        else:  # Percentile
            lower_bound = data.quantile(lower_pct / 100)
            upper_bound = data.quantile(upper_pct / 100)
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        outlier_count = outliers.sum()
        outlier_pct = (outlier_count / len(df)) * 100

        outlier_summary.append({
            'Column': col,
            'Outliers': outlier_count,
            'Percentage': f"{outlier_pct:.1f}%",
            'Lower Bound': f"{lower_bound:,.2f}",
            'Upper Bound': f"{upper_bound:,.2f}"
        })

    summary_df = pd.DataFrame(outlier_summary)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Visualization
    st.subheader("4. Outlier Visualization")

    viz_col = st.selectbox("Select column to visualize:", selected_cols)

    if viz_col:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Boxplot
        axes[0].boxplot(df[viz_col].dropna())
        axes[0].set_title(f'Boxplot: {viz_col}')
        axes[0].set_ylabel(viz_col)

        # Histogram
        axes[1].hist(df[viz_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=float(summary_df[summary_df['Column'] == viz_col]['Lower Bound'].iloc[0].replace(',', '')),
                       color='red', linestyle='--', label='Lower Bound')
        axes[1].axvline(x=float(summary_df[summary_df['Column'] == viz_col]['Upper Bound'].iloc[0].replace(',', '')),
                       color='red', linestyle='--', label='Upper Bound')
        axes[1].set_title(f'Distribution: {viz_col}')
        axes[1].set_xlabel(viz_col)
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Resolution strategy
    st.subheader("5. Outlier Resolution")

    resolution_strategy = st.radio(
        "Resolution Strategy:",
        ["Winsorize (Cap at bounds)", "Drop Outlier Rows", "Keep Outliers (No Action)"],
        help="Winsorization caps values at the boundary, preserving sample size"
    )

    # Preview impact
    if resolution_strategy == "Winsorize (Cap at bounds)":
        preview_df = df.copy()

        for col in selected_cols:
            data = df[col].dropna()

            if detection_method == "IQR (Interquartile Range)":
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
            elif detection_method == "Z-Score":
                lower_bound = data.mean() - z_threshold * data.std()
                upper_bound = data.mean() + z_threshold * data.std()
            else:
                lower_bound = data.quantile(lower_pct / 100)
                upper_bound = data.quantile(upper_pct / 100)

            preview_df[col] = preview_df[col].clip(lower=lower_bound, upper=upper_bound)

        # Show before/after stats
        st.markdown("**Impact Preview:**")

        impact_data = []
        for col in selected_cols:
            impact_data.append({
                'Column': col,
                'Original Skew': f"{df[col].skew():.2f}",
                'New Skew': f"{preview_df[col].skew():.2f}",
                'Original Std': f"{df[col].std():,.2f}",
                'New Std': f"{preview_df[col].std():,.2f}"
            })

        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, hide_index=True, use_container_width=True)

    elif resolution_strategy == "Drop Outlier Rows":
        outlier_mask = pd.Series(False, index=df.index)

        for col in selected_cols:
            data = df[col].dropna()

            if detection_method == "IQR (Interquartile Range)":
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
            elif detection_method == "Z-Score":
                lower_bound = data.mean() - z_threshold * data.std()
                upper_bound = data.mean() + z_threshold * data.std()
            else:
                lower_bound = data.quantile(lower_pct / 100)
                upper_bound = data.quantile(upper_pct / 100)

            outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)

        rows_to_drop = outlier_mask.sum()
        st.warning(f"This will remove {rows_to_drop} rows ({rows_to_drop/len(df)*100:.1f}% of data)")

    # Apply and proceed
    if st.button("Apply Outlier Resolution & Proceed", type="primary"):
        from utilities.config import OutlierConfig

        result_df = df.copy()

        for col in selected_cols:
            data = df[col].dropna()

            if detection_method == "IQR (Interquartile Range)":
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
                detection_params = {"multiplier": iqr_multiplier}
            elif detection_method == "Z-Score":
                lower_bound = data.mean() - z_threshold * data.std()
                upper_bound = data.mean() + z_threshold * data.std()
                detection_params = {"threshold": z_threshold}
            else:
                lower_bound = data.quantile(lower_pct / 100)
                upper_bound = data.quantile(upper_pct / 100)
                detection_params = {"lower_pct": lower_pct, "upper_pct": upper_pct}

            if resolution_strategy == "Winsorize (Cap at bounds)":
                result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
                resolution = "winsorize"
            elif resolution_strategy == "Drop Outlier Rows":
                outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                result_df = result_df[~outliers]
                resolution = "drop"
            else:
                resolution = "keep"

            method_map = {
                "IQR (Interquartile Range)": "iqr",
                "Z-Score": "zscore",
                "Percentile": "percentile"
            }

            outlier_configs.append(OutlierConfig(
                column=col,
                detection_method=method_map[detection_method],
                detection_params=detection_params,
                resolution_strategy=resolution
            ))

        # Update config
        st.session_state.pipeline_config.outliers = outlier_configs

        # Save result
        st.session_state['df_outlier_handled'] = result_df

        # Commit state
        delta_msg = f"Applied {resolution_strategy} to {len(selected_cols)} columns"
        if resolution_strategy == "Drop Outlier Rows":
            delta_msg += f", removed {len(df) - len(result_df)} rows"

        st.session_state.state_manager.commit_state(
            state_id="outlier_handled",
            df=result_df,
            config_snapshot=st.session_state.pipeline_config.get_snapshot(),
            delta_summary=delta_msg
        )

        st.success(f"Outliers handled! Proceed to Phase IV-A: Multicollinearity Screening")
        st.session_state.current_phase = 5
        st.rerun()
