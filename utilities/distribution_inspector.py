"""Distribution Inspector utility for visualizing and analyzing variable distributions."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from .normality_tests import (
    run_normality_test,
    get_skewness_label,
    get_kurtosis_label,
    compute_distribution_stats
)


def render_distribution_inspector(df: pd.DataFrame, state_name: str = "current"):
    """
    Render the Distribution Inspector interface.

    Args:
        df: DataFrame to inspect
        state_name: Name of the current state for display
    """
    st.subheader(f"Distribution Inspector - {state_name}")

    if df.empty:
        st.warning("No data available. Please upload and process a file first.")
        return

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        st.warning("No numerical columns found in the dataset.")
        return

    # Initialize default selection if not exists
    if 'dist_inspector_default' not in st.session_state:
        st.session_state.dist_inspector_default = numerical_cols[:min(4, len(numerical_cols))]

    # Column selection
    st.markdown("**Select variables to inspect (1-50 recommended):**")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col2:
        if st.button("Select All Numerical"):
            st.session_state.dist_inspector_default = numerical_cols[:50]
            st.rerun()

    with col3:
        if st.button("Clear Selection"):
            st.session_state.dist_inspector_default = []
            st.rerun()

    with col1:
        selected_cols = st.multiselect(
            "Variables",
            numerical_cols,
            default=st.session_state.dist_inspector_default,
            key="dist_inspector_cols"
        )

    if not selected_cols:
        st.info("Please select at least one variable to inspect.")
        return

    # Display mode selection
    view_mode = st.radio(
        "View Mode",
        ["Detailed (one at a time)", "Grid Overview"],
        horizontal=True
    )

    if view_mode == "Detailed (one at a time)":
        render_detailed_view(df, selected_cols)
    else:
        render_grid_view(df, selected_cols)


def render_detailed_view(df: pd.DataFrame, columns: List[str]):
    """Render detailed view for selected columns."""
    # Navigation
    if 'current_var_idx' not in st.session_state:
        st.session_state.current_var_idx = 0

    current_idx = st.session_state.current_var_idx
    if current_idx >= len(columns):
        current_idx = 0
        st.session_state.current_var_idx = 0

    col = columns[current_idx]

    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

    with nav_col1:
        if st.button("Previous") and current_idx > 0:
            st.session_state.current_var_idx -= 1
            st.rerun()

    with nav_col2:
        st.markdown(f"**Variable {current_idx + 1} of {len(columns)}: `{col}`**")

    with nav_col3:
        if st.button("Next") and current_idx < len(columns) - 1:
            st.session_state.current_var_idx += 1
            st.rerun()

    # Render single variable analysis
    render_single_variable(df, col)


def render_single_variable(df: pd.DataFrame, col: str):
    """Render detailed analysis for a single variable."""
    data = df[col].values

    # Create two columns for layout
    plot_col, stats_col = st.columns([2, 1])

    with plot_col:
        # KDE Plot with rug
        fig, ax = plt.subplots(figsize=(10, 6))

        # Remove NaN for plotting
        clean_data = data[~np.isnan(data)]

        if len(clean_data) > 0:
            # KDE plot
            sns.kdeplot(data=clean_data, ax=ax, fill=True, alpha=0.3)

            # Rug plot
            sns.rugplot(data=clean_data, ax=ax, alpha=0.5)

            # Mean and median lines
            mean_val = np.mean(clean_data)
            median_val = np.median(clean_data)

            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:,.2f}')
            ax.axvline(median_val, color='blue', linestyle=':', label=f'Median: {median_val:,.2f}')

            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {col}')
            ax.legend()

        st.pyplot(fig)
        plt.close()

    with stats_col:
        # Compute statistics
        stats = compute_distribution_stats(data)

        # Central Tendency
        st.markdown("**Central Tendency**")
        if stats['mean'] is not None:
            st.markdown(f"- Mean: {stats['mean']:,.4f}")
            st.markdown(f"- Median: {stats['median']:,.4f}")
            if stats['mode'] is not None:
                st.markdown(f"- Mode: {stats['mode']:,.4f}")

        # Dispersion
        st.markdown("**Dispersion**")
        if stats['std'] is not None:
            st.markdown(f"- Std Dev: {stats['std']:,.4f}")
            st.markdown(f"- IQR: {stats['iqr']:,.4f}")
            st.markdown(f"- Range: {stats['range']:,.4f}")
            st.markdown(f"- Min: {stats['min']:,.4f}")
            st.markdown(f"- Max: {stats['max']:,.4f}")

        # Shape
        st.markdown("**Shape**")
        if stats['skewness'] is not None:
            skew_label, skew_color = get_skewness_label(stats['skewness'])
            kurt_label, kurt_color = get_kurtosis_label(stats['kurtosis'])

            color_map = {"green": ":green[", "yellow": ":orange[", "red": ":red["}

            st.markdown(f"- Skewness: {stats['skewness']:.4f} {color_map[skew_color]}{skew_label}]")
            st.markdown(f"- Kurtosis: {stats['kurtosis']:.4f} {color_map[kurt_color]}{kurt_label}]")

    # Normality Assessment
    st.markdown("---")
    st.markdown("**Normality Assessment**")

    normality = run_normality_test(data)

    if normality['statistic'] is not None:
        st.markdown(f"**Test:** {normality['test_name']}")
        st.markdown(f"**{normality['stat_label']}:** {normality['statistic']:.4f}")
        st.markdown(f"**p-value:** {normality['p_value']:.4f}")

        if normality['is_normal']:
            st.success(f"Cannot reject normality (p >= 0.05)")
            st.markdown("**Recommendation:** Pearson correlation is appropriate for this variable.")
        else:
            st.warning(f"Reject normality hypothesis (p < 0.05)")
            st.markdown("""
            **Implications for correlation analysis:**
            - Pearson correlation assumes bivariate normality
            - Non-normal variables may yield misleading correlation values

            **Recommendations:**
            1. Use Spearman's rho (rank-based, does not assume normality)
            2. Transform variable: log(x), sqrt(x), or Box-Cox
            3. Accept limitation if sample size is large (n > 30)
            """)
    else:
        st.info(normality['interpretation'])


def render_grid_view(df: pd.DataFrame, columns: List[str]):
    """Render grid overview of all selected variables."""
    # Calculate grid dimensions
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, col in enumerate(columns):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx]

        data = df[col].dropna().values

        if len(data) > 0:
            # Mini KDE
            sns.kdeplot(data=data, ax=ax, fill=True, alpha=0.3)

            # Stats
            stats = compute_distribution_stats(data)
            skew_label, skew_color = get_skewness_label(stats['skewness'])
            normality = run_normality_test(data)

            # Color coding
            color_map = {"green": "g", "yellow": "orange", "red": "r"}

            ax.set_title(f"{col}\nsk={stats['skewness']:.2f}", fontsize=9)
            ax.set_xlabel('')
            ax.set_ylabel('')

            # Add normality indicator
            if normality['is_normal'] is not None:
                indicator = "" if normality['is_normal'] else ""
                ax.annotate(indicator, xy=(0.95, 0.95), xycoords='axes fraction',
                           fontsize=12, ha='right', va='top')

    # Hide empty subplots
    for idx in range(len(columns), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Legend
    st.markdown("""
    **Legend:**
    - sk = skewness
    - :green[Low skew (< 0.5)] | :orange[Moderate (0.5-1)] | :red[High (> 1)]
    """)

    # Summary table
    st.markdown("**Summary Statistics**")

    summary_data = []
    for col in columns:
        data = df[col].values
        stats = compute_distribution_stats(data)
        normality = run_normality_test(data)

        summary_data.append({
            "Variable": col,
            "Mean": f"{stats['mean']:,.2f}" if stats['mean'] is not None else "N/A",
            "Median": f"{stats['median']:,.2f}" if stats['median'] is not None else "N/A",
            "Std Dev": f"{stats['std']:,.2f}" if stats['std'] is not None else "N/A",
            "Skewness": f"{stats['skewness']:.2f}" if stats['skewness'] is not None else "N/A",
            "Normal": "" if normality['is_normal'] else "" if normality['is_normal'] is not None else "N/A"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
