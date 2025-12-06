"""Imputation selector UI component."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def render_imputation_selector(
    df: pd.DataFrame,
    column: str
) -> Optional[Dict[str, Any]]:
    """
    Render imputation strategy selector for a column.

    Args:
        df: DataFrame containing the column
        column: Column name to configure imputation for

    Returns:
        Imputation configuration dictionary or None
    """
    missing_count = df[column].isnull().sum()

    if missing_count == 0:
        st.success(f"No missing values in `{column}`")
        return None

    missing_pct = (missing_count / len(df)) * 100

    st.markdown(f"**Imputation Strategy for: `{column}`** ({missing_count} missing, {missing_pct:.1f}%)")

    col_dtype = df[column].dtype
    is_numerical = np.issubdtype(col_dtype, np.number)

    if is_numerical:
        # Calculate statistics
        col_mean = df[column].mean()
        col_median = df[column].median()

        try:
            col_mode = df[column].mode().iloc[0]
        except:
            col_mode = col_median

        # Display statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean", f"{col_mean:,.4f}")

        with col2:
            st.metric("Median", f"{col_median:,.4f}")

        with col3:
            st.metric("Mode", f"{col_mode:,.4f}")

        # Strategy selection
        strategies = [
            "Median (Recommended for skewed data)",
            "Mean (For symmetric distributions)",
            "Mode (Most frequent value)",
            "Constant Value",
            "Forward Fill (Time series)",
            "Backward Fill (Time series)",
            "Drop Rows"
        ]

    else:
        # Categorical column
        try:
            col_mode = df[column].mode().iloc[0]
        except:
            col_mode = "Unknown"

        st.metric("Mode", col_mode)

        strategies = [
            "Mode (Most Frequent)",
            "Add 'Unknown' Category",
            "Drop Rows"
        ]

    strategy = st.selectbox(
        "Select strategy:",
        strategies,
        key=f"impute_strategy_{column}"
    )

    imputation_config = {
        "column": column,
        "method": None,
        "parameters": {}
    }

    # Handle constant value input
    if "Constant" in strategy:
        if is_numerical:
            const_val = st.number_input(
                "Constant value:",
                value=0.0,
                key=f"const_val_{column}"
            )
        else:
            const_val = st.text_input(
                "Constant value:",
                value="Unknown",
                key=f"const_val_{column}"
            )

        imputation_config["parameters"]["value"] = const_val

    # Map strategy to method
    if "Median" in strategy:
        imputation_config["method"] = "median"
    elif "Mean" in strategy:
        imputation_config["method"] = "mean"
    elif "Mode" in strategy:
        imputation_config["method"] = "mode"
    elif "Constant" in strategy:
        imputation_config["method"] = "constant"
    elif "Forward Fill" in strategy:
        imputation_config["method"] = "ffill"
    elif "Backward Fill" in strategy:
        imputation_config["method"] = "bfill"
    elif "Unknown" in strategy:
        imputation_config["method"] = "constant"
        imputation_config["parameters"]["value"] = "Unknown"
    elif "Drop" in strategy:
        imputation_config["method"] = "drop"

    # Preview impact
    if st.checkbox(f"Preview impact on {column}", key=f"preview_{column}"):
        preview_df = df.copy()

        if imputation_config["method"] == "median":
            preview_df[column] = preview_df[column].fillna(col_median)
        elif imputation_config["method"] == "mean":
            preview_df[column] = preview_df[column].fillna(col_mean)
        elif imputation_config["method"] == "mode":
            preview_df[column] = preview_df[column].fillna(col_mode)
        elif imputation_config["method"] == "constant":
            preview_df[column] = preview_df[column].fillna(
                imputation_config["parameters"].get("value", 0)
            )
        elif imputation_config["method"] == "ffill":
            preview_df[column] = preview_df[column].ffill()
        elif imputation_config["method"] == "bfill":
            preview_df[column] = preview_df[column].bfill()
        elif imputation_config["method"] == "drop":
            preview_df = preview_df.dropna(subset=[column])

        st.markdown("**After Imputation:**")

        if is_numerical:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "New Mean",
                    f"{preview_df[column].mean():,.4f}",
                    delta=f"{preview_df[column].mean() - col_mean:,.4f}"
                )

            with col2:
                st.metric(
                    "New Std Dev",
                    f"{preview_df[column].std():,.4f}",
                    delta=f"{preview_df[column].std() - df[column].std():,.4f}"
                )
        else:
            st.metric(
                "Remaining Missing",
                preview_df[column].isnull().sum()
            )

    return imputation_config
