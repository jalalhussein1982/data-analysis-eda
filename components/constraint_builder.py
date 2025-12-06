"""Constraint builder UI component."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def render_constraint_builder(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Render constraint builder interface for a column.

    Args:
        df: DataFrame containing the column
        column: Column name to build constraint for

    Returns:
        Constraint configuration dictionary
    """
    st.markdown(f"**Define Constraints for: `{column}`**")

    col_dtype = df[column].dtype

    # Determine constraint types based on data type
    if np.issubdtype(col_dtype, np.number):
        constraint_types = ["Range", "Non-Negative", "Custom Expression"]
    else:
        constraint_types = ["Allowed Values", "Regex Pattern", "Not Empty"]

    constraint_type = st.selectbox(
        "Constraint Type:",
        constraint_types,
        key=f"constraint_type_{column}"
    )

    constraint_config = {
        "column": column,
        "type": constraint_type.lower().replace(" ", "_"),
        "parameters": {},
        "violation_handling": "convert_nan"
    }

    if constraint_type == "Range":
        col1, col2 = st.columns(2)

        with col1:
            min_val = st.number_input(
                "Minimum:",
                value=float(df[column].min()),
                key=f"min_{column}"
            )
            constraint_config["parameters"]["min"] = min_val

        with col2:
            max_val = st.number_input(
                "Maximum:",
                value=float(df[column].max()),
                key=f"max_{column}"
            )
            constraint_config["parameters"]["max"] = max_val

        # Check violations
        violations = (df[column] < min_val) | (df[column] > max_val)

    elif constraint_type == "Non-Negative":
        constraint_config["parameters"]["min"] = 0
        violations = df[column] < 0

    elif constraint_type == "Allowed Values":
        unique_vals = df[column].dropna().unique().tolist()

        allowed = st.multiselect(
            "Select allowed values:",
            unique_vals,
            default=unique_vals,
            key=f"allowed_{column}"
        )
        constraint_config["parameters"]["allowed"] = allowed

        violations = ~df[column].isin(allowed) & df[column].notna()

    elif constraint_type == "Regex Pattern":
        pattern = st.text_input(
            "Regex pattern:",
            value=".*",
            key=f"regex_{column}",
            help="Enter a valid regular expression"
        )
        constraint_config["parameters"]["pattern"] = pattern

        try:
            violations = ~df[column].astype(str).str.match(pattern)
        except:
            violations = pd.Series([False] * len(df))
            st.error("Invalid regex pattern")

    elif constraint_type == "Not Empty":
        violations = df[column].isna() | (df[column].astype(str).str.strip() == "")

    else:  # Custom Expression
        expression = st.text_input(
            "Custom expression (use 'x' for column value):",
            value="x >= 0",
            key=f"expr_{column}"
        )
        constraint_config["parameters"]["expression"] = expression

        try:
            x = df[column]
            violations = ~eval(expression)
        except:
            violations = pd.Series([False] * len(df))
            st.error("Invalid expression")

    # Show violation count
    violation_count = violations.sum()

    if violation_count > 0:
        st.warning(f"Violations found: {violation_count} rows ({violation_count/len(df)*100:.1f}%)")

        # Preview violations
        with st.expander("Preview Violations"):
            st.dataframe(df[violations][[column]].head(20), use_container_width=True)

        # Violation handling
        handling = st.radio(
            "Violation Handling:",
            ["Convert to NaN", "Drop Row", "Flag Only"],
            key=f"handling_{column}"
        )

        handling_map = {
            "Convert to NaN": "convert_nan",
            "Drop Row": "drop_row",
            "Flag Only": "flag_retain"
        }
        constraint_config["violation_handling"] = handling_map[handling]

    else:
        st.success("No violations found with current constraint settings.")

    return constraint_config
