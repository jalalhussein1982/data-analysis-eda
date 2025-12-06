"""Phase III: The Sanitation Layer (Data Integrity)."""

import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any


def render():
    """Render Phase III: Data Sanitation."""
    st.header("Phase III: The Sanitation Layer (Data Integrity)")
    st.markdown("""
    **Objective:** Ensure mathematical validity of the dataset through constraint enforcement and missing value imputation.
    """)

    # Check if previous phase is complete
    if 'df_scoped' not in st.session_state:
        st.warning("Please complete Phase II first.")
        if st.button("Go to Phase II"):
            st.session_state.current_phase = 2
            st.rerun()
        return

    df = st.session_state['df_scoped'].copy()

    # Tab interface for the two steps
    tab1, tab2 = st.tabs(["Step A: Constraint Enforcement", "Step B: Missing Value Imputation"])

    with tab1:
        df, constraints = render_constraint_enforcement(df)

    with tab2:
        df, imputations = render_missing_value_imputation(df)

    # Summary and Proceed
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        missing_total = df.isnull().sum().sum()
        st.metric("Remaining Missing", f"{missing_total:,}")

    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")

    # Apply and proceed
    if st.button("Apply Sanitation & Proceed", type="primary"):
        from utilities.config import ConstraintConfig, ImputationConfig

        # Update config
        st.session_state.pipeline_config.constraints = constraints
        st.session_state.pipeline_config.imputation = imputations

        # Save result
        st.session_state['df_clean'] = df

        # Commit state
        st.session_state.state_manager.commit_state(
            state_id="clean",
            df=df,
            config_snapshot=st.session_state.pipeline_config.get_snapshot(),
            delta_summary=f"Applied {len(constraints)} constraints, {len(imputations)} imputations"
        )

        st.success("Data sanitized! Proceed to Phase IV: Outlier Handling")
        st.session_state.current_phase = 4
        st.rerun()


def render_constraint_enforcement(df: pd.DataFrame) -> tuple:
    """Render constraint enforcement interface."""
    st.subheader("Step A: Logical Validity (Constraint Enforcement)")
    st.markdown("""
    Define constraints to identify and handle invalid data values.
    Invalid data is distinct from missing data - it represents values that should not logically exist.
    """)

    constraints = []

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Constraint builder for numerical columns
    if numerical_cols:
        st.markdown("**Numerical Column Constraints**")

        selected_num_col = st.selectbox(
            "Select numerical column for constraint:",
            ["(none)"] + numerical_cols,
            key="constraint_num_col"
        )

        if selected_num_col != "(none)":
            col_min = df[selected_num_col].min()
            col_max = df[selected_num_col].max()

            st.markdown(f"Current range: [{col_min:,.2f}, {col_max:,.2f}]")

            col1, col2 = st.columns(2)

            with col1:
                min_val = st.number_input(
                    "Minimum allowed value:",
                    value=float(col_min),
                    key=f"min_{selected_num_col}"
                )

            with col2:
                max_val = st.number_input(
                    "Maximum allowed value:",
                    value=float(col_max),
                    key=f"max_{selected_num_col}"
                )

            violation_handling = st.radio(
                "Violation handling:",
                ["Convert to NaN", "Drop Row", "Flag Only"],
                key=f"handling_{selected_num_col}"
            )

            # Check violations
            violations = (df[selected_num_col] < min_val) | (df[selected_num_col] > max_val)
            violation_count = violations.sum()

            if violation_count > 0:
                st.warning(f"Violations found: {violation_count} rows ({violation_count/len(df)*100:.1f}%)")

                if st.button(f"Apply Constraint to {selected_num_col}"):
                    handling_map = {
                        "Convert to NaN": "convert_nan",
                        "Drop Row": "drop_row",
                        "Flag Only": "flag_retain"
                    }

                    from utilities.config import ConstraintConfig

                    constraint = ConstraintConfig(
                        column=selected_num_col,
                        type="range",
                        parameters={"min": min_val, "max": max_val},
                        violation_handling=handling_map[violation_handling]
                    )
                    constraints.append(constraint)

                    # Apply constraint
                    if violation_handling == "Convert to NaN":
                        df.loc[violations, selected_num_col] = np.nan
                        st.success(f"Converted {violation_count} violations to NaN")
                    elif violation_handling == "Drop Row":
                        df = df[~violations]
                        st.success(f"Dropped {violation_count} rows with violations")

            else:
                st.success("No violations found with current constraint settings.")

    # Constraint builder for categorical columns
    if categorical_cols:
        st.markdown("**Categorical Column Constraints**")

        selected_cat_col = st.selectbox(
            "Select categorical column for constraint:",
            ["(none)"] + categorical_cols,
            key="constraint_cat_col"
        )

        if selected_cat_col != "(none)":
            unique_vals = df[selected_cat_col].dropna().unique().tolist()

            st.markdown(f"Current unique values ({len(unique_vals)}): {', '.join(map(str, unique_vals[:20]))}")

            allowed_values = st.multiselect(
                "Select allowed values:",
                unique_vals,
                default=unique_vals,
                key=f"allowed_{selected_cat_col}"
            )

            if allowed_values and len(allowed_values) < len(unique_vals):
                violations = ~df[selected_cat_col].isin(allowed_values) & df[selected_cat_col].notna()
                violation_count = violations.sum()

                if violation_count > 0:
                    st.warning(f"Violations found: {violation_count} rows")

                    if st.button(f"Apply Constraint to {selected_cat_col}"):
                        df.loc[violations, selected_cat_col] = np.nan
                        st.success(f"Converted {violation_count} violations to NaN")

    return df, constraints


def render_missing_value_imputation(df: pd.DataFrame) -> tuple:
    """Render missing value imputation interface."""
    st.subheader("Step B: Missing Value Imputation")

    imputations = []

    # Missing value summary
    missing_data = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            missing_data.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Missing Count': missing_count,
                'Missing %': f"{missing_pct:.1f}%"
            })

    if not missing_data:
        st.success("No missing values found in the dataset!")
        return df, imputations

    missing_df = pd.DataFrame(missing_data)
    st.dataframe(missing_df, hide_index=True, use_container_width=True)

    # Imputation interface
    st.markdown("**Configure Imputation Strategies**")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numerical imputation
    numerical_missing = [col for col in numerical_cols if df[col].isnull().any()]

    if numerical_missing:
        st.markdown("**Numerical Columns**")

        for col in numerical_missing:
            missing_count = df[col].isnull().sum()

            with st.expander(f"{col} ({missing_count} missing)"):
                # Show statistics
                col_mean = df[col].mean()
                col_median = df[col].median()

                try:
                    col_mode = df[col].mode().iloc[0]
                except:
                    col_mode = col_median

                st.markdown(f"- Mean: {col_mean:,.4f}")
                st.markdown(f"- Median: {col_median:,.4f}")
                st.markdown(f"- Mode: {col_mode:,.4f}")

                strategy = st.selectbox(
                    f"Imputation strategy for {col}:",
                    ["Median (Recommended)", "Mean", "Mode", "Constant Value", "Drop Rows"],
                    key=f"impute_{col}"
                )

                if strategy == "Constant Value":
                    const_val = st.number_input(
                        "Enter constant value:",
                        value=0.0,
                        key=f"const_{col}"
                    )

                if st.button(f"Apply to {col}", key=f"apply_impute_{col}"):
                    from utilities.config import ImputationConfig

                    if strategy == "Median (Recommended)":
                        df[col] = df[col].fillna(col_median)
                        imputations.append(ImputationConfig(column=col, method="median"))
                    elif strategy == "Mean":
                        df[col] = df[col].fillna(col_mean)
                        imputations.append(ImputationConfig(column=col, method="mean"))
                    elif strategy == "Mode":
                        df[col] = df[col].fillna(col_mode)
                        imputations.append(ImputationConfig(column=col, method="mode"))
                    elif strategy == "Constant Value":
                        df[col] = df[col].fillna(const_val)
                        imputations.append(ImputationConfig(
                            column=col,
                            method="constant",
                            parameters={"value": const_val}
                        ))
                    elif strategy == "Drop Rows":
                        df = df.dropna(subset=[col])
                        imputations.append(ImputationConfig(column=col, method="drop"))

                    st.success(f"Applied {strategy} to {col}")
                    st.rerun()

    # Categorical imputation
    categorical_missing = [col for col in categorical_cols if df[col].isnull().any()]

    if categorical_missing:
        st.markdown("**Categorical Columns**")

        for col in categorical_missing:
            missing_count = df[col].isnull().sum()

            with st.expander(f"{col} ({missing_count} missing)"):
                try:
                    col_mode = df[col].mode().iloc[0]
                except:
                    col_mode = "Unknown"

                st.markdown(f"- Mode: {col_mode}")

                strategy = st.selectbox(
                    f"Imputation strategy for {col}:",
                    ["Mode (Most Frequent)", "Add 'Unknown' Category", "Drop Rows"],
                    key=f"impute_cat_{col}"
                )

                if st.button(f"Apply to {col}", key=f"apply_impute_cat_{col}"):
                    from utilities.config import ImputationConfig

                    if strategy == "Mode (Most Frequent)":
                        df[col] = df[col].fillna(col_mode)
                        imputations.append(ImputationConfig(column=col, method="mode"))
                    elif strategy == "Add 'Unknown' Category":
                        df[col] = df[col].fillna("Unknown")
                        imputations.append(ImputationConfig(
                            column=col,
                            method="constant",
                            parameters={"value": "Unknown"}
                        ))
                    elif strategy == "Drop Rows":
                        df = df.dropna(subset=[col])
                        imputations.append(ImputationConfig(column=col, method="drop"))

                    st.success(f"Applied {strategy} to {col}")
                    st.rerun()

    # Quick imputation - apply median/mode to all
    st.markdown("---")
    st.markdown("**Quick Imputation**")

    if st.button("Apply Default Strategy (Median for numerical, Mode for categorical)"):
        from utilities.config import ImputationConfig

        for col in numerical_missing:
            col_median = df[col].median()
            df[col] = df[col].fillna(col_median)
            imputations.append(ImputationConfig(column=col, method="median"))

        for col in categorical_missing:
            try:
                col_mode = df[col].mode().iloc[0]
            except:
                col_mode = "Unknown"
            df[col] = df[col].fillna(col_mode)
            imputations.append(ImputationConfig(column=col, method="mode"))

        st.success("Applied default imputation to all columns with missing values")
        st.rerun()

    return df, imputations
