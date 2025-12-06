"""Phase I-A: Duplicate Detection & Resolution."""

import streamlit as st
import pandas as pd
from typing import List, Optional


def render():
    """Render Phase I-A: Duplicate Detection & Resolution."""
    st.header("Phase I-A: Duplicate Detection & Resolution")
    st.markdown("""
    **Objective:** Eliminate redundant observations that inflate sample size and distort variance.
    """)

    # Check if previous phase is complete
    if 'df_schema' not in st.session_state:
        st.warning("Please complete Phase I first.")
        if st.button("Go to Phase I"):
            st.session_state.current_phase = 0
            st.rerun()
        return

    df = st.session_state['df_schema'].copy()

    # Exact Duplicate Scan
    st.subheader("1. Exact Duplicate Scan")
    exact_dupes = df.duplicated(keep=False)
    exact_dupe_count = exact_dupes.sum()

    if exact_dupe_count > 0:
        st.warning(f"Found **{exact_dupe_count}** rows that are exact duplicates")

        # Preview duplicates
        with st.expander("Preview Exact Duplicates"):
            dupe_df = df[exact_dupes].sort_values(by=df.columns.tolist())
            st.dataframe(dupe_df.head(50), use_container_width=True)

        # Resolution options
        exact_resolution = st.radio(
            "Resolution for Exact Duplicates:",
            ["Keep First", "Keep Last", "Drop All Duplicates", "Keep All (No Action)"],
            key="exact_dupe_resolution"
        )
    else:
        st.success("No exact duplicates found!")
        exact_resolution = "Keep All (No Action)"

    # Subset Duplicate Scan
    st.subheader("2. Subset Duplicate Scan")
    st.markdown("Identify rows where KEY columns are identical (e.g., same transaction ID with different timestamps)")

    # Column selector for subset keys
    selected_keys = st.multiselect(
        "Select Key Columns for Subset Duplicate Detection:",
        df.columns.tolist(),
        help="Select columns that should be unique together"
    )

    subset_resolution = "Keep All (No Action)"

    if selected_keys:
        subset_dupes = df.duplicated(subset=selected_keys, keep=False)
        subset_dupe_count = subset_dupes.sum()

        if subset_dupe_count > 0:
            # Calculate number of groups
            dupe_groups = df[subset_dupes].groupby(selected_keys).ngroups

            st.warning(f"Found **{dupe_groups}** duplicate groups ({subset_dupe_count} rows) based on selected keys")

            # Preview duplicates
            with st.expander("Preview Subset Duplicates"):
                subset_dupe_df = df[subset_dupes].sort_values(by=selected_keys)
                st.dataframe(subset_dupe_df.head(50), use_container_width=True)

            # Resolution options
            subset_resolution = st.radio(
                "Resolution for Subset Duplicates:",
                ["Keep First", "Keep Last", "Drop All Duplicates", "Keep All (No Action)"],
                key="subset_dupe_resolution"
            )
        else:
            st.success("No subset duplicates found based on selected keys!")

    # Apply resolutions
    st.subheader("3. Apply Resolution")

    preview_df = df.copy()

    # Apply exact duplicate resolution
    if exact_resolution == "Keep First":
        preview_df = preview_df.drop_duplicates(keep='first')
    elif exact_resolution == "Keep Last":
        preview_df = preview_df.drop_duplicates(keep='last')
    elif exact_resolution == "Drop All Duplicates":
        preview_df = preview_df.drop_duplicates(keep=False)

    # Apply subset duplicate resolution
    if selected_keys and subset_resolution != "Keep All (No Action)":
        if subset_resolution == "Keep First":
            preview_df = preview_df.drop_duplicates(subset=selected_keys, keep='first')
        elif subset_resolution == "Keep Last":
            preview_df = preview_df.drop_duplicates(subset=selected_keys, keep='last')
        elif subset_resolution == "Drop All Duplicates":
            preview_df = preview_df.drop_duplicates(subset=selected_keys, keep=False)

    # Show impact
    rows_removed = len(df) - len(preview_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", f"{len(df):,}")
    with col2:
        st.metric("Rows Removed", f"{rows_removed:,}")
    with col3:
        st.metric("Final Rows", f"{len(preview_df):,}")

    # Confirm and proceed
    if st.button("Apply Resolution & Proceed", type="primary"):
        from utilities.config import DuplicateConfig

        # Determine method
        if exact_resolution != "Keep All (No Action)":
            method = exact_resolution.lower().replace(" ", "_")
        elif subset_resolution != "Keep All (No Action)":
            method = subset_resolution.lower().replace(" ", "_")
        else:
            method = "keep_all"

        # Update config
        st.session_state.pipeline_config.duplicates = DuplicateConfig(
            method=method,
            subset_columns=selected_keys if selected_keys else None
        )

        # Save result
        st.session_state['df_deduplicated'] = preview_df

        # Commit state
        st.session_state.state_manager.commit_state(
            state_id="deduplicated",
            df=preview_df,
            config_snapshot=st.session_state.pipeline_config.get_snapshot(),
            delta_summary=f"Removed {rows_removed} duplicate rows"
        )

        st.success(f"Duplicates resolved! Removed {rows_removed} rows. Proceed to Phase II: Scope Definition")
        st.session_state.current_phase = 2
        st.rerun()

    # Skip option
    if st.button("Skip (No Duplicates to Remove)"):
        st.session_state['df_deduplicated'] = df

        # Commit state
        st.session_state.state_manager.commit_state(
            state_id="deduplicated",
            df=df,
            config_snapshot=st.session_state.pipeline_config.get_snapshot(),
            delta_summary="No duplicates removed"
        )

        st.session_state.current_phase = 2
        st.rerun()
