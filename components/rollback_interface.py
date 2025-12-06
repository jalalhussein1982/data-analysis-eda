"""Rollback interface UI component."""

import streamlit as st
import pandas as pd
from typing import Optional


def render_rollback_interface():
    """Render the rollback and state management interface."""
    st.subheader("Pipeline History")

    if 'state_manager' not in st.session_state:
        st.warning("State manager not initialized.")
        return

    state_manager = st.session_state.state_manager

    # Current state info
    current_branch = state_manager.active_branch
    current_state_id = state_manager.get_current_state_id()

    st.markdown(f"**Current State:** `{current_state_id}` on branch `{current_branch}`")

    # List all states
    all_states = state_manager.list_all_states()

    if not all_states:
        st.info("No states recorded yet.")
        return

    st.markdown("---")

    # State history
    for state in all_states:
        is_current = (
            state['branch'] == current_branch and
            state['state_id'] == current_state_id
        )

        # Format display
        icon = "" if is_current else ""

        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"{icon} **{state['state_id']}**")
                st.caption(state['delta'])

            with col2:
                st.caption(f"Branch: {state['branch']}")
                st.caption(f"{state['rows']:,} rows, {state['columns']} cols")

            with col3:
                if not is_current:
                    if st.button("Rollback", key=f"rollback_{state['branch']}_{state['state_id']}"):
                        rollback_to_state(state['branch'], state['state_id'])

            st.markdown("---")

    # Branch management
    st.subheader("Branch Management")

    # List branches
    branches = list(state_manager.branches.keys())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Active Branch:** `{current_branch}`")

        other_branches = [b for b in branches if b != current_branch]
        if other_branches:
            switch_to = st.selectbox(
                "Switch to branch:",
                ["(current)"] + other_branches
            )

            if switch_to != "(current)" and st.button("Switch"):
                state_manager.switch_branch(switch_to)
                st.success(f"Switched to branch: {switch_to}")
                st.rerun()

    with col2:
        # Create new branch
        st.markdown("**Create New Branch:**")

        new_branch_name = st.text_input("New branch name:", key="new_branch_name")

        if current_state_id:
            if st.button("Create Branch from Current State"):
                if new_branch_name and new_branch_name not in branches:
                    state_manager.create_branch(new_branch_name, current_state_id)
                    st.success(f"Created branch: {new_branch_name}")
                    st.rerun()
                else:
                    st.error("Invalid or duplicate branch name")


def rollback_to_state(branch: str, state_id: str):
    """Execute rollback to a specific state."""
    state_manager = st.session_state.state_manager

    # Switch branch if needed
    if branch != state_manager.active_branch:
        state_manager.switch_branch(branch)

    # Rollback
    try:
        df = state_manager.rollback_to_state(state_id)

        # Update session state DataFrames based on state ID
        state_to_df_map = {
            'schema': 'df_schema',
            'deduplicated': 'df_deduplicated',
            'scoped': 'df_scoped',
            'clean': 'df_clean',
            'outlier_handled': 'df_outlier_handled',
            'collinear_resolved': 'df_collinear_resolved',
            'final': 'df_final'
        }

        if state_id in state_to_df_map:
            st.session_state[state_to_df_map[state_id]] = df

            # Clear subsequent states
            state_order = list(state_to_df_map.keys())
            state_idx = state_order.index(state_id)

            for subsequent_state in state_order[state_idx + 1:]:
                df_key = state_to_df_map[subsequent_state]
                if df_key in st.session_state:
                    del st.session_state[df_key]

        st.success(f"Rolled back to state: {state_id}")
        st.rerun()

    except Exception as e:
        st.error(f"Rollback failed: {str(e)}")


def render_state_comparison(state1_id: str, state2_id: str):
    """Render comparison between two states."""
    state_manager = st.session_state.state_manager

    try:
        df1 = state_manager.load_state(state_manager.active_branch, state1_id)
        df2 = state_manager.load_state(state_manager.active_branch, state2_id)

        st.subheader(f"Comparison: {state1_id} vs {state2_id}")

        # Basic metrics comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{state1_id}**")
            st.metric("Rows", len(df1))
            st.metric("Columns", len(df1.columns))
            st.metric("Memory (MB)", f"{df1.memory_usage(deep=True).sum() / 1024**2:.2f}")

        with col2:
            st.markdown(f"**{state2_id}**")
            row_diff = len(df2) - len(df1)
            col_diff = len(df2.columns) - len(df1.columns)

            st.metric("Rows", len(df2), delta=row_diff)
            st.metric("Columns", len(df2.columns), delta=col_diff)

            mem2 = df2.memory_usage(deep=True).sum() / 1024**2
            mem1 = df1.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory (MB)", f"{mem2:.2f}", delta=f"{mem2 - mem1:.2f}")

        # Column differences
        st.markdown("**Column Changes:**")

        cols1 = set(df1.columns)
        cols2 = set(df2.columns)

        added = cols2 - cols1
        removed = cols1 - cols2

        if added:
            st.success(f"Added: {', '.join(added)}")
        if removed:
            st.error(f"Removed: {', '.join(removed)}")
        if not added and not removed:
            st.info("No column changes")

    except Exception as e:
        st.error(f"Comparison failed: {str(e)}")
