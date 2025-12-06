"""State management for the Data Preparation Pipeline with branching support."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import tempfile


@dataclass
class StateMetadata:
    """Metadata for a single DataFrame state."""
    id: str
    timestamp: str
    row_count: int
    column_count: int
    memory_mb: float
    delta_summary: str


@dataclass
class Branch:
    """A branch containing a sequence of states."""
    created_at: str
    forked_from: Optional[dict]
    states: List[StateMetadata] = field(default_factory=list)


class StateNotFoundError(Exception):
    """Raised when a requested state doesn't exist."""
    pass


class StateManager:
    """Manages DataFrame states with branching support."""

    def __init__(self, max_memory_cache: int = 5):
        self.branches: Dict[str, Branch] = {
            "main": Branch(
                created_at=pd.Timestamp.now().isoformat(),
                forked_from=None,
                states=[]
            )
        }
        self.active_branch = "main"
        self.temp_dir = Path(tempfile.gettempdir()) / "data_pipeline"
        self.temp_dir.mkdir(exist_ok=True)

        # LRU cache
        self._memory_cache: Dict[str, pd.DataFrame] = {}
        self._access_order: List[str] = []
        self.max_memory_cache = max_memory_cache

        # Config snapshots
        self._config_snapshots: Dict[str, dict] = {}

    def commit_state(
        self,
        state_id: str,
        df: pd.DataFrame,
        config_snapshot: dict,
        delta_summary: str
    ) -> None:
        """
        Save a new state to the active branch.

        Args:
            state_id: Unique identifier (e.g., 'clean', 'outlier_handled')
            df: DataFrame to save
            config_snapshot: Configuration used to produce this state
            delta_summary: Human-readable description of changes
        """
        # Create metadata
        metadata = StateMetadata(
            id=state_id,
            timestamp=pd.Timestamp.now().isoformat(),
            row_count=len(df),
            column_count=len(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024**2,
            delta_summary=delta_summary
        )

        # Add to branch
        self.branches[self.active_branch].states.append(metadata)

        # Save config snapshot
        state_key = f"{self.active_branch}::{state_id}"
        self._config_snapshots[state_key] = config_snapshot

        # Save to cache and disk
        self._add_to_cache(state_key, df)
        self._save_to_disk(state_key, df)

    def _add_to_cache(self, key: str, df: pd.DataFrame) -> None:
        """Add state to memory cache with LRU eviction."""
        if key in self._memory_cache:
            self._access_order.remove(key)

        self._memory_cache[key] = df.copy()
        self._access_order.append(key)

        # Evict oldest if cache full
        if len(self._memory_cache) > self.max_memory_cache:
            oldest_key = self._access_order.pop(0)
            del self._memory_cache[oldest_key]

    def _save_to_disk(self, key: str, df: pd.DataFrame) -> None:
        """Persist state to disk as Parquet."""
        filepath = self.temp_dir / f"{key.replace('::', '__')}.parquet"
        df.to_parquet(filepath, index=False, compression='snappy')

    def load_state(self, branch: str, state_id: str) -> pd.DataFrame:
        """
        Load a state from cache or disk.

        Args:
            branch: Branch name
            state_id: State identifier

        Returns:
            DataFrame at the requested state

        Raises:
            StateNotFoundError: If state doesn't exist
        """
        state_key = f"{branch}::{state_id}"

        # Check memory cache
        if state_key in self._memory_cache:
            self._touch(state_key)
            return self._memory_cache[state_key].copy()

        # Load from disk
        disk_path = self.temp_dir / f"{state_key.replace('::', '__')}.parquet"
        if disk_path.exists():
            df = pd.read_parquet(disk_path)
            self._add_to_cache(state_key, df)
            return df.copy()

        raise StateNotFoundError(f"State {state_key} not found")

    def _touch(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _get_config_at_state(self, branch: str, state_id: str) -> dict:
        """Get configuration snapshot at a specific state."""
        state_key = f"{branch}::{state_id}"
        return self._config_snapshots.get(state_key, {})

    def create_branch(
        self,
        new_branch_name: str,
        fork_from_state: str
    ) -> None:
        """
        Create a new branch from a specific state.

        Args:
            new_branch_name: Name for the new branch
            fork_from_state: State ID to fork from
        """
        source_branch = self.active_branch
        source_df = self.load_state(source_branch, fork_from_state)
        source_config = self._get_config_at_state(source_branch, fork_from_state)

        # Create new branch
        self.branches[new_branch_name] = Branch(
            created_at=pd.Timestamp.now().isoformat(),
            forked_from={"branch": source_branch, "state_id": fork_from_state},
            states=[]
        )

        # Switch to new branch and commit fork point
        self.active_branch = new_branch_name
        self.commit_state(
            state_id=fork_from_state,
            df=source_df,
            config_snapshot=source_config,
            delta_summary=f"Forked from {source_branch}::{fork_from_state}"
        )

    def switch_branch(self, branch_name: str) -> pd.DataFrame:
        """
        Switch to a different branch.

        Args:
            branch_name: Branch to switch to

        Returns:
            DataFrame at the terminal state of that branch
        """
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")

        self.active_branch = branch_name
        if self.branches[branch_name].states:
            terminal_state = self.branches[branch_name].states[-1].id
            return self.load_state(branch_name, terminal_state)
        return pd.DataFrame()

    def delete_branch(self, branch_name: str) -> None:
        """
        Delete a branch and its cached states.

        Args:
            branch_name: Branch to delete (cannot be 'main')
        """
        if branch_name == "main":
            raise ValueError("Cannot delete main branch")
        if branch_name == self.active_branch:
            raise ValueError("Cannot delete active branch")

        # Remove from cache
        keys_to_remove = [k for k in self._memory_cache if k.startswith(f"{branch_name}::")]
        for key in keys_to_remove:
            del self._memory_cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        # Remove disk files
        for file in self.temp_dir.glob(f"{branch_name}__*.parquet"):
            file.unlink()

        # Remove branch
        del self.branches[branch_name]

    def get_branch_summary(self, branch_name: str) -> dict:
        """Get summary statistics for a branch."""
        branch = self.branches[branch_name]
        if not branch.states:
            return {"states": 0, "terminal_state": None}

        return {
            "name": branch_name,
            "created_at": branch.created_at,
            "forked_from": branch.forked_from,
            "state_count": len(branch.states),
            "terminal_state": branch.states[-1].id,
            "last_updated": branch.states[-1].timestamp
        }

    def list_all_states(self) -> List[dict]:
        """Get metadata for all states across all branches."""
        all_states = []
        for branch_name, branch in self.branches.items():
            for state in branch.states:
                all_states.append({
                    "branch": branch_name,
                    "state_id": state.id,
                    "timestamp": state.timestamp,
                    "rows": state.row_count,
                    "columns": state.column_count,
                    "memory_mb": state.memory_mb,
                    "delta": state.delta_summary
                })
        return sorted(all_states, key=lambda x: x["timestamp"], reverse=True)

    def get_current_state(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame at the current terminal state."""
        branch = self.branches[self.active_branch]
        if not branch.states:
            return None
        terminal_state = branch.states[-1].id
        return self.load_state(self.active_branch, terminal_state)

    def get_current_state_id(self) -> Optional[str]:
        """Get the ID of the current terminal state."""
        branch = self.branches[self.active_branch]
        if not branch.states:
            return None
        return branch.states[-1].id

    def rollback_to_state(self, state_id: str) -> pd.DataFrame:
        """
        Rollback to a previous state, discarding subsequent states.

        Args:
            state_id: State ID to rollback to

        Returns:
            DataFrame at the rollback point
        """
        branch = self.branches[self.active_branch]

        # Find the state index
        state_idx = None
        for idx, state in enumerate(branch.states):
            if state.id == state_id:
                state_idx = idx
                break

        if state_idx is None:
            raise StateNotFoundError(f"State {state_id} not found in branch {self.active_branch}")

        # Remove subsequent states
        removed_states = branch.states[state_idx + 1:]
        branch.states = branch.states[:state_idx + 1]

        # Clean up cache and disk for removed states
        for state in removed_states:
            state_key = f"{self.active_branch}::{state.id}"
            if state_key in self._memory_cache:
                del self._memory_cache[state_key]
                if state_key in self._access_order:
                    self._access_order.remove(state_key)
            disk_path = self.temp_dir / f"{state_key.replace('::', '__')}.parquet"
            if disk_path.exists():
                disk_path.unlink()

        return self.load_state(self.active_branch, state_id)

    def cleanup(self) -> None:
        """Remove all temporary files."""
        for file in self.temp_dir.glob("*.parquet"):
            file.unlink()
        if self.temp_dir.exists():
            try:
                self.temp_dir.rmdir()
            except OSError:
                pass  # Directory not empty or other issue
