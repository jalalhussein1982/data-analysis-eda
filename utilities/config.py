"""Pipeline configuration dataclasses for reproducibility."""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict
import json
import pandas as pd


@dataclass
class SchemaConfig:
    """Schema enforcement decisions."""
    type_overrides: Dict[str, str] = field(default_factory=dict)
    datetime_format: Optional[str] = None


@dataclass
class DuplicateConfig:
    """Duplicate resolution strategy."""
    method: str = "keep_first"  # "keep_first" | "keep_last" | "drop_all"
    subset_columns: Optional[List[str]] = None


@dataclass
class ScopeConfig:
    """Column selection."""
    selected_columns: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)


@dataclass
class ConstraintConfig:
    """Constraint enforcement rules."""
    column: str = ""
    type: str = "range"  # "range" | "allowed_values" | "regex" | "cross_column"
    parameters: dict = field(default_factory=dict)
    violation_handling: str = "convert_nan"  # "drop_row" | "convert_nan" | "flag_retain"


@dataclass
class ImputationConfig:
    """Missing value imputation strategies."""
    column: str = ""
    method: str = "median"  # "mean" | "median" | "mode" | "constant" | "forward_fill"
    parameters: Optional[dict] = None


@dataclass
class OutlierConfig:
    """Outlier detection and resolution."""
    column: str = ""
    detection_method: str = "iqr"  # "iqr" | "zscore" | "percentile"
    detection_params: dict = field(default_factory=dict)
    resolution_strategy: str = "winsorize"  # "keep" | "drop" | "winsorize" | "transform"
    resolution_params: Optional[dict] = None


@dataclass
class MulticollinearityConfig:
    """Multicollinearity screening."""
    vif_threshold: float = 10.0
    auto_remove: bool = False
    removed_columns: List[str] = field(default_factory=list)


@dataclass
class EncodingConfig:
    """Categorical encoding."""
    column: str = ""
    method: str = "onehot"  # "onehot" | "label" | "frequency"
    parameters: Optional[dict] = None


@dataclass
class ScalingConfig:
    """Numerical scaling."""
    method: str = "standard"  # "standard" | "minmax" | "robust" | "none"
    columns: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    version: str = "2.0"
    created_at: str = ""

    schema: Optional[SchemaConfig] = None
    duplicates: Optional[DuplicateConfig] = None
    scope: Optional[ScopeConfig] = None
    constraints: Optional[List[ConstraintConfig]] = None
    imputation: Optional[List[ImputationConfig]] = None
    outliers: Optional[List[OutlierConfig]] = None
    multicollinearity: Optional[MulticollinearityConfig] = None
    encoding: Optional[List[EncodingConfig]] = None
    scaling: Optional[ScalingConfig] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = pd.Timestamp.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Export configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        if data.get('schema'):
            data['schema'] = SchemaConfig(**data['schema'])
        if data.get('duplicates'):
            data['duplicates'] = DuplicateConfig(**data['duplicates'])
        if data.get('scope'):
            data['scope'] = ScopeConfig(**data['scope'])
        if data.get('constraints'):
            data['constraints'] = [ConstraintConfig(**c) for c in data['constraints']]
        if data.get('imputation'):
            data['imputation'] = [ImputationConfig(**i) for i in data['imputation']]
        if data.get('outliers'):
            data['outliers'] = [OutlierConfig(**o) for o in data['outliers']]
        if data.get('multicollinearity'):
            data['multicollinearity'] = MulticollinearityConfig(**data['multicollinearity'])
        if data.get('encoding'):
            data['encoding'] = [EncodingConfig(**e) for e in data['encoding']]
        if data.get('scaling'):
            data['scaling'] = ScalingConfig(**data['scaling'])

        return cls(**data)

    def get_snapshot(self) -> dict:
        """Get a snapshot of current configuration for state management."""
        return self.to_dict()
