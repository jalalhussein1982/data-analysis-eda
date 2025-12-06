"""Pipeline phase modules."""

from . import phase_1_ingestion
from . import phase_1a_duplicates
from . import phase_2_scope
from . import phase_3_sanitation
from . import phase_4_outliers
from . import phase_4a_multicollinearity
from . import phase_5_feature_eng

__all__ = [
    'phase_1_ingestion',
    'phase_1a_duplicates',
    'phase_2_scope',
    'phase_3_sanitation',
    'phase_4_outliers',
    'phase_4a_multicollinearity',
    'phase_5_feature_eng'
]
