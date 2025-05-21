"""
Utilities for dataframe operations in the China data processor.
"""

from china_data.utils.processor_dataframe.merge_operations import (
    merge_dataframe_column,
    merge_projections,
    merge_tax_data
)
from china_data.utils.processor_dataframe.metadata_operations import get_projection_metadata
from china_data.utils.processor_dataframe.output_operations import (
    prepare_final_dataframe,
    save_output_files
)

__all__ = [
    'merge_dataframe_column',
    'merge_projections',
    'merge_tax_data',
    'get_projection_metadata',
    'prepare_final_dataframe',
    'save_output_files'
]
