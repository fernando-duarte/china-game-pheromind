"""
Path constants for the China Economic Data Analysis project.
This module centralizes all path-related constants to improve maintainability.
"""

import os
from typing import Dict

# Directory structure constants
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
PACKAGE_DIR_NAME = "china_data"

# Path utility functions that leverage the constants

def get_input_dir_path(relative_to_root: bool = True) -> str:
    """
    Get the path to the input directory.
    
    Args:
        relative_to_root: If True, return path relative to project root
                          If False, return path relative to package
                          
    Returns:
        Path to the input directory
    """
    if relative_to_root:
        from china_data.utils import get_project_root
        return os.path.join(get_project_root(), PACKAGE_DIR_NAME, INPUT_DIR_NAME)
    else:
        return os.path.join(PACKAGE_DIR_NAME, INPUT_DIR_NAME)


def get_output_dir_path(relative_to_root: bool = True) -> str:
    """
    Get the path to the output directory.
    
    Args:
        relative_to_root: If True, return path relative to project root
                          If False, return path relative to package
                          
    Returns:
        Path to the output directory
    """
    if relative_to_root:
        from china_data.utils import get_project_root
        return os.path.join(get_project_root(), PACKAGE_DIR_NAME, OUTPUT_DIR_NAME)
    else:
        return os.path.join(PACKAGE_DIR_NAME, OUTPUT_DIR_NAME)


# Common file paths
def get_default_search_locations() -> Dict[str, list]:
    """
    Get default search locations for different file types.
    
    Returns:
        Dictionary of file types and their search locations
    """
    return {
        "input_files": [
            ".",
            get_input_dir_path(relative_to_root=False),
            INPUT_DIR_NAME,
            f"./{INPUT_DIR_NAME}",
            "china_data/input",
            "./input"
        ],
        "output_files": [
            ".",
            get_output_dir_path(relative_to_root=False),
            OUTPUT_DIR_NAME,
            f"./{OUTPUT_DIR_NAME}",
            "china_data/output",
            "./output"
        ],
        "general": [
            ".",
            PACKAGE_DIR_NAME,
            get_input_dir_path(relative_to_root=False),
            get_output_dir_path(relative_to_root=False),
            INPUT_DIR_NAME,
            OUTPUT_DIR_NAME,
            "china_data/input",
            "china_data/output",
            "./input",
            "./output"
        ]
    }