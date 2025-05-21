"""
Path constants for the China Economic Data Analysis project.
This module centralizes all path-related constants to improve maintainability.
"""

import os
from typing import Dict, List

# Directory structure constants
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
PACKAGE_DIR_NAME = "china_data" # This is the name of the package directory itself.

# Path utility functions

def get_absolute_input_path() -> str:
    """
    Get the absolute path to the primary input directory (project_root/china_data/input).
    """
    # Import moved inside to minimize issues if this module is imported early,
    # and to break potential top-level circular dependencies.
    from china_data.utils import get_project_root
    return os.path.join(get_project_root(), PACKAGE_DIR_NAME, INPUT_DIR_NAME)

def get_absolute_output_path() -> str:
    """
    Get the absolute path to the primary output directory (project_root/china_data/output).

    This function ensures we always use the china_data/output directory,
    not the output directory at the project root.
    """
    from china_data.utils import get_project_root
    project_root = get_project_root()
    return os.path.join(project_root, PACKAGE_DIR_NAME, OUTPUT_DIR_NAME)

# Common file paths relative to project root for searching
def get_search_locations_relative_to_root() -> Dict[str, List[str]]:
    """
    Get default search locations for different file types,
    all paths are relative to the project root.
    The find_file function will prepend get_project_root() to these.
    """
    # e.g., "china_data/input"
    primary_input_path_relative = os.path.join(PACKAGE_DIR_NAME, INPUT_DIR_NAME)
    # e.g., "china_data/output"
    primary_output_path_relative = os.path.join(PACKAGE_DIR_NAME, OUTPUT_DIR_NAME)

    return {
        "input_files": [
            primary_input_path_relative, # Only <project_root>/china_data/input
        ],
        "output_files": [
            primary_output_path_relative, # Only <project_root>/china_data/output
        ],
        "general": [ # For general searches, we might still want broader scope, but prioritize package specific
            primary_input_path_relative,
            primary_output_path_relative,
            PACKAGE_DIR_NAME, # For files directly in project_root/china_data
            "",  # Project root itself
            # Consider if root INPUT_DIR_NAME and OUTPUT_DIR_NAME are ever valid for "general"
            # For now, keeping them might be safer if some script relied on it,
            # but ideally, all specific file types use their more specific lists.
            INPUT_DIR_NAME, # <project_root>/input
            OUTPUT_DIR_NAME, # <project_root>/output
        ]
    }