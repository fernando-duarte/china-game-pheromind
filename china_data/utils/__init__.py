"""
Utilities for the China Economic Data Analysis project.
Contains common utility functions used across the codebase.
"""

import os
import logging
from typing import Optional, List, Union

logger = logging.getLogger(__name__)


def get_project_root() -> str:
    """
    Determine the project root directory.

    If we're in the china_data directory, return the parent directory.
    If we're already at the project root, return the current directory.

    Returns:
        str: Path to the project root directory
    """
    current_dir = os.path.abspath(os.getcwd())
    base_dir_name = os.path.basename(current_dir)

    if base_dir_name == "china_data":
        # We're in the china_data directory
        return os.path.dirname(current_dir)
    else:
        # We're either at the project root or somewhere else
        china_data_dir = os.path.join(current_dir, "china_data")
        if os.path.isdir(china_data_dir):
            # We're at the project root
            return current_dir
        else:
            # We're somewhere else, try to find the china_data directory
            parent_dir = os.path.dirname(current_dir)
            if os.path.isdir(os.path.join(parent_dir, "china_data")):
                return parent_dir
            else:
                # Default to current directory if we can't determine the project root
                return current_dir


def find_file(filename: str, possible_locations_relative_to_root: Optional[List[str]] = None) -> Optional[str]:
    """
    Find a file by searching multiple possible locations relative to the project root.
    
    Args:
        filename: Name of the file to find (e.g., "china_data_raw.md")
        possible_locations_relative_to_root: List of directories relative to project root to search.
                                            If None, uses default "general" locations.
        
    Returns:
        Full path to the found file, or None if not found
    """
    project_root = get_project_root()

    if possible_locations_relative_to_root is None:
        from china_data.utils.path_constants import get_search_locations_relative_to_root
        # These locations are already relative to the project root
        search_locations_relative = get_search_locations_relative_to_root()["general"]
    else:
        search_locations_relative = possible_locations_relative_to_root
    
    checked_paths = []
    for rel_location in search_locations_relative:
        # Construct absolute path by joining project_root, the relative location, and filename
        # If rel_location is an empty string (representing project root itself),
        # os.path.join handles it correctly.
        path = os.path.join(project_root, rel_location, filename)
        checked_paths.append(path)
        if os.path.exists(path):
            logger.info(f"Found file at: {path}")
            return path
            
    logger.warning(f"File '{filename}' not found. Searched in: {checked_paths}")
    return None


def ensure_directory(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        The absolute path to the directory
    """
    os.makedirs(directory, exist_ok=True)
    return os.path.abspath(directory)


def get_output_directory() -> str:
    """
    Get the path to the output directory, ensuring it exists.
    
    Returns:
        str: Path to the output directory
    """
    from china_data.utils.path_constants import get_absolute_output_path
    output_dir = get_absolute_output_path()
    return ensure_directory(output_dir)