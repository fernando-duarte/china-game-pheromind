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


def find_file(filename: str, possible_locations: Optional[List[str]] = None, 
              search_root: bool = True) -> Optional[str]:
    """
    Find a file by searching multiple possible locations.
    
    Args:
        filename: Name of the file to find (e.g., "china_data_raw.md")
        possible_locations: List of directories to search
        search_root: Whether to also search in the project root
        
    Returns:
        Full path to the found file, or None if not found
    """
    if possible_locations is None:
        possible_locations = []
    
    # Always include current directory
    possible_locations.append(".")
    
    # Add china_data directory if not already in the list
    if "china_data" not in possible_locations:
        possible_locations.append("china_data")
    
    # Add common output and input directories
    for loc in ["output", "input"]:
        if loc not in possible_locations:
            possible_locations.append(loc)
            possible_locations.append(os.path.join("china_data", loc))
    
    # Check all possible paths
    checked_paths = []
    for location in possible_locations:
        path = os.path.join(location, filename)
        checked_paths.append(path)
        if os.path.exists(path):
            logger.info(f"Found file at: {path}")
            return path
    
    # Try with project root if the file wasn't found
    if search_root:
        project_root = get_project_root()
        for location in possible_locations:
            path = os.path.join(project_root, location, filename)
            checked_paths.append(path)
            if os.path.exists(path):
                logger.info(f"Found file at: {path}")
                return path
    
    logger.warning(f"File {filename} not found in any of the expected locations: {checked_paths}")
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
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "china_data", "output")
    return ensure_directory(output_dir)