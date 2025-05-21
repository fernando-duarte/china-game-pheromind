import os
import logging
from datetime import datetime

from china_data.utils import find_file
from china_data.utils.path_constants import get_search_locations_relative_to_root

logger = logging.getLogger(__name__)


def record_download_date(source_name):
    """
    Record the download date for a specific data source.
    
    Args:
        source_name (str): The name of the data source (e.g., 'IMF', 'WDI', 'PWT')
    
    Returns:
        str: The date that was recorded (in YYYY-MM-DD format)
    """
    # Find the download_dates.txt file
    date_file = find_file('download_dates.txt', get_search_locations_relative_to_root()["input_files"])
    
    # If the file doesn't exist, create it in the input directory
    if not date_file:
        input_dirs = get_search_locations_relative_to_root()["input_files"]
        if input_dirs:
            # Use the first input directory
            input_dir = input_dirs[0]
            date_file = os.path.join(input_dir, 'download_dates.txt')
            logger.info(f"Creating new download_dates.txt file at {date_file}")
        else:
            logger.error("No input directories found, cannot create download_dates.txt")
            return datetime.today().strftime('%Y-%m-%d')
    
    # Get the current date
    today = datetime.today().strftime('%Y-%m-%d')
    
    # Read the existing dates if the file exists
    dates = {}
    if os.path.exists(date_file):
        with open(date_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                dates[key.strip()] = value.strip()
    
    # Update the date for the specified source
    dates[source_name] = today
    
    # Write the updated dates back to the file
    with open(date_file, 'w') as f:
        for key, value in dates.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Recorded download date for {source_name}: {today}")
    return today


def get_download_date(source_name):
    """
    Get the recorded download date for a specific data source.
    
    Args:
        source_name (str): The name of the data source (e.g., 'IMF', 'WDI', 'PWT')
    
    Returns:
        str: The recorded date (in YYYY-MM-DD format) or None if not found
    """
    # Find the download_dates.txt file
    date_file = find_file('download_dates.txt', get_search_locations_relative_to_root()["input_files"])
    
    if date_file and os.path.exists(date_file):
        with open(date_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                if key.strip() == source_name:
                    return value.strip()
    
    return None
