import logging
import os
import hashlib
import pandas as pd
from datetime import datetime

# Handle both import scenarios (from project root or from within china_data directory)
try:
    # When imported from project root
    from china_data.utils import find_file
    from china_data.utils.path_constants import get_search_locations_relative_to_root
except ImportError:
    # When imported from within china_data directory
    from utils import find_file
    from utils.path_constants import get_search_locations_relative_to_root

logger = logging.getLogger(__name__)


def check_and_update_hash():
    """
    Check if the IMF CSV file hash has changed and update the download_date.txt file if necessary.

    This function:
    1. Calculates the SHA-256 hash of the IMF CSV file
    2. Reads the current hash from download_date.txt
    3. If the hash has changed or download_date.txt doesn't exist, updates the file with the new hash and current date
    4. If the hash is the same, does nothing

    Returns:
        bool: True if the hash was updated, False otherwise
    """
    # Find the IMF file
    imf_filename = "dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"
    possible_locations_relative = get_search_locations_relative_to_root()["input_files"]
    imf_file = find_file(imf_filename, possible_locations_relative)

    if not imf_file:
        logger.error("IMF Fiscal Monitor file not found, cannot check hash")
        return False

    # Find the download_date.txt file
    date_file = find_file("download_date.txt", possible_locations_relative)

    # Calculate the current hash of the IMF file
    with open(imf_file, 'rb') as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()

    # Check if we need to update the hash
    hash_changed = True
    if date_file and os.path.exists(date_file):
        # Read the current metadata
        metadata = {}
        try:
            with open(date_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

            # Check if the hash has changed
            if 'hash' in metadata and metadata['hash'] == current_hash:
                hash_changed = False
                logger.info("IMF file hash unchanged, no need to update download_date.txt")
        except Exception as e:
            logger.error(f"Error reading download_date.txt: {e}")

    # Update the download_date.txt file if the hash has changed
    if hash_changed:
        logger.info("IMF file hash has changed, updating download_date.txt")
        today = datetime.today().strftime('%Y-%m-%d')

        # Create the new content
        content = f"download_date: {today}\n"
        content += f"file: {imf_filename}\n"
        content += f"hash_algorithm: SHA-256\n"
        content += f"hash: {current_hash}\n"

        # Determine where to save the file
        if date_file:
            output_path = date_file
        else:
            # If download_date.txt doesn't exist, create it in the same directory as the IMF file
            output_dir = os.path.dirname(imf_file)
            output_path = os.path.join(output_dir, "download_date.txt")

        # Write the new content
        try:
            with open(output_path, 'w') as f:
                f.write(content)
            logger.info(f"Updated download_date.txt at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error updating download_date.txt: {e}")
            return False

    return False


def load_imf_tax_data():
    """
    Load IMF Fiscal Monitor tax revenue data for China.

    This function also checks if the IMF file hash has changed and updates
    the download_date.txt file if necessary.

    Returns:
        pandas.DataFrame: DataFrame containing tax revenue data with columns 'year' and 'TAX_pct_GDP'.
                         Returns an empty DataFrame if the file is not found.
    """
    # Check if the IMF file hash has changed and update download_date.txt if needed
    check_and_update_hash()

    imf_filename = "dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"
    # find_file expects locations relative to project root
    possible_locations_relative = get_search_locations_relative_to_root()["input_files"]
    imf_file = find_file(imf_filename, possible_locations_relative)

    if imf_file:
        logger.info("Found IMF Fiscal Monitor file at: %s", imf_file)
        df = pd.read_csv(imf_file)
        df = df[(df['COUNTRY'] == 'CHN') & (df['FREQUENCY'] == 'A') & (df['INDICATOR'] == 'G1_S13_POGDP_PT')]
        tax_data = df[['TIME_PERIOD', 'OBS_VALUE']].rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'TAX_pct_GDP'})
        tax_data['year'] = tax_data['year'].astype(int)
        tax_data['TAX_pct_GDP'] = pd.to_numeric(tax_data['TAX_pct_GDP'], errors='coerce')
        return tax_data
    else:
        logger.error("IMF Fiscal Monitor file not found in any of the expected locations")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['year', 'TAX_pct_GDP'])
