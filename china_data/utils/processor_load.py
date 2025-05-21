import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

from china_data.utils import find_file, get_project_root, get_output_directory
from china_data.utils.path_constants import get_default_search_locations, get_output_dir_path, get_input_dir_path

logger = logging.getLogger(__name__)


def load_raw_data(data_dir: str = ".", input_file: str = "china_data_raw.md") -> pd.DataFrame:
    """
    Load raw data from a markdown table file.
    
    Args:
        data_dir: Directory to start searching from
        input_file: Name of the input file
        
    Returns:
        DataFrame containing the raw data
        
    Raises:
        FileNotFoundError: If the input file cannot be found
    """
    # Use the common find_file utility to locate the file
    possible_locations = [
        data_dir,
        os.path.join(data_dir, "output"),
        get_output_dir_path(relative_to_root=False)
    ]
    
    md_file = find_file(input_file, possible_locations)
    
    if md_file is None:
        raise FileNotFoundError(
            f"Raw data file not found: {input_file} in any of the expected locations.")

    with open(md_file, 'r') as f:
        lines = f.readlines()
    
    # Print the first 20 lines to debug
    print("\nDebug: First few lines of markdown file:")
    for i, line in enumerate(lines[:10]):
        print(f"{i}: {line.strip()}")
        
    header_idx = None
    for i, line in enumerate(lines):
        if "| Year |" in line and "GDP" in line:
            header_idx = i
            print(f"Found header at line {i}: {line.strip()}")
            break
            
    if header_idx is None:
        raise ValueError("Could not find table header in the markdown file.")
        
    header_line = lines[header_idx].strip()
    # Clean up header line by removing leading/trailing |
    if header_line.startswith('|'):
        header_line = header_line[1:]
    if header_line.endswith('|'):
        header_line = header_line[:-1]
        
    # Split by | and strip whitespace
    header = [h.strip() for h in header_line.split('|') if h.strip()]
    print(f"Parsed header columns: {header}")
    
    mapping = {
        'Year': 'year',
        'GDP (USD)': 'GDP_USD',
        'Consumption (USD)': 'C_USD',
        'Government (USD)': 'G_USD',
        'Investment (USD)': 'I_USD',
        'Exports (USD)': 'X_USD',
        'Imports (USD)': 'M_USD',
        'FDI (% of GDP)': 'FDI_pct_GDP',
        'Tax Revenue (% of GDP)': 'TAX_pct_GDP',
        'Population': 'POP',
        'Labor Force': 'LF',
        'PWT rgdpo': 'rgdpo',
        'PWT rkna': 'rkna',
        'PWT pl_gdpo': 'pl_gdpo',
        'PWT cgdpo': 'cgdpo',
        'PWT hc': 'hc'
    }
    
    # Print all available columns and their mappings
    renamed = []
    for col in header:
        mapped_col = mapping.get(col, col)
        renamed.append(mapped_col)
        print(f"Column '{col}' -> '{mapped_col}'")
    data_start_idx = header_idx + 2
    data = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('**Notes'):
            break
        row = [c.strip() for c in line.split('|') if c.strip()]
        if len(row) == len(header):
            processed = []
            for j, value in enumerate(row):
                if j == 0:
                    processed.append(int(value))
                elif value == 'N/A':
                    processed.append(np.nan)
                elif renamed[j] in ['FDI_pct_GDP', 'TAX_pct_GDP']:
                    processed.append(float(value) if value != 'N/A' else np.nan)
                elif renamed[j] in ['POP', 'LF']:
                    processed.append(int(value.replace(',', '')) if value != 'N/A' else np.nan)
                else:
                    processed.append(float(value) if value != 'N/A' else np.nan)
            data.append(processed)
    return pd.DataFrame(data, columns=renamed)


def load_imf_tax_revenue_data(data_dir: str = ".") -> pd.DataFrame:
    """
    Load IMF tax revenue data from CSV file.
    
    Args:
        data_dir: Directory to start searching from
        
    Returns:
        DataFrame containing the tax revenue data
    """
    imf_filename = "dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"
    
    # Use the common find_file utility to locate the file
    possible_locations = [
        os.path.join(data_dir, "china_data", "input"),
        os.path.join(data_dir, "input"),
        "input",
        get_input_dir_path(relative_to_root=False)
    ]
    
    imf_file = find_file(imf_filename, possible_locations)
    
    if imf_file is None:
        logger.warning("IMF tax revenue data file not found in any of the expected locations")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["year", "TAX_pct_GDP"])

    logger.info(f"Found IMF tax revenue data file at: {imf_file}")
    df = pd.read_csv(imf_file)
    tax_data = df[df["INDICATOR"] == "G1_S13_POGDP_PT"][["TIME_PERIOD", "OBS_VALUE"]]
    tax_data = tax_data.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "TAX_pct_GDP"})
    tax_data["year"] = tax_data["year"].astype(int)
    return tax_data
