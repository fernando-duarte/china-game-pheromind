import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

from china_data.utils import find_file, get_project_root, get_output_directory
from china_data.utils.path_constants import get_search_locations_relative_to_root, get_absolute_output_path, get_absolute_input_path

logger = logging.getLogger(__name__)


def load_raw_data(input_file: str = "china_data_raw.md") -> pd.DataFrame:
    """
    Load raw data from a markdown table file.
    This file is expected to be in one of the standard output locations.

    Args:
        input_file: Name of the input file

    Returns:
        DataFrame containing the raw data

    Raises:
        FileNotFoundError: If the input file cannot be found
    """
    # Use the common find_file utility. It searches relative to project root.
    # china_data_raw.md is an output file.
    possible_locations_relative = get_search_locations_relative_to_root()["output_files"]

    md_file = find_file(input_file, possible_locations_relative)

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


def load_imf_tax_revenue_data() -> pd.DataFrame:
    """
    Load IMF tax revenue data from CSV file.
    This file is expected to be in one of the standard input locations.

    Returns:
        DataFrame containing the tax revenue data
    """
    # Use the dedicated IMF loader module
    from china_data.utils.data_sources.imf_loader import load_imf_tax_data
    return load_imf_tax_data()
