import os
import pandas as pd
import numpy as np
import logging


def load_raw_data(data_dir=".", input_file="china_data_raw.md"):
    md_file = os.path.join(data_dir, input_file)
    if not os.path.exists(md_file):
        output_dir = os.path.join(data_dir, "output")
        md_file = os.path.join(output_dir, input_file)
    if not os.path.exists(md_file):
        raise FileNotFoundError(
            f"Raw data file not found: {input_file} in either {data_dir} or {os.path.join(data_dir, 'output')}.")
    with open(md_file, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "| Year | GDP (USD)" in line or "| Year | GDP (USD) |" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find table header in the markdown file.")
    header = [h.strip() for h in lines[header_idx].strip().split('|') if h.strip()]
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
    renamed = [mapping.get(col, col) for col in header]
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


def load_imf_tax_revenue_data(data_dir="."):
    # Try multiple possible locations for the IMF file
    possible_paths = [
        os.path.join(data_dir, "china_data/input/dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"),
        os.path.join(data_dir, "input/dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"),
        os.path.join("input/dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"),
        os.path.join("china_data/input/dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv")
    ]

    imf_file = None
    for path in possible_paths:
        if os.path.exists(path):
            imf_file = path
            break

    if imf_file is None:
        logging.warning(f"IMF tax revenue data file not found in any of the expected locations")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["year", "TAX_pct_GDP"])

    logging.info(f"Found IMF tax revenue data file at: {imf_file}")
    df = pd.read_csv(imf_file)
    tax_data = df[df["INDICATOR"] == "G1_S13_POGDP_PT"][["TIME_PERIOD", "OBS_VALUE"]]
    tax_data = tax_data.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "TAX_pct_GDP"})
    tax_data["year"] = tax_data["year"].astype(int)
    return tax_data
