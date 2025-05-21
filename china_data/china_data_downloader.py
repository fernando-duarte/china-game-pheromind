#!/usr/bin/env python3
"""Download and aggregate raw economic data for China."""

import os
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from china_data.utils import get_output_directory
from china_data.utils.wdi_downloader import download_wdi_data
from china_data.utils.pwt_downloader import get_pwt_data
from china_data.utils.imf_loader import load_imf_tax_data
from china_data.utils.markdown_utils import render_markdown_table

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download and integrate China economic data.")
    parser.add_argument('--end-year', type=int, default=None,
                        help='Last year to include in the output (default: current year)')
    args = parser.parse_args()
    end_year = args.end_year if args.end_year else datetime.now().year

    # Get output directory using the common utility function
    output_dir = get_output_directory()
    logger.info("Output files will be saved to: %s", output_dir)

    indicators = {
        'NY.GDP.MKTP.CD': 'GDP_USD',
        'NE.CON.PRVT.CD': 'C_USD',
        'NE.CON.GOVT.CD': 'G_USD',
        'NE.GDI.TOTL.CD': 'I_USD',
        'NE.EXP.GNFS.CD': 'X_USD',
        'NE.IMP.GNFS.CD': 'M_USD',
        'BX.KLT.DINV.WD.GD.ZS': 'FDI_pct_GDP',
        'SP.POP.TOTL': 'POP',
        'SL.TLF.TOTL.IN': 'LF',
    }

    all_data = {}
    # Record the download date for WDI data
    wdi_download_date = datetime.now().strftime('%Y-%m-%d')

    for code, name in indicators.items():
        data = download_wdi_data(code, end_year=end_year)
        if not data.empty:
            data = data[['year', code.replace('.', '_')]].rename(columns={code.replace('.', '_'): name})
            data['year'] = data['year'].astype(int)
            all_data[name] = data
        time.sleep(1)

    # Load IMF tax data using the dedicated loader
    tax_data = load_imf_tax_data()
    if not tax_data.empty:
        all_data['TAX_pct_GDP'] = tax_data

    # Record the download date for PWT data
    pwt_download_date = datetime.now().strftime('%Y-%m-%d')

    try:
        pwt_data = get_pwt_data()
    except Exception as e:
        logger.warning("Could not get PWT data: %s", e)
        pwt_data = pd.DataFrame(columns=['year', 'rgdpo', 'rkna', 'pl_gdpo', 'cgdpo', 'hc'])
    pwt_data['year'] = pwt_data['year'].astype(int)
    all_data['PWT'] = pwt_data

    merged_data = None
    for data in all_data.values():
        if merged_data is None:
            merged_data = data
        else:
            merged_data['year'] = merged_data['year'].astype(int)
            data['year'] = data['year'].astype(int)
            merged_data = pd.merge(merged_data, data, on='year', how='outer')

    merged_data['year'] = pd.to_numeric(merged_data['year'], errors='coerce')
    merged_data = merged_data.dropna(subset=['year'])
    merged_data['year'] = merged_data['year'].astype(int)
    merged_data = merged_data.sort_values('year')

    all_years = pd.DataFrame({'year': range(1960, end_year + 1)})
    merged_data = pd.merge(all_years, merged_data, on='year', how='left')

    # Get the IMF download date from the download_date.txt file
    imf_date = "2025-05-20"  # Default fallback
    try:
        from china_data.utils import find_file
        from china_data.utils.path_constants import get_search_locations_relative_to_root

        date_file = find_file('download_date.txt', get_search_locations_relative_to_root()["input_files"])
        if date_file and os.path.exists(date_file):
            with open(date_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    if key.strip() == 'download_date':
                        imf_date = value.strip()
                        break
    except Exception as e:
        logger.warning("Could not read IMF download date: %s", e)

    # Pass all download dates to the markdown renderer
    markdown_output = render_markdown_table(merged_data,
                                           wdi_date=wdi_download_date,
                                           pwt_date=pwt_download_date,
                                           imf_date=imf_date)

    with open(os.path.join(output_dir, 'china_data_raw.md'), 'w') as f:
        f.write(markdown_output)
    logger.info("Data download and integration complete!")


if __name__ == '__main__':
    main()
