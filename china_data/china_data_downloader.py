#!/usr/bin/env python3
"""
China Economic Data Downloader

This script downloads raw economic data for China from various sources:
1. GDP and its components (C, I, G, X, M) from World Bank
2. Foreign Direct Investment (FDI) as percentage of GDP from World Bank
3. Population and labor force data from World Bank
4. Penn World Table (PWT) data including:
   - Real GDP (rgdpo)
   - Capital stock (rkna)
   - Price level (pl_gdpo)
   - Nominal GDP (cgdpo)
   - Human capital index (hc)

The script creates a single unified dataset in markdown format with raw data only.
No unit conversions or calculations are performed.

Primary data sources:
- World Bank World Development Indicators (WDI)
- Penn World Table (PWT)

DEPENDENCIES:
This script relies on external APIs and services:
1. World Bank API (via pandas-datareader): Used to fetch WDI data
   - May experience rate limiting or service interruptions
   - Requires internet connectivity
2. Penn World Table data hosted at dataverse.nl
   - Requires internet connectivity
   - URL: https://dataverse.nl/api/access/datafile/354095
   - If this URL changes, the script will need to be updated

If any of these services are unavailable, the script will attempt to handle
errors gracefully but may not be able to download complete data.
"""

# Standard library imports
import os
import time
import io
import tempfile
import logging
from datetime import datetime
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Third-party imports
import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
import requests
from tabulate import tabulate
from jinja2 import Template

# Add to_markdown method to pandas DataFrame if it doesn't exist
if not hasattr(pd.DataFrame, 'to_markdown'):
    def to_markdown(df, index=True, **kwargs):
        return tabulate(df, headers='keys', tablefmt='pipe', showindex=index, **kwargs)
    pd.DataFrame.to_markdown = to_markdown

def download_wdi_data(indicator_code, country_code='CN', start_year=1960, end_year=None):
    """
    Download data from World Bank World Development Indicators.

    Parameters:
    -----------
    indicator_code : str
        World Bank indicator code
    country_code : str
        Country code (default: 'CN' for China)
    start_year : int
        Start year for data (default: 1960)
    end_year : int
        End year for data (default: current year)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with the requested data
    """
    if end_year is None:
        end_year = datetime.now().year

    logger.info(f"Downloading {indicator_code} data...")

    # Add retry logic to handle potential connection issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = wb.download(country=country_code,
                              indicator=indicator_code,
                              start=start_year,
                              end=end_year)
            data = data.reset_index()
            # Rename the indicator column to something more meaningful
            data = data.rename(columns={indicator_code: indicator_code.replace('.', '_')})
            logger.debug(f"Successfully downloaded {indicator_code} data with {len(data)} rows")
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt+1} failed. Retrying in 5 seconds... Error: {e}")
                time.sleep(5)
            else:
                logger.error(f"Failed to download {indicator_code} after {max_retries} attempts. Error: {e}")
                # Return empty DataFrame with expected structure
                return pd.DataFrame(columns=['country', 'year', indicator_code.replace('.', '_')])

def get_pwt_data():
    """
    Download raw data from Penn World Table for China.
    Returns a pandas DataFrame with years as index and raw PWT variables.
    """
    logger.info("Downloading Penn World Table data...")

    # Download PWT 10.01 Excel file
    # Source: Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015),
    # "The Next Generation of the Penn World Table" American Economic Review, 105(10), 3150-3182
    # Available at www.ggdc.net/pwt
    excel_url = "https://dataverse.nl/api/access/datafile/354095"  # Excel file from PWT 10.01

    tmp_path = None
    try:
        # Use stream=True to avoid loading the entire file into memory at once
        with requests.get(excel_url, stream=True) as response:
            # Check response status and raise exception for 4xx/5xx errors
            response.raise_for_status()

            # Create a temporary file to store the downloaded content
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                # Write the content in chunks to avoid memory issues with large files
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        tmp.write(chunk)
                tmp_path = tmp.name
                logger.debug(f"Downloaded PWT data to temporary file: {tmp_path}")

        # Read the Excel file
        pwt = pd.read_excel(tmp_path, sheet_name="Data")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred while downloading PWT data: {e}")
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error occurred while downloading PWT data: {e}")
        raise
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout occurred while downloading PWT data: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred while downloading PWT data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while processing PWT data: {e}")
        raise
    finally:
        # Delete the temporary file even if an exception occurs
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug(f"Deleted temporary file: {tmp_path}")

    # Get China's data
    chn = pwt[pwt.countrycode == "CHN"].copy()

    # Extract relevant columns
    chn_data = chn[['year', 'rgdpo', 'rkna', 'pl_gdpo', 'cgdpo', 'hc']].copy()

    # Ensure year is an integer
    chn_data['year'] = chn_data['year'].astype(int)

    return chn_data

def main():
    """Main function to download and integrate all data."""
    logger.info("Starting China Economic Data Downloader...")

    # Create output directory if it doesn't exist
    output_dir = os.path.join(".", "output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output files will be saved to: {output_dir}")

    # Define the indicators to download
    indicators = {
        # GDP and components
        'NY.GDP.MKTP.CD': 'GDP_USD',  # GDP (current US$)
        'NE.CON.PRVT.CD': 'C_USD',    # Household consumption (current US$)
        'NE.CON.GOVT.CD': 'G_USD',    # Government consumption (current US$)
        'NE.GDI.TOTL.CD': 'I_USD',    # Gross capital formation (current US$)
        'NE.EXP.GNFS.CD': 'X_USD',    # Exports (current US$)
        'NE.IMP.GNFS.CD': 'M_USD',    # Imports (current US$)

        # FDI as percentage of GDP
        'BX.KLT.DINV.WD.GD.ZS': 'FDI_pct_GDP',  # Foreign direct investment, net inflows (% of GDP)

        # Population and labor force
        'SP.POP.TOTL': 'POP',         # Population, total
        'SL.TLF.TOTL.IN': 'LF'        # Labor force, total
    }

    # Download all indicators
    all_data = {}
    for code, name in indicators.items():
        data = download_wdi_data(code)
        if not data.empty:
            # Keep only year and the indicator value
            data = data[['year', code.replace('.', '_')]]
            data = data.rename(columns={code.replace('.', '_'): name})
            # Ensure year is an integer
            data['year'] = data['year'].astype(int)
            all_data[name] = data

        # Add a small delay to avoid hitting API rate limits
        time.sleep(1)

    # Get Penn World Table data
    try:
        pwt_data = get_pwt_data()
        all_data['PWT'] = pwt_data
    except Exception as e:
        logger.warning(f"Could not get PWT data: {e}")
        # Create empty DataFrame with the expected structure
        pwt_data = pd.DataFrame(columns=['year', 'rgdpo', 'rkna', 'pl_gdpo', 'cgdpo', 'hc'])
        pwt_data['year'] = pwt_data['year'].astype(int)
        all_data['PWT'] = pwt_data

    # Merge all datasets on year
    logger.info("Merging datasets...")
    merged_data = None

    for name, data in all_data.items():
        if merged_data is None:
            merged_data = data
        else:
            # Ensure both dataframes have year as integer
            merged_data['year'] = merged_data['year'].astype(int)
            data['year'] = data['year'].astype(int)
            merged_data = pd.merge(merged_data, data, on='year', how='outer')

    # Ensure year is an integer
    try:
        merged_data['year'] = merged_data['year'].astype(int)
    except:
        # If conversion fails, try to clean the data first
        merged_data['year'] = pd.to_numeric(merged_data['year'], errors='coerce')
        merged_data = merged_data.dropna(subset=['year'])
        merged_data['year'] = merged_data['year'].astype(int)

    # No unit conversions - keep raw data

    # Sort by year
    merged_data = merged_data.sort_values('year')

    # Create a complete range of years from 1960 to the present
    current_year = datetime.now().year
    all_years = pd.DataFrame({'year': range(1960, current_year + 1)})

    # Merge with the data to ensure all years are included
    merged_data = pd.merge(all_years, merged_data, on='year', how='left')

    # Format the data for display
    display_data = merged_data.copy()

    # Rename columns for better readability in the table
    column_mapping = {
        'year': 'Year',
        'GDP_USD': 'GDP (USD)',
        'C_USD': 'Consumption (USD)',
        'G_USD': 'Government (USD)',
        'I_USD': 'Investment (USD)',
        'X_USD': 'Exports (USD)',
        'M_USD': 'Imports (USD)',
        'FDI_pct_GDP': 'FDI (% of GDP)',
        'POP': 'Population',
        'LF': 'Labor Force',
        'rgdpo': 'PWT rgdpo',
        'rkna': 'PWT rkna',
        'pl_gdpo': 'PWT pl_gdpo',
        'cgdpo': 'PWT cgdpo',
        'hc': 'PWT hc'
    }
    display_data = display_data.rename(columns=column_mapping)

    # Format numeric columns
    for col in display_data.columns:
        if col == 'Year':
            display_data[col] = display_data[col].astype(int)
        elif col in ['Population', 'Labor Force']:
            display_data[col] = display_data[col].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A")
        elif col != 'Year':
            display_data[col] = display_data[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")

    # Create markdown output using Jinja2 template for header
    header_template = Template("""# China Economic Data

Data sources:
- World Bank World Development Indicators (WDI)
- Penn World Table (PWT) version 10.01

## Economic Data (1960-present)

""")

    # Render the header template
    markdown_output = header_template.render()

    # Use pandas to_markdown to generate the table
    markdown_output += display_data.to_markdown(index=False)

    # Use Jinja2 template for notes and sources
    notes_template = Template("""
**Notes:**
- GDP and its components (Consumption, Government, Investment, Exports, Imports) are in current US dollars
- FDI is shown as a percentage of GDP (net inflows)
- Population and Labor Force are in number of people
- PWT rgdpo: Output-side real GDP at chained PPPs (in millions of 2017 USD)
- PWT rkna: Capital stock at constant 2017 national prices (index: 2017=1)
- PWT pl_gdpo: Price level of GDP (price level of USA GDPo in 2017=1)
- PWT cgdpo: Output-side real GDP at current PPPs (in millions of USD)
- PWT hc: Human capital index, based on years of schooling and returns to education

Sources:
- World Bank WDI data: World Development Indicators, The World Bank. Available at https://databank.worldbank.org/source/world-development-indicators
- PWT data: Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "The Next Generation of the Penn World Table" American Economic Review, 105(10), 3150-3182. Available at www.ggdc.net/pwt
""")

    # Render the template and add to markdown output
    markdown_output += notes_template.render()

    # Save the markdown file
    output_file = os.path.join(output_dir, 'china_data_raw.md')
    with open(output_file, 'w') as f:
        f.write(markdown_output)
    logger.info(f"Markdown dataset saved to {output_file}")

    logger.info("Data download and integration complete!")

if __name__ == "__main__":
    main()
