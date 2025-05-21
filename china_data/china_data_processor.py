#!/usr/bin/env python3
"""
China Economic Data Processor

This script processes raw economic data for China and performs various transformations and calculations
to produce a comprehensive dataset of economic variables from 1960 to 2025. The data processing includes:

1. Unit Conversions:
   - Monetary values (GDP, consumption, etc.) from USD to billions USD
   - Population and labor force from people to millions of people

2. Capital Stock Calculations:
   - Uses Penn World Table (PWT) real capital stock index (rkna)
   - Applies price level adjustments using PWT price level data (pl_gdpo)
   - Uses 2017 as base year with capital-output ratio of 3.0 (configurable)
   - Projects forward using investment data with 5% annual depreciation rate

3. Derived Economic Variables:
   - Net exports (exports minus imports)
   - Total Factor Productivity (TFP) using Cobb-Douglas production function
   - Capital-to-output ratio over time
   - Tax revenue in billions USD and as percentage of GDP
   - Openness ratio (sum of exports and imports divided by GDP)
   - Total saving (GDP minus consumption minus government spending)
   - Private saving (GDP minus tax revenue minus consumption)
   - Public saving (tax revenue minus government spending)
   - Saving rate (saving divided by GDP)

4. Extrapolation to 2025 using multiple statistical methods:
   - ARIMA(1,1,1) models for GDP and its components (consumption, government, investment, exports, imports)
   - Linear regression for population, labor force, and human capital
   - Investment-based projection for physical capital (accumulation with depreciation)
   - Average growth rate for variables with stable trends (FDI, tax revenue percentage, TFP)
   - IMF projections for tax revenue (when available through 2030)
   - Derived calculations for other variables based on the extrapolated components

The script takes raw data downloaded by china_data_downloader.py and produces
processed datasets for analysis. The output files include:

1. china_data_processed.csv - Complete dataset with all variables in CSV format
2. china_data_processed.md - Markdown version with detailed notes on computation methods and data sources

Data sources:
- World Bank World Development Indicators (WDI):
  - GDP and its components (consumption, government, investment, exports, imports)
  - Foreign Direct Investment (FDI)
  - Exchange rates
  - Population and labor force data

- Penn World Table (PWT) version 10.01:
  - Real GDP (rgdpo)
  - Capital stock (rkna)
  - Price levels (pl_gdpo)
  - Human capital index (hc)

- International Monetary Fund (IMF) Fiscal Monitor:
  - Tax revenue data (both historical and projections through 2030)
  - Replaces and extends World Bank tax revenue data

Compatible with Python 3.7+ (Python 3.13+ recommended)
"""

# Standard library imports
import os
import argparse
import logging
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# (The custom to_markdown method previously here is no longer needed as we use Jinja2 directly for table generation)

def load_raw_data(data_dir=".", input_file="china_data_raw.md"):
    """
    Load raw data from the data directory.

    Parameters:
    -----------
    data_dir : str
        Directory containing the raw data files
    input_file : str
        Name of the input file containing raw data (markdown format)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with the raw economic data
    """
    # First check if the file exists in the specified directory
    md_file = os.path.join(data_dir, input_file)

    # If not found, check in the output directory
    if not os.path.exists(md_file):
        output_dir = os.path.join(data_dir, "output")
        md_file = os.path.join(output_dir, input_file)

    # If still not found, raise an error
    if not os.path.exists(md_file):
        raise FileNotFoundError(f"Raw data file not found: {input_file} in either {data_dir} or {os.path.join(data_dir, 'output')}. Run china_data_downloader.py first.")

    # Parse the markdown file to extract the data
    # This is a simple approach - in a production environment, you might want to save CSV files instead
    with open(md_file, 'r') as f:
        lines = f.readlines()

    # Find the table header and data rows
    header_idx = None
    for i, line in enumerate(lines):
        if "|   Year | GDP (USD)" in line or "| Year | GDP (USD) |" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find table header in the markdown file.")

    # Extract column names
    header = lines[header_idx].strip().split('|')
    header = [h.strip() for h in header if h.strip()]

    # Rename columns to more code-friendly names
    column_mapping = {
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

    renamed_header = [column_mapping.get(col, col) for col in header]

    # Skip the separator line
    data_start_idx = header_idx + 2

    # Extract data rows
    data = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('**Notes'):
            break

        row = line.split('|')
        row = [cell.strip() for cell in row if cell.strip()]

        if len(row) == len(header):
            # Convert values to appropriate types
            processed_row = []
            for j, value in enumerate(row):
                if j == 0:  # Year
                    processed_row.append(int(value))
                elif value == 'N/A':
                    processed_row.append(np.nan)
                elif renamed_header[j] in ['FDI_pct_GDP', 'TAX_pct_GDP']:  # Percentage indicators
                    processed_row.append(float(value) if value != 'N/A' else np.nan)
                elif renamed_header[j] in ['POP', 'LF']:  # Population, Labor Force
                    processed_row.append(int(value.replace(',', '')) if value != 'N/A' else np.nan)
                else:  # Other numeric values
                    processed_row.append(float(value) if value != 'N/A' else np.nan)

            data.append(processed_row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=renamed_header)

    return df

def load_imf_tax_revenue_data(data_dir="."):
    """
    Load tax revenue data from the IMF CSV file.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the IMF data file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with years as rows and tax revenue (% of GDP) values
    """
    # Path to the IMF data file
    imf_file = os.path.join(data_dir, "china_data/input/dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv")
    
    # Check if the file exists
    if not os.path.exists(imf_file):
        raise FileNotFoundError(f"IMF tax revenue data file not found: {imf_file}")
    
    # Load the IMF data
    imf_data = pd.read_csv(imf_file)
    
    # Filter for the tax revenue indicator and extract relevant columns
    tax_revenue_data = imf_data[imf_data["INDICATOR"] == "G1_S13_POGDP_PT"][["TIME_PERIOD", "OBS_VALUE"]]
    
    # Rename columns for consistency with our data model
    tax_revenue_data = tax_revenue_data.rename(columns={
        "TIME_PERIOD": "year",
        "OBS_VALUE": "TAX_pct_GDP"
    })
    
    # Convert year to integer
    tax_revenue_data["year"] = tax_revenue_data["year"].astype(int)
    
    return tax_revenue_data

def convert_units(raw_data):
    """
    Convert units in the raw data for easier analysis and visualization.

    Conversions performed:
    1. World Bank nominal values (GDP, C, G, I, X, M): USD to billions USD
    2. PWT nominal values (rgdpo, cgdpo): millions USD to billions USD
    3. Population: people to millions of people
    4. Labor Force: people to millions of people

    Parameters:
    -----------
    raw_data : pandas.DataFrame
        DataFrame with raw economic data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with converted units, renamed columns, and missing data filled-in up to 2025 by extrapolation
    """
    # Make a copy of the raw data
    df = raw_data.copy()

    # Convert World Bank nominal values from USD to billions USD
    # These values are in current USD (not millions or billions)
    monetary_cols = ['GDP_USD', 'C_USD', 'G_USD', 'I_USD', 'X_USD', 'M_USD']
    for col in monetary_cols:
        if col in df.columns:
            df[col] = df[col] / 1e9  # Convert to billions

    # Convert PWT monetary values from millions to billions
    # PWT reports these values in millions of USD
    pwt_monetary_cols = ['rgdpo', 'cgdpo']
    for col in pwt_monetary_cols:
        if col in df.columns:
            df[col] = df[col] / 1000  # Convert millions to billions

    # Convert population and labor force from people to millions of people
    demographic_cols = ['POP', 'LF']
    for col in demographic_cols:
        if col in df.columns:
            df[col] = df[col] / 1e6  # Convert to millions

    # Rename columns to reflect the unit conversion
    df = df.rename(columns={
        'GDP_USD': 'GDP_USD_bn',  # Gross Domestic Product (billions USD)
        'C_USD': 'C_USD_bn',      # Household consumption (billions USD)
        'G_USD': 'G_USD_bn',      # Government consumption (billions USD)
        'I_USD': 'I_USD_bn',      # Gross capital formation (billions USD)
        'X_USD': 'X_USD_bn',      # Exports of goods and services (billions USD)
        'M_USD': 'M_USD_bn',      # Imports of goods and services (billions USD)
        'rgdpo': 'rgdpo_bn',      # Real GDP, output side (billions 2017 USD)
        'cgdpo': 'cgdpo_bn',      # Current GDP, output side (billions USD)
        'POP': 'POP_mn',          # Population (millions of people)
        'LF': 'LF_mn'             # Labor force (millions of people)
    })

    return df

def calculate_capital_stock(raw_data, capital_output_ratio=3.0):
    """
    Calculate capital stock in billions of USD from raw Penn World Table data.

    Method:
    1. Use 2017 as the reference year (PWT 10.01 uses 2017 as base year)
    2. Calculate nominal capital stock for 2017 using specified capital-output ratio
    3. Use the real capital stock index (rkna) to extrapolate to other years
    4. Adjust for price level changes using pl_gdpo
    5. Convert from millions to billions of USD

    Parameters:
    -----------
    raw_data : pandas.DataFrame
        DataFrame with raw economic data including PWT variables
    capital_output_ratio : float
        Capital-to-output ratio for base year (2017) capital stock calculation (default: 3.0)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated capital stock in billions of USD (K_USD_bn)
    """
    # Make a copy of the raw data
    df = raw_data.copy()

    # Check if we have the necessary PWT data
    if not all(col in df.columns for col in ['rkna', 'pl_gdpo', 'cgdpo']):
        print("Warning: Missing required PWT data for capital stock calculation")
        df['K_USD_bn'] = np.nan
        return df

    # Get China's nominal GDP in 2017 (in millions of USD)
    try:
        gdp_2017 = df.loc[df.year == 2017, 'cgdpo'].values[0]

        # Use the provided capital-output ratio for China in 2017
        print(f"Using capital-output ratio of {capital_output_ratio} for 2017 capital stock calculation")

        # Calculate the nominal capital stock in 2017 (millions USD)
        capital_stock_2017 = gdp_2017 * capital_output_ratio

        # Get the rkna value for 2017
        rkna_2017 = df.loc[df.year == 2017, 'rkna'].values[0]

        # Get the pl_gdpo value for 2017
        pl_gdpo_2017 = df.loc[df.year == 2017, 'pl_gdpo'].values[0]

        # Calculate capital stock for all years
        df['K_USD_bn'] = np.nan

        for idx, row in df.iterrows():
            if not pd.isna(row['rkna']) and not pd.isna(row['pl_gdpo']):
                df.loc[idx, 'K_USD_bn'] = (
                    (row['rkna'] / rkna_2017) *  # Scale by 2017 value
                    capital_stock_2017 *  # Base value in 2017
                    (row['pl_gdpo'] / pl_gdpo_2017)  # Adjust for price level changes
                ) / 1000  # Convert from millions to billions

        # Round to 2 decimal places
        df['K_USD_bn'] = df['K_USD_bn'].round(2)

    except Exception as e:
        print(f"Error calculating capital stock: {e}")
        df['K_USD_bn'] = np.nan

    return df

def project_capital_stock(processed_data, end_year=2025):
    """
    Project capital stock data using investment and depreciation.

    Parameters:
    -----------
    processed_data : pandas.DataFrame
        DataFrame with processed economic data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with projected capital stock data
    """
    # Extract the latest available capital stock data
    k_data = processed_data[['year', 'K_USD_bn']].copy()

    # Find the last year with available capital stock data
    last_year_with_data = k_data.dropna(subset=['K_USD_bn']).year.max()

    # If we already have data up to end_year, no need to project
    if last_year_with_data >= end_year:
        return k_data

    # Get the last known capital stock value
    last_k = k_data.loc[k_data.year == last_year_with_data, 'K_USD_bn'].values[0]

    # Depreciation rate
    δ = 0.05  # 5% annual depreciation

    # Project capital stock using investment data from processed_data
    try:
        # Get investment data from processed_data
        inv_data = processed_data[['year', 'I_USD_bn']].copy().dropna()

        # Calculate average growth rate from the last 3-4 years of investment data
        last_inv_years = sorted(inv_data.year.tolist())[-4:]
        last_inv_values = [inv_data.loc[inv_data.year == y, 'I_USD_bn'].values[0] for y in last_inv_years]
        inv_growth_rates = [(last_inv_values[i] / last_inv_values[i-1] - 1) for i in range(1, len(last_inv_values))]
        avg_inv_growth = sum(inv_growth_rates) / len(inv_growth_rates)

        # Project investment for future years
        years_to_project = list(range(last_year_with_data + 1, end_year + 1))
        projected_inv = {}

        # Get the last known investment value
        last_inv_year = max(inv_data.year)
        last_inv_value = inv_data.loc[inv_data.year == last_inv_year, 'I_USD_bn'].values[0]

        # Project investment for future years
        for y in years_to_project:
            years_from_last_inv = y - last_inv_year
            projected_inv[y] = last_inv_value * (1 + avg_inv_growth) ** years_from_last_inv

        # Project capital stock
        proj = {}
        proj[last_year_with_data] = last_k

        for y in years_to_project:
            # Use projected investment
            inv_value = projected_inv[y]
            # K_t = (1-δ) * K_{t-1} + I_t
            proj[y] = round((1-δ) * proj[y-1] + inv_value, 2)

        # Create DataFrame with projections
        proj_df = pd.DataFrame(list(proj.items()), columns=['year', 'K_USD_bn'])

        # Merge with original data
        k_data = pd.merge(k_data, proj_df, on='year', how='outer', suffixes=('', '_proj'))

        # Use projected values where original data is missing
        mask = k_data['K_USD_bn'].isna()
        k_data.loc[mask, 'K_USD_bn'] = k_data.loc[mask, 'K_USD_bn_proj']

        # Drop the projection column
        k_data = k_data.drop(columns=['K_USD_bn_proj'])

        return k_data

    except Exception as e:
        # Fallback to simple growth rate method if the above fails
        logger.warning(f"Investment-based capital stock projection failed, falling back to average growth rate. Error: {str(e)}")
        try:
            # Calculate average growth rate from the last 3 years of capital stock data
            k_years = sorted(k_data.dropna(subset=['K_USD_bn']).year.tolist())[-4:]
            k_values = [k_data.loc[k_data.year == y, 'K_USD_bn'].values[0] for y in k_years]
            k_growth_rates = [(k_values[i] / k_values[i-1] - 1) for i in range(1, len(k_values))]
            avg_k_growth = sum(k_growth_rates) / len(k_growth_rates)

            # Project for future years
            years_to_project = list(range(last_year_with_data + 1, end_year + 1))
            for y in years_to_project:
                years_from_last = y - last_year_with_data
                projected_value = last_k * (1 + avg_k_growth) ** years_from_last
                k_data.loc[k_data.year == y, 'K_USD_bn'] = round(projected_value, 2)

            return k_data
        except Exception as e:
            logger.warning(f"Average growth rate fallback for capital stock projection also failed. Error: {str(e)}")
            return k_data

def project_human_capital(processed_data, end_year=2025):
    """
    Project human capital data using exponential smoothing.

    Parameters:
    -----------
    processed_data : pandas.DataFrame
        DataFrame with processed economic data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with projected human capital data
    """
    # Extract the human capital data
    hc_data = processed_data[['year', 'hc']].copy()

    # Find the last year with available human capital data
    last_year_with_data = hc_data.dropna(subset=['hc']).year.max()

    # If we already have data up to end_year, no need to project
    if last_year_with_data >= end_year:
        return hc_data

    # Get the historical data for exponential smoothing
    historical = hc_data.dropna(subset=['hc'])

    # Get all years that need projection (including years between last_year_with_data and end_year)
    all_years = list(range(1960, end_year + 1))
    years_to_project = [y for y in all_years if y > last_year_with_data or
                        (y in hc_data['year'].values and pd.isna(hc_data.loc[hc_data['year'] == y, 'hc'].values[0]))]

    try:
        # Try exponential smoothing first
        model = ExponentialSmoothing(historical['hc'], trend='add', seasonal=None)
        model_fit = model.fit()

        # Project for all years that need projection
        forecast_steps = max(years_to_project) - last_year_with_data
        forecast = model_fit.forecast(steps=forecast_steps)

        # Create a dictionary to map years to forecasted values
        forecast_dict = {}
        for i, year in enumerate(range(last_year_with_data + 1, last_year_with_data + forecast_steps + 1)):
            forecast_dict[year] = round(forecast[i], 4)

        # Create DataFrame with projections
        proj_rows = []
        for year in years_to_project:
            if year > last_year_with_data:
                proj_rows.append({'year': year, 'hc': forecast_dict[year]})

        if proj_rows:
            proj_df = pd.DataFrame(proj_rows)

            # Merge with original data
            hc_data = pd.merge(hc_data, proj_df, on='year', how='outer', suffixes=('', '_proj'))

            # Use projected values where original data is missing
            for _, row in proj_df.iterrows():
                year = row['year']
                hc_value = row['hc']
                hc_data.loc[hc_data['year'] == year, 'hc'] = hc_value

        return hc_data
    except Exception as e:
        # Fallback to linear regression method
        logger.warning(f"Exponential smoothing failed for human capital projection, falling back to linear regression. Error: {str(e)}")
        try:
            # Create a linear regression model for human capital
            X = historical['year'].values.reshape(-1, 1)
            y = historical['hc'].values
            model = LinearRegression()
            model.fit(X, y)

            # Project for all years that need projection
            proj_rows = []
            for year in years_to_project:
                predicted_value = model.predict([[year]])[0]
                proj_rows.append({'year': year, 'hc': round(predicted_value, 4)})

            if proj_rows:
                proj_df = pd.DataFrame(proj_rows)

                # Merge with original data
                hc_data = pd.merge(hc_data, proj_df, on='year', how='outer', suffixes=('', '_proj'))

                # Use projected values where original data is missing
                for _, row in proj_df.iterrows():
                    year = row['year']
                    hc_value = row['hc']
                    hc_data.loc[hc_data['year'] == year, 'hc'] = hc_value

            return hc_data
        except Exception as e:
            logger.warning(f"Linear regression also failed for human capital projection. Error: {str(e)}")
            return hc_data

def calculate_tfp(data, alpha=1/3):
    """
    Calculate Total Factor Productivity (TFP) using the Cobb-Douglas production function:
    Y_t = A_t * K_t^α * (L_t * H_t)^(1-α)

    Solving for A_t:
    A_t = Y_t / (K_t^α * (L_t * H_t)^(1-α))

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with economic data including GDP, capital stock, labor force, and human capital
    alpha : float
        Capital share parameter in the production function (default: 1/3)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with TFP values added
    """
    # Make a copy of the data
    df = data.copy()

    # Check if we have all the necessary data
    required_cols = ['GDP_USD_bn', 'K_USD_bn', 'LF_mn']
    if not all(col in df.columns for col in required_cols):
        df['TFP'] = np.nan
        return df

    # Ensure we have human capital data
    if 'hc' not in df.columns:
        df['hc'] = np.nan

    # Fill missing human capital values using linear interpolation and extrapolation
    if df['hc'].isna().any():
        # Get the available human capital data
        hc_data = df[['year', 'hc']].dropna(subset=['hc'])

        if len(hc_data) >= 2:
            # Create a linear regression model for human capital
            X = hc_data['year'].values.reshape(-1, 1)
            y = hc_data['hc'].values
            model = LinearRegression()
            model.fit(X, y)

            # Predict human capital for years with missing values
            missing_years = df[df['hc'].isna()]['year'].values
            if len(missing_years) > 0:
                predictions = model.predict(missing_years.reshape(-1, 1))

                # Update the missing values
                for i, year in enumerate(missing_years):
                    df.loc[df['year'] == year, 'hc'] = round(predictions[i], 4)

    # Calculate TFP
    try:
        # A_t = Y_t / (K_t^α * (L_t * H_t)^(1-α))
        df['TFP'] = df['GDP_USD_bn'] / (
            (df['K_USD_bn'] ** alpha) *
            ((df['LF_mn'] * df['hc']) ** (1 - alpha))
        )

        # Round to 4 decimal places
        df['TFP'] = df['TFP'].round(4)

    except Exception:
        df['TFP'] = np.nan

    return df

def extrapolate_series_to_end_year(data, end_year=2025, raw_data=None):
    """
    Extrapolate all time series in the data to 2025.

    Uses a combination of methods depending on the series:
    1. For GDP and components: ARIMA(1,1,1) model
    2. For population and labor force: Linear regression
    3. For human capital: Exponential smoothing
    4. For other series: Average growth rate

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with economic data

    Returns:
    --------
    tuple
        (DataFrame with extrapolated data to 2025,
         Dictionary tracking which method and years were used for each variable)
    """
    # Make a copy of the data
    df = data.copy()

    # Dictionary to track which method and years were used for each variable
    extrapolation_info = {}

    # Get the maximum year in the data
    max_year = df.year.max()

    # Check if we have complete data up to end_year
    if max_year >= end_year:
        # Check if we have missing values for key variables in the last two years
        missing_values = False
        key_variables = ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn', 'X_USD_bn', 'M_USD_bn', 'POP_mn', 'LF_mn']
        for year in [end_year-1, end_year]:
            for var in key_variables:
                if var in df.columns and pd.isna(df.loc[df.year == year, var].values[0]):
                    missing_values = True
                    break
            if missing_values:
                break
        if not missing_values:
            return df, {}
        else:
            years_to_add = [end_year-1, end_year]
    else:
        years_to_add = list(range(max_year + 1, end_year + 1))

    # Create a DataFrame for the new years
    new_years_df = pd.DataFrame({'year': years_to_add})

    # Add the new years to the data
    df = pd.concat([df, new_years_df], ignore_index=True)

    # List of columns to extrapolate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_extrapolate = [col for col in numeric_cols if col != 'year']

    # Group columns by extrapolation method
    gdp_components = ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn', 'X_USD_bn', 'M_USD_bn', 'NX_USD_bn']
    demographic = ['POP_mn', 'LF_mn']
    human_capital = ['hc']
    # FDI and other variables will use the default average growth rate method

    # For each column, extrapolate from the last year with data up to end_year (filling all missing years in between)
    for col in cols_to_extrapolate:
        # Skip columns with all NaN values
        if df[col].isna().all():
            continue

        # Get the historical data for this column
        historical = df[['year', col]].dropna()
        if len(historical) == 0:
            continue

        # Find the last year with data for this column
        last_year_with_data = int(historical['year'].max())
        last_value = historical[historical['year'] == last_year_with_data][col].values[0]

        # Determine which years need extrapolation for this column
        years_to_extrapolate = [year for year in range(last_year_with_data + 1, end_year + 1)]
        if not years_to_extrapolate:
            continue

        # Extrapolation logic (ARIMA, linear regression, etc.)
        # For GDP and components, use ARIMA
        if col in gdp_components and len(historical) >= 5:
            try:
                model = ARIMA(historical[col], order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(years_to_extrapolate))
                forecast_values = forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)
                for i, year in enumerate(years_to_extrapolate):
                    df.loc[df.year == year, col] = round(max(0, forecast_values[i]), 4)
                if years_to_extrapolate:
                    extrapolation_info[col] = {
                        'method': 'ARIMA(1,1,1)',
                        'years': years_to_extrapolate
                    }
                continue
            except Exception as e:
                logger.warning(f"ARIMA failed for {col}, falling back to average growth rate. Error: {str(e)}")
        # For population and labor force, use linear regression
        if col in demographic and len(historical) >= 2:
            try:
                X = historical['year'].values.reshape(-1, 1)
                y = historical[col].values
                model = LinearRegression()
                model.fit(X, y)
                for i, year in enumerate(years_to_extrapolate):
                    pred = model.predict([[year]])[0]
                    df.loc[df.year == year, col] = round(max(0, pred), 4)
                if years_to_extrapolate:
                    extrapolation_info[col] = {
                        'method': 'Linear regression',
                        'years': years_to_extrapolate
                    }
                continue
            except Exception as e:
                logger.warning(f"Linear regression failed for {col}, falling back to average growth rate. Error: {str(e)}")
        # For human capital, use exponential smoothing
        if col in human_capital and len(historical) >= 2:
            try:
                model = ExponentialSmoothing(historical[col], trend='add', seasonal=None)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(years_to_extrapolate))
                for i, year in enumerate(years_to_extrapolate):
                    df.loc[df.year == year, col] = round(max(0, forecast[i]), 4)
                if years_to_extrapolate:
                    extrapolation_info[col] = {
                        'method': 'Exponential smoothing',
                        'years': years_to_extrapolate
                    }
                continue
            except Exception as e:
                logger.warning(f"Exponential smoothing failed for {col}, falling back to average growth rate. Error: {str(e)}")
        # Fallback: average growth rate
        if len(historical) >= 2:
            n_years = min(4, len(historical))
            last_years = historical[col].iloc[-n_years:].values
            growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
            avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0.03
        else:
            avg_growth = 0.03
        for i, year in enumerate(years_to_extrapolate):
            years_from_last = year - last_year_with_data
            projected_value = last_value * (1 + avg_growth) ** years_from_last
            df.loc[df.year == year, col] = round(projected_value, 4)
        if years_to_extrapolate:
            extrapolation_info[col] = {
                'method': 'Average growth rate',
                'years': years_to_extrapolate
            }

    # Calculate Net Exports for extrapolated years if Exports and Imports are available
    for year in years_to_add:
        if 'X_USD_bn' in df.columns and 'M_USD_bn' in df.columns:
            x_value = df.loc[df.year == year, 'X_USD_bn'].values[0]
            m_value = df.loc[df.year == year, 'M_USD_bn'].values[0]
            if not pd.isna(x_value) and not pd.isna(m_value):
                df.loc[df.year == year, 'NX_USD_bn'] = round(x_value - m_value, 4)

    # Ensure all variables are extrapolated through end_year
    # Define key variables that must be extrapolated
    key_variables = ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn', 'X_USD_bn', 'M_USD_bn',
                     'POP_mn', 'LF_mn', 'FDI_pct_GDP', 'TAX_pct_GDP', 'hc', 'K_USD_bn']

    for year in years_to_add:
        for col in key_variables:
            if col in df.columns and pd.isna(df.loc[df.year == year, col].values[0]):
                # Try to extrapolate using average growth rate from the last available data
                last_valid_data = df[df.year < year][[col]].dropna()
                if not last_valid_data.empty:
                    last_value = last_valid_data.iloc[-1].values[0]
                    last_year = df.loc[last_valid_data.index[-1], 'year']
                    years_diff = year - last_year

                    # Set default growth rates based on variable type
                    if col in ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn', 'X_USD_bn', 'M_USD_bn']:
                        default_growth = 0.05  # 5% for economic variables
                    elif col in ['POP_mn']:
                        default_growth = 0.005  # 0.5% for population
                    elif col in ['LF_mn']:
                        default_growth = 0.01  # 1% for labor force
                    elif col in ['hc']:
                        default_growth = 0.01  # 1% for human capital
                    elif col in ['K_USD_bn']:
                        default_growth = 0.04  # 4% for capital stock
                    else:
                        default_growth = 0.03  # 3% default

                    # Try to calculate growth rate from historical data
                    historical = df[df.year <= max_year][[col]].dropna()
                    if len(historical) >= 2:
                        # Use up to 5 years of historical data, but at least 2
                        n_years = min(5, len(historical))
                        last_years = historical.iloc[-n_years:].values.flatten()
                        if len(last_years) > 1:
                            growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                            avg_growth = sum(growth_rates) / len(growth_rates)
                        else:
                            avg_growth = default_growth
                    else:
                        avg_growth = default_growth

                    projected_value = last_value * (1 + avg_growth) ** years_diff
                    df.loc[df.year == year, col] = round(projected_value, 4)

    # Recalculate Net Exports for extrapolated years
    for year in years_to_add:
        if all(col in df.columns for col in ['X_USD_bn', 'M_USD_bn']):
            x_value = df.loc[df.year == year, 'X_USD_bn'].values[0]
            m_value = df.loc[df.year == year, 'M_USD_bn'].values[0]
            if not pd.isna(x_value) and not pd.isna(m_value):
                df.loc[df.year == year, 'NX_USD_bn'] = round(x_value - m_value, 4)

    # Add tracking for physical capital (K_USD_bn) which is handled by project_capital_stock
    if 'K_USD_bn' in df.columns:
        # Check if there are any years after 2019 (PWT data ends in 2019)
        if any(year > 2019 for year in df['year']):
            extrapolation_info['K_USD_bn'] = {
                'method': 'Investment-based projection',
                'years': [year for year in years_to_add if year > 2019]
            }

    # Add tracking for derived variables
    if 'NX_USD_bn' in df.columns:
        nx_years = [year for year in years_to_add if year > df['year'][df['NX_USD_bn'].notna()].max()]
        if nx_years:
            extrapolation_info['NX_USD_bn'] = {
                'method': 'Derived calculation',
                'years': nx_years
            }
    if 'TFP' in df.columns:
        tfp_years = [year for year in years_to_add if year > df['year'][df['TFP'].notna()].max()]
        if tfp_years:
            extrapolation_info['TFP'] = {
                'method': 'Derived calculation',
                'years': tfp_years
            }

    # Add tracking for human capital if it's not already tracked
    if 'hc' not in extrapolation_info:
        # Check if we have human capital data
        if 'hc' in df.columns and not df['hc'].isna().all():
            # Find the last year with data
            last_year_with_data = None
            for year in range(2023, 2018, -1):
                if year in df['year'].values:
                    if not pd.isna(df.loc[df['year'] == year, 'hc'].values[0]):
                        last_year_with_data = year
                        break

            if last_year_with_data is not None:
                # Don't modify last_years dictionary directly
                extrapolation_info['hc'] = {
                    'method': 'Linear regression',
                    'years': [year for year in years_to_add if year > last_year_with_data]
                }

    # Add special handling for TFP description
    if 'TFP' in extrapolation_info and extrapolation_info['TFP']['method'] == "Derived calculation":
        # Find the years for TFP
        tfp_years = ""
        if 'TFP' in last_years and last_years['TFP'] is not None:
            start_year = last_years['TFP'] + 1
            end_year = end_year
            tfp_years = f" ({start_year}-{end_year})"
        else:
            tfp_years = " (2024-2025)"

        # Create a special key for TFP with description
        extrapolation_info.pop('TFP')
        extrapolation_info['TFP' + tfp_years] = {
            'method': 'Derived calculation',
            'years': [year for year in years_to_add if year > df['year'][df['TFP'].notna()].max()]
        }

    # After extrapolation, update extrapolation_info to include all years after the last year with actual data from the original raw data for each variable
    for col in cols_to_extrapolate:
        # Find the last year with actual (non-extrapolated) data for this column in the original raw data
        if col in raw_data.columns:
            raw_non_nan = raw_data[['year', col]].dropna()
            if len(raw_non_nan) == 0:
                continue
            last_actual_year = int(raw_non_nan['year'].max())
        else:
            # Fallback: use the last year with any data in the processed DataFrame
            historical = df[['year', col]].dropna()
            if len(historical) == 0:
                continue
            last_actual_year = int(historical['year'].max())
        # If the column was extrapolated, update the years to include all years after last_actual_year up to end_year
        if last_actual_year < end_year:
            extrap_years = [year for year in range(last_actual_year + 1, end_year + 1)]
            if extrap_years:
                # Determine method used (prefer existing, else fallback to 'Extrapolated')
                method = extrapolation_info.get(col, {}).get('method', 'Extrapolated')
                extrapolation_info[col] = {
                    'method': method,
                    'years': extrap_years
                }

    return df, extrapolation_info

def format_data_for_output(data_df):
    """
    Formats the DataFrame for consistent output in CSV and Markdown.
    Numeric values are converted to strings with specific formatting.
    NaNs are converted to 'nan'.
    """
    formatted_df = data_df.copy()
    for col_name in formatted_df.columns:
        new_col_values = []
        for val in formatted_df[col_name]:
            if pd.isna(val):
                new_col_values.append('nan')
            elif isinstance(val, float):
                # Columns requiring higher precision (e.g., percentages, indices)
                if col_name in ['FDI (% of GDP)', 'TFP', 'Human Capital', 'Openness Ratio', 'Saving Rate']: # Ratios and Indices
                    new_col_values.append(f"{val:.4f}".rstrip('0').rstrip('.'))
                # Columns representing billions USD, also with potentially more precision
                elif col_name in ['GDP', 'Consumption', 'Government', 'Investment', 'Exports', 'Imports', 'Net Exports', 'Physical Capital', 'Tax Revenue (bn USD)', 'Saving (bn USD)', 'Private Saving (bn USD)', 'Public Saving (bn USD)']:
                    new_col_values.append(f"{val:.4f}".rstrip('0').rstrip('.'))
                # Other float columns (e.g., Population, Labor Force in millions)
                elif col_name in ['Population', 'Labor Force']: # Millions
                    new_col_values.append(f"{val:.2f}".rstrip('0').rstrip('.'))
                else: # Default for any other floats not specifically handled
                    new_col_values.append(f"{val:.2f}".rstrip('0').rstrip('.'))
            elif isinstance(val, int) and col_name == 'Year':
                new_col_values.append(str(val))
            # Handle cases where Population/Labor Force might already be float due to calculations
            elif col_name in ['Population', 'Labor Force'] and isinstance(val, (int, float)): # Millions
                 new_col_values.append(f"{val:.2f}".rstrip('0').rstrip('.'))
            else:
                new_col_values.append(str(val))
        formatted_df[col_name] = new_col_values
    return formatted_df

def create_markdown_table(data, output_path, extrapolation_info, alpha=1/3, capital_output_ratio=3.0, input_file="china_data_raw.md", end_year=2025):
    """
    Create a markdown file with a table of the processed data and notes on computation.
    Assumes 'data' DataFrame is already formatted with string values.
    """
    # Define column mapping for determining extrapolated years (used for notes, not table formatting here)
    column_mapping = {
        'Year': 'year',
        'GDP': 'GDP_USD_bn',
        'Consumption': 'C_USD_bn',
        'Government': 'G_USD_bn',
        'Investment': 'I_USD_bn',
        'Exports': 'X_USD_bn',
        'Imports': 'M_USD_bn',
        'Net Exports': 'NX_USD_bn',
        'Population': 'POP_mn',
        'Labor Force': 'LF_mn',
        'Physical Capital': 'K_USD_bn',
        'TFP': 'TFP',
        'FDI (% of GDP)': 'FDI_pct_GDP',
        'Human Capital': 'hc',
        'Tax Revenue (bn USD)': 'T_USD_bn',
        'Openness Ratio': 'Openness_Ratio',
        'Saving (bn USD)': 'S_USD_bn',
        'Private Saving (bn USD)': 'S_priv_USD_bn',
        'Public Saving (bn USD)': 'S_pub_USD_bn',
        'Saving Rate': 'Saving_Rate'
    }
    # Data is already formatted, prepare for Jinja2
    table_headers = list(data.columns)
    table_rows = data.values.tolist()

    # The `extrapolation_info` (passed to this function) is the primary source for notes on extrapolation.
    # The `column_mapping` (defined at the start of this function) maps display names to internal DataFrame column names
    # which are keys in `extrapolation_info`.

    # Group variables by extrapolation method and list years
    methods_to_variables = {}
    for var, info in extrapolation_info.items():
        method = info['method']
        years = info['years']
        if not years:
            continue
        if method not in methods_to_variables:
            methods_to_variables[method] = []
        display_name = var
        for display, internal in column_mapping.items():
            if internal == var:
                display_name = display
                break
        # List all years, or as a range if consecutive
        if len(years) == 1:
            years_str = f"{years[0]}"
        else:
            # Check if years are consecutive
            if years == list(range(years[0], years[-1]+1)):
                years_str = f"{years[0]}-{years[-1]}"
            else:
                years_str = ', '.join(str(y) for y in years)
        methods_to_variables[method].append(f"{display_name} ({years_str})")

    # Prepare the extrapolation methods markdown
    methods_md = ""
    for method, variables in methods_to_variables.items():
        if method == "ARIMA(1,1,1)":
            methods_md += f"   - **ARIMA(1,1,1) model**: \n"
        elif method == "Linear regression":
            methods_md += f"   - **Linear regression**: \n"
        elif method == "Exponential smoothing":
            methods_md += f"   - **Exponential smoothing**: \n"
        elif method == "Average growth rate":
            methods_md += f"   - **Average growth rate of historical data**: \n"
        elif method == "Investment-based projection":
            methods_md += f"   - **Investment-based projection**: \n"
        elif method == "Derived calculation":
            methods_md += f"   - **Derived calculations**: \n"
        else:
            methods_md += f"   - **{method}**: \n"
        for var in variables:
            var_parts = var.split(" (")
            var_name = var_parts[0]
            years_part = f" ({var_parts[1]}" if len(var_parts) > 1 else ""
            if var_name == "Physical Capital" and method == "Investment-based projection":
                methods_md += f"     - {var_name}{years_part}: Projected using the formula K_t = K_{{t-1}} * (1-delta) + I_t, where delta = 0.05 (5% depreciation rate) and I_t is investment in year t\n"
            elif var_name == "Net Exports" and method == "Derived calculation":
                methods_md += f"     - {var_name}{years_part}: Calculated as Exports - Imports for each projected year\n"
            elif "TFP" in var and method == "Derived calculation":
                methods_md += f"     - {var_name}{years_part}: Recalculated using the Cobb-Douglas formula for each projected year\n"
            else:
                methods_md += f"     - {var}\n"
    methods_md += "\n"

    # Jinja2 template for the markdown report
    template_str = f'''# China Economic Variables

## Economic Variables (1960-{{{{end_year}}}})

|{{% for h in table_headers %}} {{{{ h }}}} |{{% endfor %}}
|{{% for h in table_headers %}} --- |{{% endfor %}}
{{% for row in table_rows %}}|{{% for cell in row %}} {{{{ cell }}}} |{{% endfor %}}
{{% endfor %}}

## Notes on Computation

The raw data in `{{{{ input_file }}}}` comes from the following sources:

1. **Original Data Sources**:
   - World Bank World Development Indicators (WDI) for GDP components, FDI, population, and labor force
   - Penn World Table (PWT) version 10.01 for human capital index and capital stock related variables

This processed dataset was created by applying the following transformations to the raw data:

2. **Unit Conversions**:
   - GDP and its components (Consumption, Government, Investment, Exports, Imports) were converted from USD to billions USD
   - Population and Labor Force were converted from people to millions of people

3. **Derived Variables**:
   - **Net Exports**: Calculated as Exports - Imports (in billions USD)
     ```
     Net Exports = Exports - Imports
     ```

   - **Physical Capital**: Calculated using PWT data with the following formula:
     ```
     K_t = (rkna_t / rkna_2017) * K_2017 * (pl_gdpo_t / pl_gdpo_2017)
     ```
     Where:
     - K_t is the capital stock in year t (billions USD)
     - rkna_t is the real capital stock index in year t (from PWT)
     - rkna_2017 is the real capital stock index in 2017 (from PWT)
     - K_2017 is the nominal capital stock in 2017, estimated as GDP_2017 * {{{{ capital_output_ratio }}}} (capital-output ratio)
     - pl_gdpo_t is the price level of GDP in year t (from PWT)
     - pl_gdpo_2017 is the price level of GDP in 2017 (from PWT)

   - **TFP (Total Factor Productivity)**: Calculated using the Cobb-Douglas production function:
     ```
     TFP_t = Y_t / (K_t^alpha * (L_t * H_t)^(1-alpha))
     ```
     Where:
     - Y_t is GDP in year t (billions USD)
     - K_t is Physical Capital in year t (billions USD)
     - L_t is Labor Force in year t (millions of people)
     - H_t is Human Capital index in year t
     - alpha = 0.33 (capital share parameter)

4. **Extrapolation to {{{{end_year}}}}**:
   Each series was extrapolated using the following methods:

{{{{ methods_md }}}}

Data processing was performed using Python 3.7+ (Python 3.13+ recommended).
'''
    template = Template(template_str)
    markdown_content = template.render(
        table_headers=table_headers,
        table_rows=table_rows,
        input_file=input_file,
        capital_output_ratio=capital_output_ratio,
        alpha=alpha,
        methods_md=methods_md,
        end_year=end_year
    )

    # Write to file
    with open(output_path, 'w') as f:
        f.write(markdown_content)

    print(f"Markdown table saved to {output_path}")

def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
    --------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process China economic data and calculate economic variables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--input-file",
        default="china_data_raw.md",
        help="Input file containing raw economic data (markdown format)"
    )

    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=1/3,
        help="Capital share parameter (alpha) for TFP calculation in Cobb-Douglas production function"
    )

    parser.add_argument(
        "-o", "--output-file",
        default="china_data_processed",
        help="Base name for output files (without extension)"
    )

    parser.add_argument(
        "-k", "--capital-output-ratio",
        type=float,
        default=3.0,
        help="Capital-to-output ratio for base year (2017) capital stock calculation"
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to extrapolate/process (default: 2025)"
    )

    return parser.parse_args()

def main():
    """Main function to process the data."""
    print("Starting China Economic Data Processor...")

    # Parse command-line arguments
    args = parse_arguments()

    # Extract arguments
    input_file = args.input_file
    alpha = args.alpha
    output_base = args.output_file
    capital_output_ratio = args.capital_output_ratio
    end_year = args.end_year

    # Create output directory if it doesn't exist
    output_dir = os.path.join(".", "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved to: {output_dir}")

    print(f"Using input file: {input_file}")
    print(f"Using alpha value: {alpha}")
    print(f"Using capital-output ratio: {capital_output_ratio}")
    print(f"Output will be saved with base name: {output_base}")

    # Load raw data
    try:
        raw_data = load_raw_data(data_dir=".", input_file=input_file)
        print(f"Loaded raw data with {len(raw_data)} rows.")
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return
    
    # Load IMF tax revenue data
    try:
        imf_tax_data = load_imf_tax_revenue_data(data_dir=".")
        print(f"Loaded IMF tax revenue data with {len(imf_tax_data)} rows (years {imf_tax_data['year'].min()}-{imf_tax_data['year'].max()}).")
    except Exception as e:
        print(f"Error loading IMF tax revenue data: {e}")
        return

    # Convert units
    converted_data = convert_units(raw_data)
    print("Converted units in raw data.")

    # Calculate capital stock from PWT data
    data_with_capital = calculate_capital_stock(raw_data, capital_output_ratio=capital_output_ratio)

    # Merge the calculated capital stock with the converted data
    processed_data = converted_data.copy()
    processed_data['K_USD_bn'] = data_with_capital['K_USD_bn']

    # Project capital stock
    k_data = project_capital_stock(processed_data, end_year=end_year)

    # Project human capital
    hc_data = project_human_capital(raw_data, end_year=end_year)

    # Combine all data
    # Start with the processed data
    merged_data = processed_data.copy()

    # Update capital stock with projections
    for _, row in k_data.iterrows():
        year = row['year']
        k_value = row['K_USD_bn']
        if not pd.isna(k_value):
            merged_data.loc[merged_data['year'] == year, 'K_USD_bn'] = k_value

    # Update human capital with projections
    for _, row in hc_data.iterrows():
        year = row['year']
        if 'hc' in row and not pd.isna(row['hc']):
            hc_value = row['hc']
            merged_data.loc[merged_data['year'] == year, 'hc'] = hc_value

    # Ensure we have human capital values for all years (2020-2025)
    # Use linear interpolation for missing values
    if 'hc' in merged_data.columns:
        # Get the available human capital data
        available_hc = merged_data[['year', 'hc']].dropna(subset=['hc'])

        if len(available_hc) >= 2:
            # Create a linear regression model
            X = available_hc['year'].values.reshape(-1, 1)
            y = available_hc['hc'].values
            model = LinearRegression()
            model.fit(X, y)

            # Fill in missing values for all years
            for year in range(1960, end_year+1):
                if year in merged_data['year'].values:
                    idx = merged_data[merged_data['year'] == year].index
                    if pd.isna(merged_data.loc[idx, 'hc'].values[0]):
                        predicted_value = model.predict([[year]])[0]
                        merged_data.loc[idx, 'hc'] = round(predicted_value, 4)

            # Recalculate TFP after filling in human capital values
            merged_data = calculate_tfp(merged_data, alpha=alpha)
    
    # Replace tax revenue data with IMF data
    if 'TAX_pct_GDP' in merged_data.columns:
        # First, remove the existing tax revenue data
        merged_data['TAX_pct_GDP'] = np.nan
        
        # Then merge with the IMF tax revenue data
        merged_data = pd.merge(
            merged_data,
            imf_tax_data,
            on='year',
            how='left',
            suffixes=('', '_imf')
        )
        
        print("Replaced tax revenue data with IMF data.")

    # Calculate additional variables
    # Net exports
    if all(col in merged_data.columns for col in ['X_USD_bn', 'M_USD_bn']):
        merged_data['NX_USD_bn'] = merged_data['X_USD_bn'] - merged_data['M_USD_bn']
        print("Calculated net exports.")

    # Capital-to-output ratio
    if all(col in merged_data.columns for col in ['K_USD_bn', 'GDP_USD_bn']):
        merged_data['K_Y_ratio'] = merged_data['K_USD_bn'] / merged_data['GDP_USD_bn']
        print("Calculated capital-to-output ratio.")

    # Calculate TFP with provided alpha value
    merged_data = calculate_tfp(merged_data, alpha=alpha)
    print(f"Calculated total factor productivity (TFP) with alpha={alpha}.")

    # Extrapolate all series to end_year
    merged_data, extrapolation_info = extrapolate_series_to_end_year(merged_data, end_year=end_year, raw_data=raw_data)
    
    # Update extrapolation info for tax revenue since we're using IMF data with existing projections
    if 'TAX_pct_GDP' in extrapolation_info:
        # Get the years from IMF data that are beyond the last year of actual data
        projected_years = [y for y in imf_tax_data['year'] if y > 2023]  # Assuming 2023 is the last year of actual data
        if projected_years:
            extrapolation_info['TAX_pct_GDP'] = {
                'method': 'IMF projections',
                'years': projected_years
            }

    # Recalculate derived variables for extrapolated years
    # Recalculate Net Exports for all years
    if all(col in merged_data.columns for col in ['X_USD_bn', 'M_USD_bn']):
        merged_data['NX_USD_bn'] = merged_data['X_USD_bn'] - merged_data['M_USD_bn']

    # Recalculate Capital-to-output ratio for all years
    if all(col in merged_data.columns for col in ['K_USD_bn', 'GDP_USD_bn']):
        merged_data['K_Y_ratio'] = merged_data['K_USD_bn'] / merged_data['GDP_USD_bn']

    # Recalculate TFP for all years with the provided alpha value
    merged_data = calculate_tfp(merged_data, alpha=alpha)

    # Calculate Tax Revenue in billions USD
    if 'TAX_pct_GDP' in merged_data.columns and 'GDP_USD_bn' in merged_data.columns:
        merged_data['T_USD_bn'] = (merged_data['TAX_pct_GDP'] / 100) * merged_data['GDP_USD_bn']
        print("Calculated Tax Revenue (bn USD).")

    # Calculate Openness Ratio
    if all(col in merged_data.columns for col in ['X_USD_bn', 'M_USD_bn', 'GDP_USD_bn']):
        merged_data['Openness_Ratio'] = (merged_data['X_USD_bn'] + merged_data['M_USD_bn']) / merged_data['GDP_USD_bn']
        print("Calculated Openness Ratio.")

    # Calculate Saving (bn USD)
    if all(col in merged_data.columns for col in ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn']):
        merged_data['S_USD_bn'] = merged_data['GDP_USD_bn'] - merged_data['C_USD_bn'] - merged_data['G_USD_bn']
        print("Calculated Saving (bn USD).")

    # Calculate Private Saving (bn USD)
    if all(col in merged_data.columns for col in ['GDP_USD_bn', 'T_USD_bn', 'C_USD_bn']):
        merged_data['S_priv_USD_bn'] = merged_data['GDP_USD_bn'] - merged_data['T_USD_bn'] - merged_data['C_USD_bn']
        print("Calculated Private Saving (bn USD).")

    # Calculate Public Saving (bn USD)
    if all(col in merged_data.columns for col in ['T_USD_bn', 'G_USD_bn']):
        merged_data['S_pub_USD_bn'] = merged_data['T_USD_bn'] - merged_data['G_USD_bn']
        print("Calculated Public Saving (bn USD).")

    # Calculate Saving Rate
    if all(col in merged_data.columns for col in ['S_USD_bn', 'GDP_USD_bn']):
        merged_data['Saving_Rate'] = merged_data['S_USD_bn'] / merged_data['GDP_USD_bn']
        print("Calculated Saving Rate.")


    # Extrapolate all series to end_year
    merged_data, extrapolation_info = extrapolate_series_to_end_year(merged_data, end_year=end_year, raw_data=raw_data)

    # Recalculate derived variables for extrapolated years
    # Recalculate Net Exports for all years
    if all(col in merged_data.columns for col in ['X_USD_bn', 'M_USD_bn']):
        merged_data['NX_USD_bn'] = merged_data['X_USD_bn'] - merged_data['M_USD_bn']

    # Recalculate Capital-to-output ratio for all years
    if all(col in merged_data.columns for col in ['K_USD_bn', 'GDP_USD_bn']):
        merged_data['K_Y_ratio'] = merged_data['K_USD_bn'] / merged_data['GDP_USD_bn']
        
    # Recalculate Tax Revenue (bn USD) for all years
    if 'TAX_pct_GDP' in merged_data.columns and 'GDP_USD_bn' in merged_data.columns:
        merged_data['T_USD_bn'] = (merged_data['TAX_pct_GDP'] / 100) * merged_data['GDP_USD_bn']
        if 'T_USD_bn' not in extrapolation_info: # Add to extrapolation info if not already handled by base series
             extrapolation_info['T_USD_bn'] = {'method': 'Derived calculation', 'years': [y for y in range(merged_data['year'].min(), end_year + 1) if y > raw_data['year'].max()]}


    # Recalculate Openness Ratio for all years
    if all(col in merged_data.columns for col in ['X_USD_bn', 'M_USD_bn', 'GDP_USD_bn']):
        merged_data['Openness_Ratio'] = (merged_data['X_USD_bn'] + merged_data['M_USD_bn']) / merged_data['GDP_USD_bn']
        if 'Openness_Ratio' not in extrapolation_info:
            extrapolation_info['Openness_Ratio'] = {'method': 'Derived calculation', 'years': [y for y in range(merged_data['year'].min(), end_year + 1) if y > raw_data['year'].max()]}


    # Recalculate Saving (bn USD) for all years
    if all(col in merged_data.columns for col in ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn']):
        merged_data['S_USD_bn'] = merged_data['GDP_USD_bn'] - merged_data['C_USD_bn'] - merged_data['G_USD_bn']
        if 'S_USD_bn' not in extrapolation_info:
            extrapolation_info['S_USD_bn'] = {'method': 'Derived calculation', 'years': [y for y in range(merged_data['year'].min(), end_year + 1) if y > raw_data['year'].max()]}

    # Recalculate Private Saving (bn USD) for all years
    if all(col in merged_data.columns for col in ['GDP_USD_bn', 'T_USD_bn', 'C_USD_bn']):
        merged_data['S_priv_USD_bn'] = merged_data['GDP_USD_bn'] - merged_data['T_USD_bn'] - merged_data['C_USD_bn']
        if 'S_priv_USD_bn' not in extrapolation_info:
             extrapolation_info['S_priv_USD_bn'] = {'method': 'Derived calculation', 'years': [y for y in range(merged_data['year'].min(), end_year + 1) if y > raw_data['year'].max()]}


    # Recalculate Public Saving (bn USD) for all years
    if all(col in merged_data.columns for col in ['T_USD_bn', 'G_USD_bn']):
        merged_data['S_pub_USD_bn'] = merged_data['T_USD_bn'] - merged_data['G_USD_bn']
        if 'S_pub_USD_bn' not in extrapolation_info:
            extrapolation_info['S_pub_USD_bn'] = {'method': 'Derived calculation', 'years': [y for y in range(merged_data['year'].min(), end_year + 1) if y > raw_data['year'].max()]}

    # Recalculate Saving Rate for all years
    if all(col in merged_data.columns for col in ['S_USD_bn', 'GDP_USD_bn']):
        merged_data['Saving_Rate'] = merged_data['S_USD_bn'] / merged_data['GDP_USD_bn']
        if 'Saving_Rate' not in extrapolation_info:
            extrapolation_info['Saving_Rate'] = {'method': 'Derived calculation', 'years': [y for y in range(merged_data['year'].min(), end_year + 1) if y > raw_data['year'].max()]}

    # Recalculate TFP for all years with the provided alpha value
    merged_data = calculate_tfp(merged_data, alpha=alpha)

    # Select and reorder columns as requested
    output_columns = [
        'year', 'GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn',
        'X_USD_bn', 'M_USD_bn', 'NX_USD_bn',
        'T_USD_bn', # Tax Revenue bn USD
        'Openness_Ratio',
        'S_USD_bn', # Saving
        'S_priv_USD_bn', # Private Saving
        'S_pub_USD_bn', # Public Saving
        'Saving_Rate',
        'POP_mn', 'LF_mn',
        'K_USD_bn', 'TFP', 'FDI_pct_GDP', 'TAX_pct_GDP', 'hc' # TAX_pct_GDP is kept for reference
    ]

    # Rename columns to match requested format
    column_mapping = {
        'year': 'Year',
        'GDP_USD_bn': 'GDP',
        'C_USD_bn': 'Consumption',
        'G_USD_bn': 'Government',
        'I_USD_bn': 'Investment',
        'X_USD_bn': 'Exports',
        'M_USD_bn': 'Imports',
        'NX_USD_bn': 'Net Exports',
        'T_USD_bn': 'Tax Revenue (bn USD)',
        'Openness_Ratio': 'Openness Ratio',
        'S_USD_bn': 'Saving (bn USD)',
        'S_priv_USD_bn': 'Private Saving (bn USD)',
        'S_pub_USD_bn': 'Public Saving (bn USD)',
        'Saving_Rate': 'Saving Rate',
        'POP_mn': 'Population',
        'LF_mn': 'Labor Force',
        'K_USD_bn': 'Physical Capital',
        'TFP': 'TFP',
        'FDI_pct_GDP': 'FDI (% of GDP)',
        'TAX_pct_GDP': 'Tax Revenue (% of GDP)', # This is the original percentage
        'hc': 'Human Capital'
    }

    # Remove duplicate rows for 2024 and 2025
    # Keep only one row per year by dropping duplicates
    merged_data = merged_data.drop_duplicates(subset=['year'], keep='first')

    # Filter and rename columns
    final_data = merged_data[output_columns].copy()
    final_data = final_data.rename(columns=column_mapping)

    # Format data for output
    # This ensures that the data written to CSV and used for MD table is consistently formatted
    formatted_final_data = format_data_for_output(final_data.copy())

    # Save the processed data to CSV
    csv_path = os.path.join(output_dir, f"{output_base}.csv")
    # Use 'nan' for N/A to match markdown formatting
    formatted_final_data.to_csv(csv_path, index=False, na_rep='nan')
    print(f"Final processed data saved to {csv_path}")

    # Create markdown table using the pre-formatted data
    md_path = os.path.join(output_dir, f"{output_base}.md")
    create_markdown_table(
        formatted_final_data, # Pass the already formatted data
        md_path,
        extrapolation_info,
        alpha=alpha,
        capital_output_ratio=capital_output_ratio,
        input_file=input_file,
        end_year=end_year
    )

    print("\nData processing complete!")

if __name__ == "__main__":
    main()
