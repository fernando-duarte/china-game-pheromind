#!/usr/bin/env python3
"""
China Economic Data Processor

This script processes raw economic data for China and performs various transformations and calculations:
1. Converts units (e.g., USD to billions USD for nominal values)
2. Calculates capital stock data from Penn World Table (PWT) raw data
3. Projects capital stock data using investment and depreciation (5% annual rate)
4. Projects human capital data using trend extrapolation
5. Calculates additional economic variables (e.g., net exports, capital-to-output ratio, TFP)
6. Extrapolates all time series to 2025 using appropriate statistical methods

The script takes raw data downloaded by china_data_downloader.py and produces
processed datasets for analysis. The output files include:

1. china_data_processed.csv - Main processed dataset with all variables
2. china_data_processed.md - Markdown version with detailed notes on computation methods

Data sources:
- World Bank World Development Indicators (WDI) for GDP components, FDI, population, and labor force
- Penn World Table (PWT) version 10.01 for human capital index and capital stock related variables

Extrapolation methods used:
- ARIMA(1,1,1) for GDP and components (with fallback to average growth rate)
- Linear regression for population and labor force (with fallback to average growth rate)
- Exponential smoothing for human capital (with fallback to average growth rate)
- Capital stock projection using investment and 5% depreciation rate
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

# Add to_markdown method to pandas DataFrame if it doesn't exist
if not hasattr(pd.DataFrame, 'to_markdown'):
    def to_markdown(df, index=True, **kwargs):
        return tabulate(df, headers='keys', tablefmt='pipe', showindex=index, **kwargs)
    pd.DataFrame.to_markdown = to_markdown

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

def extrapolate_series_to_2025(data, end_year=2025):
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
         Dictionary tracking which method was actually used for each variable)
    """
    # Make a copy of the data
    df = data.copy()

    # Dictionary to track which method was actually used for each variable
    extrapolation_methods_used = {}

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

    # Extrapolate each column
    for col in cols_to_extrapolate:
        # Skip columns with all NaN values
        if df[col].isna().all():
            continue

        # Get the historical data for this column
        historical = df[df.year <= max_year][[col]].dropna()

        if len(historical) == 0:
            continue

        # Find the last year with data for this column
        last_year_with_data = historical.index[-1]
        last_year_value = df.loc[last_year_with_data, 'year']

        # If we already have data up to end_year for this column, skip extrapolation
        if last_year_value >= end_year:
            continue

        # Determine which years need extrapolation for this column
        years_to_extrapolate = [year for year in years_to_add if year > last_year_value]
        if not years_to_extrapolate:
            continue

        # If we have less than 5 years of data, use average growth rate
        if len(historical) < 5:
            # Use average growth rate of the last 3 years or all available years
            n_years = min(3, len(historical) - 1)
            if n_years > 0:
                last_years = historical.iloc[-n_years-1:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)
            else:
                avg_growth = 0.03  # Default 3% growth

            # Project using average growth rate
            last_value = historical.iloc[-1].values[0]
            for i, year in enumerate(years_to_extrapolate):
                years_from_last = year - last_year_value
                projected_value = last_value * (1 + avg_growth) ** years_from_last
                df.loc[df.year == year, col] = round(projected_value, 4)

            # Track the method used
            extrapolation_methods_used[col] = "Average growth rate"

        # For GDP and components, use ARIMA
        elif col in gdp_components:
            try:
                # Fit ARIMA model
                logger.info(f"Attempting ARIMA model for {col}")
                model = ARIMA(historical, order=(1, 1, 1))
                model_fit = model.fit()

                # Forecast
                forecast = model_fit.forecast(steps=len(years_to_extrapolate))

                # Convert forecast to list to ensure we can access by index
                forecast_values = forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)

                # Log the forecast values
                logger.info(f"ARIMA forecast for {col}: {forecast_values}")

                # Update the DataFrame
                for i, year in enumerate(years_to_extrapolate):
                    df.loc[df.year == year, col] = round(max(0, forecast_values[i]), 4)

                # Track the method used
                extrapolation_methods_used[col] = "ARIMA(1,1,1)"
            except Exception as e:
                # Fallback to average growth rate if ARIMA fails
                logger.warning(f"ARIMA failed for {col}, falling back to average growth rate. Error: {str(e)}")
                last_years = historical.iloc[-4:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)

                last_value = historical.iloc[-1].values[0]
                for i, year in enumerate(years_to_extrapolate):
                    years_from_last = year - last_year_value
                    projected_value = last_value * (1 + avg_growth) ** years_from_last
                    df.loc[df.year == year, col] = round(projected_value, 4)

                # Track the method used
                extrapolation_methods_used[col] = "Average growth rate"

        # For population and labor force, use linear regression
        elif col in demographic:
            try:
                # Prepare data for linear regression
                X = np.arange(len(historical)).reshape(-1, 1)
                y = historical.values.flatten()

                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Predict for future years
                # Calculate how many steps ahead we need to predict
                steps_ahead = [year - last_year_value for year in years_to_extrapolate]
                X_future = np.array(range(len(historical), len(historical) + len(steps_ahead))).reshape(-1, 1)
                predictions = model.predict(X_future)

                # Update the DataFrame
                for i, year in enumerate(years_to_extrapolate):
                    df.loc[df.year == year, col] = round(max(0, predictions[i]), 4)

                # Track the method used
                extrapolation_methods_used[col] = "Linear regression"
            except Exception as e:
                # Fallback to average growth rate
                logger.warning(f"Linear regression failed for {col}, falling back to average growth rate. Error: {str(e)}")
                last_years = historical.iloc[-4:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)

                last_value = historical.iloc[-1].values[0]
                for i, year in enumerate(years_to_extrapolate):
                    years_from_last = year - last_year_value
                    projected_value = last_value * (1 + avg_growth) ** years_from_last
                    df.loc[df.year == year, col] = round(projected_value, 4)

                # Track the method used
                extrapolation_methods_used[col] = "Average growth rate"

        # For human capital, use exponential smoothing
        elif col in human_capital:
            try:
                # Fit exponential smoothing model
                model = ExponentialSmoothing(historical, trend='add', seasonal=None)
                model_fit = model.fit()

                # Forecast
                forecast = model_fit.forecast(steps=len(years_to_extrapolate))

                # Update the DataFrame
                for i, year in enumerate(years_to_extrapolate):
                    df.loc[df.year == year, col] = round(max(0, forecast[i]), 4)

                # Track the method used
                extrapolation_methods_used[col] = "Exponential smoothing"
            except Exception as e:
                # Fallback to average growth rate
                logger.warning(f"Exponential smoothing failed for {col}, falling back to average growth rate. Error: {str(e)}")
                last_years = historical.iloc[-4:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)

                last_value = historical.iloc[-1].values[0]
                for i, year in enumerate(years_to_extrapolate):
                    years_from_last = year - last_year_value
                    projected_value = last_value * (1 + avg_growth) ** years_from_last
                    df.loc[df.year == year, col] = round(projected_value, 4)

                # Track the method used
                extrapolation_methods_used[col] = "Average growth rate"

        # For other series, use average growth rate
        else:
            if len(historical) >= 4:
                last_years = historical.iloc[-4:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)
            else:
                # Use all available data if less than 4 years
                last_years = historical.values.flatten()
                if len(last_years) > 1:
                    growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                    avg_growth = sum(growth_rates) / len(growth_rates)
                else:
                    avg_growth = 0.03  # Default 3% growth

            last_value = historical.iloc[-1].values[0]
            for i, year in enumerate(years_to_extrapolate):
                years_from_last = year - last_year_value
                projected_value = last_value * (1 + avg_growth) ** years_from_last
                df.loc[df.year == year, col] = round(projected_value, 4)

            # Track the method used
            extrapolation_methods_used[col] = "Average growth rate"

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
            extrapolation_methods_used['K_USD_bn'] = "Investment-based projection"

    # Add tracking for derived variables
    extrapolation_methods_used['NX_USD_bn'] = "Derived calculation"
    extrapolation_methods_used['TFP'] = "Derived calculation"

    # Add tracking for human capital if it's not already tracked
    if 'hc' not in extrapolation_methods_used:
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
                extrapolation_methods_used['hc'] = "Linear regression"

    # Add special handling for TFP description
    if 'TFP' in extrapolation_methods_used and extrapolation_methods_used['TFP'] == "Derived calculation":
        # Find the years for TFP
        tfp_years = ""
        if 'TFP' in last_years and last_years['TFP'] is not None:
            start_year = last_years['TFP'] + 1
            end_year = end_year
            tfp_years = f" ({start_year}-{end_year})"
        else:
            tfp_years = " (2024-2025)"

        # Create a special key for TFP with description
        extrapolation_methods_used.pop('TFP')
        extrapolation_methods_used['TFP' + tfp_years] = "Derived calculation"

    return df, extrapolation_methods_used

def create_markdown_table(data, output_path, extrapolation_methods_used, alpha=1/3, capital_output_ratio=3.0, input_file="china_data_raw.md"):
    """
    Create a markdown file with a table of the processed data and notes on computation.
    """
    # Define column mapping for determining extrapolated years
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
        'Human Capital': 'hc'
    }
    # Convert DataFrame to markdown table
    table = data.to_markdown(index=False)

    # Determine the last year with data for each variable
    last_years = {}
    for col in ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn', 'X_USD_bn', 'M_USD_bn', 'NX_USD_bn',
                'POP_mn', 'LF_mn', 'FDI_pct_GDP', 'TAX_pct_GDP', 'hc', 'K_USD_bn', 'TFP', 'rgdpo_bn', 'rkna', 'pl_gdpo', 'cgdpo_bn', 'K_Y_ratio']:
        if col in data.columns:
            col_name = column_mapping.get(col, col)
            last_year = None
            for year in range(2023, 2018, -1):
                if year in data['Year'].values:
                    idx = data[data['Year'] == year].index[0]
                    if not pd.isna(data.loc[idx, col_name]):
                        last_year = year
                        break
            if last_year:
                if col not in last_years:
                    last_years[col] = last_year

    # Group variables by extrapolation method actually used
    methods_to_variables = {}
    for var, method in extrapolation_methods_used.items():
        if method not in methods_to_variables:
            methods_to_variables[method] = []
        display_name = var
        for display, internal in column_mapping.items():
            if internal == var:
                display_name = display
                break
        if var in last_years and last_years[var] is not None:
            start_year = last_years[var] + 1
            end_year = end_year
            years_extrapolated = f" ({start_year}-{end_year})"
        else:
            years_extrapolated = " (2024-2025)"
        methods_to_variables[method].append(f"{display_name}{years_extrapolated}")

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
    template_str = '''# China Economic Variables

## Economic Variables (1960-2025)

{{ table }}

## Notes on Computation

The raw data in `{{ input_file }}` comes from the following sources:

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
     - K_2017 is the nominal capital stock in 2017, estimated as GDP_2017 * {{ capital_output_ratio }} (capital-output ratio)
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

4. **Extrapolation to 2025**:
   Each series was extrapolated using the following methods:

{{ methods_md }}
'''
    template = Template(template_str)
    markdown_content = template.render(
        table=table,
        input_file=input_file,
        capital_output_ratio=capital_output_ratio,
        alpha=alpha,
        methods_md=methods_md
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
    merged_data, extrapolation_methods_used = extrapolate_series_to_2025(merged_data, end_year=end_year)

    # Recalculate derived variables for extrapolated years
    # Recalculate Net Exports for all years
    if all(col in merged_data.columns for col in ['X_USD_bn', 'M_USD_bn']):
        merged_data['NX_USD_bn'] = merged_data['X_USD_bn'] - merged_data['M_USD_bn']

    # Recalculate Capital-to-output ratio for all years
    if all(col in merged_data.columns for col in ['K_USD_bn', 'GDP_USD_bn']):
        merged_data['K_Y_ratio'] = merged_data['K_USD_bn'] / merged_data['GDP_USD_bn']

    # Recalculate TFP for all years with the provided alpha value
    merged_data = calculate_tfp(merged_data, alpha=alpha)

    # Select and reorder columns as requested
    output_columns = [
        'year', 'GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn',
        'X_USD_bn', 'M_USD_bn', 'NX_USD_bn', 'POP_mn', 'LF_mn',
        'K_USD_bn', 'TFP', 'FDI_pct_GDP', 'TAX_pct_GDP', 'hc'
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
        'POP_mn': 'Population',
        'LF_mn': 'Labor Force',
        'K_USD_bn': 'Physical Capital',
        'TFP': 'TFP',
        'FDI_pct_GDP': 'FDI (% of GDP)',
        'TAX_pct_GDP': 'Tax Revenue (% of GDP)',
        'hc': 'Human Capital'
    }

    # Remove duplicate rows for 2024 and 2025
    # Keep only one row per year by dropping duplicates
    merged_data = merged_data.drop_duplicates(subset=['year'], keep='first')

    # Filter and rename columns
    final_data = merged_data[output_columns].copy()
    final_data = final_data.rename(columns=column_mapping)

    # Save the processed data to CSV
    csv_path = os.path.join(output_dir, f"{output_base}.csv")
    final_data.to_csv(csv_path, index=False)
    print(f"Final processed data saved to {csv_path}")

    # Create markdown table
    md_path = os.path.join(output_dir, f"{output_base}.md")
    create_markdown_table(
        final_data,
        md_path,
        extrapolation_methods_used,
        alpha=alpha,
        capital_output_ratio=capital_output_ratio,
        input_file=input_file
    )

    print("\nData processing complete!")

if __name__ == "__main__":
    main()
