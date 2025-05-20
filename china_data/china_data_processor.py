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
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

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
                elif renamed_header[j] == 'FDI_pct_GDP':  # FDI (% of GDP)
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

def project_capital_stock(processed_data):
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
    print(f"Last year with capital stock data: {last_year_with_data}")

    # If we already have data up to the current year, no need to project
    current_year = datetime.now().year
    if last_year_with_data >= current_year:
        return k_data

    # Get investment data for projection
    try:
        # Get nominal gross capital formation from WDI (current USD bn)
        inv = wb.download("CN", "NE.GDI.TOTL.CD", last_year_with_data + 1, current_year)["NE.GDI.TOTL.CD"].div(1e9)
        inv.index = inv.index.get_level_values("year").astype(int)

        # Drop NaN values (years where data is not available)
        inv = inv.dropna()

        if inv.empty:
            print("No investment data available for projection.")
            return k_data

        # Get the last known capital stock value
        last_k = k_data.loc[k_data.year == last_year_with_data, 'K_USD_bn'].values[0]

        # Depreciation rate
        δ = 0.05  # 5% annual depreciation

        # Project capital stock
        proj = {}
        proj[last_year_with_data] = last_k

        # Process years with available investment data
        available_years = sorted(inv.index.tolist())
        for y in available_years:
            if y > last_year_with_data:
                proj[y] = round(proj[y-1] * (1-δ) + inv.loc[y], 2)

        # For years without investment data, use average growth rate
        last_available_year = max(available_years)
        if last_available_year < current_year:
            # Calculate average growth rate from the last 3 years
            if len(proj) >= 4:  # Need at least 4 years to calculate 3 growth rates
                years = sorted(list(proj.keys()))[-4:]
                growth_rates = [(proj[years[i]] / proj[years[i-1]] - 1) for i in range(1, len(years))]
                avg_growth = sum(growth_rates) / len(growth_rates)
            else:
                avg_growth = 0.03  # Default 3% growth

            # Project for remaining years
            for y in range(last_available_year + 1, current_year + 1):
                proj[y] = round(proj[y-1] * (1 + avg_growth), 2)

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
        print(f"Error projecting capital stock: {e}")
        return k_data

def project_human_capital(processed_data):
    """
    Project human capital data using trend extrapolation.

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
    print(f"Last year with human capital data: {last_year_with_data}")

    # If we already have data up to the current year, no need to project
    current_year = datetime.now().year
    if last_year_with_data >= current_year:
        return hc_data

    try:
        # Get the last 5 years of data for trend extrapolation
        last_5_years = hc_data[(hc_data.year >= last_year_with_data - 4) & (hc_data.year <= last_year_with_data)]
        last_5_years = last_5_years.dropna(subset=['hc'])

        if len(last_5_years) < 2:
            print("Not enough data for human capital projection.")
            return hc_data

        # Calculate average annual growth rate
        first_year = last_5_years.year.min()
        last_year = last_5_years.year.max()
        first_value = last_5_years.loc[last_5_years.year == first_year, 'hc'].values[0]
        last_value = last_5_years.loc[last_5_years.year == last_year, 'hc'].values[0]

        years_diff = last_year - first_year
        avg_growth = (last_value / first_value) ** (1 / years_diff) - 1

        # Project human capital for future years
        proj = {}
        proj[last_year_with_data] = hc_data.loc[hc_data.year == last_year_with_data, 'hc'].values[0]

        for y in range(last_year_with_data + 1, current_year + 1):
            proj[y] = round(proj[y-1] * (1 + avg_growth), 2)

        # Create DataFrame with projections
        proj_df = pd.DataFrame(list(proj.items()), columns=['year', 'hc'])

        # Merge with original data
        hc_data = pd.merge(hc_data, proj_df, on='year', how='outer', suffixes=('', '_proj'))

        # Use projected values where original data is missing
        mask = hc_data['hc'].isna()
        hc_data.loc[mask, 'hc'] = hc_data.loc[mask, 'hc_proj']

        # Drop the projection column
        hc_data = hc_data.drop(columns=['hc_proj'])

        return hc_data

    except Exception as e:
        print(f"Error projecting human capital: {e}")
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
    required_cols = ['GDP_USD_bn', 'K_USD_bn', 'LF_mn', 'hc']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing required data for TFP calculation")
        df['TFP'] = np.nan
        return df

    # Calculate TFP
    try:
        # A_t = Y_t / (K_t^α * (L_t * H_t)^(1-α))
        df['TFP'] = df['GDP_USD_bn'] / (
            (df['K_USD_bn'] ** alpha) *
            ((df['LF_mn'] * df['hc']) ** (1 - alpha))
        )

        # Round to 4 decimal places
        df['TFP'] = df['TFP'].round(4)

    except Exception as e:
        print(f"Error calculating TFP: {e}")
        df['TFP'] = np.nan

    return df

def extrapolate_series_to_2025(data):
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
    pandas.DataFrame
        DataFrame with extrapolated data to 2025
    """
    # Make a copy of the data
    df = data.copy()

    # Get the maximum year in the data
    max_year = df.year.max()

    # If we already have data up to 2025, no need to extrapolate
    if max_year >= 2025:
        return df

    # Years to extrapolate
    years_to_add = list(range(max_year + 1, 2026))

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

    # Extrapolate each column
    for col in cols_to_extrapolate:
        # Skip columns with all NaN values
        if df[col].isna().all():
            continue

        # Get the historical data for this column
        historical = df[df.year <= max_year][[col]].dropna()

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
            for i, year in enumerate(years_to_add):
                projected_value = last_value * (1 + avg_growth) ** (i + 1)
                df.loc[df.year == year, col] = round(projected_value, 4)

        # For GDP and components, use ARIMA
        elif col in gdp_components:
            try:
                # Fit ARIMA model
                model = ARIMA(historical, order=(1, 1, 1))
                model_fit = model.fit()

                # Forecast
                forecast = model_fit.forecast(steps=len(years_to_add))

                # Update the DataFrame
                for i, year in enumerate(years_to_add):
                    df.loc[df.year == year, col] = round(max(0, forecast[i]), 4)
            except:
                # Fallback to average growth rate if ARIMA fails
                last_years = historical.iloc[-4:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)

                last_value = historical.iloc[-1].values[0]
                for i, year in enumerate(years_to_add):
                    projected_value = last_value * (1 + avg_growth) ** (i + 1)
                    df.loc[df.year == year, col] = round(projected_value, 4)

        # For population and labor force, use linear regression
        elif col in demographic:
            try:
                # Prepare data for linear regression
                X = historical.index.values.reshape(-1, 1)
                y = historical.values.flatten()

                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Predict for future years
                X_future = np.array(range(len(historical), len(historical) + len(years_to_add))).reshape(-1, 1)
                predictions = model.predict(X_future)

                # Update the DataFrame
                for i, year in enumerate(years_to_add):
                    df.loc[df.year == year, col] = round(max(0, predictions[i]), 4)
            except:
                # Fallback to average growth rate
                last_years = historical.iloc[-4:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)

                last_value = historical.iloc[-1].values[0]
                for i, year in enumerate(years_to_add):
                    projected_value = last_value * (1 + avg_growth) ** (i + 1)
                    df.loc[df.year == year, col] = round(projected_value, 4)

        # For human capital, use exponential smoothing
        elif col in human_capital:
            try:
                # Fit exponential smoothing model
                model = ExponentialSmoothing(historical, trend='add', seasonal=None)
                model_fit = model.fit()

                # Forecast
                forecast = model_fit.forecast(steps=len(years_to_add))

                # Update the DataFrame
                for i, year in enumerate(years_to_add):
                    df.loc[df.year == year, col] = round(max(0, forecast[i]), 4)
            except:
                # Fallback to average growth rate
                last_years = historical.iloc[-4:].values.flatten()
                growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
                avg_growth = sum(growth_rates) / len(growth_rates)

                last_value = historical.iloc[-1].values[0]
                for i, year in enumerate(years_to_add):
                    projected_value = last_value * (1 + avg_growth) ** (i + 1)
                    df.loc[df.year == year, col] = round(projected_value, 4)

        # For other series, use average growth rate
        else:
            last_years = historical.iloc[-4:].values.flatten()
            growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1, len(last_years))]
            avg_growth = sum(growth_rates) / len(growth_rates)

            last_value = historical.iloc[-1].values[0]
            for i, year in enumerate(years_to_add):
                projected_value = last_value * (1 + avg_growth) ** (i + 1)
                df.loc[df.year == year, col] = round(projected_value, 4)

    return df

def create_markdown_table(data, output_path, alpha=1/3, capital_output_ratio=3.0, input_file="china_data_raw.md"):
    """
    Create a markdown file with a table of the processed data and notes on computation.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with the processed data
    output_path : str
        Path to save the markdown file
    alpha : float
        Capital share parameter used in TFP calculation
    capital_output_ratio : float
        Capital-to-output ratio used for base year capital stock calculation
    input_file : str
        Name of the input file containing raw data
    """
    # Create the markdown content
    markdown_content = "# China Economic Variables\n\n"

    # Add the table
    markdown_content += "## Economic Variables (1960-2025)\n\n"

    # Convert DataFrame to markdown table
    table = data.to_markdown(index=False)
    markdown_content += table + "\n\n"

    # Add notes on computation
    markdown_content += "## Notes on Computation\n\n"
    markdown_content += f"The raw data in `{input_file}` comes from the following sources:\n\n"
    markdown_content += "1. **Original Data Sources**:\n"
    markdown_content += "   - World Bank World Development Indicators (WDI) for GDP components, FDI, population, and labor force\n"
    markdown_content += "   - Penn World Table (PWT) version 10.01 for human capital index and capital stock related variables\n\n"
    markdown_content += "This processed dataset was created by applying the following transformations to the raw data:\n\n"
    markdown_content += "2. **Unit Conversions**:\n"
    markdown_content += "   - GDP and its components (Consumption, Government, Investment, Exports, Imports) were converted from USD to billions USD\n"
    markdown_content += "   - Population and Labor Force were converted from people to millions of people\n\n"
    markdown_content += "3. **Derived Variables**:\n"
    markdown_content += "   - **Net Exports**: Calculated as Exports - Imports (in billions USD)\n"
    markdown_content += "     ```\n"
    markdown_content += "     Net Exports = Exports - Imports\n"
    markdown_content += "     ```\n\n"
    markdown_content += "   - **Physical Capital**: Calculated using PWT data with the following formula:\n"
    markdown_content += "     ```\n"
    markdown_content += "     K_t = (rkna_t / rkna_2017) * K_2017 * (pl_gdpo_t / pl_gdpo_2017)\n"
    markdown_content += "     ```\n"
    markdown_content += "     Where:\n"
    markdown_content += "     - K_t is the capital stock in year t (billions USD)\n"
    markdown_content += "     - rkna_t is the real capital stock index in year t (from PWT)\n"
    markdown_content += "     - rkna_2017 is the real capital stock index in 2017 (from PWT)\n"
    markdown_content += f"     - K_2017 is the nominal capital stock in 2017, estimated as GDP_2017 * {capital_output_ratio} (capital-output ratio)\n"
    markdown_content += "     - pl_gdpo_t is the price level of GDP in year t (from PWT)\n"
    markdown_content += "     - pl_gdpo_2017 is the price level of GDP in 2017 (from PWT)\n\n"
    markdown_content += "   - **TFP (Total Factor Productivity)**: Calculated using the Cobb-Douglas production function:\n"
    markdown_content += "     ```\n"
    markdown_content += "     TFP_t = Y_t / (K_t^α * (L_t * H_t)^(1-α))\n"
    markdown_content += "     ```\n"
    markdown_content += "     Where:\n"
    markdown_content += "     - Y_t is GDP in year t (billions USD)\n"
    markdown_content += "     - K_t is Physical Capital in year t (billions USD)\n"
    markdown_content += "     - L_t is Labor Force in year t (millions of people)\n"
    markdown_content += "     - H_t is Human Capital index in year t\n"
    markdown_content += f"     - α = {alpha} (capital share parameter)\n\n"
    markdown_content += "4. **Extrapolation to 2025**:\n"
    markdown_content += "   Each series was extrapolated using the following methods:\n\n"
    markdown_content += "   - **ARIMA(1,1,1) model with fallback to average growth rate of last 4 years**: GDP, Consumption, Government, Investment, Exports, Imports\n"
    markdown_content += "   - **Linear regression on historical values with fallback to average growth rate**: Population, Labor Force\n"
    markdown_content += "   - **Exponential smoothing with fallback to average growth rate of last 4 years**: Human Capital\n"
    markdown_content += "   - **Average growth rate of last 4 years**: FDI (% of GDP)\n"
    markdown_content += "   - **Calculated as Exports - Imports for each projected year**: Net Exports\n"
    markdown_content += "   - **Projected using the formula K_t = K_{t-1} * (1-δ) + I_t, where δ = 0.05 (5% depreciation rate) and I_t is investment in year t**: Physical Capital\n"
    markdown_content += "   - **Recalculated using the Cobb-Douglas formula for each projected year**: TFP\n\n"

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
    k_data = project_capital_stock(processed_data)

    # Project human capital
    hc_data = project_human_capital(raw_data)  # Use raw data for human capital projection

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
        hc_value = row['hc']
        if not pd.isna(hc_value):
            merged_data.loc[merged_data['year'] == year, 'hc'] = hc_value

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

    # Extrapolate all series to 2025
    merged_data = extrapolate_series_to_2025(merged_data)
    print("Extrapolated all series to 2025.")

    # Select and reorder columns as requested
    output_columns = [
        'year', 'GDP_USD_bn', 'C_USD_bn', 'G_USD_bn', 'I_USD_bn',
        'X_USD_bn', 'M_USD_bn', 'NX_USD_bn', 'POP_mn', 'LF_mn',
        'K_USD_bn', 'TFP', 'FDI_pct_GDP', 'hc'
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
        'hc': 'Human Capital'
    }

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
        alpha=alpha,
        capital_output_ratio=capital_output_ratio,
        input_file=input_file
    )

    print("\nData processing complete!")

if __name__ == "__main__":
    main()
