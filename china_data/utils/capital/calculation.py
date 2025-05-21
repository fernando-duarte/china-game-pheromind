"""
Capital stock calculation module.

This module provides functions for calculating capital stock based on Penn World Table data
and a specified capital-output ratio.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_capital_stock(raw_data, capital_output_ratio=3.0):
    """
    Calculate capital stock using PWT data and capital-output ratio.
    
    This function calculates physical capital stock based on Penn World Table data
    using relative real capital stock and price level indices, normalized to a
    baseline year (2017), and calibrated with a capital-output ratio.
    
    Args:
        raw_data: DataFrame with PWT data including rkna, pl_gdpo, and cgdpo columns
        capital_output_ratio: Capital-output ratio to use (default: 3.0)
        
    Returns:
        DataFrame with K_USD_bn column added (capital stock in billions of USD)
    """
    logger.info(f"Calculating capital stock using K/Y ratio = {capital_output_ratio}")
    
    # Validate input
    if not isinstance(raw_data, pd.DataFrame):
        logger.error("Invalid input type: raw_data must be a pandas DataFrame")
        return pd.DataFrame({'year': [], 'K_USD_bn': []})
    
    # Create a copy to avoid modifying the original
    df = raw_data.copy()
    
    # Log available columns for debugging
    logger.debug(f"Available columns for capital stock calculation: {df.columns.tolist()}")
    if 'year' not in df.columns:
        logger.error("Critical: 'year' column missing from input data")
        return pd.DataFrame({'year': [], 'K_USD_bn': []})
    
    # Check for required columns
    required_columns = ['rkna', 'pl_gdpo', 'cgdpo']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns for capital stock calculation: {missing_columns}")
        
        # Look for alternative columns that might contain the required data
        pwt_cols = [col for col in df.columns if col.startswith('PWT') or col.lower().startswith('pwt')]
        if pwt_cols:
            logger.info(f"Found PWT columns that might contain needed data: {pwt_cols}")
            # Try to map PWT columns to required columns
            for col in pwt_cols:
                for req_col in missing_columns:
                    if req_col.lower() in col.lower():
                        logger.info(f"Potential match: '{col}' might contain '{req_col}' data")
        
        # Create empty K_USD_bn column
        logger.info("Adding empty K_USD_bn column due to missing data")
        df['K_USD_bn'] = np.nan
        return df
        
    # Check if we have data for 2017 (baseline year)
    baseline_year = 2017
    if baseline_year not in df['year'].values:
        logger.warning(f"Missing {baseline_year} data for capital stock calculation")
        
        years_available = sorted(df['year'].unique())
        logger.info(f"Available years: {min(years_available)} to {max(years_available)}")
        
        # Try to find an alternative baseline year (closest to 2017)
        alt_years = [y for y in years_available if y >= 2010 and y <= 2020]
        if alt_years:
            # Choose closest year to 2017
            baseline_year = min(alt_years, key=lambda y: abs(y - 2017))
            logger.info(f"Using alternative baseline year: {baseline_year}")
        else:
            logger.error("No suitable baseline year found in range 2010-2020")
            df['K_USD_bn'] = np.nan
            return df
    
    try:
        # Get baseline values
        logger.info(f"Using {baseline_year} as baseline year for capital stock calculation")
        
        # Get GDP (cgdpo) for baseline year
        gdp_baseline_rows = df.loc[df.year == baseline_year, 'cgdpo']
        if gdp_baseline_rows.empty or pd.isna(gdp_baseline_rows.iloc[0]):
            raise ValueError(f"No cgdpo data for {baseline_year}")
        gdp_baseline = gdp_baseline_rows.iloc[0]
        
        # Get capital stock at constant prices (rkna) for baseline year
        rkna_baseline_rows = df.loc[df.year == baseline_year, 'rkna']
        if rkna_baseline_rows.empty or pd.isna(rkna_baseline_rows.iloc[0]):
            raise ValueError(f"No rkna data for {baseline_year}")
        rkna_baseline = rkna_baseline_rows.iloc[0]
        
        # Get price level (pl_gdpo) for baseline year
        pl_gdpo_baseline_rows = df.loc[df.year == baseline_year, 'pl_gdpo']
        if pl_gdpo_baseline_rows.empty or pd.isna(pl_gdpo_baseline_rows.iloc[0]):
            raise ValueError(f"No pl_gdpo data for {baseline_year}")
        pl_gdpo_baseline = pl_gdpo_baseline_rows.iloc[0]
        
        # Calculate capital in baseline constant USD
        k_baseline_usd = (rkna_baseline * gdp_baseline) / capital_output_ratio
        logger.info(f"Baseline year ({baseline_year}) calculated capital: {k_baseline_usd:.2f} billion USD")
        
        # Calculate capital stock for all years
        df['K_USD_bn'] = np.nan  # Initialize with NaN
        
        # Calculate capital stock for each year with data
        for _, row in df.iterrows():
            try:
                year = row['year']
                rkna_value = row['rkna']
                pl_gdpo_value = row['pl_gdpo']
                
                # Skip if we have missing values
                if pd.isna(rkna_value) or pd.isna(pl_gdpo_value):
                    logger.debug(f"Missing required data for year {year}")
                    continue
                
                # Calculate capital in USD
                k_usd = (rkna_value / rkna_baseline) * (pl_gdpo_value / pl_gdpo_baseline) * k_baseline_usd
                
                # Store in DataFrame
                df.loc[df.year == year, 'K_USD_bn'] = k_usd
                
            except Exception as e:
                logger.warning(f"Error calculating capital for year {row.get('year', '?')}: {str(e)}")
                
        # Round to 2 decimal places
        if 'K_USD_bn' in df.columns:
            df['K_USD_bn'] = df['K_USD_bn'].round(2)
            
        # Log summary statistics
        k_data = df.dropna(subset=['K_USD_bn'])
        logger.info(f"Calculated capital stock for {k_data.shape[0]} years")
        
        if not k_data.empty:
            min_k = k_data['K_USD_bn'].min()
            max_k = k_data['K_USD_bn'].max()
            logger.info(f"Capital stock range: {min_k:.2f} to {max_k:.2f} billion USD")
            
        return df
        
    except Exception as e:
        logger.error(f"Error in capital stock calculation: {str(e)}")
        df['K_USD_bn'] = np.nan
        return df
