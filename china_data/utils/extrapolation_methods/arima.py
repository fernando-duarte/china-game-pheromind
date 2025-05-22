"""
ARIMA (Auto-Regressive Integrated Moving Average) extrapolation method.

This module provides a function to extrapolate time series data using the ARIMA model.
"""

import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


def extrapolate_with_arima(df, col, years_to_project, min_data_points=5, order=(1, 1, 1)):
    """
    Extrapolate a time series using ARIMA model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the time series data
        col (str): Column name of the series to extrapolate
        years_to_project (list): List of years to project values for
        min_data_points (int): Minimum number of data points required for ARIMA (default: 5)
        order (tuple): ARIMA order parameters as (p, d, q) (default: (1, 1, 1))
        
    Returns:
        tuple: (updated_df, success, method_info)
            - updated_df: DataFrame with extrapolated values
            - success: Boolean indicating if ARIMA was successful
            - method_info: String describing the method used
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_result = df.copy()
    
    # Check if the column exists and has sufficient data
    if col not in df_result.columns or df_result[col].isna().all():
        return df_result, False, "No data"
    
    # Get historical data (non-NA values)
    historical = df_result[['year', col]].dropna()
    
    if len(historical) < min_data_points:
        logger.info(f"Insufficient data for ARIMA on {col} (need {min_data_points}, have {len(historical)})")
        return df_result, False, f"Insufficient data (need {min_data_points})"
    
    # Get the last observed year and value
    last_year = int(historical['year'].max())
    last_value = historical[historical['year'] == last_year][col].values[0]
    
    # Filter years to actually project (might be fewer than requested if some already exist)
    yrs = [y for y in years_to_project if y > last_year]
    if not yrs:
        return df_result, False, "No years to project"
    
    try:
        # Fit ARIMA model
        model = ARIMA(historical[col], order=order)
        model_fit = model.fit()
        
        # Generate forecasts
        fc = model_fit.forecast(steps=len(yrs))
        vals = fc.tolist() if hasattr(fc, 'tolist') else list(fc)
        
        # Update the dataframe with projected values
        for i, year in enumerate(yrs):
            df_result.loc[df_result.year == year, col] = round(max(0, vals[i]), 4)
        
        logger.info(f"Successfully applied ARIMA({order[0]},{order[1]},{order[2]}) to {col} for years {min(yrs)}-{max(yrs)}")
        return df_result, True, f"ARIMA({order[0]},{order[1]},{order[2]})"
        
    except Exception as e:
        logger.warning(f"ARIMA failed for {col}, error: {e}")
        return df_result, False, f"ARIMA failed: {str(e)}"