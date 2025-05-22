"""
Linear Regression extrapolation method.

This module provides a function to extrapolate time series data using linear regression.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def extrapolate_with_linear_regression(df, col, years_to_project, min_data_points=2):
    """
    Extrapolate a time series using linear regression.
    
    Args:
        df (pd.DataFrame): DataFrame containing the time series data
        col (str): Column name of the series to extrapolate
        years_to_project (list): List of years to project values for
        min_data_points (int): Minimum number of data points required (default: 2)
        
    Returns:
        tuple: (updated_df, success, method_info)
            - updated_df: DataFrame with extrapolated values
            - success: Boolean indicating if the projection was successful
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
        logger.info(f"Insufficient data for linear regression on {col} (need {min_data_points}, have {len(historical)})")
        return df_result, False, f"Insufficient data (need {min_data_points})"
    
    # Get the last observed year and value
    last_year = int(historical['year'].max())
    
    # Filter years to actually project (might be fewer than requested if some already exist)
    yrs = [y for y in years_to_project if y > last_year]
    if not yrs:
        return df_result, False, "No years to project"
    
    try:
        # Prepare data for linear regression
        X = historical['year'].values.reshape(-1, 1)
        y = historical[col].values
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate predictions for future years
        for year in yrs:
            pred = model.predict([[year]])[0]
            # Ensure predictions are non-negative and rounded appropriately
            df_result.loc[df_result.year == year, col] = round(max(0, pred), 4)
        
        logger.info(f"Successfully applied linear regression to {col} for years {min(yrs)}-{max(yrs)}")
        return df_result, True, "Linear regression"
        
    except Exception as e:
        logger.warning(f"Linear regression failed for {col}, error: {e}")
        return df_result, False, f"Linear regression failed: {str(e)}"