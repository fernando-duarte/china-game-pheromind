"""
Capital stock smoothing module.

This module provides functions for smoothing capital and investment time series data
to handle outliers and gaps.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def smooth_capital_data(df, window_size=3, outlier_threshold=3.0, interpolate_gaps=True):
    """
    Smooth capital and investment time series data to handle outliers and gaps.
    
    This function performs several operations to improve capital stock and investment time series:
    1. Detects and handles outliers using rolling statistics
    2. Fills gaps in the time series using interpolation
    3. Applies smoothing to reduce noise in the data
    
    Args:
        df: DataFrame with 'year', and at least one of 'K_USD_bn' or 'I_USD_bn' columns
        window_size: Size of the rolling window for outlier detection (default: 3)
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)
        interpolate_gaps: Whether to interpolate missing values (default: True)
        
    Returns:
        DataFrame with smoothed capital and investment data
    """
    logger.info(f"Smoothing capital data with window_size={window_size}, outlier_threshold={outlier_threshold}")
    
    # Validate input
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame")
        return pd.DataFrame()
    
    if 'year' not in df.columns:
        logger.error("'year' column missing from input data")
        return df
    
    # Check if we have any capital or investment data to smooth
    has_k = 'K_USD_bn' in df.columns
    has_i = 'I_USD_bn' in df.columns
    
    if not has_k and not has_i:
        logger.error("Neither capital (K_USD_bn) nor investment (I_USD_bn) columns found")
        return df
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by year to ensure proper processing
    result = result.sort_values('year').reset_index(drop=True)
    
    # Create backup columns to track changes
    if has_k:
        result['K_USD_bn_original'] = result['K_USD_bn'].copy()
    if has_i:
        result['I_USD_bn_original'] = result['I_USD_bn'].copy()
    
    # Process capital stock data if available
    if has_k:
        result = _smooth_capital_stock(result, window_size, outlier_threshold, interpolate_gaps)
    
    # Process investment data if available
    if has_i:
        result = _smooth_investment(result, window_size, outlier_threshold, interpolate_gaps)
    
    # Ensure consistency between capital and investment (if both columns exist)
    if has_k and has_i and result.dropna(subset=['K_USD_bn', 'I_USD_bn']).shape[0] >= 2:
        result = _ensure_consistency(result)
    
    # Remove backup columns
    if 'K_USD_bn_original' in result.columns:
        result = result.drop('K_USD_bn_original', axis=1)
    if 'I_USD_bn_original' in result.columns:
        result = result.drop('I_USD_bn_original', axis=1)
    
    # Round final values for consistency
    if has_k:
        result['K_USD_bn'] = result['K_USD_bn'].round(2)
    if has_i:
        result['I_USD_bn'] = result['I_USD_bn'].round(2)
    
    # Final summary
    modified_count = (result != df).any(axis=1).sum()
    if modified_count > 0:
        logger.info(f"Smoothing complete - modified data for {modified_count} out of {len(result)} rows")
    else:
        logger.info("Smoothing complete - no changes were necessary")
    
    return result


def _smooth_capital_stock(result, window_size, outlier_threshold, interpolate_gaps):
    """Helper function to smooth capital stock data."""
    try:
        logger.info("Processing capital stock data")
        
        # Drop rows with missing capital stock values for analysis
        k_data = result.dropna(subset=['K_USD_bn']).copy()
        
        if k_data.shape[0] > window_size:
            # Step 1: Detect outliers using rolling statistics
            # Calculate rolling median and standard deviation
            k_data['rolling_median'] = k_data['K_USD_bn'].rolling(window=window_size, center=True, min_periods=2).median()
            
            # For first and last points, use the nearest valid rolling median
            k_data['rolling_median'].iloc[0] = k_data['rolling_median'].iloc[1] if pd.notna(k_data['rolling_median'].iloc[1]) else k_data['K_USD_bn'].iloc[0]
            k_data['rolling_median'].iloc[-1] = k_data['rolling_median'].iloc[-2] if pd.notna(k_data['rolling_median'].iloc[-2]) else k_data['K_USD_bn'].iloc[-1]
            
            # Calculate rolling MAD (Median Absolute Deviation) for robust standard deviation
            k_data['rolling_mad'] = abs(k_data['K_USD_bn'] - k_data['rolling_median']).rolling(window=window_size, center=True, min_periods=2).median()
            k_data['rolling_mad'] = k_data['rolling_mad'].fillna(k_data['rolling_mad'].median())
            
            # Ensure MAD is never zero to avoid division issues
            k_data['rolling_mad'] = k_data['rolling_mad'].replace(0, k_data['K_USD_bn'].std() * 0.1)
            
            # Calculate modified z-scores using MAD
            k_data['zscore'] = (k_data['K_USD_bn'] - k_data['rolling_median']) / k_data['rolling_mad']
            
            # Identify outliers
            outliers = k_data[abs(k_data['zscore']) > outlier_threshold]
            
            if not outliers.empty:
                outlier_years = outliers['year'].tolist()
                logger.warning(f"Detected {len(outlier_years)} outliers in capital stock data for years: {outlier_years}")
                
                # Replace outliers with estimates based on surrounding values
                for idx in outliers.index:
                    year = k_data.loc[idx, 'year']
                    old_value = k_data.loc[idx, 'K_USD_bn']
                    new_value = k_data.loc[idx, 'rolling_median']
                    
                    # Update the main dataframe
                    result.loc[result['year'] == year, 'K_USD_bn'] = new_value
                    logger.info(f"Replaced outlier for year {year}: {old_value:.2f} -> {new_value:.2f}")
            
            # Step 2: Apply smoothing to reduce noise
            # Use a simple rolling average for smoothing
            smooth_window = min(3, k_data.shape[0])
            
            if smooth_window >= 2:
                # Create a temporary series with interpolated values for better smoothing
                k_series = result['K_USD_bn'].copy()
                
                # Apply rolling mean smoothing
                smoothed = k_series.rolling(window=smooth_window, center=True, min_periods=1).mean()
                
                # Calculate how much change the smoothing made
                change_pct = abs((smoothed - result['K_USD_bn']) / result['K_USD_bn'] * 100)
                avg_change = change_pct.mean()
                
                # Only apply smoothing if it doesn't change values too much
                if avg_change < 5.0:  # Less than 5% average change
                    result['K_USD_bn'] = smoothed
                    logger.info(f"Applied rolling window smoothing to capital stock data (avg change: {avg_change:.2f}%)")
                else:
                    logger.warning(f"Skipped smoothing as it would change values too much (avg change: {avg_change:.2f}%)")
            else:
                logger.info("Not enough data points for smoothing capital stock")
        else:
            logger.info(f"Not enough capital stock data points for outlier detection ({k_data.shape[0]} < {window_size+1})")
        
        # Step 3: Interpolate gaps if requested
        if interpolate_gaps and result['K_USD_bn'].isna().any():
            # Count missing values before interpolation
            na_count_before = result['K_USD_bn'].isna().sum()
            
            # Only interpolate if we have enough data points
            if result.dropna(subset=['K_USD_bn']).shape[0] >= 2:
                # Use linear interpolation for missing values
                result['K_USD_bn'] = result['K_USD_bn'].interpolate(method='linear')
                
                # Count missing values after interpolation
                na_count_after = result['K_USD_bn'].isna().sum()
                filled_count = na_count_before - na_count_after
                
                if filled_count > 0:
                    logger.info(f"Filled {filled_count} missing capital stock values using linear interpolation")
            else:
                logger.warning("Not enough non-missing capital stock values for interpolation")
        
        # Summarize changes
        if 'K_USD_bn_original' in result.columns:
            changed = result[result['K_USD_bn'] != result['K_USD_bn_original']].dropna(subset=['K_USD_bn', 'K_USD_bn_original'])
            if not changed.empty:
                avg_change_pct = abs((changed['K_USD_bn'] - changed['K_USD_bn_original']) / changed['K_USD_bn_original'] * 100).mean()
                logger.info(f"Modified {changed.shape[0]} capital stock values (avg change: {avg_change_pct:.2f}%)")
            else:
                logger.info("No capital stock values were modified")
        
    except Exception as e:
        logger.error(f"Error smoothing capital stock data: {str(e)}")
        # Revert to original values if there was an error
        if 'K_USD_bn_original' in result.columns:
            result['K_USD_bn'] = result['K_USD_bn_original']
    
    return result


def _smooth_investment(result, window_size, outlier_threshold, interpolate_gaps):
    """Helper function to smooth investment data."""
    try:
        logger.info("Processing investment data")
        
        # Drop rows with missing investment values for analysis
        i_data = result.dropna(subset=['I_USD_bn']).copy()
        
        if i_data.shape[0] > window_size:
            # Step 1: Detect outliers using rolling statistics
            # Calculate rolling median and standard deviation
            i_data['rolling_median'] = i_data['I_USD_bn'].rolling(window=window_size, center=True, min_periods=2).median()
            
            # For first and last points, use the nearest valid rolling median
            i_data['rolling_median'].iloc[0] = i_data['rolling_median'].iloc[1] if pd.notna(i_data['rolling_median'].iloc[1]) else i_data['I_USD_bn'].iloc[0]
            i_data['rolling_median'].iloc[-1] = i_data['rolling_median'].iloc[-2] if pd.notna(i_data['rolling_median'].iloc[-2]) else i_data['I_USD_bn'].iloc[-1]
            
            # Calculate rolling MAD (Median Absolute Deviation) for robust standard deviation
            i_data['rolling_mad'] = abs(i_data['I_USD_bn'] - i_data['rolling_median']).rolling(window=window_size, center=True, min_periods=2).median()
            i_data['rolling_mad'] = i_data['rolling_mad'].fillna(i_data['rolling_mad'].median())
            
            # Ensure MAD is never zero to avoid division issues
            i_data['rolling_mad'] = i_data['rolling_mad'].replace(0, i_data['I_USD_bn'].std() * 0.1)
            
            # Calculate modified z-scores using MAD
            i_data['zscore'] = (i_data['I_USD_bn'] - i_data['rolling_median']) / i_data['rolling_mad']
            
            # Identify outliers (more liberal threshold for investment which is naturally more volatile)
            outlier_threshold_i = outlier_threshold * 1.5  # 50% higher threshold for investment
            outliers = i_data[abs(i_data['zscore']) > outlier_threshold_i]
            
            if not outliers.empty:
                outlier_years = outliers['year'].tolist()
                logger.warning(f"Detected {len(outlier_years)} outliers in investment data for years: {outlier_years}")
                
                # Replace outliers with estimates based on surrounding values
                for idx in outliers.index:
                    year = i_data.loc[idx, 'year']
                    old_value = i_data.loc[idx, 'I_USD_bn']
                    new_value = i_data.loc[idx, 'rolling_median']
                    
                    # Update the main dataframe
                    result.loc[result['year'] == year, 'I_USD_bn'] = new_value
                    logger.info(f"Replaced outlier for year {year}: {old_value:.2f} -> {new_value:.2f}")
            
            # Step 2: Apply smoothing to reduce noise (less smoothing for investment which is naturally volatile)
            # Use a shorter window for investment data
            smooth_window = min(2, i_data.shape[0])
            
            if smooth_window >= 2:
                # Create a temporary series with interpolated values for better smoothing
                i_series = result['I_USD_bn'].copy()
                
                # Apply rolling mean smoothing
                smoothed = i_series.rolling(window=smooth_window, center=True, min_periods=1).mean()
                
                # Calculate how much change the smoothing made
                change_pct = abs((smoothed - result['I_USD_bn']) / result['I_USD_bn'] * 100)
                avg_change = change_pct.mean()
                
                # Only apply smoothing if it doesn't change values too much
                if avg_change < 7.5:  # Allow higher change for investment (7.5%)
                    result['I_USD_bn'] = smoothed
                    logger.info(f"Applied rolling window smoothing to investment data (avg change: {avg_change:.2f}%)")
                else:
                    logger.warning(f"Skipped smoothing as it would change values too much (avg change: {avg_change:.2f}%)")
            else:
                logger.info("Not enough data points for smoothing investment")
        else:
            logger.info(f"Not enough investment data points for outlier detection ({i_data.shape[0]} < {window_size+1})")
        
        # Step 3: Interpolate gaps if requested
        if interpolate_gaps and result['I_USD_bn'].isna().any():
            # Count missing values before interpolation
            na_count_before = result['I_USD_bn'].isna().sum()
            
            # Only interpolate if we have enough data points
            if result.dropna(subset=['I_USD_bn']).shape[0] >= 2:
                # Use linear interpolation for missing values
                result['I_USD_bn'] = result['I_USD_bn'].interpolate(method='linear')
                
                # Count missing values after interpolation
                na_count_after = result['I_USD_bn'].isna().sum()
                filled_count = na_count_before - na_count_after
                
                if filled_count > 0:
                    logger.info(f"Filled {filled_count} missing investment values using linear interpolation")
            else:
                logger.warning("Not enough non-missing investment values for interpolation")
        
        # Summarize changes
        if 'I_USD_bn_original' in result.columns:
            changed = result[result['I_USD_bn'] != result['I_USD_bn_original']].dropna(subset=['I_USD_bn', 'I_USD_bn_original'])
            if not changed.empty:
                avg_change_pct = abs((changed['I_USD_bn'] - changed['I_USD_bn_original']) / changed['I_USD_bn_original'] * 100).mean()
                logger.info(f"Modified {changed.shape[0]} investment values (avg change: {avg_change_pct:.2f}%)")
            else:
                logger.info("No investment values were modified")
        
    except Exception as e:
        logger.error(f"Error smoothing investment data: {str(e)}")
        # Revert to original values if there was an error
        if 'I_USD_bn_original' in result.columns:
            result['I_USD_bn'] = result['I_USD_bn_original']
    
    return result


def _ensure_consistency(result):
    """Helper function to ensure consistency between capital and investment data."""
    try:
        logger.info("Checking consistency between capital and investment")
        
        # Sort by year
        df_clean = result.dropna(subset=['K_USD_bn', 'I_USD_bn']).sort_values('year')
        
        if df_clean.shape[0] >= 2:
            # Create a new DataFrame for the check
            check_df = df_clean.copy()
            
            # Calculate implied investment using capital stock differences
            check_df['K_prev'] = check_df['K_USD_bn'].shift(1)
            check_df['implied_I'] = check_df['K_USD_bn'] - (1 - 0.05) * check_df['K_prev']
            
            # Calculate discrepancy
            check_df['I_diff'] = check_df['I_USD_bn'] - check_df['implied_I']
            check_df['I_diff_pct'] = abs(check_df['I_diff'] / check_df['implied_I'] * 100)
            
            # Identify large inconsistencies
            large_diff = check_df.iloc[1:].loc[check_df['I_diff_pct'] > 25]
            
            if not large_diff.empty:
                diff_years = large_diff['year'].tolist()
                logger.warning(f"Large inconsistencies between capital and investment for years: {diff_years}")
                
                # We prefer to keep capital stock as is and adjust investment
                # This choice reflects that capital stock is usually more reliable
                for _, row in large_diff.iterrows():
                    year = row['year']
                    implied_i = row['implied_I']
                    
                    # Only adjust if the implied investment is positive
                    if implied_i > 0:
                        old_i = row['I_USD_bn']
                        
                        # Use a weighted average to preserve some of the original data
                        new_i = 0.7 * implied_i + 0.3 * old_i
                        
                        # Update the main dataframe
                        result.loc[result['year'] == year, 'I_USD_bn'] = round(new_i, 2)
                        logger.info(f"Adjusted investment for year {year} to improve consistency: {old_i:.2f} -> {new_i:.2f}")
    except Exception as e:
        logger.error(f"Error ensuring consistency between capital and investment: {str(e)}")
    
    return result
