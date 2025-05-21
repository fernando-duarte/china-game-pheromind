"""
Investment calculation module.

This module provides functions for calculating investment data based on capital stock
and depreciation rates.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_investment(capital_data, delta=0.05):
    """
    Calculate investment data using changes in capital stock and depreciation.
    
    This function calculates investment using the perpetual inventory method in reverse:
    I_t = K_t - (1-delta) * K_{t-1}
    
    Args:
        capital_data: DataFrame with 'year' and 'K_USD_bn' columns
        delta: Depreciation rate (default: 0.05 or 5% per year)
        
    Returns:
        DataFrame with 'year' and 'I_USD_bn' columns
    """
    logger.info(f"Estimating investment data using delta={delta}")
    
    # Validate input
    if not isinstance(capital_data, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame")
        return pd.DataFrame({'year': [], 'I_USD_bn': []})
    
    if 'year' not in capital_data.columns:
        logger.error("'year' column missing from input data")
        return pd.DataFrame({'year': [], 'I_USD_bn': []})
    
    if 'K_USD_bn' not in capital_data.columns:
        logger.error("'K_USD_bn' column missing from input data")
        return pd.DataFrame({'year': [], 'I_USD_bn': np.nan})
    
    # Create a copy to avoid modifying the original
    df = capital_data.copy()
    
    # Drop rows with missing capital stock values
    df_clean = df.dropna(subset=['K_USD_bn'])
    
    if df_clean.shape[0] < 2:
        logger.error("Not enough non-NA capital stock data points to calculate investment")
        return pd.DataFrame({'year': df['year'], 'I_USD_bn': np.nan})
    
    # Sort by year to ensure proper calculation
    df_clean = df_clean.sort_values('year')
    logger.info(f"Using {df_clean.shape[0]} years of capital stock data from {df_clean['year'].min()} to {df_clean['year'].max()}")
    
    # Create result DataFrame with all original years to maintain consistency
    result = pd.DataFrame({'year': df['year']})
    
    try:
        # Dictionary to store calculated investments
        investments = {}
        
        # Initialize counters for logging
        valid_years = []
        
        # Iterate through years to calculate investment
        for i in range(1, len(df_clean)):
            curr_year = df_clean.iloc[i]['year']
            prev_year = df_clean.iloc[i-1]['year']
            
            # Only calculate if years are consecutive
            if curr_year == prev_year + 1:
                curr_k = df_clean.iloc[i]['K_USD_bn']
                prev_k = df_clean.iloc[i-1]['K_USD_bn']
                
                # Calculate investment using I_t = K_t - (1-delta) * K_{t-1}
                inv = curr_k - (1 - delta) * prev_k
                
                # Store calculated investment
                investments[curr_year] = inv
                valid_years.append(curr_year)
                
                # Apply sanity checks
                if inv < 0:
                    logger.warning(f"Calculated negative investment for year {curr_year}: {inv:.2f}")
                    if inv < -0.1 * curr_k:  # If negative investment is large relative to capital
                        logger.warning(f"Large negative investment ({inv:.2f}) in year {curr_year}, capping to zero")
                        investments[curr_year] = 0
            else:
                logger.debug(f"Skipping non-consecutive years {prev_year} to {curr_year}")
        
        if investments:
            # Create a DataFrame with the calculated investments
            inv_df = pd.DataFrame(list(investments.items()), columns=['year', 'I_USD_bn'])
            
            # Merge with result DataFrame
            result = pd.merge(result, inv_df, on='year', how='left')
            
            # Log statistics for validation
            non_na = result.dropna(subset=['I_USD_bn'])
            if not non_na.empty:
                min_i = non_na['I_USD_bn'].min()
                max_i = non_na['I_USD_bn'].max()
                mean_i = non_na['I_USD_bn'].mean()
                logger.info(f"Calculated investment for {len(valid_years)} years")
                logger.info(f"Investment range: {min_i:.2f} to {max_i:.2f} billion USD, average: {mean_i:.2f} billion USD")
                
                # Check for outlier investments
                if non_na.shape[0] > 5:
                    std_i = non_na['I_USD_bn'].std()
                    
                    # Calculate z-scores
                    z_scores = (non_na['I_USD_bn'] - mean_i) / std_i
                    outliers = non_na[abs(z_scores) > 3]
                    
                    if not outliers.empty:
                        outlier_years = outliers['year'].tolist()
                        logger.warning(f"Outlier investment values detected for years: {outlier_years}")
                        
                # Calculate investment as a percentage of capital stock
                non_na['I_K_ratio'] = non_na['I_USD_bn'] / non_na['K_USD_bn']
                avg_i_k_ratio = non_na['I_K_ratio'].mean()
                logger.info(f"Average investment-to-capital ratio: {avg_i_k_ratio:.4f} ({avg_i_k_ratio*100:.2f}%)")
            else:
                logger.warning("No valid investment calculations")
        else:
            logger.warning("Could not calculate investment for any year")
            
        # Round results to 2 decimal places
        if 'I_USD_bn' in result.columns:
            result['I_USD_bn'] = result['I_USD_bn'].round(2)
        
    except Exception as e:
        logger.error(f"Error calculating investment: {str(e)}")
        # Ensure the result has an I_USD_bn column even if calculation failed
        result['I_USD_bn'] = np.nan
    
    return result
