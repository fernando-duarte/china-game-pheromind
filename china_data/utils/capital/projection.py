"""
Capital stock projection module.

This module provides functions for projecting capital stock into the future
using a perpetual inventory method.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def project_capital_stock(processed_data, end_year, delta=0.05):
    """
    Project capital stock into the future using a perpetual inventory method.

    This method projects capital stock using the perpetual inventory equation:
    K_t = (1-delta) * K_{t-1} + I_t

    Args:
        processed_data: DataFrame containing 'year', 'K_USD_bn', and 'I_USD_bn' columns
                       (Investment should already be extrapolated to end_year)
        end_year: Final year to project capital stock to
        delta: Depreciation rate (default: 0.05)

    Returns:
        DataFrame with projected capital stock values
    """
    logger.info(f"Projecting capital stock to year {end_year} with delta={delta}")

    # Validate inputs
    if not isinstance(processed_data, pd.DataFrame):
        logger.error("Invalid input type: processed_data must be a pandas DataFrame")
        return pd.DataFrame({'year': [], 'K_USD_bn': []})

    # Create a copy to avoid modifying the original
    df = processed_data.copy()

    # Verify required columns exist
    required_columns = ['year', 'K_USD_bn', 'I_USD_bn']
    for column in required_columns:
        if column not in df.columns:
            logger.error(f"Required '{column}' column not found in data")
            return df

    # Check if we have data to project from
    k_data_not_na = df.dropna(subset=['K_USD_bn'])
    if k_data_not_na.empty:
        logger.error("No non-NA capital stock data available for projection")
        return df

    # Sort by year to ensure correct order
    df = df.sort_values('year').reset_index(drop=True)
    logger.info(f"Capital stock data available: {k_data_not_na.shape[0]} rows")

    # Check if we need to project at all (if end_year is already covered)
    max_year = df['year'].max()
    if max_year >= end_year:
        logger.info(f"Data already extends to year {max_year}, no projection needed")
        return df

    # Get the last valid capital stock value for projection
    try:
        last_year_with_data = k_data_not_na['year'].max()
        last_k = k_data_not_na.loc[k_data_not_na.year == last_year_with_data, 'K_USD_bn'].iloc[0]

        if pd.isna(last_k) or last_k <= 0:
            raise ValueError(f"Invalid capital stock value for year {last_year_with_data}: {last_k}")

        logger.info(f"Last capital stock value: {last_k:.2f} billion USD (year {last_year_with_data})")
    except Exception as e:
        logger.error(f"Error retrieving last capital stock value: {str(e)}")
        return df

    # Define years to project
    years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))
    if not years_to_project:
        logger.info("No years to project - returning original data")
        return df

    logger.info(f"Years to project: {min(years_to_project)} to {max(years_to_project)}")

    # Initialize the projection dictionary with the last known value
    proj = {last_year_with_data: last_k}

    # Project capital stock using perpetual inventory method: K_t = (1-delta) * K_{t-1} + I_t
    try:
        # Project forward using the perpetual inventory method
        for y in years_to_project:
            # Get investment value for this year
            inv_row = df.loc[df['year'] == y, 'I_USD_bn']

            if inv_row.empty or pd.isna(inv_row.iloc[0]):
                logger.warning(f"No investment data for year {y}, skipping projection")
                continue

            inv_value = inv_row.iloc[0]
            previous_k = proj[y-1]

            # Apply the perpetual inventory method
            projected_k = (1-delta) * previous_k + inv_value

            # Store the projected value
            proj[y] = round(projected_k, 2)

        logger.info(f"Successfully projected capital stock for {len(proj) - 1} years")

        # Create DataFrame with projections
        proj_df = pd.DataFrame(list(proj.items()), columns=['year', 'K_USD_bn'])

        # Merge with original data
        result = df.copy()

        # For each projection year, update the capital stock
        for _, row in proj_df.iterrows():
            year_mask = result['year'] == row['year']
            if year_mask.any():
                result.loc[year_mask, 'K_USD_bn'] = row['K_USD_bn']
            else:
                # Add missing years
                new_row = pd.DataFrame({'year': [row['year']], 'K_USD_bn': [row['K_USD_bn']]})
                result = pd.concat([result, new_row], ignore_index=True)

        # Sort by year for consistency
        result = result.sort_values('year').reset_index(drop=True)

        logger.info(f"Final result has capital stock data for {result.dropna(subset=['K_USD_bn']).shape[0]} years")
        return result

    except Exception as e:
        logger.error(f"Error projecting capital stock: {str(e)}")
        return df
