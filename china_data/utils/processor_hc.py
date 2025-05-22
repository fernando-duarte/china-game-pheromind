"""
Human capital projection module.

This module provides functions for projecting human capital values to future years
using various statistical methods.
"""

import pandas as pd
import numpy as np
import logging

from china_data.utils.extrapolation_methods import (
    extrapolate_with_linear_regression,
    extrapolate_with_average_growth_rate
)

logger = logging.getLogger(__name__)


def project_human_capital(processed_data, end_year=2025):
    """
    Project human capital values to a specified end year using linear regression.

    Parameters:
    -----------
    processed_data : pandas DataFrame
        Data containing at least 'year' and possibly 'hc' columns
    end_year : int, default=2025
        The final year to project human capital values to

    Returns:
    --------
    pandas DataFrame
        DataFrame with years and human capital values, including projections to end_year
    """
    logger.info(f"Projecting human capital to year {end_year}")

    # Validate input data
    try:
        if not isinstance(processed_data, pd.DataFrame):
            logger.error("Input data is not a pandas DataFrame")
            raise TypeError("Input data must be a pandas DataFrame")

        if 'year' not in processed_data.columns:
            logger.error("Year column missing from input data")
            raise ValueError("Input data must contain a 'year' column")

        # Debug available columns
        logger.info(f"Data contains {processed_data.shape[0]} rows and {processed_data.shape[1]} columns")
        logger.debug(f"Available columns in data: {processed_data.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to validate input data: {str(e)}")
        # Return empty DataFrame with year and hc columns
        logger.info("Returning empty DataFrame due to validation failure")
        return pd.DataFrame({'year': range(1960, end_year + 1), 'hc': np.nan})

    # Check if 'hc' column exists in the data
    if 'hc' not in processed_data.columns:
        logger.warning("Human capital (hc) column not found in the data")
        # Look for potential alternative columns
        pwt_cols = [col for col in processed_data.columns if col.startswith('PWT') or col.lower().startswith('pwt')]
        if pwt_cols:
            logger.info(f"Found potential human capital columns: {pwt_cols}")

        # Create a DataFrame with just years and NaN for human capital
        logger.info(f"Creating placeholder DataFrame with years from 1960 to {end_year}")
        return pd.DataFrame({'year': range(1960, end_year + 1), 'hc': np.nan})

    try:
        # Extract human capital data
        logger.info("Extracting human capital data")
        hc_data = processed_data[['year', 'hc']].copy()

        # Data quality checks
        total_rows = hc_data.shape[0]
        non_na_rows = hc_data.dropna(subset=['hc']).shape[0]
        na_percentage = (total_rows - non_na_rows) / total_rows * 100 if total_rows > 0 else 0

        logger.info(f"Human capital data: {total_rows} total rows, {non_na_rows} non-NA rows ({na_percentage:.1f}% missing)")

        if non_na_rows > 0:
            min_hc = hc_data['hc'].min()
            max_hc = hc_data['hc'].max()
            logger.info(f"Human capital range: {min_hc} to {max_hc}")

            if min_hc < 0:
                logger.warning(f"Found negative human capital values (min={min_hc})")

            if max_hc > 5:
                logger.warning(f"Unusually high human capital values detected (max={max_hc})")

            # Sample data for debugging
            sample = hc_data.dropna(subset=['hc']).head(3)
            logger.debug(f"Sample human capital values:\n{sample}")

        # Check if we have any non-NA human capital data
        hc_data_not_na = hc_data.dropna(subset=['hc'])
        if hc_data_not_na.empty:
            logger.warning("No non-NA human capital data available for projection")
            return hc_data

        # Find the last year with data (moved up to be available for all code paths)
        last_year_with_data = hc_data_not_na['year'].max()
        first_year_with_data = hc_data_not_na['year'].min()
        logger.info(f"Human capital data available from {first_year_with_data} to {last_year_with_data}")

        # Check if we have enough data for regression
        if len(hc_data_not_na) < 2:
            logger.warning(f"Insufficient data for regression (only {len(hc_data_not_na)} points)")
            # Instead of returning early, we'll fall back to using the last value method
            if len(hc_data_not_na) > 0:
                logger.info("Falling back to last value carry-forward due to insufficient data points")
                years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))

                # Make sure all years exist in the dataframe
                for year in years_to_project:
                    if year not in hc_data['year'].values:
                        hc_data = pd.concat([hc_data, pd.DataFrame({'year': [year], 'hc': [np.nan]})], ignore_index=True)

                # Use average growth rate with 0 growth (equivalent to last value carry-forward)
                df_updated, success, method = extrapolate_with_average_growth_rate(
                    hc_data, 'hc', years_to_project, default_growth=0.0, min_data_points=1
                )
                if success:
                    logger.info(f"Successfully projected human capital using {method}")
                    return df_updated
            return hc_data

        # Check if we already have data up to the end year
        if last_year_with_data >= end_year:
            logger.info(f"No projection needed - data already available up to {last_year_with_data}")
            return hc_data

        # Determine years that need projection
        years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))
        logger.info(f"Will project human capital for {len(years_to_project)} years: {min(years_to_project)} to {max(years_to_project)}")

        # Only continue if we have years to project
        if not years_to_project:
            logger.info("No years to project")
            return hc_data

        # Try linear regression as primary method
        df_updated, success, method = extrapolate_with_linear_regression(
            hc_data, 'hc', years_to_project, min_data_points=2
        )

        if success:
            logger.info(f"Successfully projected human capital using {method}")
            return df_updated
        else:
            logger.error(f"Linear regression failed: {method}")

        # Try average growth rate with a small lookback period (trend extrapolation)
        df_updated, success, method = extrapolate_with_average_growth_rate(
            hc_data, 'hc', years_to_project, lookback_years=2, default_growth=0.01
        )

        if success:
            logger.info(f"Successfully projected human capital using {method}")
            return df_updated
        else:
            logger.error(f"Growth trend extrapolation failed: {method}")

        # Last resort: just copy the last available value (using growth rate of 0)
        df_updated, success, method = extrapolate_with_average_growth_rate(
            hc_data, 'hc', years_to_project, default_growth=0.0, min_data_points=1
        )

        if success:
            logger.info(f"Successfully projected human capital using {method}")
            return df_updated
        else:
            logger.error(f"Last value carry-forward failed: {method}")

    except Exception as e:
        logger.error(f"Unexpected error projecting human capital: {str(e)}")

    # If all else fails, return the original data
    logger.warning("All projection methods failed, returning original data")
    return hc_data


# The private helper functions are no longer needed as they're replaced
# by the reusable extrapolation methods in china_data/utils/extrapolation_methods/
