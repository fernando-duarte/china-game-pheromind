"""
Human capital projection module.

This module provides functions for projecting human capital values to future years
using various statistical methods.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

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
                historical = hc_data_not_na
                years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))
                result = _project_with_last_value(hc_data, historical, years_to_project)
                if result is not None:
                    return result
            return hc_data

        # Check if we already have data up to the end year
        if last_year_with_data >= end_year:
            logger.info(f"No projection needed - data already available up to {last_year_with_data}")
            return hc_data

        # Determine years that need projection
        historical = hc_data_not_na
        years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))
        logger.info(f"Will project human capital for {len(years_to_project)} years: {min(years_to_project)} to {max(years_to_project)}")

        # Only continue if we have years to project
        if not years_to_project:
            logger.info("No years to project")
            return hc_data

        # Try linear regression as primary method
        try:
            result = _project_with_linear_regression(hc_data, historical, years_to_project)
            if result is not None:
                return result
        except Exception as e:
            logger.error(f"Linear regression failed: {str(e)}")

        # Try trend extrapolation as fallback
        try:
            result = _project_with_trend_extrapolation(hc_data, historical, years_to_project)
            if result is not None:
                return result
        except Exception as e:
            logger.error(f"Simple trend extrapolation failed: {str(e)}")

        # Last resort: just copy the last available value
        try:
            result = _project_with_last_value(hc_data, historical, years_to_project)
            if result is not None:
                return result
        except Exception as e:
            logger.error(f"Last value carry-forward failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error projecting human capital: {str(e)}")

    # If all else fails, return the original data
    logger.warning("All projection methods failed, returning original data")
    return hc_data


def _project_with_linear_regression(hc_data, historical, years_to_project):
    """
    Project human capital using linear regression.

    Parameters:
    -----------
    hc_data : pandas DataFrame
        Original data with 'year' and 'hc' columns
    historical : pandas DataFrame
        Non-NA subset of hc_data
    years_to_project : list
        List of years to project

    Returns:
    --------
    pandas DataFrame or None
        DataFrame with projections if successful, None otherwise
    """
    logger.info(f"Using linear regression based on {len(historical)} historical data points")

    model = LinearRegression()
    X = historical['year'].values.reshape(-1, 1)
    y = historical['hc'].values
    model.fit(X, y)

    # Evaluate model fit
    r_squared = model.score(X, y)
    logger.info(f"Linear regression model fit: R² = {r_squared:.4f}")

    if r_squared < 0.5:
        logger.warning(f"Poor linear regression fit (R² = {r_squared:.4f})")

    # Generate projections
    proj_rows = []
    for year in years_to_project:
        predicted_value = model.predict([[year]])[0]

        # Sanity check on projected values
        if predicted_value < 0:
            logger.warning(f"Negative human capital projection for year {year}: {predicted_value}")
            predicted_value = max(0.1, historical['hc'].min())  # Use minimum historical value but at least 0.1

        if predicted_value > 5:
            logger.warning(f"Unusually high human capital projection for year {year}: {predicted_value}")
            predicted_value = min(5, historical['hc'].max() * 1.2)  # Cap at 5 or 20% above max historical

        proj_rows.append({'year': year, 'hc': round(predicted_value, 4)})

    logger.info(f"Generated {len(proj_rows)} human capital projections")

    if proj_rows:
        # Create DataFrame with projections
        proj_df = pd.DataFrame(proj_rows)
        logger.debug(f"Projection summary: {proj_df['hc'].describe()}")

        # Merge with original data
        try:
            logger.info("Merging projections with original data")
            result = hc_data.copy()

            # Update values using safe merge approach
            for _, row in proj_df.iterrows():
                result.loc[result['year'] == row['year'], 'hc'] = row['hc']

            # Check if all projections were properly merged
            merged_years = set(result.loc[result['hc'].notna(), 'year'])
            projected_years = set(proj_df['year'])
            missing_years = projected_years - merged_years

            if missing_years:
                logger.warning(f"Some projected years were not merged: {missing_years}")
                # Append missing years
                for year in missing_years:
                    proj_row = proj_df[proj_df['year'] == year].iloc[0]
                    new_row = pd.DataFrame({'year': [year], 'hc': [proj_row['hc']]})
                    result = pd.concat([result, new_row], ignore_index=True)

            # Sort by year for consistency
            result = result.sort_values('year').reset_index(drop=True)
            logger.info("Successfully merged projections with original data")
            return result

        except Exception as e:
            logger.error(f"Error merging projections: {str(e)}")
            # Fall back to just returning the original data with projections appended
            logger.info("Falling back to simple append method for projections")
            return pd.concat([hc_data, proj_df], ignore_index=True).sort_values('year')

    return None


def _project_with_trend_extrapolation(hc_data, historical, years_to_project):
    """
    Project human capital using simple trend extrapolation.

    Parameters:
    -----------
    hc_data : pandas DataFrame
        Original data with 'year' and 'hc' columns
    historical : pandas DataFrame
        Non-NA subset of hc_data
    years_to_project : list
        List of years to project

    Returns:
    --------
    pandas DataFrame or None
        DataFrame with projections if successful, None otherwise
    """
    logger.info("Falling back to simple trend extrapolation")

    # Fallback: Simple trend based on last 5 years
    recent_data = historical.sort_values('year').tail(min(5, len(historical)))
    if len(recent_data) >= 2:
        avg_change = (recent_data['hc'].iloc[-1] - recent_data['hc'].iloc[0]) / (len(recent_data) - 1)
        last_value = recent_data['hc'].iloc[-1]

        logger.info(f"Using simple trend with average change of {avg_change:.4f} per year")
        proj_rows = []

        for i, year in enumerate(years_to_project):
            projected_value = last_value + avg_change * (i + 1)
            # Ensure reasonable values
            projected_value = max(0.1, min(5, projected_value))
            proj_rows.append({'year': year, 'hc': round(projected_value, 4)})

        if proj_rows:
            proj_df = pd.DataFrame(proj_rows)
            # Merge with original data using simple concatenation
            result = pd.concat([hc_data, proj_df], ignore_index=True)
            # Remove any duplicates by year
            result = result.drop_duplicates(subset=['year'], keep='last')
            result = result.sort_values('year').reset_index(drop=True)
            return result
    else:
        logger.warning("Not enough recent data for trend extrapolation")

    return None


def _project_with_last_value(hc_data, historical, years_to_project):
    """
    Project human capital by carrying forward the last value.

    Parameters:
    -----------
    hc_data : pandas DataFrame
        Original data with 'year' and 'hc' columns
    historical : pandas DataFrame
        Non-NA subset of hc_data
    years_to_project : list
        List of years to project

    Returns:
    --------
    pandas DataFrame or None
        DataFrame with projections if successful, None otherwise
    """
    logger.info("Falling back to last value carry-forward")

    last_value = historical['hc'].iloc[-1]
    proj_rows = [{'year': year, 'hc': round(last_value, 4)} for year in years_to_project]
    if proj_rows:
        proj_df = pd.DataFrame(proj_rows)
        result = pd.concat([hc_data, proj_df], ignore_index=True)
        result = result.drop_duplicates(subset=['year'], keep='last')
        result = result.sort_values('year').reset_index(drop=True)
        return result

    return None
