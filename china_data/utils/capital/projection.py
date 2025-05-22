"""
Capital stock projection module.

This module provides functions for projecting capital stock into the future
using a perpetual inventory method.
"""

import pandas as pd
import numpy as np
import logging

from china_data.utils.capital.investment import calculate_investment

logger = logging.getLogger(__name__)


def project_capital_stock(processed_data, end_year, delta=0.05):
    """
    Project capital stock into the future using a perpetual inventory method.

    This method projects capital stock using the perpetual inventory equation:
    K_t = (1-delta) * K_{t-1} + I_t

    Args:
        processed_data: DataFrame containing at least 'year' and 'K_USD_bn' columns
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
    if 'year' not in df.columns:
        logger.error("Required 'year' column not found in data")
        return df

    if 'K_USD_bn' not in df.columns:
        logger.error("Required 'K_USD_bn' column not found in data")
        return df

    # Log available columns
    logger.debug(f"Available columns for capital projection: {df.columns.tolist()}")

    # Check if we have data to project from
    k_data_not_na = df.dropna(subset=['K_USD_bn'])
    if k_data_not_na.empty:
        logger.error("No non-NA capital stock data available for projection")
        return df

    # Sort by year to ensure correct order
    k_data = df[['year', 'K_USD_bn']].copy().sort_values('year').reset_index(drop=True)
    logger.info(f"Capital stock data available: {k_data_not_na.shape[0]} rows")

    # Check if we need to project at all (if end_year is already covered)
    max_year = k_data['year'].max()
    if max_year >= end_year:
        logger.info(f"Data already extends to year {max_year}, no projection needed")
        return df

    # Get the last valid capital stock value for projection
    try:
        last_year_with_data = k_data_not_na['year'].max()
        last_k_rows = k_data_not_na.loc[k_data_not_na.year == last_year_with_data, 'K_USD_bn']

        if last_k_rows.empty:
            raise ValueError(f"No capital stock data found for year {last_year_with_data}")

        last_k = last_k_rows.iloc[0]
        if pd.isna(last_k) or last_k <= 0:
            raise ValueError(f"Invalid capital stock value for year {last_year_with_data}: {last_k}")

        logger.info(f"Last capital stock value: {last_k:.2f} billion USD (year {last_year_with_data})")
    except Exception as e:
        logger.error(f"Error retrieving last capital stock value: {str(e)}")
        return k_data

    # Get investment data for projecting future values
    try:
        if 'I_USD_bn' not in processed_data.columns:
            raise ValueError("Investment (I_USD_bn) column not found in data")

        inv_data = processed_data[['year', 'I_USD_bn']].copy().dropna()
        logger.info(f"Investment data available: {inv_data.shape[0]} rows")

        if inv_data.empty:
            raise ValueError("No non-NA investment data available")
    except Exception as e:
        logger.warning(f"Error retrieving investment data: {str(e)}")
        logger.info("Will use estimated investment based on capital stock")

        # Create synthetic investment data based on capital stock and depreciation
        inv_data = calculate_investment(k_data_not_na, delta=delta)

        if inv_data.empty or inv_data['I_USD_bn'].isna().all():
            logger.warning("Could not create synthetic investment data")
            # Last resort: assume investment is a fraction of capital stock
            last_inv_year = last_year_with_data
            last_inv_value = last_k * 0.1  # Assume investment is 10% of capital stock
            logger.info(f"Using estimated investment of {last_inv_value:.2f} billion USD (10% of capital stock)")

            # Create a simple investment dataframe
            inv_data = pd.DataFrame({'year': [last_inv_year], 'I_USD_bn': [last_inv_value]})
        else:
            logger.info(f"Created synthetic investment data for {inv_data.dropna(subset=['I_USD_bn']).shape[0]} years")

    # Calculate average investment growth rate from recent data
    try:
        # Use the most recent data for growth rate calculation (up to 5 years)
        if not inv_data.empty:
            # Sort investments by year
            inv_data = inv_data.sort_values('year')

            # Get the most recent years of data (up to 5)
            recent_years = 5
            if len(inv_data) < recent_years:
                recent_years = len(inv_data)

            last_inv_data = inv_data.tail(recent_years)
            logger.info(f"Using investment data from {recent_years} most recent years: {last_inv_data['year'].tolist()}")

            if len(last_inv_data) >= 2:
                # Calculate year-over-year growth rates
                growth_rates = []
                prev_value = None

                for _, row in last_inv_data.iterrows():
                    if prev_value is not None and prev_value > 0:
                        growth_rate = (row['I_USD_bn'] / prev_value) - 1
                        growth_rates.append(growth_rate)
                    prev_value = row['I_USD_bn']

                if growth_rates:
                    # Remove extreme values (more than 3 std dev from mean)
                    if len(growth_rates) > 3:
                        mean_growth = np.mean(growth_rates)
                        std_growth = np.std(growth_rates)
                        growth_rates = [g for g in growth_rates if abs(g - mean_growth) <= 3 * std_growth]

                    avg_inv_growth = np.mean(growth_rates)
                    logger.info(f"Average investment growth rate: {avg_inv_growth:.4f} ({avg_inv_growth*100:.2f}%)")

                    # Cap growth rate at reasonable bounds
                    if avg_inv_growth > 0.15:
                        logger.warning(f"Capping high investment growth rate {avg_inv_growth:.4f} to 0.15 (15%)")
                        avg_inv_growth = 0.15
                    elif avg_inv_growth < -0.10:
                        logger.warning(f"Capping low investment growth rate {avg_inv_growth:.4f} to -0.10 (-10%)")
                        avg_inv_growth = -0.10
                else:
                    logger.warning("Could not calculate investment growth rates")
                    avg_inv_growth = 0.05  # Default to 5%
            else:
                logger.warning("Insufficient data points for growth rate calculation")
                avg_inv_growth = 0.05  # Default to 5%
        else:
            logger.warning("No investment data available for growth rate calculation")
            avg_inv_growth = 0.05  # Default to 5%
    except Exception as e:
        logger.error(f"Error calculating investment growth rate: {str(e)}")
        avg_inv_growth = 0.05  # Default to 5%

    logger.info(f"Using investment growth rate of {avg_inv_growth:.4f} for projections")

    # Define years to project
    years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))
    if years_to_project:
        logger.info(f"Years to project: {min(years_to_project)} to {max(years_to_project)}")
    else:
        logger.info("No years to project - returning original data")
        return k_data

    # Get last known investment value for projections
    try:
        if not inv_data.empty:
            last_inv_year = inv_data['year'].max()
            last_inv_value = inv_data.loc[inv_data['year'] == last_inv_year, 'I_USD_bn'].iloc[0]
            logger.info(f"Last investment value: {last_inv_value:.2f} billion USD (year {last_inv_year})")
        else:
            # Fallback if no investment data is available
            last_inv_year = last_year_with_data
            last_inv_value = last_k * 0.1  # Assume investment is 10% of capital stock
            logger.info(f"Using estimated investment of {last_inv_value:.2f} billion USD (10% of capital stock)")
    except Exception as e:
        logger.error(f"Error retrieving last investment value: {str(e)}")
        last_inv_year = last_year_with_data
        last_inv_value = last_k * 0.1  # Assume investment is 10% of capital stock
        logger.info(f"Using estimated investment of {last_inv_value:.2f} billion USD (10% of capital stock)")

    # Initialize the projection dictionary with the last known value
    proj = {last_year_with_data: last_k}

    # Project investment for future years
    try:
        projected_inv = {}
        for y in years_to_project:
            years_from_last_inv = y - last_inv_year
            projected_value = last_inv_value * (1 + avg_inv_growth) ** years_from_last_inv

            # Sanity check - investment shouldn't exceed 40% of previous year's capital
            if y > years_to_project[0]:
                max_reasonable_inv = proj[y-1] * 0.40
            else:
                max_reasonable_inv = last_k * 0.40

            if projected_value > max_reasonable_inv:
                logger.warning(f"Capping projected investment for year {y}: {projected_value:.2f} to {max_reasonable_inv:.2f}")
                projected_value = max_reasonable_inv

            projected_inv[y] = projected_value

        logger.info(f"Projected investment from {min(projected_inv.keys())} to {max(projected_inv.keys())}")
    except Exception as e:
        logger.error(f"Error projecting investment values: {str(e)}")
        # Simple fallback - assume constant investment at last value
        projected_inv = {y: last_inv_value for y in years_to_project}
        logger.info("Using constant investment value for projections due to error")

    # Project capital stock using perpetual inventory method: K_t = (1-delta) * K_{t-1} + I_t
    try:
        # Project forward using the perpetual inventory method
        for y in years_to_project:
            inv_value = projected_inv.get(y, 0)
            previous_k = proj[y-1]

            # Apply the perpetual inventory method
            projected_k = (1-delta) * previous_k + inv_value

            # Sanity check - capital shouldn't decrease too much
            if projected_k < previous_k * 0.85 and inv_value > 0:
                logger.warning(f"Unusual drop in projected capital for year {y}: {projected_k:.2f} < {previous_k:.2f}*0.85")

            # Ensure capital remains positive
            if projected_k <= 0:
                logger.warning(f"Negative capital projection for year {y}: {projected_k:.2f}")
                projected_k = previous_k * 0.9  # Fallback: 10% decline from previous year

            # Store the projected value
            proj[y] = round(projected_k, 2)

        logger.info(f"Successfully projected capital stock for {len(years_to_project)} years")

        # Quality check - examine the growth pattern
        if years_to_project:
            k_growth_rates = [(proj[y]/proj[y-1] - 1) for y in years_to_project]
            avg_k_growth = sum(k_growth_rates) / len(k_growth_rates) if k_growth_rates else 0
            logger.info(f"Average projected capital growth rate: {avg_k_growth:.4f} ({avg_k_growth*100:.2f}%)")

            if any(rate < -0.1 for rate in k_growth_rates):
                years_with_decline = [years_to_project[i] for i, rate in enumerate(k_growth_rates) if rate < -0.1]
                logger.warning(f"Large capital stock declines in years: {years_with_decline}")

        # Create DataFrame with projections
        proj_df = pd.DataFrame(list(proj.items()), columns=['year', 'K_USD_bn'])
        logger.info(f"Created projection dataframe with {proj_df.shape[0]} rows")

        # Merge with original data
        try:
            # Create a clean result DataFrame
            result = k_data.copy()

            # For each projection year, update the capital stock
            for _, row in proj_df.iterrows():
                result.loc[result['year'] == row['year'], 'K_USD_bn'] = row['K_USD_bn']

            # Check for missing projection years (might happen if they weren't in the original data)
            projected_years = set(proj_df['year'])
            result_years = set(result['year'])
            missing_years = projected_years - result_years

            if missing_years:
                logger.warning(f"Some projected years were not in the original data: {missing_years}")
                # Add the missing years
                missing_df = proj_df[proj_df['year'].isin(missing_years)]
                result = pd.concat([result, missing_df], ignore_index=True)

            # Sort by year for consistency
            result = result.sort_values('year').reset_index(drop=True)

            logger.info(f"Final result has capital stock data for {result.dropna(subset=['K_USD_bn']).shape[0]} years")
            return result

        except Exception as e:
            logger.error(f"Error merging projections with original data: {str(e)}")
            # Fallback to simple method
            logger.info("Falling back to simple merge method")
            return pd.merge(k_data, proj_df, on='year', how='outer').sort_values('year')

    except Exception as e:
        logger.error(f"Error projecting capital stock: {str(e)}")
        return k_data
