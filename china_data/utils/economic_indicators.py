"""
Utility functions for calculating economic indicators for China data.

This module contains functions to calculate various economic indicators:
- Total Factor Productivity (TFP)
- Net exports
- Capital-output ratio
- Tax revenue in USD billions
- Openness ratio
- Total savings
- Private savings
- Public savings
- Saving rate
"""

import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def calculate_tfp(data, alpha=1/3):
    """
    Calculate Total Factor Productivity using Cobb-Douglas production function.

    Args:
        data: DataFrame with GDP_USD_bn, K_USD_bn, LF_mn, and hc columns
        alpha: Capital share parameter (default: 1/3)

    Returns:
        DataFrame with TFP column added
    """
    df = data.copy()
    required = ['GDP_USD_bn', 'K_USD_bn', 'LF_mn']
    if not all(col in df.columns for col in required):
        df['TFP'] = np.nan
        return df
    if 'hc' not in df.columns:
        df['hc'] = np.nan
    if df['hc'].isna().any():
        hc_data = df[['year', 'hc']].dropna(subset=['hc'])
        if len(hc_data) >= 2:
            X = hc_data['year'].values.reshape(-1, 1)
            y = hc_data['hc'].values
            model = LinearRegression()
            model.fit(X, y)
            missing = df[df['hc'].isna()]['year'].values
            if len(missing) > 0:
                preds = model.predict(missing.reshape(-1, 1))
                for i, year in enumerate(missing):
                    df.loc[df['year'] == year, 'hc'] = round(preds[i], 4)
    try:
        df['TFP'] = df['GDP_USD_bn'] / (
            (df['K_USD_bn'] ** alpha) * ((df['LF_mn'] * df['hc']) ** (1 - alpha))
        )
        df['TFP'] = df['TFP'].round(4)
    except Exception:
        df['TFP'] = np.nan
    return df


def calculate_economic_indicators(merged, alpha=1/3, logger=None):
    """
    Calculate various economic indicators based on the input data.

    This function calculates:
    - Net exports
    - Capital-output ratio
    - Total Factor Productivity
    - Tax revenue in USD billions
    - Openness ratio
    - Total savings
    - Private savings
    - Public savings
    - Saving rate

    Args:
        merged: DataFrame with economic data
        alpha: Capital share parameter for TFP calculation (default: 1/3)
        logger: Logger object (optional)

    Returns:
        DataFrame with additional economic indicators
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df = merged.copy()

    # Net exports
    if all(c in df.columns for c in ['X_USD_bn', 'M_USD_bn']):
        logger.info("Calculating net exports (NX_USD_bn)")
        df['NX_USD_bn'] = df['X_USD_bn'] - df['M_USD_bn']
    else:
        missing = [c for c in ['X_USD_bn', 'M_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate net exports - missing columns: {missing}")

    # Capital-output ratio
    if all(c in df.columns for c in ['K_USD_bn', 'GDP_USD_bn']):
        logger.info("Calculating capital-output ratio (K_Y_ratio)")
        df['K_Y_ratio'] = df['K_USD_bn'] / df['GDP_USD_bn']
    else:
        missing = [c for c in ['K_USD_bn', 'GDP_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate K/Y ratio - missing columns: {missing}")

    # Calculate TFP
    logger.info(f"Calculating Total Factor Productivity with alpha={alpha}")
    try:
        df = calculate_tfp(df, alpha=alpha)
        if 'TFP' in df.columns:
            non_na_count = df['TFP'].notna().sum()
            logger.info(f"TFP calculated for {non_na_count} years")
        else:
            logger.warning("TFP calculation failed - TFP column not found")
    except Exception as e:
        logger.error(f"Error calculating TFP: {e}")
        # Ensure TFP column exists even if calculation fails
        if 'TFP' not in df.columns:
            df['TFP'] = np.nan

    # Tax revenue in USD billions
    if all(c in df.columns for c in ['TAX_pct_GDP', 'GDP_USD_bn']):
        logger.info("Calculating tax revenue in USD billions (T_USD_bn)")
        df['T_USD_bn'] = (df['TAX_pct_GDP'] / 100) * df['GDP_USD_bn']
        non_na_count = df['T_USD_bn'].notna().sum()
        logger.info(f"Calculated T_USD_bn for {non_na_count} years")
    else:
        missing = [c for c in ['TAX_pct_GDP', 'GDP_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate T_USD_bn - missing columns: {missing}")
        df['T_USD_bn'] = np.nan

    # Openness ratio (trade as % of GDP)
    if all(c in df.columns for c in ['X_USD_bn', 'M_USD_bn', 'GDP_USD_bn']):
        logger.info("Calculating trade openness ratio")
        # This is the ratio of total trade (exports + imports) to GDP
        df['Openness_Ratio'] = (df['X_USD_bn'] + df['M_USD_bn']) / df['GDP_USD_bn']
        non_na_count = df['Openness_Ratio'].notna().sum()
        logger.info(f"Calculated Openness_Ratio for {non_na_count} years")
    else:
        missing = [c for c in ['X_USD_bn', 'M_USD_bn', 'GDP_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate Openness_Ratio - missing columns: {missing}")

    # Total savings
    if all(c in df.columns for c in ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn']):
        logger.info("Calculating total savings (S_USD_bn)")
        df['S_USD_bn'] = df['GDP_USD_bn'] - df['C_USD_bn'] - df['G_USD_bn']
        non_na_count = df['S_USD_bn'].notna().sum()
        logger.info(f"Calculated S_USD_bn for {non_na_count} years")
    else:
        missing = [c for c in ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate S_USD_bn - missing columns: {missing}")

    # Private savings
    if all(c in df.columns for c in ['GDP_USD_bn', 'T_USD_bn', 'C_USD_bn']):
        logger.info("Calculating private savings (S_priv_USD_bn)")
        df['S_priv_USD_bn'] = df['GDP_USD_bn'] - df['T_USD_bn'] - df['C_USD_bn']
        non_na_count = df['S_priv_USD_bn'].notna().sum()
        logger.info(f"Calculated S_priv_USD_bn for {non_na_count} years")
    else:
        missing = [c for c in ['GDP_USD_bn', 'T_USD_bn', 'C_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate S_priv_USD_bn - missing columns: {missing}")

    # Public savings
    if all(c in df.columns for c in ['T_USD_bn', 'G_USD_bn']):
        logger.info("Calculating public savings (S_pub_USD_bn)")
        df['S_pub_USD_bn'] = df['T_USD_bn'] - df['G_USD_bn']
        non_na_count = df['S_pub_USD_bn'].notna().sum()
        logger.info(f"Calculated S_pub_USD_bn for {non_na_count} years")
    else:
        missing = [c for c in ['T_USD_bn', 'G_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate S_pub_USD_bn - missing columns: {missing}")

    # Saving rate
    if all(c in df.columns for c in ['S_USD_bn', 'GDP_USD_bn']):
        logger.info("Calculating saving rate")
        df['Saving_Rate'] = df['S_USD_bn'] / df['GDP_USD_bn']
        non_na_count = df['Saving_Rate'].notna().sum()
        logger.info(f"Calculated Saving_Rate for {non_na_count} years")
    else:
        missing = [c for c in ['S_USD_bn', 'GDP_USD_bn'] if c not in df.columns]
        logger.warning(f"Cannot calculate Saving_Rate - missing columns: {missing}")

    return df
