import pandas as pd
import numpy as np
import logging
from china_data.utils.extrapolation_methods import (
    extrapolate_with_arima,
    extrapolate_with_linear_regression,
    extrapolate_with_average_growth_rate
)

logger = logging.getLogger(__name__)


def _prepare(df, end_year):
    max_year = df.year.max()
    if max_year >= end_year:
        missing = False
        key = ['GDP_USD_bn','C_USD_bn','G_USD_bn','I_USD_bn','X_USD_bn','M_USD_bn','POP_mn','LF_mn']
        for year in [end_year-1, end_year]:
            for var in key:
                if var in df.columns and pd.isna(df.loc[df.year == year, var].values[0]):
                    missing = True
                    break
            if missing:
                break
        if not missing:
            return df, {}, []
        years_to_add = [end_year-1, end_year]
    else:
        years_to_add = list(range(max_year + 1, end_year + 1))
    new_years_df = pd.DataFrame({'year': years_to_add})
    df = pd.concat([df, new_years_df], ignore_index=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_extrapolate = [c for c in numeric_cols if c != 'year']
    return df, { }, years_to_add, cols_to_extrapolate


def _apply_methods(df, years_to_add, cols, info):
    """
    Apply appropriate extrapolation methods to each column based on column type.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        years_to_add (list): List of years to add projections for
        cols (list): List of columns to extrapolate
        info (dict): Dictionary to store extrapolation method information

    Returns:
        tuple: (updated DataFrame, updated info dictionary)
    """
    gdp = ['GDP_USD_bn','C_USD_bn','G_USD_bn','I_USD_bn','X_USD_bn','M_USD_bn','NX_USD_bn']
    demographic = ['POP_mn','LF_mn']
    human = ['hc']

    for col in cols:
        if df[col].isna().all():
            continue

        historical = df[['year', col]].dropna()
        if len(historical) == 0:
            continue

        last_year = int(historical['year'].max())
        yrs = [y for y in range(last_year + 1, years_to_add[-1] + 1)]
        if not yrs:
            continue

        # Determine which method to try first based on column type
        success = False

        # For GDP-related columns, try ARIMA first
        if col in gdp:
            df_updated, success, method = extrapolate_with_arima(
                df, col, yrs, min_data_points=5, order=(1, 1, 1)
            )
            if success:
                df = df_updated
                info[col] = {'method': method, 'years': yrs}
                continue

        # For demographic and human capital columns, try linear regression
        if (col in demographic or col in human) and not success:
            df_updated, success, method = extrapolate_with_linear_regression(
                df, col, yrs, min_data_points=2
            )
            if success:
                df = df_updated
                info[col] = {'method': method, 'years': yrs}
                continue

        # For all columns that couldn't be extrapolated with the above methods,
        # fall back to average growth rate
        if not success:
            # For demographic data, use different default growth rates
            default_growth = 0.03  # Default for most series
            lookback = 4  # Default lookback period

            df_updated, success, method = extrapolate_with_average_growth_rate(
                df, col, yrs, lookback_years=lookback, default_growth=default_growth
            )
            if success:
                df = df_updated
                info[col] = {'method': method, 'years': yrs}

    return df, info


def _finalize(df, years_to_add, raw_data, cols, info, end_year):
    # Net exports calculation removed - this is now handled in economic_indicators.py
    key_vars = ['GDP_USD_bn','C_USD_bn','G_USD_bn','I_USD_bn','X_USD_bn','M_USD_bn','POP_mn','LF_mn','FDI_pct_GDP','TAX_pct_GDP','hc','K_USD_bn']
    for year in years_to_add:
        for col in key_vars:
            if col in df.columns and pd.isna(df.loc[df.year == year, col].values[0]):
                last_valid = df[df.year < year][[col]].dropna()
                if not last_valid.empty:
                    last_value = last_valid.iloc[-1].values[0]
                    last_year = df.loc[last_valid.index[-1], 'year']
                    default_growth = 0.03
                    if col in ['GDP_USD_bn','C_USD_bn','G_USD_bn','I_USD_bn','X_USD_bn','M_USD_bn']:
                        default_growth = 0.05
                    elif col == 'POP_mn':
                        default_growth = 0.005
                    elif col == 'LF_mn':
                        default_growth = 0.01
                    elif col == 'hc':
                        default_growth = 0.01
                    elif col == 'K_USD_bn':
                        default_growth = 0.04
                    historical = df[df.year <= df.year.max()][[col]].dropna()
                    if len(historical) >= 2:
                        n_years = min(5, len(historical))
                        last_years = historical.iloc[-n_years:].values.flatten()
                        if len(last_years) > 1:
                            growth_rates = [(last_years[i] / last_years[i-1]) - 1 for i in range(1,len(last_years))]
                            avg_growth = sum(growth_rates) / len(growth_rates)
                        else:
                            avg_growth = default_growth
                    else:
                        avg_growth = default_growth
                    projected_value = last_value * (1 + avg_growth) ** (year - last_year)
                    df.loc[df.year == year, col] = round(projected_value, 4)
    for col in cols:
        if col in raw_data.columns:
            raw_non_nan = raw_data[['year', col]].dropna()
            if len(raw_non_nan) == 0:
                continue
            last_actual_year = int(raw_non_nan['year'].max())
        else:
            hist = df[['year', col]].dropna()
            if len(hist) == 0:
                continue
            last_actual_year = int(hist['year'].max())
        if last_actual_year < end_year:
            extrap_years = [y for y in range(last_actual_year + 1, end_year + 1)]
            if extrap_years:
                method = info.get(col, {}).get('method', 'Extrapolated')
                info[col] = {'method': method, 'years': extrap_years}
    return df, info


def extrapolate_series_to_end_year(data, end_year=2025, raw_data=None):
    df, info, years_to_add, cols = _prepare(data.copy(), end_year)
    if years_to_add == [] and info == {}:
        return df, info
    df, info = _apply_methods(df, years_to_add, cols, info)
    df, info = _finalize(df, years_to_add, raw_data if raw_data is not None else data, cols, info, end_year)
    return df, info
