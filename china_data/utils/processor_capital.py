import pandas as pd
import numpy as np


def calculate_capital_stock(raw_data, capital_output_ratio=3.0):
    df = raw_data.copy()
    if not all(c in df.columns for c in ['rkna', 'pl_gdpo', 'cgdpo']):
        df['K_USD_bn'] = np.nan
        return df
    try:
        gdp_2017 = df.loc[df.year == 2017, 'cgdpo'].values[0]
        capital_stock_2017 = gdp_2017 * capital_output_ratio
        rkna_2017 = df.loc[df.year == 2017, 'rkna'].values[0]
        pl_gdpo_2017 = df.loc[df.year == 2017, 'pl_gdpo'].values[0]
        df['K_USD_bn'] = np.nan
        for idx, row in df.iterrows():
            if not pd.isna(row['rkna']) and not pd.isna(row['pl_gdpo']):
                df.loc[idx, 'K_USD_bn'] = (
                    (row['rkna'] / rkna_2017) * capital_stock_2017 * (row['pl_gdpo'] / pl_gdpo_2017)
                ) / 1000
        df['K_USD_bn'] = df['K_USD_bn'].round(2)
    except Exception:
        df['K_USD_bn'] = np.nan
    return df


def project_capital_stock(processed_data, end_year=2025):
    k_data = processed_data[['year', 'K_USD_bn']].copy()
    last_year_with_data = k_data.dropna(subset=['K_USD_bn']).year.max()
    if last_year_with_data >= end_year:
        return k_data
    last_k = k_data.loc[k_data.year == last_year_with_data, 'K_USD_bn'].values[0]
    delta = 0.05
    inv_data = processed_data[['year', 'I_USD_bn']].copy().dropna()
    last_inv_years = sorted(inv_data.year.tolist())[-4:]
    last_inv_values = [inv_data.loc[inv_data.year == y, 'I_USD_bn'].values[0] for y in last_inv_years]
    inv_growth_rates = [(last_inv_values[i] / last_inv_values[i-1] - 1) for i in range(1, len(last_inv_values))]
    avg_inv_growth = sum(inv_growth_rates) / len(inv_growth_rates)
    years_to_project = list(range(last_year_with_data + 1, end_year + 1))
    projected_inv = {}
    last_inv_year = max(inv_data.year)
    last_inv_value = inv_data.loc[inv_data.year == last_inv_year, 'I_USD_bn'].values[0]
    for y in years_to_project:
        years_from_last_inv = y - last_inv_year
        projected_inv[y] = last_inv_value * (1 + avg_inv_growth) ** years_from_last_inv
    proj = {last_year_with_data: last_k}
    for y in years_to_project:
        inv_value = projected_inv[y]
        proj[y] = round((1-delta) * proj[y-1] + inv_value, 2)
    proj_df = pd.DataFrame(list(proj.items()), columns=['year', 'K_USD_bn'])
    k_data = pd.merge(k_data, proj_df, on='year', how='outer', suffixes=('', '_proj'))
    mask = k_data['K_USD_bn'].isna()
    k_data.loc[mask, 'K_USD_bn'] = k_data.loc[mask, 'K_USD_bn_proj']
    return k_data.drop(columns=['K_USD_bn_proj'])
