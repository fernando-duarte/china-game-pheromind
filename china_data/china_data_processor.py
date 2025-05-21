#!/usr/bin/env python3
"""Process raw China economic data and produce analysis files."""

import os
import logging

import pandas as pd
import numpy as np

from .utils.processor_cli import parse_arguments
from .utils.processor_load import load_raw_data, load_imf_tax_revenue_data
from .utils.processor_units import convert_units
from .utils.processor_capital import calculate_capital_stock, project_capital_stock
from .utils.processor_hc import project_human_capital
from .utils.processor_tfp import calculate_tfp
from .utils.processor_extrapolation import extrapolate_series_to_end_year
from .utils.processor_output import format_data_for_output, create_markdown_table

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    args = parse_arguments()
    input_file = args.input_file
    alpha = args.alpha
    output_base = args.output_file
    capital_output_ratio = args.capital_output_ratio
    end_year = args.end_year

    output_dir = os.path.join('.', 'output')
    os.makedirs(output_dir, exist_ok=True)

    raw_data = load_raw_data(data_dir='.', input_file=input_file)
    imf_tax_data = load_imf_tax_revenue_data(data_dir='.')

    converted = convert_units(raw_data)
    capital_df = calculate_capital_stock(raw_data, capital_output_ratio)
    processed = converted.copy()
    processed['K_USD_bn'] = capital_df['K_USD_bn']

    k_proj = project_capital_stock(processed, end_year=end_year)
    hc_proj = project_human_capital(raw_data, end_year=end_year)

    merged = processed.copy()
    for _, row in k_proj.iterrows():
        if not pd.isna(row['K_USD_bn']):
            merged.loc[merged['year'] == row['year'], 'K_USD_bn'] = row['K_USD_bn']
    for _, row in hc_proj.iterrows():
        if 'hc' in row and not pd.isna(row['hc']):
            merged.loc[merged['year'] == row['year'], 'hc'] = row['hc']

    if 'TAX_pct_GDP' in merged.columns:
        merged['TAX_pct_GDP'] = np.nan
        merged = pd.merge(merged, imf_tax_data, on='year', how='left', suffixes=('', '_imf'))

    if all(c in merged.columns for c in ['X_USD_bn', 'M_USD_bn']):
        merged['NX_USD_bn'] = merged['X_USD_bn'] - merged['M_USD_bn']
    if all(c in merged.columns for c in ['K_USD_bn', 'GDP_USD_bn']):
        merged['K_Y_ratio'] = merged['K_USD_bn'] / merged['GDP_USD_bn']
    merged = calculate_tfp(merged, alpha=alpha)

    merged, info = extrapolate_series_to_end_year(merged, end_year=end_year, raw_data=raw_data)

    if 'TAX_pct_GDP' in info:
        projected_years = [y for y in imf_tax_data['year'] if y > 2023]
        if projected_years:
            info['TAX_pct_GDP'] = {'method': 'IMF projections', 'years': projected_years}

    if 'TAX_pct_GDP' in merged.columns and 'GDP_USD_bn' in merged.columns:
        merged['T_USD_bn'] = (merged['TAX_pct_GDP'] / 100) * merged['GDP_USD_bn']
    if all(c in merged.columns for c in ['X_USD_bn', 'M_USD_bn', 'GDP_USD_bn']):
        merged['Openness_Ratio'] = (merged['X_USD_bn'] + merged['M_USD_bn']) / merged['GDP_USD_bn']
    if all(c in merged.columns for c in ['GDP_USD_bn', 'C_USD_bn', 'G_USD_bn']):
        merged['S_USD_bn'] = merged['GDP_USD_bn'] - merged['C_USD_bn'] - merged['G_USD_bn']
    if all(c in merged.columns for c in ['GDP_USD_bn', 'T_USD_bn', 'C_USD_bn']):
        merged['S_priv_USD_bn'] = merged['GDP_USD_bn'] - merged['T_USD_bn'] - merged['C_USD_bn']
    if all(c in merged.columns for c in ['T_USD_bn', 'G_USD_bn']):
        merged['S_pub_USD_bn'] = merged['T_USD_bn'] - merged['G_USD_bn']
    if all(c in merged.columns for c in ['S_USD_bn', 'GDP_USD_bn']):
        merged['Saving_Rate'] = merged['S_USD_bn'] / merged['GDP_USD_bn']

    merged = calculate_tfp(merged, alpha=alpha)

    output_columns = ['year','GDP_USD_bn','C_USD_bn','G_USD_bn','I_USD_bn','X_USD_bn','M_USD_bn','NX_USD_bn','T_USD_bn','Openness_Ratio','S_USD_bn','S_priv_USD_bn','S_pub_USD_bn','Saving_Rate','POP_mn','LF_mn','K_USD_bn','TFP','FDI_pct_GDP','TAX_pct_GDP','hc']
    column_map = {'year':'Year','GDP_USD_bn':'GDP','C_USD_bn':'Consumption','G_USD_bn':'Government','I_USD_bn':'Investment','X_USD_bn':'Exports','M_USD_bn':'Imports','NX_USD_bn':'Net Exports','T_USD_bn':'Tax Revenue (bn USD)','Openness_Ratio':'Openness Ratio','S_USD_bn':'Saving (bn USD)','S_priv_USD_bn':'Private Saving (bn USD)','S_pub_USD_bn':'Public Saving (bn USD)','Saving_Rate':'Saving Rate','POP_mn':'Population','LF_mn':'Labor Force','K_USD_bn':'Physical Capital','TFP':'TFP','FDI_pct_GDP':'FDI (% of GDP)','TAX_pct_GDP':'Tax Revenue (% of GDP)','hc':'Human Capital'}
    merged = merged.drop_duplicates(subset=['year'], keep='first')
    final_df = merged[output_columns].rename(columns=column_map)
    formatted = format_data_for_output(final_df.copy())

    csv_path = os.path.join(output_dir, f"{output_base}.csv")
    formatted.to_csv(csv_path, index=False, na_rep='nan')
    md_path = os.path.join(output_dir, f"{output_base}.md")
    create_markdown_table(formatted, md_path, info, alpha=alpha, capital_output_ratio=capital_output_ratio, input_file=input_file, end_year=end_year)


if __name__ == '__main__':
    main()
