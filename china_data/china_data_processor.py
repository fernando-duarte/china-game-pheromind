#!/usr/bin/env python3
"""Process raw China economic data and produce analysis files."""

import os
import logging

import pandas as pd
import numpy as np

# Use absolute imports
from china_data.utils import get_output_directory
from china_data.utils.processor_cli import parse_arguments
from china_data.utils.processor_load import load_raw_data, load_imf_tax_revenue_data
from china_data.utils.processor_units import convert_units
from china_data.utils.processor_capital import calculate_capital_stock, project_capital_stock
from china_data.utils.processor_hc import project_human_capital
from china_data.utils.economic_indicators import calculate_economic_indicators
from china_data.utils.processor_extrapolation import extrapolate_series_to_end_year
from china_data.utils.processor_output import format_data_for_output, create_markdown_table

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

    # Get output directory using the common utility function
    output_dir = get_output_directory()
    logger.info(f"Output files will be saved to: {output_dir}")

    raw_data = load_raw_data(input_file=input_file)
    imf_tax_data = load_imf_tax_revenue_data()

    converted = convert_units(raw_data)
    # Handle capital stock calculation
    logger.info("Calculating capital stock")
    capital_df = calculate_capital_stock(raw_data, capital_output_ratio)
    processed = converted.copy()

    # Check if capital stock calculation was successful
    if 'K_USD_bn' in capital_df.columns:
        processed['K_USD_bn'] = capital_df['K_USD_bn']
        non_na_count = processed['K_USD_bn'].notna().sum()
        logger.info(f"Added capital stock data for {non_na_count} years")
    else:
        logger.warning("Capital stock calculation failed - K_USD_bn column not found")
        processed['K_USD_bn'] = np.nan

    # Project capital stock
    logger.info(f"Projecting capital stock to {end_year}")
    k_proj = project_capital_stock(processed, end_year=end_year)

    # Project human capital
    logger.info(f"Projecting human capital to {end_year}")
    hc_proj = project_human_capital(raw_data, end_year=end_year)

    # Capital smoothing has been removed
    logger.info("Using unsmoothed capital data")

    # Merge projected data back into main dataframe
    merged = processed.copy()

    # Merge capital stock projections
    if 'K_USD_bn' in k_proj.columns:
        k_proj_count = 0
        for _, row in k_proj.iterrows():
            if not pd.isna(row['K_USD_bn']):
                merged.loc[merged['year'] == row['year'], 'K_USD_bn'] = row['K_USD_bn']
                k_proj_count += 1
        logger.info(f"Added {k_proj_count} years of projected capital stock data")
    else:
        logger.warning("No capital stock projections available")

    # Merge human capital projections
    if 'hc' in hc_proj.columns:
        hc_proj_count = 0
        for _, row in hc_proj.iterrows():
            if not pd.isna(row['hc']):
                year_rows = merged.loc[merged['year'] == row['year']]
                if not year_rows.empty:
                    merged.loc[merged['year'] == row['year'], 'hc'] = row['hc']
                    hc_proj_count += 1
                else:
                    logger.warning(f"Year {row['year']} not found in main dataframe for hc projection")
        logger.info(f"Added {hc_proj_count} years of projected human capital data")
    else:
        logger.warning("No human capital projections available")

    # Merge IMF tax revenue data
    logger.info("Processing tax revenue data")
    if 'TAX_pct_GDP' in merged.columns:
        # Initialize with NaN
        merged['TAX_pct_GDP'] = np.nan

        # Check if IMF tax data is available
        if not imf_tax_data.empty:
            logger.info(f"Found IMF tax data for {imf_tax_data.shape[0]} years")
            try:
                merged = pd.merge(merged, imf_tax_data, on='year', how='left', suffixes=('', '_imf'))
                # Copy the IMF tax data to the TAX_pct_GDP column
                if 'TAX_pct_GDP_imf' in merged.columns:
                    non_na_count = (~merged['TAX_pct_GDP_imf'].isna()).sum()
                    logger.info(f"Adding tax revenue data for {non_na_count} years")
                    merged.loc[~merged['TAX_pct_GDP_imf'].isna(), 'TAX_pct_GDP'] = merged.loc[~merged['TAX_pct_GDP_imf'].isna(), 'TAX_pct_GDP_imf']
                    merged = merged.drop(columns=['TAX_pct_GDP_imf'])
                else:
                    logger.warning("No IMF tax data column found after merge")
            except Exception as e:
                logger.error(f"Error merging IMF tax data: {e}")
        else:
            logger.warning("No IMF tax data available")
    else:
        logger.warning("TAX_pct_GDP column not found in processed data")

    # Calculate derived economic indicators
    logger.info("Calculating derived economic indicators")
    merged = calculate_economic_indicators(merged, alpha=alpha, logger=logger)

    # Extrapolate series to end year
    logger.info(f"Extrapolating series to end year {end_year}")
    try:
        merged, info = extrapolate_series_to_end_year(merged, end_year=end_year, raw_data=raw_data)
        logger.info(f"Extrapolation complete - info contains {len(info)} series")
    except Exception as e:
        logger.error(f"Error during extrapolation: {e}")
        info = {}  # Initialize empty info if extrapolation fails

    # Document projection methods in the info dictionary
    logger.info("Recording projection methods for human capital and other variables")

    # Set Human Capital method to Linear Regression if 'hc' column exists
    if 'hc' in raw_data.columns:
        try:
            hc_data = raw_data[['year', 'hc']].dropna()
            if not hc_data.empty:
                logger.info(f"Found human capital data for {hc_data.shape[0]} years")
                last_hc_year = hc_data['year'].max()
                logger.info(f"Last year with human capital data: {last_hc_year}")
                if last_hc_year < end_year:
                    hc_years = list(range(int(last_hc_year) + 1, end_year + 1))
                    info['hc'] = {'method': 'Linear regression', 'years': hc_years}
                    logger.info(f"Set human capital projection method to Linear regression for years {min(hc_years)}-{max(hc_years)}")
                else:
                    logger.info("No need to project human capital (data already available to end year)")
            else:
                logger.warning("No non-NA human capital data found")
        except Exception as e:
            logger.warning(f"Error processing human capital data: {e}")
    else:
        logger.warning("Human capital (hc) column not found in raw data")

    # Physical Capital is already projected using investment-based method in project_capital_stock
    # Just make sure it's correctly categorized in the info dictionary
    if 'K_USD_bn' in merged.columns:
        try:
            # Find the years that were projected
            k_data = k_proj[['year', 'K_USD_bn']].copy()
            orig_k_data = processed[['year', 'K_USD_bn']].dropna()

            if not orig_k_data.empty and not k_data.empty:
                projected_years = [y for y in k_data['year'].tolist() if y not in orig_k_data['year'].tolist()]
                if projected_years:
                    info['K_USD_bn'] = {'method': 'Investment-based projection', 'years': projected_years}
                    logger.info(f"Set physical capital projection method to Investment-based projection for years {min(projected_years)}-{max(projected_years)}")
                else:
                    logger.info("No physical capital projections needed")
            else:
                if orig_k_data.empty:
                    logger.warning("No original capital stock data available")
                if k_data.empty:
                    logger.warning("No projected capital stock data available")
        except Exception as e:
            logger.warning(f"Error recording capital stock projection info: {e}")
    else:
        logger.warning("Physical capital (K_USD_bn) column not found")

    # Set IMF tax revenue projections
    if 'TAX_pct_GDP' in merged.columns and not imf_tax_data.empty:
        try:
            projected_years = [y for y in imf_tax_data['year'] if y > 2023]
            if projected_years:
                info['TAX_pct_GDP'] = {'method': 'IMF projections', 'years': projected_years}
                logger.info(f"Set tax revenue projection method to IMF projections for years {min(projected_years)}-{max(projected_years)}")
            else:
                logger.info("No IMF tax revenue projections available")
        except Exception as e:
            logger.warning(f"Error recording tax projection info: {e}")
    else:
        if 'TAX_pct_GDP' not in merged.columns:
            logger.warning("Tax revenue (TAX_pct_GDP) column not found")
        if imf_tax_data.empty:
            logger.warning("No IMF tax data available")



    # Prepare output data
    logger.info("Preparing data for output")

    # Define all possible output columns
    all_output_columns = ['year','GDP_USD_bn','C_USD_bn','G_USD_bn','I_USD_bn','X_USD_bn','M_USD_bn','NX_USD_bn','T_USD_bn','Openness_Ratio','S_USD_bn','S_priv_USD_bn','S_pub_USD_bn','Saving_Rate','POP_mn','LF_mn','K_USD_bn','TFP','FDI_pct_GDP','TAX_pct_GDP','hc']
    column_map = {'year':'Year','GDP_USD_bn':'GDP','C_USD_bn':'Consumption','G_USD_bn':'Government','I_USD_bn':'Investment','X_USD_bn':'Exports','M_USD_bn':'Imports','NX_USD_bn':'Net Exports','T_USD_bn':'Tax Revenue (bn USD)','Openness_Ratio':'Openness Ratio','S_USD_bn':'Saving (bn USD)','S_priv_USD_bn':'Private Saving (bn USD)','S_pub_USD_bn':'Public Saving (bn USD)','Saving_Rate':'Saving Rate','POP_mn':'Population','LF_mn':'Labor Force','K_USD_bn':'Physical Capital','TFP':'TFP','FDI_pct_GDP':'FDI (% of GDP)','TAX_pct_GDP':'Tax Revenue (% of GDP)','hc':'Human Capital'}

    # Filter to only include columns that exist in the DataFrame
    output_columns = [col for col in all_output_columns if col in merged.columns]
    missing_columns = [col for col in all_output_columns if col not in merged.columns]

    if missing_columns:
        logger.warning(f"Some output columns are missing from the data: {missing_columns}")

    logger.info(f"Using {len(output_columns)} output columns: {output_columns}")

    try:
        # Drop duplicate years
        year_counts_before = merged['year'].value_counts()
        duplicated_years = year_counts_before[year_counts_before > 1].index.tolist()

        if duplicated_years:
            logger.warning(f"Found duplicate years in data: {duplicated_years}. Will keep first occurrence only.")

        merged = merged.drop_duplicates(subset=['year'], keep='first')
        logger.info(f"Data contains {merged.shape[0]} unique years from {merged['year'].min()} to {merged['year'].max()}")

        # Select only the output columns and rename them
        final_df = merged[output_columns].rename(columns={col: column_map[col] for col in output_columns})
        logger.info(f"Final data frame has {final_df.shape[0]} rows and {final_df.shape[1]} columns")

        # Format the data for output (handle special formatting, rounding, etc.)
        formatted = format_data_for_output(final_df.copy())

        # Save outputs
        logger.info("Saving output files")

        # Save CSV output
        csv_path = os.path.join(output_dir, f"{output_base}.csv")
        logger.info(f"Writing CSV to: {csv_path}")
        try:
            formatted.to_csv(csv_path, index=False, na_rep='nan')
            logger.info(f"Successfully wrote CSV data to {csv_path}")
        except Exception as e:
            logger.error(f"Error writing CSV file: {e}")

        # Save markdown output
        md_path = os.path.join(output_dir, f"{output_base}.md")
        logger.info(f"Creating markdown table at: {md_path}")
        try:
            create_markdown_table(
                formatted,
                md_path,
                info,
                alpha=alpha,
                capital_output_ratio=capital_output_ratio,
                input_file=input_file,
                end_year=end_year
            )
            logger.info(f"Successfully created markdown table at {md_path}")
        except Exception as e:
            logger.error(f"Error creating markdown table: {e}")

    except Exception as e:
        logger.error(f"Error preparing output data: {e}")
        raise


if __name__ == '__main__':
    main()
