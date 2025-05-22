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
from china_data.utils.capital import calculate_capital_stock, project_capital_stock
from china_data.utils.processor_hc import project_human_capital
from china_data.utils.economic_indicators import calculate_economic_indicators
from china_data.utils.processor_extrapolation import extrapolate_series_to_end_year
from china_data.utils.processor_output import format_data_for_output

# Import the refactored functions from their new locations
from china_data.utils.processor_dataframe.merge_operations import (
    merge_dataframe_column,
    merge_projections,
    merge_tax_data
)
from china_data.utils.processor_dataframe.metadata_operations import get_projection_metadata
from china_data.utils.processor_dataframe.output_operations import (
    prepare_final_dataframe,
    save_output_files
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    # 1. INITIALIZATION
    args = parse_arguments()
    input_file = args.input_file
    alpha = args.alpha
    output_base = args.output_file
    capital_output_ratio = args.capital_output_ratio
    end_year = args.end_year

    output_dir = get_output_directory()
    logger.info(f"Output files will be saved to: {output_dir}")

    # 2. DATA LOADING
    logger.info("Loading raw data sources")
    raw_data = load_raw_data(input_file=input_file)
    imf_tax_data = load_imf_tax_revenue_data()

    # 3. DATA PREPROCESSING
    logger.info("Converting units")
    processed = convert_units(raw_data)

    # 4. PROJECTIONS & CALCULATIONS
    projection_info = {}

    # 4.1 Capital Stock Calculation
    logger.info("Calculating capital stock")
    capital_df = calculate_capital_stock(raw_data, capital_output_ratio)
    processed, _ = merge_dataframe_column(processed, capital_df, 'K_USD_bn', "capital stock")

    # 4.2 Human Capital Projection
    # Human capital projection
    logger.info(f"Projecting human capital to {end_year}")
    hc_proj = project_human_capital(raw_data, end_year=end_year)

    # Merge human capital projections
    processed, hc_info = merge_projections(processed, hc_proj, 'hc',
                                          "Linear regression", "human capital")

    # Merge tax data
    logger.info("Processing tax revenue data")
    processed, tax_info = merge_tax_data(processed, imf_tax_data)

    # 4.3 Extrapolate base series to end year
    logger.info(f"Extrapolating base series to end year {end_year}")
    try:
        processed, extrapolation_info = extrapolate_series_to_end_year(processed, end_year=end_year, raw_data=raw_data)
        logger.info(f"Extrapolation complete - info contains {len(extrapolation_info)} series")
    except Exception as e:
        logger.error(f"Error during extrapolation: {e}")
        extrapolation_info = {}

    # 4.4 Capital Stock Projection (after investment has been extrapolated)
    logger.info(f"Projecting capital stock to {end_year} using extrapolated investment")
    logger.info("Using unsmoothed capital data")
    k_proj = project_capital_stock(processed, end_year=end_year)

    # Merge capital stock projections
    processed, k_info = merge_projections(processed, k_proj, 'K_USD_bn',
                                         "Investment-based projection", "capital stock")

    # Calculate economic indicators using extrapolated variables
    logger.info("Calculating derived economic indicators from extrapolated variables")
    processed = calculate_economic_indicators(processed, alpha=alpha, logger=logger)

    # 5. DOCUMENTATION - Record projection methods
    logger.info("Recording projection methods for all variables")

    # Human Capital metadata
    hc_metadata = get_projection_metadata(processed, hc_proj, raw_data,
                                         'hc', 'Linear regression', end_year)
    if hc_metadata:
        projection_info['hc'] = hc_metadata

    # Physical Capital metadata
    k_metadata = get_projection_metadata(processed, k_proj, processed,
                                        'K_USD_bn', 'Investment-based projection', end_year)
    if k_metadata:
        projection_info['K_USD_bn'] = k_metadata

    # Tax revenue metadata
    if 'TAX_pct_GDP' in processed.columns and not imf_tax_data.empty:
        try:
            projected_years = [y for y in imf_tax_data['year'] if y > 2023]
            if projected_years:
                projection_info['TAX_pct_GDP'] = {'method': 'IMF projections', 'years': projected_years}
                logger.info(f"Set tax revenue projection method to IMF projections for years {min(projected_years)}-{max(projected_years)}")
        except Exception as e:
            logger.warning(f"Error recording tax projection info: {e}")

    # Update projection info with extrapolation info
    projection_info.update(extrapolation_info)

    # 6. OUTPUT PREPARATION
    # Prepare output data
    logger.info("Preparing data for output")

    # Define column mapping
    column_map = {
        'year': 'Year',
        'GDP_USD_bn': 'GDP',
        'C_USD_bn': 'Consumption',
        'G_USD_bn': 'Government',
        'I_USD_bn': 'Investment',
        'X_USD_bn': 'Exports',
        'M_USD_bn': 'Imports',
        'NX_USD_bn': 'Net Exports',
        'T_USD_bn': 'Tax Revenue (bn USD)',
        'Openness_Ratio': 'Openness Ratio',
        'S_USD_bn': 'Saving (bn USD)',
        'S_priv_USD_bn': 'Private Saving (bn USD)',
        'S_pub_USD_bn': 'Public Saving (bn USD)',
        'Saving_Rate': 'Saving Rate',
        'POP_mn': 'Population',
        'LF_mn': 'Labor Force',
        'K_USD_bn': 'Physical Capital',
        'TFP': 'TFP',
        'FDI_pct_GDP': 'FDI (% of GDP)',
        'TAX_pct_GDP': 'Tax Revenue (% of GDP)',
        'hc': 'Human Capital'
    }

    try:
        # Prepare final dataframe
        final_df = prepare_final_dataframe(processed, column_map)

        # Format data for output
        formatted = format_data_for_output(final_df.copy())

        # Save to output files
        save_output_files(
            formatted,
            output_dir,
            output_base,
            projection_info,
            alpha,
            capital_output_ratio,
            input_file,
            end_year
        )
    except Exception as e:
        logger.error(f"Error preparing output data: {e}")
        raise


if __name__ == '__main__':
    main()
