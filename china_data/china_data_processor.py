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


def merge_dataframe_column(target_df, source_df, column_name, description):
    """
    Merge a single column from source dataframe into target dataframe.
    
    Args:
        target_df (pd.DataFrame): The target dataframe to merge into
        source_df (pd.DataFrame): The source dataframe with data to merge
        column_name (str): The name of the column to merge
        description (str): Description for logging purposes
        
    Returns:
        pd.DataFrame: The target dataframe with merged column
        int: Count of non-NA values merged
    """
    result_df = target_df.copy()
    
    if column_name in source_df.columns:
        result_df[column_name] = source_df[column_name]
        non_na_count = result_df[column_name].notna().sum()
        logger.info(f"Added {description} data for {non_na_count} years")
    else:
        logger.warning(f"{description.capitalize()} calculation failed - {column_name} column not found")
        result_df[column_name] = np.nan
        non_na_count = 0
        
    return result_df, non_na_count


def merge_projections(target_df, projection_df, column_name, method_name, description):
    """
    Merge projection data into the main dataframe with proper metadata tracking.
    
    Args:
        target_df (pd.DataFrame): The target dataframe to merge projections into
        projection_df (pd.DataFrame): The dataframe containing projections
        column_name (str): The name of the column to merge
        method_name (str): The method used for projection (for metadata)
        description (str): Description for logging purposes
        
    Returns:
        pd.DataFrame: The updated dataframe with projections
        dict: Projection metadata if projections were applied, None otherwise
    """
    result_df = target_df.copy()
    projection_info = None
    
    if column_name in projection_df.columns:
        # Get valid projections (non-NA values)
        valid_proj = projection_df[['year', column_name]].dropna()
        
        if not valid_proj.empty:
            # Find years that have data in original dataframe
            orig_years = set(result_df.loc[result_df[column_name].notna(), 'year'])
            
            # Use efficient dataframe operations for merging
            result_df = pd.merge(
                result_df,
                valid_proj,
                on='year',
                how='left',
                suffixes=('', f'_proj')
            )
            
            # Create a mask for rows where we want to use the projection
            # (original is NA or missing and projection is available)
            mask = result_df[column_name].isna() & result_df[f'{column_name}_proj'].notna()
            proj_count = mask.sum()
            
            if proj_count > 0:
                # Apply projections where mask is True
                result_df.loc[mask, column_name] = result_df.loc[mask, f'{column_name}_proj']
                
                # Drop the temporary projection column
                result_df = result_df.drop(columns=[f'{column_name}_proj'])
                
                # Get the projected years for metadata
                projected_years = sorted(result_df.loc[mask, 'year'].tolist())
                
                logger.info(f"Added {proj_count} years of projected {description} data")
                
                # Create projection metadata
                projection_info = {
                    'method': method_name,
                    'years': projected_years
                }
            else:
                # No projections were needed/applied
                result_df = result_df.drop(columns=[f'{column_name}_proj'])
                logger.info(f"No {description} projections needed or applied")
        else:
            logger.warning(f"No valid {description} projections available (all NA)")
    else:
        logger.warning(f"No {description} projections available (column missing)")
    
    return result_df, projection_info


def merge_tax_data(target_df, tax_data):
    """
    Merge tax revenue data from IMF into the target dataframe.
    
    Args:
        target_df (pd.DataFrame): The target dataframe
        tax_data (pd.DataFrame): The IMF tax revenue data
        
    Returns:
        pd.DataFrame: The dataframe with tax data merged
        dict: Tax projection metadata if applicable
    """
    result_df = target_df.copy()
    
    # Check if TAX_pct_GDP column exists or needs to be created
    if 'TAX_pct_GDP' not in result_df.columns:
        result_df['TAX_pct_GDP'] = np.nan
        
    # Process tax data if available
    if tax_data.empty:
        logger.warning("No IMF tax data available")
        return result_df, None
    
    logger.info(f"Found IMF tax data for {tax_data.shape[0]} years")
    
    try:
        # Perform merge
        result_df = pd.merge(
            result_df,
            tax_data,
            on='year',
            how='left', 
            suffixes=('', '_imf')
        )
        
        # Check if merge was successful
        if 'TAX_pct_GDP_imf' in result_df.columns:
            # Create mask for non-NA tax data
            mask = ~result_df['TAX_pct_GDP_imf'].isna()
            non_na_count = mask.sum()
            
            if non_na_count > 0:
                logger.info(f"Adding tax revenue data for {non_na_count} years")
                result_df.loc[mask, 'TAX_pct_GDP'] = result_df.loc[mask, 'TAX_pct_GDP_imf']
            
            # Clean up by dropping the temporary column
            result_df = result_df.drop(columns=['TAX_pct_GDP_imf'])
        else:
            logger.warning("No IMF tax data column found after merge")
    except Exception as e:
        logger.error(f"Error merging IMF tax data: {e}")
    
    return result_df


def get_projection_metadata(processed_df, projection_df, original_df, 
                           column_name, method_name, end_year, cutoff_year=None):
    """
    Generate projection metadata for a specific column.
    
    Args:
        processed_df (pd.DataFrame): The processed dataframe
        projection_df (pd.DataFrame): The projection dataframe
        original_df (pd.DataFrame): The original dataframe with raw data
        column_name (str): The column to generate metadata for
        method_name (str): The projection method used
        end_year (int): The end year for projections
        cutoff_year (int): Optional cutoff year to consider projections after
        
    Returns:
        dict: Projection metadata if applicable, None otherwise
    """
    metadata = None
    
    try:
        if column_name not in processed_df.columns:
            return None
        
        if column_name in original_df.columns:
            # For columns present in original data, find last non-NA year
            col_data = original_df[['year', column_name]].dropna()
            
            if not col_data.empty:
                last_data_year = col_data['year'].max()
                
                # If projection extends beyond last available data
                if last_data_year < end_year:
                    # Determine the years that were projected
                    if cutoff_year:
                        projected_years = list(range(max(int(cutoff_year), int(last_data_year) + 1), end_year + 1))
                    else:
                        projected_years = list(range(int(last_data_year) + 1, end_year + 1))
                    
                    if projected_years:
                        metadata = {
                            'method': method_name,
                            'years': projected_years
                        }
                        logger.info(f"Set {column_name} projection method to {method_name} for years {min(projected_years)}-{max(projected_years)}")
        
        # Special case for projection dataframes
        elif projection_df is not None and column_name in projection_df.columns:
            proj_data = projection_df[['year', column_name]].dropna()
            orig_data = processed_df[['year', column_name]].dropna()
            
            if not proj_data.empty and not orig_data.empty:
                # Find years that were projected (in projection but not in original)
                orig_years = set(orig_data['year'].tolist())
                projected_years = [y for y in proj_data['year'].tolist() if y not in orig_years]
                
                if projected_years:
                    metadata = {
                        'method': method_name,
                        'years': projected_years
                    }
                    logger.info(f"Set {column_name} projection method to {method_name} for years {min(projected_years)}-{max(projected_years)}")
    except Exception as e:
        logger.warning(f"Error generating metadata for {column_name}: {e}")
    
    return metadata


def prepare_final_dataframe(processed_df, column_map):
    """
    Prepare the final DataFrame for output by selecting columns and handling duplicates.
    
    Args:
        processed_df (pd.DataFrame): The processed dataframe
        column_map (dict): Mapping of internal column names to display names
        
    Returns:
        pd.DataFrame: The final dataframe ready for output
    """
    # Get available columns
    all_output_columns = list(column_map.keys())
    output_columns = [col for col in all_output_columns if col in processed_df.columns]
    missing_columns = [col for col in all_output_columns if col not in processed_df.columns]
    
    if missing_columns:
        logger.warning(f"Some output columns are missing from the data: {missing_columns}")
    
    logger.info(f"Using {len(output_columns)} output columns: {output_columns}")
    
    # Check for duplicate years
    year_counts = processed_df['year'].value_counts()
    duplicated_years = year_counts[year_counts > 1].index.tolist()
    
    if duplicated_years:
        logger.warning(f"Found duplicate years in data: {duplicated_years}. Will keep first occurrence only.")
    
    # Drop duplicates
    df_unique = processed_df.drop_duplicates(subset=['year'], keep='first')
    logger.info(f"Data contains {df_unique.shape[0]} unique years from {df_unique['year'].min()} to {df_unique['year'].max()}")
    
    # Select and rename columns
    final_df = df_unique[output_columns].rename(columns={col: column_map[col] for col in output_columns})
    logger.info(f"Final data frame has {final_df.shape[0]} rows and {final_df.shape[1]} columns")
    
    return final_df


def save_output_files(formatted_df, output_dir, output_base, projection_info, 
                     alpha, capital_output_ratio, input_file, end_year):
    """
    Save the processed data to output files (CSV and markdown).
    
    Args:
        formatted_df (pd.DataFrame): The formatted dataframe to save
        output_dir (str): The output directory path
        output_base (str): The base name for output files
        projection_info (dict): Projection metadata
        alpha (float): Alpha parameter for capital share
        capital_output_ratio (float): Capital-output ratio
        input_file (str): Input file path
        end_year (int): End year for projections
        
    Returns:
        bool: True if successful, False otherwise
    """
    success = True
    
    # Save CSV output
    csv_path = os.path.join(output_dir, f"{output_base}.csv")
    logger.info(f"Writing CSV to: {csv_path}")
    try:
        formatted_df.to_csv(csv_path, index=False, na_rep='nan')
        logger.info(f"Successfully wrote CSV data to {csv_path}")
    except Exception as e:
        logger.error(f"Error writing CSV file: {e}")
        success = False
    
    # Save markdown output
    md_path = os.path.join(output_dir, f"{output_base}.md")
    logger.info(f"Creating markdown table at: {md_path}")
    try:
        create_markdown_table(
            formatted_df,
            md_path,
            projection_info,
            alpha=alpha,
            capital_output_ratio=capital_output_ratio,
            input_file=input_file,
            end_year=end_year
        )
        logger.info(f"Successfully created markdown table at {md_path}")
    except Exception as e:
        logger.error(f"Error creating markdown table: {e}")
        success = False
    
    return success


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
    
    # 4.2 Projections
    # Capital stock projection
    logger.info(f"Projecting capital stock to {end_year}")
    k_proj = project_capital_stock(processed, end_year=end_year)
    
    # Human capital projection
    logger.info(f"Projecting human capital to {end_year}")
    hc_proj = project_human_capital(raw_data, end_year=end_year)
    
    logger.info("Using unsmoothed capital data")
    
    # 4.3 Merge Projections
    # Merge capital stock projections
    processed, k_info = merge_projections(processed, k_proj, 'K_USD_bn', 
                                         "Investment-based projection", "capital stock")
    
    # Merge human capital projections
    processed, hc_info = merge_projections(processed, hc_proj, 'hc', 
                                          "Linear regression", "human capital")
    
    # Merge tax data
    logger.info("Processing tax revenue data")
    processed = merge_tax_data(processed, imf_tax_data)
    
    # Calculate economic indicators
    logger.info("Calculating derived economic indicators")
    processed = calculate_economic_indicators(processed, alpha=alpha, logger=logger)
    
    # Extrapolate series to end year
    logger.info(f"Extrapolating series to end year {end_year}")
    try:
        processed, extrapolation_info = extrapolate_series_to_end_year(processed, end_year=end_year, raw_data=raw_data)
        logger.info(f"Extrapolation complete - info contains {len(extrapolation_info)} series")
    except Exception as e:
        logger.error(f"Error during extrapolation: {e}")
        extrapolation_info = {}
    
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
