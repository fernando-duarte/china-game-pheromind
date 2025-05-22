"""
Functions for preparing and saving output in the China data processor.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union

from china_data.utils.processor_output import create_markdown_table

logger = logging.getLogger(__name__)


def prepare_final_dataframe(processed_df: pd.DataFrame, 
                           column_map: Dict[str, str]) -> pd.DataFrame:
    """
    Prepare the final DataFrame for output by selecting columns and handling duplicates.
    
    Args:
        processed_df: The processed dataframe
        column_map: Mapping of internal column names to display names
        
    Returns:
        The final dataframe ready for output
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


def save_output_files(formatted_df: pd.DataFrame, 
                     output_dir: str, 
                     output_base: str, 
                     projection_info: Dict[str, Any], 
                     alpha: float, 
                     capital_output_ratio: float, 
                     input_file: str, 
                     end_year: int) -> bool:
    """
    Save the processed data to output files (CSV and markdown).
    
    Args:
        formatted_df: The formatted dataframe to save
        output_dir: The output directory path
        output_base: The base name for output files
        projection_info: Projection metadata
        alpha: Alpha parameter for capital share
        capital_output_ratio: Capital-output ratio
        input_file: Input file path
        end_year: End year for projections
        
    Returns:
        True if successful, False otherwise
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
