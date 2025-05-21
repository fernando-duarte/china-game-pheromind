"""
Functions for merging dataframes and data in the China data processor.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)


def merge_dataframe_column(target_df: pd.DataFrame, 
                          source_df: pd.DataFrame, 
                          column_name: str, 
                          description: str) -> Tuple[pd.DataFrame, int]:
    """
    Merge a single column from source dataframe into target dataframe.
    
    Args:
        target_df: The target dataframe to merge into
        source_df: The source dataframe with data to merge
        column_name: The name of the column to merge
        description: Description for logging purposes
        
    Returns:
        The target dataframe with merged column and count of non-NA values merged
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


def merge_projections(target_df: pd.DataFrame, 
                     projection_df: pd.DataFrame, 
                     column_name: str, 
                     method_name: str, 
                     description: str) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Merge projection data into the main dataframe with proper metadata tracking.
    
    Args:
        target_df: The target dataframe to merge projections into
        projection_df: The dataframe containing projections
        column_name: The name of the column to merge
        method_name: The method used for projection (for metadata)
        description: Description for logging purposes
        
    Returns:
        The updated dataframe with projections and projection metadata if projections were applied
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


def merge_tax_data(target_df: pd.DataFrame, 
                  tax_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Merge tax revenue data from IMF into the target dataframe.
    
    Args:
        target_df: The target dataframe
        tax_data: The IMF tax revenue data
        
    Returns:
        The dataframe with tax data merged and tax projection metadata if applicable
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
    
    return result_df, None
