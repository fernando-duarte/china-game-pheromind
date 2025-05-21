"""
Functions for handling metadata in the China data processor.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def get_projection_metadata(processed_df: pd.DataFrame, 
                           projection_df: Optional[pd.DataFrame], 
                           original_df: pd.DataFrame,
                           column_name: str, 
                           method_name: str, 
                           end_year: int, 
                           cutoff_year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Generate projection metadata for a specific column.
    
    Args:
        processed_df: The processed dataframe
        projection_df: The projection dataframe
        original_df: The original dataframe with raw data
        column_name: The column to generate metadata for
        method_name: The projection method used
        end_year: The end year for projections
        cutoff_year: Optional cutoff year to consider projections after
        
    Returns:
        Projection metadata if applicable, None otherwise
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
