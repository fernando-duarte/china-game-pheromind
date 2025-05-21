"""
Capital stock validation module.

This module provides functions for validating capital stock and investment data
for consistency and quality.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_capital_data(df, verbose=True):
    """
    Validate capital stock and investment data for consistency and quality.
    
    This function checks for common issues in capital and investment data:
    - Missing values
    - Negative values
    - Unrealistic growth rates
    - Consistency between capital stock and investment (perpetual inventory relation)
    
    Args:
        df: DataFrame with 'year', 'K_USD_bn', and 'I_USD_bn' columns
        verbose: Whether to print detailed validation information
        
    Returns:
        Dict with validation results and summary statistics
    """
    logger.info("Validating capital stock and investment data")
    
    # Initialize results dictionary
    results = {
        'valid': False,
        'issues': [],
        'stats': {},
        'issue_count': 0
    }
    
    # Validate input
    if not isinstance(df, pd.DataFrame):
        results['issues'].append("Input is not a pandas DataFrame")
        results['issue_count'] += 1
        return results
    
    required_cols = ['year']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        results['issues'].append(f"Missing required columns: {missing_cols}")
        results['issue_count'] += 1
        return results
    
    # Check which of the optional columns are available
    has_k = 'K_USD_bn' in df.columns
    has_i = 'I_USD_bn' in df.columns
    
    if not has_k and not has_i:
        results['issues'].append("Neither capital (K_USD_bn) nor investment (I_USD_bn) columns found")
        results['issue_count'] += 1
        return results
    
    # Analyze years
    years = sorted(df['year'].unique())
    results['stats']['years'] = {
        'min': min(years) if years else None,
        'max': max(years) if years else None,
        'count': len(years),
        'consecutive': all(years[i+1] == years[i] + 1 for i in range(len(years) - 1)) if len(years) > 1 else None
    }
    
    # Check for gaps in years
    if results['stats']['years']['consecutive'] is False:
        # Find gaps
        gaps = []
        for i in range(len(years) - 1):
            if years[i+1] > years[i] + 1:
                gaps.append((years[i], years[i+1]))
        if gaps:
            results['issues'].append(f"Gap(s) in years: {gaps}")
            results['issue_count'] += 1
            results['stats']['years']['gaps'] = gaps
    
    # Analyze capital stock data if available
    if has_k:
        k_data = df.dropna(subset=['K_USD_bn'])
        k_na_count = df.shape[0] - k_data.shape[0]
        
        results['stats']['capital'] = {
            'count': k_data.shape[0],
            'na_count': k_na_count,
            'min': k_data['K_USD_bn'].min() if not k_data.empty else None,
            'max': k_data['K_USD_bn'].max() if not k_data.empty else None,
            'mean': k_data['K_USD_bn'].mean() if not k_data.empty else None
        }
        
        # Check for issues in capital stock data
        if k_na_count > 0:
            # If more than 20% of values are missing, flag as an issue
            if k_na_count / df.shape[0] > 0.2:
                results['issues'].append(f"High proportion of missing capital data: {k_na_count}/{df.shape[0]} rows ({k_na_count/df.shape[0]*100:.1f}%)")
                results['issue_count'] += 1
        
        if not k_data.empty:
            # Check for negative capital stock
            neg_k = k_data[k_data['K_USD_bn'] <= 0]
            if not neg_k.empty:
                neg_years = neg_k['year'].tolist()
                results['issues'].append(f"Non-positive capital stock for years: {neg_years}")
                results['issue_count'] += 1
            
            # Check for unrealistic capital stock growth rates
            if k_data.shape[0] > 1:
                # Sort by year
                k_data_sorted = k_data.sort_values('year')
                
                # Calculate growth rates
                k_data_sorted['prev_K'] = k_data_sorted['K_USD_bn'].shift(1)
                k_data_sorted['growth'] = k_data_sorted['K_USD_bn'] / k_data_sorted['prev_K'] - 1
                
                # Identify extreme growth rates (>30% per year)
                extreme_growth = k_data_sorted.iloc[1:].loc[k_data_sorted['growth'] > 0.3]
                if not extreme_growth.empty:
                    extreme_years = extreme_growth['year'].tolist()
                    results['issues'].append(f"Extreme capital stock growth (>30%) for years: {extreme_years}")
                    results['issue_count'] += 1
                
                # Identify extreme declines (>15% per year)
                extreme_decline = k_data_sorted.iloc[1:].loc[k_data_sorted['growth'] < -0.15]
                if not extreme_decline.empty:
                    extreme_years = extreme_decline['year'].tolist()
                    results['issues'].append(f"Extreme capital stock decline (>15%) for years: {extreme_years}")
                    results['issue_count'] += 1
    
    # Analyze investment data if available
    if has_i:
        i_data = df.dropna(subset=['I_USD_bn'])
        i_na_count = df.shape[0] - i_data.shape[0]
        
        results['stats']['investment'] = {
            'count': i_data.shape[0],
            'na_count': i_na_count,
            'min': i_data['I_USD_bn'].min() if not i_data.empty else None,
            'max': i_data['I_USD_bn'].max() if not i_data.empty else None,
            'mean': i_data['I_USD_bn'].mean() if not i_data.empty else None
        }
        
        # Check for issues in investment data
        if i_na_count > 0:
            # If more than 20% of values are missing, flag as an issue
            if i_na_count / df.shape[0] > 0.2:
                results['issues'].append(f"High proportion of missing investment data: {i_na_count}/{df.shape[0]} rows ({i_na_count/df.shape[0]*100:.1f}%)")
                results['issue_count'] += 1
        
        if not i_data.empty:
            # Check for large negative investment
            neg_i = i_data[i_data['I_USD_bn'] < 0]
            if not neg_i.empty:
                neg_years = neg_i['year'].tolist()
                results['issues'].append(f"Negative investment for years: {neg_years}")
                results['issue_count'] += 1
            
            # Check for unrealistic investment growth rates
            if i_data.shape[0] > 1:
                # Sort by year
                i_data_sorted = i_data.sort_values('year')
                
                # Calculate growth rates
                i_data_sorted['prev_I'] = i_data_sorted['I_USD_bn'].shift(1)
                i_data_sorted['growth'] = i_data_sorted['I_USD_bn'] / i_data_sorted['prev_I'] - 1
                
                # Identify extreme growth rates (>100% per year)
                extreme_growth = i_data_sorted.iloc[1:].loc[i_data_sorted['growth'] > 1.0]
                if not extreme_growth.empty:
                    extreme_growth_years = extreme_growth['year'].tolist()
                    results['issues'].append(f"Extreme investment growth (>100%) for years: {extreme_growth_years}")
                    results['issue_count'] += 1
                    
                # Identify extreme declines (>50% per year)
                extreme_decline = i_data_sorted.iloc[1:].loc[i_data_sorted['growth'] < -0.5]
                if not extreme_decline.empty:
                    extreme_decline_years = extreme_decline['year'].tolist()
                    results['issues'].append(f"Extreme investment decline (>50%) for years: {extreme_decline_years}")
                    results['issue_count'] += 1
        
        # Check consistency between capital and investment if both are available
        if has_k and has_i:
            # Create a merged dataset with both capital and investment
            ki_data = df[['year', 'K_USD_bn', 'I_USD_bn']].copy().dropna()
            
            if ki_data.shape[0] > 1:
                # Sort by year
                ki_data = ki_data.sort_values('year')
                
                # Calculate implied vs actual capital growth
                prev_k = ki_data['K_USD_bn'].shift(1)
                prev_i = ki_data['I_USD_bn'].shift(1)
                
                # Assume 5% depreciation for checking consistency
                delta = 0.05
                
                # Calculate expected capital based on previous capital, depreciation, and investment
                ki_data['expected_K'] = (1 - delta) * prev_k + prev_i
                
                # Calculate discrepancy between actual and expected capital
                ki_data['discrepancy'] = ki_data['K_USD_bn'] - ki_data['expected_K']
                ki_data['discrepancy_pct'] = ki_data['discrepancy'] / ki_data['expected_K'] * 100
                
                # Identify large discrepancies
                large_discrepancy = ki_data.iloc[1:].loc[abs(ki_data['discrepancy_pct']) > 20]
                
                if not large_discrepancy.empty:
                    discrepancy_years = large_discrepancy['year'].tolist()
                    results['issues'].append(f"Large discrepancy (>20%) between actual capital and expected capital for years: {discrepancy_years}")
                    results['issue_count'] += 1
                    
                    # Add discrepancy statistics to results
                    results['stats']['discrepancy'] = {
                        'mean_abs_pct': abs(ki_data['discrepancy_pct']).mean(),
                        'max_abs_pct': abs(ki_data['discrepancy_pct']).max(),
                        'years_with_large_discrepancy': discrepancy_years
                    }
                else:
                    results['stats']['discrepancy'] = {
                        'mean_abs_pct': abs(ki_data['discrepancy_pct']).mean(),
                        'max_abs_pct': abs(ki_data['discrepancy_pct']).max(),
                        'years_with_large_discrepancy': []
                    }
        
        # Determine overall validity
        results['valid'] = results['issue_count'] == 0
        
        # Log validation results
        if results['valid']:
            logger.info("Capital data validation passed with no issues")
        else:
            logger.warning(f"Capital data validation found {results['issue_count']} issues")
            for issue in results['issues']:
                logger.warning(f"Validation issue: {issue}")
        
        # Print detailed validation information if requested
        if verbose:
            logger.info("Capital data validation statistics:")
            if 'years' in results['stats']:
                yr_stats = results['stats']['years']
                logger.info(f"Years: {yr_stats['min']} to {yr_stats['max']} ({yr_stats['count']} years)")
                if yr_stats['consecutive'] is not None:
                    logger.info(f"Years are {'consecutive' if yr_stats['consecutive'] else 'not consecutive'}")
            
            if 'capital' in results['stats']:
                k_stats = results['stats']['capital']
                logger.info(f"Capital stock: {k_stats['count']} data points, range: {k_stats['min']:.2f} to {k_stats['max']:.2f} billion USD")
                logger.info(f"Missing capital data: {k_stats['na_count']} points ({k_stats['na_count']/len(df)*100:.1f}%)")
            
            if 'investment' in results['stats']:
                i_stats = results['stats']['investment']
                logger.info(f"Investment: {i_stats['count']} data points, range: {i_stats['min']:.2f} to {i_stats['max']:.2f} billion USD")
                logger.info(f"Missing investment data: {i_stats['na_count']} points ({i_stats['na_count']/len(df)*100:.1f}%)")
            
            if 'discrepancy' in results['stats']:
                d_stats = results['stats']['discrepancy']
                logger.info(f"Perpetual inventory consistency: mean abs discrepancy = {d_stats['mean_abs_pct']:.2f}%, max = {d_stats['max_abs_pct']:.2f}%")
        
        return results
