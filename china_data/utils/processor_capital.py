import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_capital_stock(raw_data, capital_output_ratio=3.0):
    """
    Calculate capital stock using PWT data and capital-output ratio.
    
    This function calculates physical capital stock based on Penn World Table data
    using relative real capital stock and price level indices, normalized to a
    baseline year (2017), and calibrated with a capital-output ratio.
    
    Args:
        raw_data: DataFrame with PWT data including rkna, pl_gdpo, and cgdpo columns
        capital_output_ratio: Capital-output ratio to use (default: 3.0)
        
    Returns:
        DataFrame with K_USD_bn column added (capital stock in billions of USD)
    """
    logger.info(f"Calculating capital stock using K/Y ratio = {capital_output_ratio}")
    
    # Validate input
    if not isinstance(raw_data, pd.DataFrame):
        logger.error("Invalid input type: raw_data must be a pandas DataFrame")
        return pd.DataFrame({'year': [], 'K_USD_bn': []})
    
    # Create a copy to avoid modifying the original
    df = raw_data.copy()
    
    # Log available columns for debugging
    logger.debug(f"Available columns for capital stock calculation: {df.columns.tolist()}")
    if 'year' not in df.columns:
        logger.error("Critical: 'year' column missing from input data")
        return pd.DataFrame({'year': [], 'K_USD_bn': []})
    
    # Check for required columns
    required_columns = ['rkna', 'pl_gdpo', 'cgdpo']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns for capital stock calculation: {missing_columns}")
        
        # Look for alternative columns that might contain the required data
        pwt_cols = [col for col in df.columns if col.startswith('PWT') or col.lower().startswith('pwt')]
        if pwt_cols:
            logger.info(f"Found PWT columns that might contain needed data: {pwt_cols}")
            # Try to map PWT columns to required columns
            for col in pwt_cols:
                for req_col in missing_columns:
                    if req_col.lower() in col.lower():
                        logger.info(f"Potential match: '{col}' might contain '{req_col}' data")
        
        # Create empty K_USD_bn column
        logger.info("Adding empty K_USD_bn column due to missing data")
        df['K_USD_bn'] = np.nan
        return df
        
    # Check if we have data for 2017 (baseline year)
    baseline_year = 2017
    if baseline_year not in df['year'].values:
        logger.warning(f"Missing {baseline_year} data for capital stock calculation")
        
        years_available = sorted(df['year'].unique())
        logger.info(f"Available years: {min(years_available)} to {max(years_available)}")
        
        # Try to find an alternative baseline year (closest to 2017)
        alt_years = [y for y in years_available if y >= 2010 and y <= 2020]
        if alt_years:
            # Choose closest year to 2017
            baseline_year = min(alt_years, key=lambda y: abs(y - 2017))
            logger.info(f"Using alternative baseline year: {baseline_year}")
        else:
            logger.error("No suitable baseline year found in range 2010-2020")
            df['K_USD_bn'] = np.nan
            return df
    
    try:
        # Get baseline values
        logger.info(f"Using {baseline_year} as baseline year for capital stock calculation")
        
        # Get GDP (cgdpo) for baseline year
        gdp_baseline_rows = df.loc[df.year == baseline_year, 'cgdpo']
        if gdp_baseline_rows.empty or pd.isna(gdp_baseline_rows.iloc[0]):
            raise ValueError(f"No cgdpo data for {baseline_year}")
        gdp_baseline = gdp_baseline_rows.iloc[0]
        
        # Get capital stock at constant prices (rkna) for baseline year
        rkna_baseline_rows = df.loc[df.year == baseline_year, 'rkna']
        if rkna_baseline_rows.empty or pd.isna(rkna_baseline_rows.iloc[0]):
            raise ValueError(f"No rkna data for {baseline_year}")
        rkna_baseline = rkna_baseline_rows.iloc[0]
        
        # Get price level (pl_gdpo) for baseline year
        pl_gdpo_baseline_rows = df.loc[df.year == baseline_year, 'pl_gdpo']
        if pl_gdpo_baseline_rows.empty or pd.isna(pl_gdpo_baseline_rows.iloc[0]):
            raise ValueError(f"No pl_gdpo data for {baseline_year}")
        pl_gdpo_baseline = pl_gdpo_baseline_rows.iloc[0]
        
        # Calculate capital in baseline constant USD
        k_baseline_usd = (rkna_baseline * gdp_baseline) / capital_output_ratio
        logger.info(f"Baseline year ({baseline_year}) calculated capital: {k_baseline_usd:.2f} billion USD")
        
        # Calculate capital stock for all years
        df['K_USD_bn'] = np.nan  # Initialize with NaN
        
        # Calculate capital stock for each year with data
        for _, row in df.iterrows():
            try:
                year = row['year']
                rkna_value = row['rkna']
                pl_gdpo_value = row['pl_gdpo']
                
                # Skip if we have missing values
                if pd.isna(rkna_value) or pd.isna(pl_gdpo_value):
                    logger.debug(f"Missing required data for year {year}")
                    continue
                
                # Calculate capital in USD
                k_usd = (rkna_value / rkna_baseline) * (pl_gdpo_value / pl_gdpo_baseline) * k_baseline_usd
                
                # Store in DataFrame
                df.loc[df.year == year, 'K_USD_bn'] = k_usd
                
            except Exception as e:
                logger.warning(f"Error calculating capital for year {row.get('year', '?')}: {str(e)}")
                
        # Round to 2 decimal places
        if 'K_USD_bn' in df.columns:
            df['K_USD_bn'] = df['K_USD_bn'].round(2)
            
        # Log summary statistics
        k_data = df.dropna(subset=['K_USD_bn'])
        logger.info(f"Calculated capital stock for {k_data.shape[0]} years")
        
        if not k_data.empty:
            min_k = k_data['K_USD_bn'].min()
            max_k = k_data['K_USD_bn'].max()
            logger.info(f"Capital stock range: {min_k:.2f} to {max_k:.2f} billion USD")
            
        return df
        
    except Exception as e:
        logger.error(f"Error in capital stock calculation: {str(e)}")
        df['K_USD_bn'] = np.nan
        return df


def project_capital_stock(processed_data, end_year, delta=0.05):
    """
    Project capital stock into the future using a perpetual inventory method.
    
    This method projects capital stock using the perpetual inventory equation:
    K_t = (1-delta) * K_{t-1} + I_t
    
    Args:
        processed_data: DataFrame containing at least 'year' and 'K_USD_bn' columns
        end_year: Final year to project capital stock to
        delta: Depreciation rate (default: 0.05)
        
    Returns:
        DataFrame with projected capital stock values
    """
    logger.info(f"Projecting capital stock to year {end_year} with delta={delta}")
    
    # Validate inputs
    if not isinstance(processed_data, pd.DataFrame):
        logger.error("Invalid input type: processed_data must be a pandas DataFrame")
        return pd.DataFrame({'year': [], 'K_USD_bn': []})
    
    # Create a copy to avoid modifying the original
    df = processed_data.copy()
    
    # Verify required columns exist
    if 'year' not in df.columns:
        logger.error("Required 'year' column not found in data")
        return df
    
    if 'K_USD_bn' not in df.columns:
        logger.error("Required 'K_USD_bn' column not found in data")
        return df
    
    # Log available columns
    logger.debug(f"Available columns for capital projection: {df.columns.tolist()}")
    
    # Check if we have data to project from
    k_data_not_na = df.dropna(subset=['K_USD_bn'])
    if k_data_not_na.empty:
        logger.error("No non-NA capital stock data available for projection")
        return df
    
    # Sort by year to ensure correct order
    k_data = df[['year', 'K_USD_bn']].copy().sort_values('year').reset_index(drop=True)
    logger.info(f"Capital stock data available: {k_data_not_na.shape[0]} rows")
    
    # Check if we need to project at all (if end_year is already covered)
    max_year = k_data['year'].max()
    if max_year >= end_year:
        logger.info(f"Data already extends to year {max_year}, no projection needed")
        return df
    
    # Get the last valid capital stock value for projection
    try:
        last_year_with_data = k_data_not_na['year'].max()
        last_k_rows = k_data_not_na.loc[k_data_not_na.year == last_year_with_data, 'K_USD_bn']
        
        if last_k_rows.empty:
            raise ValueError(f"No capital stock data found for year {last_year_with_data}")
        
        last_k = last_k_rows['K_USD_bn'].iloc[0]
        if pd.isna(last_k) or last_k <= 0:
            raise ValueError(f"Invalid capital stock value for year {last_year_with_data}: {last_k}")
        
        logger.info(f"Last capital stock value: {last_k:.2f} billion USD (year {last_year_with_data})")
    except Exception as e:
        logger.error(f"Error retrieving last capital stock value: {str(e)}")
        return k_data
    
    
    def calculate_investment(capital_data, delta=0.05):
        """
        Calculate investment data using changes in capital stock and depreciation.
        
        This function calculates investment using the perpetual inventory method in reverse:
        I_t = K_t - (1-delta) * K_{t-1}
        
        Args:
            capital_data: DataFrame with 'year' and 'K_USD_bn' columns
            delta: Depreciation rate (default: 0.05 or 5% per year)
            
        Returns:
            DataFrame with 'year' and 'I_USD_bn' columns
        """
        logger.info(f"Estimating investment data using delta={delta}")
        
        # Validate input
        if not isinstance(capital_data, pd.DataFrame):
            logger.error("Input is not a pandas DataFrame")
            return pd.DataFrame({'year': [], 'I_USD_bn': []})
        
        if 'year' not in capital_data.columns:
            logger.error("'year' column missing from input data")
            return pd.DataFrame({'year': [], 'I_USD_bn': []})
        
        if 'K_USD_bn' not in capital_data.columns:
            logger.error("'K_USD_bn' column missing from input data")
            return pd.DataFrame({'year': [], 'I_USD_bn': np.nan})
        
        # Create a copy to avoid modifying the original
        df = capital_data.copy()
        
        # Drop rows with missing capital stock values
        df_clean = df.dropna(subset=['K_USD_bn'])
        
        if df_clean.shape[0] < 2:
            logger.error("Not enough non-NA capital stock data points to calculate investment")
            return pd.DataFrame({'year': df['year'], 'I_USD_bn': np.nan})
        
        # Sort by year to ensure proper calculation
        df_clean = df_clean.sort_values('year')
        logger.info(f"Using {df_clean.shape[0]} years of capital stock data from {df_clean['year'].min()} to {df_clean['year'].max()}")
        
        # Create result DataFrame with all original years to maintain consistency
        result = pd.DataFrame({'year': df['year']})
        
        try:
            # Dictionary to store calculated investments
            investments = {}
            
            # Initialize counters for logging
            valid_years = []
            
            # Iterate through years to calculate investment
            for i in range(1, len(df_clean)):
                curr_year = df_clean.iloc[i]['year']
                prev_year = df_clean.iloc[i-1]['year']
                
                # Only calculate if years are consecutive
                if curr_year == prev_year + 1:
                    curr_k = df_clean.iloc[i]['K_USD_bn']
                    prev_k = df_clean.iloc[i-1]['K_USD_bn']
                    
                    # Calculate investment using I_t = K_t - (1-delta) * K_{t-1}
                    inv = curr_k - (1 - delta) * prev_k
                    
                    # Store calculated investment
                    investments[curr_year] = inv
                    valid_years.append(curr_year)
                    
                    # Apply sanity checks
                    if inv < 0:
                        logger.warning(f"Calculated negative investment for year {curr_year}: {inv:.2f}")
                        if inv < -0.1 * curr_k:  # If negative investment is large relative to capital
                            logger.warning(f"Large negative investment ({inv:.2f}) in year {curr_year}, capping to zero")
                            investments[curr_year] = 0
                else:
                    logger.debug(f"Skipping non-consecutive years {prev_year} to {curr_year}")
            
            if investments:
                # Create a DataFrame with the calculated investments
                inv_df = pd.DataFrame(list(investments.items()), columns=['year', 'I_USD_bn'])
                
                # Merge with result DataFrame
                result = pd.merge(result, inv_df, on='year', how='left')
                
                # Log statistics for validation
                non_na = result.dropna(subset=['I_USD_bn'])
                if not non_na.empty:
                    min_i = non_na['I_USD_bn'].min()
                    max_i = non_na['I_USD_bn'].max()
                    mean_i = non_na['I_USD_bn'].mean()
                    logger.info(f"Calculated investment for {len(valid_years)} years")
                    logger.info(f"Investment range: {min_i:.2f} to {max_i:.2f} billion USD, average: {mean_i:.2f} billion USD")
                    
                    # Check for outlier investments
                    if non_na.shape[0] > 5:
                        std_i = non_na['I_USD_bn'].std()
                        
                        # Calculate z-scores
                        z_scores = (non_na['I_USD_bn'] - mean_i) / std_i
                        outliers = non_na[abs(z_scores) > 3]
                        
                        if not outliers.empty:
                            outlier_years = outliers['year'].tolist()
                            logger.warning(f"Outlier investment values detected for years: {outlier_years}")
                            
                    # Calculate investment as a percentage of capital stock
                    non_na['I_K_ratio'] = non_na['I_USD_bn'] / non_na['K_USD_bn']
                    avg_i_k_ratio = non_na['I_K_ratio'].mean()
                    logger.info(f"Average investment-to-capital ratio: {avg_i_k_ratio:.4f} ({avg_i_k_ratio*100:.2f}%)")
                else:
                    logger.warning("No valid investment calculations")
            else:
                logger.warning("Could not calculate investment for any year")
                
            # Round results to 2 decimal places
            if 'I_USD_bn' in result.columns:
                result['I_USD_bn'] = result['I_USD_bn'].round(2)
            
        except Exception as e:
            logger.error(f"Error calculating investment: {str(e)}")
            # Ensure the result has an I_USD_bn column even if calculation failed
            result['I_USD_bn'] = np.nan
        
        return result
    
    
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
        
        # Get investment data for projecting future values
        try:
            if 'I_USD_bn' not in processed_data.columns:
                raise ValueError("Investment (I_USD_bn) column not found in data")
            
            inv_data = processed_data[['year', 'I_USD_bn']].copy().dropna()
            logger.info(f"Investment data available: {inv_data.shape[0]} rows")
            
            if inv_data.empty:
                raise ValueError("No non-NA investment data available")
        except Exception as e:
            logger.warning(f"Error retrieving investment data: {str(e)}")
            logger.info("Will use estimated investment based on capital stock")
            
            # Create synthetic investment data based on capital stock and depreciation
            inv_data = pd.DataFrame({'year': k_data_not_na['year'], 'I_USD_bn': np.nan})
            
            # Sort by year to calculate investment: I_t = K_t - (1-delta) * K_{t-1}
            sorted_k = k_data_not_na.sort_values('year')
            
            for i in range(1, len(sorted_k)):
                prev_year = sorted_k.iloc[i-1]['year']
                curr_year = sorted_k.iloc[i]['year']
                prev_k = sorted_k.iloc[i-1]['K_USD_bn']
                curr_k = sorted_k.iloc[i]['K_USD_bn']
                
                # Only calculate if years are consecutive
                if curr_year == prev_year + 1:
                    implied_inv = curr_k - (1 - delta) * prev_k
                    inv_data.loc[inv_data['year'] == curr_year, 'I_USD_bn'] = implied_inv
            
            # Drop any remaining NA values
            inv_data = inv_data.dropna()
            
            if inv_data.empty:
                logger.warning("Could not create synthetic investment data")
                # Last resort: assume investment is a fraction of capital stock
                last_inv_year = last_year_with_data
                last_inv_value = last_k * 0.1  # Assume investment is 10% of capital stock
                logger.info(f"Using estimated investment of {last_inv_value:.2f} billion USD (10% of capital stock)")
            else:
                logger.info(f"Created synthetic investment data for {inv_data.shape[0]} years")
        
        # Calculate average investment growth rate from recent data
        try:
            # Use the most recent data for growth rate calculation (up to 5 years)
            if not inv_data.empty:
                # Sort investments by year
                inv_data = inv_data.sort_values('year')
                
                # Get the most recent years of data (up to 5)
                recent_years = 5
                if len(inv_data) < recent_years:
                    recent_years = len(inv_data)
                
                last_inv_data = inv_data.tail(recent_years)
                logger.info(f"Using investment data from {recent_years} most recent years: {last_inv_data['year'].tolist()}")
                
                if len(last_inv_data) >= 2:
                    # Calculate year-over-year growth rates
                    growth_rates = []
                    prev_value = None
                    
                    for _, row in last_inv_data.iterrows():
                        if prev_value is not None and prev_value > 0:
                            growth_rate = (row['I_USD_bn'] / prev_value) - 1
                            growth_rates.append(growth_rate)
                        prev_value = row['I_USD_bn']
                    
                    if growth_rates:
                        # Remove extreme values (more than 3 std dev from mean)
                        if len(growth_rates) > 3:
                            mean_growth = np.mean(growth_rates)
                            std_growth = np.std(growth_rates)
                            growth_rates = [g for g in growth_rates if abs(g - mean_growth) <= 3 * std_growth]
                        
                        avg_inv_growth = np.mean(growth_rates)
                        logger.info(f"Average investment growth rate: {avg_inv_growth:.4f} ({avg_inv_growth*100:.2f}%)")
                        
                        # Cap growth rate at reasonable bounds
                        if avg_inv_growth > 0.15:
                            logger.warning(f"Capping high investment growth rate {avg_inv_growth:.4f} to 0.15 (15%)")
                            avg_inv_growth = 0.15
                        elif avg_inv_growth < -0.10:
                            logger.warning(f"Capping low investment growth rate {avg_inv_growth:.4f} to -0.10 (-10%)")
                            avg_inv_growth = -0.10
                    else:
                        logger.warning("Could not calculate investment growth rates")
                        avg_inv_growth = 0.05  # Default to 5%
                else:
                    logger.warning("Insufficient data points for growth rate calculation")
                    avg_inv_growth = 0.05  # Default to 5%
            else:
                logger.warning("No investment data available for growth rate calculation")
                avg_inv_growth = 0.05  # Default to 5%
        except Exception as e:
            logger.error(f"Error calculating investment growth rate: {str(e)}")
            avg_inv_growth = 0.05  # Default to 5%
            
        logger.info(f"Using investment growth rate of {avg_inv_growth:.4f} for projections")
                
        # Define years to project
        years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))
        if years_to_project:
            logger.info(f"Years to project: {min(years_to_project)} to {max(years_to_project)}")
        else:
            logger.info("No years to project - returning original data")
            return k_data
            
        # Get last known investment value for projections
        try:
            if not inv_data.empty:
                last_inv_year = inv_data['year'].max()
                last_inv_value = inv_data.loc[inv_data['year'] == last_inv_year, 'I_USD_bn'].iloc[0]
                logger.info(f"Last investment value: {last_inv_value:.2f} billion USD (year {last_inv_year})")
            else:
                # Fallback if no investment data is available
                last_inv_year = last_year_with_data
                last_inv_value = last_k * 0.1  # Assume investment is 10% of capital stock
                logger.info(f"Using estimated investment of {last_inv_value:.2f} billion USD (10% of capital stock)")
        except Exception as e:
            logger.error(f"Error retrieving last investment value: {str(e)}")
            last_inv_year = last_year_with_data
            last_inv_value = last_k * 0.1  # Assume investment is 10% of capital stock
            logger.info(f"Using estimated investment of {last_inv_value:.2f} billion USD (10% of capital stock)")
        
        # Initialize the projection dictionary with the last known value
        proj = {last_year_with_data: last_k}
            
        # Project investment for future years
        try:
            projected_inv = {}
            for y in years_to_project:
                years_from_last_inv = y - last_inv_year
                projected_value = last_inv_value * (1 + avg_inv_growth) ** years_from_last_inv
                
                # Sanity check - investment shouldn't exceed 40% of previous year's capital
                if y > years_to_project[0]:
                    max_reasonable_inv = proj[y-1] * 0.40
                else:
                    max_reasonable_inv = last_k * 0.40
                    
                if projected_value > max_reasonable_inv:
                    logger.warning(f"Capping projected investment for year {y}: {projected_value:.2f} to {max_reasonable_inv:.2f}")
                    projected_value = max_reasonable_inv
                    
                projected_inv[y] = projected_value
                
            logger.info(f"Projected investment from {min(projected_inv.keys())} to {max(projected_inv.keys())}")
        except Exception as e:
            logger.error(f"Error projecting investment values: {str(e)}")
            # Simple fallback - assume constant investment at last value
            projected_inv = {y: last_inv_value for y in years_to_project}
            logger.info("Using constant investment value for projections due to error")
        
        # Project capital stock using perpetual inventory method: K_t = (1-delta) * K_{t-1} + I_t
        try:
            # Project forward using the perpetual inventory method
            for y in years_to_project:
                inv_value = projected_inv.get(y, 0)
                previous_k = proj[y-1]
                
                # Apply the perpetual inventory method
                projected_k = (1-delta) * previous_k + inv_value
                
                # Sanity check - capital shouldn't decrease too much
                if projected_k < previous_k * 0.85 and inv_value > 0:
                    logger.warning(f"Unusual drop in projected capital for year {y}: {projected_k:.2f} < {previous_k:.2f}*0.85")
                
                # Ensure capital remains positive
                if projected_k <= 0:
                    logger.warning(f"Negative capital projection for year {y}: {projected_k:.2f}")
                    projected_k = previous_k * 0.9  # Fallback: 10% decline from previous year
                    
                # Store the projected value
                proj[y] = round(projected_k, 2)
                
            logger.info(f"Successfully projected capital stock for {len(years_to_project)} years")
            
            # Quality check - examine the growth pattern
            if years_to_project:
                k_growth_rates = [(proj[y]/proj[y-1] - 1) for y in years_to_project]
                avg_k_growth = sum(k_growth_rates) / len(k_growth_rates) if k_growth_rates else 0
                logger.info(f"Average projected capital growth rate: {avg_k_growth:.4f} ({avg_k_growth*100:.2f}%)")
                
                if any(rate < -0.1 for rate in k_growth_rates):
                    years_with_decline = [years_to_project[i] for i, rate in enumerate(k_growth_rates) if rate < -0.1]
                    logger.warning(f"Large capital stock declines in years: {years_with_decline}")
                
            # Create DataFrame with projections
            proj_df = pd.DataFrame(list(proj.items()), columns=['year', 'K_USD_bn'])
            logger.info(f"Created projection dataframe with {proj_df.shape[0]} rows")
            
            # Merge with original data
            try:
                # Create a clean result DataFrame
                result = k_data.copy()
                
                # For each projection year, update the capital stock
                for _, row in proj_df.iterrows():
                    result.loc[result['year'] == row['year'], 'K_USD_bn'] = row['K_USD_bn']
                
                # Check for missing projection years (might happen if they weren't in the original data)
                projected_years = set(proj_df['year'])
                result_years = set(result['year'])
                missing_years = projected_years - result_years
                
                if missing_years:
                    logger.warning(f"Some projected years were not in the original data: {missing_years}")
                    # Add the missing years
                    missing_df = proj_df[proj_df['year'].isin(missing_years)]
                    result = pd.concat([result, missing_df], ignore_index=True)
                    
                # Sort by year for consistency
                result = result.sort_values('year').reset_index(drop=True)
                
                logger.info(f"Final result has capital stock data for {result.dropna(subset=['K_USD_bn']).shape[0]} years")
                return result
                
            except Exception as e:
                logger.error(f"Error merging projections with original data: {str(e)}")
                # Fallback to simple method
                logger.info("Falling back to simple merge method")
                return pd.merge(k_data, proj_df, on='year', how='outer').sort_values('year')
            
        except Exception as e:
            logger.error(f"Error projecting capital stock: {str(e)}")
            return k_data


def smooth_capital_data(df, window_size=3, outlier_threshold=3.0, interpolate_gaps=True):
    """
    Smooth capital and investment time series data to handle outliers and gaps.
    
    This function performs several operations to improve capital stock and investment time series:
    1. Detects and handles outliers using rolling statistics
    2. Fills gaps in the time series using interpolation
    3. Applies smoothing to reduce noise in the data
    
    Args:
        df: DataFrame with 'year', and at least one of 'K_USD_bn' or 'I_USD_bn' columns
        window_size: Size of the rolling window for outlier detection (default: 3)
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)
        interpolate_gaps: Whether to interpolate missing values (default: True)
        
    Returns:
        DataFrame with smoothed capital and investment data
    """
    logger.info(f"Smoothing capital data with window_size={window_size}, outlier_threshold={outlier_threshold}")
    
    # Validate input
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame")
        return pd.DataFrame()
    
    if 'year' not in df.columns:
        logger.error("'year' column missing from input data")
        return df
    
    # Check if we have any capital or investment data to smooth
    has_k = 'K_USD_bn' in df.columns
    has_i = 'I_USD_bn' in df.columns
    
    if not has_k and not has_i:
        logger.error("Neither capital (K_USD_bn) nor investment (I_USD_bn) columns found")
        return df
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by year to ensure proper processing
    result = result.sort_values('year').reset_index(drop=True)
    
    # Create backup columns to track changes
    if has_k:
        result['K_USD_bn_original'] = result['K_USD_bn'].copy()
    if has_i:
        result['I_USD_bn_original'] = result['I_USD_bn'].copy()
    
    # Process capital stock data if available
    if has_k:
        try:
            logger.info("Processing capital stock data")
            
            # Drop rows with missing capital stock values for analysis
            k_data = result.dropna(subset=['K_USD_bn']).copy()
            
            if k_data.shape[0] > window_size:
                # Step 1: Detect outliers using rolling statistics
                # Calculate rolling median and standard deviation
                k_data['rolling_median'] = k_data['K_USD_bn'].rolling(window=window_size, center=True, min_periods=2).median()
                
                # For first and last points, use the nearest valid rolling median
                k_data['rolling_median'].iloc[0] = k_data['rolling_median'].iloc[1] if pd.notna(k_data['rolling_median'].iloc[1]) else k_data['K_USD_bn'].iloc[0]
                k_data['rolling_median'].iloc[-1] = k_data['rolling_median'].iloc[-2] if pd.notna(k_data['rolling_median'].iloc[-2]) else k_data['K_USD_bn'].iloc[-1]
                
                # Calculate rolling MAD (Median Absolute Deviation) for robust standard deviation
                k_data['rolling_mad'] = abs(k_data['K_USD_bn'] - k_data['rolling_median']).rolling(window=window_size, center=True, min_periods=2).median()
                k_data['rolling_mad'] = k_data['rolling_mad'].fillna(k_data['rolling_mad'].median())
                
                # Ensure MAD is never zero to avoid division issues
                k_data['rolling_mad'] = k_data['rolling_mad'].replace(0, k_data['K_USD_bn'].std() * 0.1)
                
                # Calculate modified z-scores using MAD
                k_data['zscore'] = (k_data['K_USD_bn'] - k_data['rolling_median']) / k_data['rolling_mad']
                
                # Identify outliers
                outliers = k_data[abs(k_data['zscore']) > outlier_threshold]
                
                if not outliers.empty:
                    outlier_years = outliers['year'].tolist()
                    logger.warning(f"Detected {len(outlier_years)} outliers in capital stock data for years: {outlier_years}")
                    
                    # Replace outliers with estimates based on surrounding values
                    for idx in outliers.index:
                        year = k_data.loc[idx, 'year']
                        old_value = k_data.loc[idx, 'K_USD_bn']
                        new_value = k_data.loc[idx, 'rolling_median']
                        
                        # Update the main dataframe
                        result.loc[result['year'] == year, 'K_USD_bn'] = new_value
                        logger.info(f"Replaced outlier for year {year}: {old_value:.2f} -> {new_value:.2f}")
                
                # Step 2: Apply smoothing to reduce noise
                # Use a simple rolling average for smoothing
                smooth_window = min(3, k_data.shape[0])
                
                if smooth_window >= 2:
                    # Create a temporary series with interpolated values for better smoothing
                    k_series = result['K_USD_bn'].copy()
                    
                    # Apply rolling mean smoothing
                    smoothed = k_series.rolling(window=smooth_window, center=True, min_periods=1).mean()
                    
                    # Calculate how much change the smoothing made
                    change_pct = abs((smoothed - result['K_USD_bn']) / result['K_USD_bn'] * 100)
                    avg_change = change_pct.mean()
                    
                    # Only apply smoothing if it doesn't change values too much
                    if avg_change < 5.0:  # Less than 5% average change
                        result['K_USD_bn'] = smoothed
                        logger.info(f"Applied rolling window smoothing to capital stock data (avg change: {avg_change:.2f}%)")
                    else:
                        logger.warning(f"Skipped smoothing as it would change values too much (avg change: {avg_change:.2f}%)")
                else:
                    logger.info("Not enough data points for smoothing capital stock")
            else:
                logger.info(f"Not enough capital stock data points for outlier detection ({k_data.shape[0]} < {window_size+1})")
            
            # Step 3: Interpolate gaps if requested
            if interpolate_gaps and result['K_USD_bn'].isna().any():
                # Count missing values before interpolation
                na_count_before = result['K_USD_bn'].isna().sum()
                
                # Only interpolate if we have enough data points
                if result.dropna(subset=['K_USD_bn']).shape[0] >= 2:
                    # Use linear interpolation for missing values
                    result['K_USD_bn'] = result['K_USD_bn'].interpolate(method='linear')
                    
                    # Count missing values after interpolation
                    na_count_after = result['K_USD_bn'].isna().sum()
                    filled_count = na_count_before - na_count_after
                    
                    if filled_count > 0:
                        logger.info(f"Filled {filled_count} missing capital stock values using linear interpolation")
                else:
                    logger.warning("Not enough non-missing capital stock values for interpolation")
            
            # Summarize changes
            if has_k and 'K_USD_bn_original' in result.columns:
                changed = result[result['K_USD_bn'] != result['K_USD_bn_original']].dropna(subset=['K_USD_bn', 'K_USD_bn_original'])
                if not changed.empty:
                    avg_change_pct = abs((changed['K_USD_bn'] - changed['K_USD_bn_original']) / changed['K_USD_bn_original'] * 100).mean()
                    logger.info(f"Modified {changed.shape[0]} capital stock values (avg change: {avg_change_pct:.2f}%)")
                else:
                    logger.info("No capital stock values were modified")
            
        except Exception as e:
            logger.error(f"Error smoothing capital stock data: {str(e)}")
            # Revert to original values if there was an error
            if 'K_USD_bn_original' in result.columns:
                result['K_USD_bn'] = result['K_USD_bn_original']
    
    # Process investment data if available
    if has_i:
        try:
            logger.info("Processing investment data")
            
            # Drop rows with missing investment values for analysis
            i_data = result.dropna(subset=['I_USD_bn']).copy()
            
            if i_data.shape[0] > window_size:
                # Step 1: Detect outliers using rolling statistics
                # Calculate rolling median and standard deviation
                i_data['rolling_median'] = i_data['I_USD_bn'].rolling(window=window_size, center=True, min_periods=2).median()
                
                # For first and last points, use the nearest valid rolling median
                i_data['rolling_median'].iloc[0] = i_data['rolling_median'].iloc[1] if pd.notna(i_data['rolling_median'].iloc[1]) else i_data['I_USD_bn'].iloc[0]
                i_data['rolling_median'].iloc[-1] = i_data['rolling_median'].iloc[-2] if pd.notna(i_data['rolling_median'].iloc[-2]) else i_data['I_USD_bn'].iloc[-1]
                
                # Calculate rolling MAD (Median Absolute Deviation) for robust standard deviation
                i_data['rolling_mad'] = abs(i_data['I_USD_bn'] - i_data['rolling_median']).rolling(window=window_size, center=True, min_periods=2).median()
                i_data['rolling_mad'] = i_data['rolling_mad'].fillna(i_data['rolling_mad'].median())
                
                # Ensure MAD is never zero to avoid division issues
                i_data['rolling_mad'] = i_data['rolling_mad'].replace(0, i_data['I_USD_bn'].std() * 0.1)
                
                # Calculate modified z-scores using MAD
                i_data['zscore'] = (i_data['I_USD_bn'] - i_data['rolling_median']) / i_data['rolling_mad']
                
                # Identify outliers (more liberal threshold for investment which is naturally more volatile)
                outlier_threshold_i = outlier_threshold * 1.5  # 50% higher threshold for investment
                outliers = i_data[abs(i_data['zscore']) > outlier_threshold_i]
                
                if not outliers.empty:
                    outlier_years = outliers['year'].tolist()
                    logger.warning(f"Detected {len(outlier_years)} outliers in investment data for years: {outlier_years}")
                    
                    # Replace outliers with estimates based on surrounding values
                    for idx in outliers.index:
                        year = i_data.loc[idx, 'year']
                        old_value = i_data.loc[idx, 'I_USD_bn']
                        new_value = i_data.loc[idx, 'rolling_median']
                        
                        # Update the main dataframe
                        result.loc[result['year'] == year, 'I_USD_bn'] = new_value
                        logger.info(f"Replaced outlier for year {year}: {old_value:.2f} -> {new_value:.2f}")
                
                # Step 2: Apply smoothing to reduce noise (less smoothing for investment which is naturally volatile)
                # Use a shorter window for investment data
                smooth_window = min(2, i_data.shape[0])
                
                if smooth_window >= 2:
                    # Create a temporary series with interpolated values for better smoothing
                    i_series = result['I_USD_bn'].copy()
                    
                    # Apply rolling mean smoothing
                    smoothed = i_series.rolling(window=smooth_window, center=True, min_periods=1).mean()
                    
                    # Calculate how much change the smoothing made
                    change_pct = abs((smoothed - result['I_USD_bn']) / result['I_USD_bn'] * 100)
                    avg_change = change_pct.mean()
                    
                    # Only apply smoothing if it doesn't change values too much
                    if avg_change < 7.5:  # Allow higher change for investment (7.5%)
                        result['I_USD_bn'] = smoothed
                        logger.info(f"Applied rolling window smoothing to investment data (avg change: {avg_change:.2f}%)")
                    else:
                        logger.warning(f"Skipped smoothing as it would change values too much (avg change: {avg_change:.2f}%)")
                else:
                    logger.info("Not enough data points for smoothing investment")
            else:
                logger.info(f"Not enough investment data points for outlier detection ({i_data.shape[0]} < {window_size+1})")
            
            # Step 3: Interpolate gaps if requested
            if interpolate_gaps and result['I_USD_bn'].isna().any():
                # Count missing values before interpolation
                na_count_before = result['I_USD_bn'].isna().sum()
                
                # Only interpolate if we have enough data points
                if result.dropna(subset=['I_USD_bn']).shape[0] >= 2:
                    # Use linear interpolation for missing values
                    result['I_USD_bn'] = result['I_USD_bn'].interpolate(method='linear')
                    
                    # Count missing values after interpolation
                    na_count_after = result['I_USD_bn'].isna().sum()
                    filled_count = na_count_before - na_count_after
                    
                    if filled_count > 0:
                        logger.info(f"Filled {filled_count} missing investment values using linear interpolation")
                else:
                    logger.warning("Not enough non-missing investment values for interpolation")
            
            # Summarize changes
            if has_i and 'I_USD_bn_original' in result.columns:
                changed = result[result['I_USD_bn'] != result['I_USD_bn_original']].dropna(subset=['I_USD_bn', 'I_USD_bn_original'])
                if not changed.empty:
                    avg_change_pct = abs((changed['I_USD_bn'] - changed['I_USD_bn_original']) / changed['I_USD_bn_original'] * 100).mean()
                    logger.info(f"Modified {changed.shape[0]} investment values (avg change: {avg_change_pct:.2f}%)")
                else:
                    logger.info("No investment values were modified")
            
        except Exception as e:
            logger.error(f"Error smoothing investment data: {str(e)}")
            # Revert to original values if there was an error
            if 'I_USD_bn_original' in result.columns:
                result['I_USD_bn'] = result['I_USD_bn_original']
    
    # Step 4: Ensure consistency between capital and investment (if both columns exist)
    if has_k and has_i and result.dropna(subset=['K_USD_bn', 'I_USD_bn']).shape[0] >= 2:
        try:
            logger.info("Checking consistency between capital and investment")
            
            # Sort by year
            df_clean = result.dropna(subset=['K_USD_bn', 'I_USD_bn']).sort_values('year')
            
            if df_clean.shape[0] >= 2:
                # Create a new DataFrame for the check
                check_df = df_clean.copy()
                
                # Calculate implied investment using capital stock differences
                check_df['K_prev'] = check_df['K_USD_bn'].shift(1)
                check_df['implied_I'] = check_df['K_USD_bn'] - (1 - 0.05) * check_df['K_prev']
                
                # Calculate discrepancy
                check_df['I_diff'] = check_df['I_USD_bn'] - check_df['implied_I']
                check_df['I_diff_pct'] = abs(check_df['I_diff'] / check_df['implied_I'] * 100)
                
                # Identify large inconsistencies
                large_diff = check_df.iloc[1:].loc[check_df['I_diff_pct'] > 25]
                
                if not large_diff.empty:
                    diff_years = large_diff['year'].tolist()
                    logger.warning(f"Large inconsistencies between capital and investment for years: {diff_years}")
                    
                    # We prefer to keep capital stock as is and adjust investment
                    # This choice reflects that capital stock is usually more reliable
                    for _, row in large_diff.iterrows():
                        year = row['year']
                        implied_i = row['implied_I']
                        
                        # Only adjust if the implied investment is positive
                        if implied_i > 0:
                            old_i = row['I_USD_bn']
                            
                            # Use a weighted average to preserve some of the original data
                            new_i = 0.7 * implied_i + 0.3 * old_i
                            
                            # Update the main dataframe
                            result.loc[result['year'] == year, 'I_USD_bn'] = round(new_i, 2)
                            logger.info(f"Adjusted investment for year {year} to improve consistency: {old_i:.2f} -> {new_i:.2f}")
        except Exception as e:
            logger.error(f"Error ensuring consistency between capital and investment: {str(e)}")
    
    # Remove backup columns
    if 'K_USD_bn_original' in result.columns:
        result = result.drop('K_USD_bn_original', axis=1)
    if 'I_USD_bn_original' in result.columns:
        result = result.drop('I_USD_bn_original', axis=1)
    
    # Round final values for consistency
    if has_k:
        result['K_USD_bn'] = result['K_USD_bn'].round(2)
    if has_i:
        result['I_USD_bn'] = result['I_USD_bn'].round(2)
    
    # Final summary
    modified_count = (result != df).any(axis=1).sum()
    if modified_count > 0:
        logger.info(f"Smoothing complete - modified data for {modified_count} out of {len(result)} rows")
    else:
        logger.info("Smoothing complete - no changes were necessary")
    
    return result
