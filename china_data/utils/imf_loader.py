import logging
import pandas as pd

from china_data.utils import find_file
from china_data.utils.path_constants import get_search_locations_relative_to_root

logger = logging.getLogger(__name__)


def load_imf_tax_data():
    """
    Load IMF Fiscal Monitor tax revenue data for China.
    
    Returns:
        pandas.DataFrame: DataFrame containing tax revenue data with columns 'year' and 'TAX_pct_GDP'.
                         Returns an empty DataFrame if the file is not found.
    """
    imf_filename = "dataset_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"
    # find_file expects locations relative to project root
    possible_locations_relative = get_search_locations_relative_to_root()["input_files"]
    imf_file = find_file(imf_filename, possible_locations_relative)

    if imf_file:
        logger.info("Found IMF Fiscal Monitor file at: %s", imf_file)
        df = pd.read_csv(imf_file)
        df = df[(df['COUNTRY'] == 'CHN') & (df['FREQUENCY'] == 'A') & (df['INDICATOR'] == 'G1_S13_POGDP_PT')]
        tax_data = df[['TIME_PERIOD', 'OBS_VALUE']].rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'TAX_pct_GDP'})
        tax_data['year'] = tax_data['year'].astype(int)
        tax_data['TAX_pct_GDP'] = pd.to_numeric(tax_data['TAX_pct_GDP'], errors='coerce')
        return tax_data
    else:
        logger.error("IMF Fiscal Monitor file not found in any of the expected locations")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['year', 'TAX_pct_GDP'])
