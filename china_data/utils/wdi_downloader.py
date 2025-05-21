import logging
import time
from datetime import datetime
import pandas as pd
import pandas_datareader.wb as wb

from china_data.utils.download_date_utils import record_download_date

logger = logging.getLogger(__name__)


def download_wdi_data(indicator_code, country_code="CN", start_year=1960, end_year=None):
    if end_year is None:
        end_year = datetime.now().year

    logger.info(f"Downloading {indicator_code} data...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = wb.download(country=country_code,
                               indicator=indicator_code,
                               start=start_year,
                               end=end_year)
            data = data.reset_index()
            data = data.rename(columns={indicator_code: indicator_code.replace('.', '_')})
            logger.debug(
                "Successfully downloaded %s data with %d rows", indicator_code, len(data)
            )
            # Record the download date
            record_download_date('WDI')
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("Attempt %d failed. Retrying in 5 seconds... Error: %s", attempt + 1, e)
                time.sleep(5)
            else:
                logger.error(
                    "Failed to download %s after %d attempts. Error: %s", indicator_code, max_retries, e
                )
                return pd.DataFrame(columns=["country", "year", indicator_code.replace('.', '_')])
