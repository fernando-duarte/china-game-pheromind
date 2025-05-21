import logging
import os
import tempfile
from datetime import datetime
import time
import requests
import pandas as pd
import pandas_datareader.wb as wb

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


def get_pwt_data():
    logger.info("Downloading Penn World Table data...")
    excel_url = "https://dataverse.nl/api/access/datafile/354095"
    tmp_path = None
    try:
        with requests.get(excel_url, stream=True) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
                logger.debug("Downloaded PWT data to temporary file: %s", tmp_path)
        pwt = pd.read_excel(tmp_path, sheet_name="Data")
    except requests.exceptions.RequestException as e:
        logger.error("Error occurred while downloading PWT data: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while processing PWT data: %s", e)
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("Deleted temporary file: %s", tmp_path)

    chn = pwt[pwt.countrycode == "CHN"].copy()
    chn_data = chn[["year", "rgdpo", "rkna", "pl_gdpo", "cgdpo", "hc"]].copy()
    chn_data["year"] = chn_data["year"].astype(int)
    return chn_data
