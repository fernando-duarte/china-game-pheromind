import logging
import os
import tempfile
import requests
import pandas as pd

logger = logging.getLogger(__name__)


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
