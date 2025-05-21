# This file is kept for backward compatibility
# It re-exports functions from the new module files

import logging
from china_data.utils.wdi_downloader import download_wdi_data
from china_data.utils.pwt_downloader import get_pwt_data

logger = logging.getLogger(__name__)

# Re-export the functions
__all__ = ['download_wdi_data', 'get_pwt_data']
