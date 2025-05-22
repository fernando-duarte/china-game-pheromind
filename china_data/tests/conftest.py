import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pytest
from china_data.utils.processor_load import load_raw_data

def pytest_configure(config):
    os.environ.setdefault('PYTHONPATH', os.path.dirname(os.path.dirname(__file__)))

@pytest.fixture(scope="session")
def raw_df():
    return load_raw_data('china_data_raw.md')
