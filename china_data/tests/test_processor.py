import os
import pandas as pd
import numpy as np
import pytest
from unittest import mock

from china_data.utils.processor_load import load_raw_data
from china_data.utils.processor_units import convert_units
from china_data.utils.processor_capital import calculate_capital_stock, project_capital_stock
from china_data.utils.processor_hc import project_human_capital
from china_data.utils.processor_tfp import calculate_tfp
from china_data.utils.processor_extrapolation import extrapolate_series_to_end_year
from china_data.utils.processor_output import create_markdown_table
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

RAW_MD = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'china_data_raw.md')


def test_load_raw_data_success():
    # Ensure the output directory and a dummy raw file exist for this test
    # as the main script might not have run if --test is used.
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    dummy_raw_md_path = os.path.join(output_dir, 'china_data_raw.md')
    if not os.path.exists(dummy_raw_md_path):
        with open(dummy_raw_md_path, 'w') as f:
            f.write("| Year | GDP (USD) |\n")
            f.write("|------|-----------|\n")
            f.write("| 2020 | 100       |\n")

    df = load_raw_data(data_dir='output', input_file='china_data_raw.md')
    assert not df.empty
    assert 'GDP_USD' in df.columns


def test_load_raw_data_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(data_dir=str(tmp_path), input_file='missing.md')


def test_convert_units():
    raw = pd.DataFrame({
        'year': [2017],
        'GDP_USD': [1e12],
        'rgdpo': [1000],
        'POP': [1000000],
        'LF': [500000],
    })
    out = convert_units(raw)
    assert out['GDP_USD_bn'].iloc[0] == 1000
    assert out['rgdpo_bn'].iloc[0] == 1
    assert out['POP_mn'].iloc[0] == 1
    assert out['LF_mn'].iloc[0] == 0.5


def test_calculate_capital_stock_missing():
    raw = pd.DataFrame({'year':[2017]})
    result = calculate_capital_stock(raw)
    assert 'K_USD_bn' in result.columns
    assert np.isnan(result['K_USD_bn']).all()


def test_calculate_capital_stock_basic():
    raw = pd.DataFrame({
        'year': [2017,2018],
        'rkna': [2.0, 2.1],
        'pl_gdpo': [3.0, 3.1],
        'cgdpo': [4.0, 4.4],
    })
    df = calculate_capital_stock(raw, capital_output_ratio=3)
    assert not df['K_USD_bn'].isna().all()


def test_project_capital_stock(monkeypatch):
    data = pd.DataFrame({
        'year':[2017,2018],
        'K_USD_bn':[10.0,11.0],
        'I_USD_bn':[5.0,5.25]
    })
    projected = project_capital_stock(data, end_year=2019)
    assert projected['year'].tolist()[-1] == 2019


def test_project_human_capital_fallback(monkeypatch):
    data = pd.DataFrame({'year':[2017,2018],'hc':[1.0,np.nan]})
    # This test doesn't need to mock ExponentialSmoothing since we're using LinearRegression now
    df = project_human_capital(data, end_year=2019)
    assert 2019 in df['year'].values


def test_calculate_tfp_with_missing_hc():
    data = pd.DataFrame({'year':[2017,2018],'GDP_USD_bn':[2.0,2.1],'K_USD_bn':[1.0,1.1],'LF_mn':[1,1.1],'hc':[1.0,np.nan]})
    out = calculate_tfp(data)
    assert 'TFP' in out.columns


def test_extrapolate_series_to_end_year(monkeypatch):
    df = pd.DataFrame({
        'year':[2023],
        'GDP_USD_bn':[1.0],
        'C_USD_bn':[1.0],
        'G_USD_bn':[1.0],
        'I_USD_bn':[1.0],
        'X_USD_bn':[1.0],
        'M_USD_bn':[1.0],
        'POP_mn':[1.0],
        'LF_mn':[1.0],
    })
    class Dummy:
        def fit(self):
            return self
        def forecast(self, steps):
            return [1.0]*steps
    # Import the module where ARIMA is used
    import china_data.utils.processor_extrapolation as extrapolation_module
    monkeypatch.setattr(extrapolation_module, 'ARIMA', lambda *a, **k: Dummy())
    monkeypatch.setattr(extrapolation_module, 'LinearRegression', lambda *a, **k: Dummy())
    out, info = extrapolate_series_to_end_year(df, end_year=2024, raw_data=df)
    assert 2024 in out['year'].values
    assert isinstance(info, dict)


def test_create_markdown_table(tmp_path):
    data = pd.DataFrame({
        'Year':[2024],
        'GDP_USD_bn':[1.0],
        'C_USD_bn':[1.0],
        'G_USD_bn':[1.0],
        'I_USD_bn':[1.0],
        'X_USD_bn':[1.0],
        'M_USD_bn':[1.0],
        'NX_USD_bn':[0.0],
        'POP_mn':[1.0],
        'LF_mn':[1.0],
        'K_USD_bn':[2.0],
        'TFP':[1.0],
        'FDI_pct_GDP':[0.1],
        'TAX_pct_GDP':[0.2],
        'hc':[1.0]
    })
    out = tmp_path/'out.md'
    create_markdown_table(data, str(out), {'GDP_USD_bn': {'method':'test','years':[2024]}}, end_year=2024)
    assert out.exists()