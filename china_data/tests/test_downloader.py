import builtins
import io
import os
import types
import pandas as pd
import pytest
from unittest import mock

# Use absolute imports consistently
from china_data.utils.data_sources import download_wdi_data, get_pwt_data

# Create module-like objects for backward compatibility with the test code
class wdi_downloader:
    download_wdi_data = download_wdi_data
    wb = __import__('pandas_datareader', fromlist=['wb']).wb
    time = __import__('time')

class pwt_downloader:
    get_pwt_data = get_pwt_data
    requests = __import__('requests')
    pd = __import__('pandas', fromlist=['pd'])
    os = __import__('os')


def make_df(rows):
    return pd.DataFrame(rows)


def test_download_wdi_data_success(monkeypatch):
    sample = pd.DataFrame({"country": ["CN"], "year": [2020], "NY_GDP_MKTP_CD": [1.0]})
    def fake_download(country, indicator, start, end):
        return sample
    monkeypatch.setattr(wdi_downloader.wb, "download", fake_download)
    monkeypatch.setattr(wdi_downloader.time, "sleep", lambda s: None)
    df = wdi_downloader.download_wdi_data("NY.GDP.MKTP.CD")
    assert not df.empty
    assert list(df.columns) == ["index", "country", "year", "NY_GDP_MKTP_CD"]


def test_download_wdi_data_failure(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("fail")
    monkeypatch.setattr(wdi_downloader.wb, "download", fail)
    monkeypatch.setattr(wdi_downloader.time, "sleep", lambda s: None)
    df = wdi_downloader.download_wdi_data("BAD")
    assert df.empty


class DummyResponse:
    def __init__(self):
        self.content = b"dummy"
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size=8192):
        yield b"data"
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass


def test_get_pwt_data_success(monkeypatch, tmp_path):
    monkeypatch.setattr(pwt_downloader.requests, "get", lambda url, stream=True: DummyResponse())
    expected = pd.DataFrame({
        "countrycode": ["CHN"],
        "year": [2017],
        "rgdpo": [1],
        "rkna": [2],
        "pl_gdpo": [3],
        "cgdpo": [4],
        "hc": [5],
    })
    monkeypatch.setattr(pwt_downloader.pd, "read_excel", lambda path, sheet_name="Data": expected)
    df = pwt_downloader.get_pwt_data()
    assert list(df.columns) == ["year", "rgdpo", "rkna", "pl_gdpo", "cgdpo", "hc"]
    assert df.iloc[0]["year"] == 2017


def test_get_pwt_data_error(monkeypatch):
    def boom(*a, **k):
        raise pwt_downloader.requests.exceptions.HTTPError("bad")
    monkeypatch.setattr(pwt_downloader.requests, "get", boom)
    with pytest.raises(pwt_downloader.requests.exceptions.HTTPError):
        pwt_downloader.get_pwt_data()