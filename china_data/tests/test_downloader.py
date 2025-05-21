import builtins
import io
import os
import types
import pandas as pd
import pytest
from unittest import mock

import china_data.china_data_downloader as downloader


def make_df(rows):
    return pd.DataFrame(rows)


def test_download_wdi_data_success(monkeypatch):
    sample = pd.DataFrame({"country": ["CN"], "year": [2020], "NY_GDP_MKTP_CD": [1.0]})
    def fake_download(country, indicator, start, end):
        return sample
    monkeypatch.setattr(downloader.wb, "download", fake_download)
    monkeypatch.setattr(downloader.time, "sleep", lambda s: None)
    df = downloader.download_wdi_data("NY.GDP.MKTP.CD")
    assert not df.empty
    assert list(df.columns) == ["index", "country", "year", "NY_GDP_MKTP_CD"]


def test_download_wdi_data_failure(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("fail")
    monkeypatch.setattr(downloader.wb, "download", fail)
    monkeypatch.setattr(downloader.time, "sleep", lambda s: None)
    df = downloader.download_wdi_data("BAD")
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
    monkeypatch.setattr(downloader.requests, "get", lambda url, stream=True: DummyResponse())
    expected = pd.DataFrame({
        "countrycode": ["CHN"],
        "year": [2017],
        "rgdpo": [1],
        "rkna": [2],
        "pl_gdpo": [3],
        "cgdpo": [4],
        "hc": [5],
    })
    monkeypatch.setattr(downloader.pd, "read_excel", lambda path, sheet_name="Data": expected)
    df = downloader.get_pwt_data()
    assert list(df.columns) == ["year", "rgdpo", "rkna", "pl_gdpo", "cgdpo", "hc"]
    assert df.iloc[0]["year"] == 2017


def test_get_pwt_data_error(monkeypatch):
    def boom(*a, **k):
        raise downloader.requests.exceptions.HTTPError("bad")
    monkeypatch.setattr(downloader.requests, "get", boom)
    with pytest.raises(downloader.requests.exceptions.HTTPError):
        downloader.get_pwt_data()