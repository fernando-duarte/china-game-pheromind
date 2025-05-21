import builtins
import io
import os
import types
import pandas as pd
import pytest
from unittest import mock

import china_data.utils.downloader_utils as downloader_utils


def make_df(rows):
    return pd.DataFrame(rows)


def test_download_wdi_data_success(monkeypatch):
    sample = pd.DataFrame({"country": ["CN"], "year": [2020], "NY_GDP_MKTP_CD": [1.0]})
    def fake_download(country, indicator, start, end):
        return sample
    monkeypatch.setattr(downloader_utils.wb, "download", fake_download)
    monkeypatch.setattr(downloader_utils.time, "sleep", lambda s: None)
    df = downloader_utils.download_wdi_data("NY.GDP.MKTP.CD")
    assert not df.empty
    assert list(df.columns) == ["index", "country", "year", "NY_GDP_MKTP_CD"]


def test_download_wdi_data_failure(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("fail")
    monkeypatch.setattr(downloader_utils.wb, "download", fail)
    monkeypatch.setattr(downloader_utils.time, "sleep", lambda s: None)
    df = downloader_utils.download_wdi_data("BAD")
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
    monkeypatch.setattr(downloader_utils.requests, "get", lambda url, stream=True: DummyResponse())
    expected = pd.DataFrame({
        "countrycode": ["CHN"],
        "year": [2017],
        "rgdpo": [1],
        "rkna": [2],
        "pl_gdpo": [3],
        "cgdpo": [4],
        "hc": [5],
    })
    monkeypatch.setattr(downloader_utils.pd, "read_excel", lambda path, sheet_name="Data": expected)
    df = downloader_utils.get_pwt_data()
    assert list(df.columns) == ["year", "rgdpo", "rkna", "pl_gdpo", "cgdpo", "hc"]
    assert df.iloc[0]["year"] == 2017


def test_get_pwt_data_error(monkeypatch):
    def boom(*a, **k):
        raise downloader_utils.requests.exceptions.HTTPError("bad")
    monkeypatch.setattr(downloader_utils.requests, "get", boom)
    with pytest.raises(downloader_utils.requests.exceptions.HTTPError):
        downloader_utils.get_pwt_data()