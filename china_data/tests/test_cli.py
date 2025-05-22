import sys
import subprocess
from pathlib import Path
import pandas as pd
import types

import china_data.china_data_downloader as downloader
import china_data.china_data_processor as processor


def test_setup_help():
    script = Path(__file__).resolve().parents[1] / "setup.sh"
    result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
    assert "Usage: ./setup.sh" in result.stdout
    assert result.returncode == 0


def test_downloader_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "get_output_directory", lambda: tmp_path)
    monkeypatch.setattr(downloader, "download_wdi_data", lambda *a, **k: pd.DataFrame({"year": [2020], a[0].replace('.', '_'): [1]}))
    monkeypatch.setattr(downloader, "load_imf_tax_data", lambda: pd.DataFrame())
    monkeypatch.setattr(downloader, "get_pwt_data", lambda: pd.DataFrame({"year": [2020], "rgdpo": [1], "rkna": [1], "pl_gdpo": [1], "cgdpo": [1], "hc": [1]}))
    monkeypatch.setattr(downloader, "render_markdown_table", lambda df, **k: "markdown")
    monkeypatch.setattr(downloader.time, "sleep", lambda s: None)

    sys.argv = ["china_data_downloader.py", "--end-year", "2021"]
    downloader.main()

    assert (tmp_path / "china_data_raw.md").read_text() == "markdown"


def test_processor_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(processor, "get_output_directory", lambda: tmp_path)
    base_df = pd.DataFrame({"year": [2020]})
    monkeypatch.setattr(processor, "load_raw_data", lambda input_file=None: base_df)
    monkeypatch.setattr(processor, "load_imf_tax_revenue_data", lambda: pd.DataFrame())
    monkeypatch.setattr(processor, "convert_units", lambda df: df)
    monkeypatch.setattr(processor, "calculate_capital_stock", lambda raw, capital_output_ratio: pd.DataFrame({"year": [2020], "K_USD_bn": [1]}))
    monkeypatch.setattr(processor, "project_human_capital", lambda raw, end_year=None: pd.DataFrame({"year": [2020], "hc": [1]}))
    monkeypatch.setattr(processor, "merge_dataframe_column", lambda df, src, col, desc: (df, 0))
    monkeypatch.setattr(processor, "merge_projections", lambda df, proj, col, method, desc: (df, {}))
    monkeypatch.setattr(processor, "merge_tax_data", lambda df, tax: (df, None))
    monkeypatch.setattr(processor, "extrapolate_series_to_end_year", lambda df, end_year=None, raw_data=None: (df, {}))
    monkeypatch.setattr(processor, "project_capital_stock", lambda df, end_year=None: pd.DataFrame({"year": [2020], "K_USD_bn": [1]}))
    monkeypatch.setattr(processor, "calculate_economic_indicators", lambda df, alpha=None, logger=None: df)
    monkeypatch.setattr(processor, "get_projection_metadata", lambda *a, **k: {})
    monkeypatch.setattr(processor, "prepare_final_dataframe", lambda df, column_map: df)
    monkeypatch.setattr(processor, "format_data_for_output", lambda df: df)
    saved = {}

    def fake_save(data, out_dir, out_base, proj_info, alpha, ratio, input_file, end_year):
        saved["base"] = out_base
        saved["year"] = end_year

    monkeypatch.setattr(processor, "save_output_files", fake_save)

    sys.argv = [
        "china_data_processor.py",
        "-i",
        "in.md",
        "-a",
        "0.4",
        "-o",
        "out",
        "-k",
        "2.5",
        "--end-year",
        "2026",
    ]
    processor.main()

    assert saved["base"] == "out"
    assert saved["year"] == 2026
