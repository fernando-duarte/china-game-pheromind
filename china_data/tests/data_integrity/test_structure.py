import os
from datetime import datetime
from china_data.utils.path_constants import get_absolute_output_path

EXPECTED_COLS = [
    "year", "GDP_USD", "C_USD", "G_USD", "I_USD", "X_USD", "M_USD",
    "FDI_pct_GDP", "POP", "LF", "TAX_pct_GDP",
    "rgdpo", "rkna", "pl_gdpo", "cgdpo", "hc"
]


def test_expected_columns_present(raw_df):
    assert set(raw_df.columns) == set(EXPECTED_COLS)


def test_column_count(raw_df):
    assert raw_df.shape[1] == len(EXPECTED_COLS)


def test_year_min_value(raw_df):
    assert int(raw_df["year"].min()) == 1960


def test_year_max_value_current(raw_df):
    current_year = datetime.now().year
    max_year = int(raw_df["year"].max())
    assert max_year in {current_year, current_year - 1, current_year - 2}


def test_year_sequence_no_gaps(raw_df):
    years = sorted(raw_df["year"].dropna().astype(int))
    assert years == list(range(years[0], years[-1] + 1))


def test_year_unique(raw_df):
    years = raw_df["year"].dropna().astype(int)
    assert years.is_unique


def test_no_nan_entries_in_markdown():
    path = os.path.join(get_absolute_output_path(), "china_data_raw.md")
    with open(path, "r") as f:
        content = f.read()
    assert "| nan " not in content
