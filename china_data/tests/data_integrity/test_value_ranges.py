import numpy as np


def test_gdp_values_positive(raw_df):
    assert (raw_df["GDP_USD"].dropna() > 0).all()


def test_consumption_values_positive(raw_df):
    assert (raw_df["C_USD"].dropna() > 0).all()


def test_government_values_positive(raw_df):
    assert (raw_df["G_USD"].dropna() > 0).all()


def test_investment_values_positive(raw_df):
    assert (raw_df["I_USD"].dropna() > 0).all()


def test_exports_values_non_negative(raw_df):
    assert (raw_df["X_USD"].dropna() >= 0).all()


def test_imports_values_non_negative(raw_df):
    assert (raw_df["M_USD"].dropna() >= 0).all()


def test_population_values_positive(raw_df):
    assert (raw_df["POP"].dropna() > 0).all()


def test_labor_force_values_positive(raw_df):
    assert (raw_df["LF"].dropna() > 0).all()


def test_rgdpo_values_positive(raw_df):
    assert (raw_df["rgdpo"].dropna() > 0).all()


def test_capital_index_positive(raw_df):
    assert (raw_df["rkna"].dropna() > 0).all()


def test_price_level_values_positive(raw_df):
    assert (raw_df["pl_gdpo"].dropna() > 0).all()


def test_human_capital_positive(raw_df):
    assert (raw_df["hc"].dropna() > 0).all()


def test_population_and_labor_values_integral(raw_df):
    assert all(float(x).is_integer() for x in raw_df["POP"].dropna())
    assert all(float(x).is_integer() for x in raw_df["LF"].dropna())


def test_fdi_pct_gdp_valid_range(raw_df):
    vals = raw_df["FDI_pct_GDP"].dropna()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_tax_pct_gdp_valid_range(raw_df):
    if "TAX_pct_GDP" not in raw_df.columns:
        return
    vals = raw_df["TAX_pct_GDP"].dropna()
    if vals.empty:
        return
    assert ((vals >= 0) & (vals <= 100)).all()
