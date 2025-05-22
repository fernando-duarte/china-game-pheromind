def test_consumption_gdp_ratio_within_bounds(raw_df):
    for _, row in raw_df.dropna(subset=["GDP_USD", "C_USD"]).iterrows():
        share = row["C_USD"] / row["GDP_USD"]
        assert 0 <= share <= 1


def test_investment_gdp_ratio_within_bounds(raw_df):
    for _, row in raw_df.dropna(subset=["GDP_USD", "I_USD"]).iterrows():
        share = row["I_USD"] / row["GDP_USD"]
        assert 0 <= share <= 1


def test_government_gdp_ratio_within_bounds(raw_df):
    for _, row in raw_df.dropna(subset=["GDP_USD", "G_USD"]).iterrows():
        share = row["G_USD"] / row["GDP_USD"]
        assert 0 <= share <= 1


def test_exports_gdp_ratio_within_bounds(raw_df):
    for _, row in raw_df.dropna(subset=["GDP_USD", "X_USD"]).iterrows():
        share = row["X_USD"] / row["GDP_USD"]
        assert 0 <= share <= 1


def test_imports_gdp_ratio_within_bounds(raw_df):
    for _, row in raw_df.dropna(subset=["GDP_USD", "M_USD"]).iterrows():
        share = row["M_USD"] / row["GDP_USD"]
        assert 0 <= share <= 1


def test_gdp_equals_sum_c_i_g_plus_net_exports(raw_df):
    tol = 0.06
    for _, row in raw_df.dropna(subset=["GDP_USD", "C_USD", "I_USD", "G_USD", "X_USD", "M_USD"]).iterrows():
        gdp = row["GDP_USD"]
        total = row["C_USD"] + row["I_USD"] + row["G_USD"] + (row["X_USD"] - row["M_USD"])
        if gdp:
            diff = abs(gdp - total) / gdp
            assert diff < tol


def test_labor_force_less_than_population(raw_df):
    subset = raw_df.dropna(subset=["POP", "LF"])
    assert (subset["LF"] <= subset["POP"]).all()


def test_complete_year_coverage(raw_df):
    years = raw_df["year"].astype(int)
    assert years.min() == 1960
    assert len(years) == years.max() - 1960 + 1
