import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)


def project_human_capital(processed_data, end_year=2025):
    hc_data = processed_data[['year', 'hc']].copy()
    last_year_with_data = hc_data.dropna(subset=['hc']).year.max()
    if last_year_with_data >= end_year:
        return hc_data
    historical = hc_data.dropna()
    years_to_project = list(range(int(last_year_with_data) + 1, end_year + 1))

    # Use Linear Regression as primary method for Human Capital
    logger.info("Using linear regression for human capital projection")
    model = LinearRegression()
    X = historical['year'].values.reshape(-1, 1)
    y = historical['hc'].values
    model.fit(X, y)
    proj_rows = [{'year': year, 'hc': round(model.predict([[year]])[0], 4)} for year in years_to_project]

    if proj_rows:
        proj_df = pd.DataFrame(proj_rows)
        hc_data = pd.merge(hc_data, proj_df, on='year', how='outer', suffixes=('', '_proj'))
        for _, row in proj_df.iterrows():
            hc_data.loc[hc_data['year'] == row['year'], 'hc'] = row['hc']
    return hc_data
