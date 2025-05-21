import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_tfp(data, alpha=1/3):
    df = data.copy()
    required = ['GDP_USD_bn', 'K_USD_bn', 'LF_mn']
    if not all(col in df.columns for col in required):
        df['TFP'] = np.nan
        return df
    if 'hc' not in df.columns:
        df['hc'] = np.nan
    if df['hc'].isna().any():
        hc_data = df[['year', 'hc']].dropna(subset=['hc'])
        if len(hc_data) >= 2:
            X = hc_data['year'].values.reshape(-1, 1)
            y = hc_data['hc'].values
            model = LinearRegression()
            model.fit(X, y)
            missing = df[df['hc'].isna()]['year'].values
            if len(missing) > 0:
                preds = model.predict(missing.reshape(-1, 1))
                for i, year in enumerate(missing):
                    df.loc[df['year'] == year, 'hc'] = round(preds[i], 4)
    try:
        df['TFP'] = df['GDP_USD_bn'] / (
            (df['K_USD_bn'] ** alpha) * ((df['LF_mn'] * df['hc']) ** (1 - alpha))
        )
        df['TFP'] = df['TFP'].round(4)
    except Exception:
        df['TFP'] = np.nan
    return df
