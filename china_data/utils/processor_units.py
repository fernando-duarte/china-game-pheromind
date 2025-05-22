import pandas as pd


def convert_units(raw_data):
    df = raw_data.copy()
    for col in ['GDP_USD', 'C_USD', 'G_USD', 'I_USD', 'X_USD', 'M_USD']:
        if col in df.columns:
            df[col] = df[col] / 1e9
    for col in ['rgdpo', 'cgdpo']:
        if col in df.columns:
            df[col] = df[col] / 1000
    for col in ['POP', 'LF']:
        if col in df.columns:
            df[col] = df[col] / 1e6
    df = df.rename(columns={
        'GDP_USD': 'GDP_USD_bn',
        'C_USD': 'C_USD_bn',
        'G_USD': 'G_USD_bn',
        'I_USD': 'I_USD_bn',
        'X_USD': 'X_USD_bn',
        'M_USD': 'M_USD_bn',
        'rgdpo': 'rgdpo_bn',
        'cgdpo': 'cgdpo_bn',
        'POP': 'POP_mn',
        'LF': 'LF_mn'
    })
    return df
