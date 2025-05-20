# Open-Economy Growth Model for China (1980–2025)

## Variables

### Endogenous

| Symbol       | Definition                      | Units       |
| :----------- | :------------------------------ | :---------- |
| $Y_t$        | Real GDP                        | bn USD      |
| $K_t$        | Physical capital stock          | bn USD      |
| $A_t$        | Total factor productivity (TFP) | index       |
| $X_t$        | Exports                         | bn USD      |
| $M_t$        | Imports                         | bn USD      |
| $NX_t$       | Net exports                     | bn USD      |
| $C_t$        | Consumption                     | bn USD      |
| $I_t$        | Investment                      | bn USD      |
| $openness_t$ | (Exports + Imports) / GDP       | fraction    |


### Exogenous

| Symbol         | Definition                                    | Units                                 |
| :------------- | :-------------------------------------------- | :------------------------------------ |
| $e_t$        | Nominal exchange rate           | CNY per USD |
| $L_t$        | Labor force                     | million     |
| $fdi\_ratio_t$ | FDI inflows \/ GDP                            | fraction                              |
| $Y^*_t$        | Foreign income                                | index (1980 = 1000)                   |
| $H_t$          | Human capital index                           | index (2015 = Penn World Table value) |

### Parameters

| Symbol                          | Definition                                   | Units    | Value       |
| :------------------------------ | :------------------------------------------- | :------- | :---------- |
| $\alpha$                        | Capital share in production                  | unitless | $0.30$      |
| $\delta$                        | Depreciation rate                            | per year | $0.10$      |
| $g$                             | Baseline TFP growth rate                     | per year | $0.005$     |
| $\theta$                        | Openness contribution to TFP growth          | unitless | $0.1453$    |
| $\phi$                          | FDI contribution to TFP growth               | unitless | $0.10$      |
| $K_0$                           | Initial level of physical capital (1980)     | bn USD   | $2050.10$   |
| $X_0$                           | Initial level of exports (1980)              | bn USD   | $18.10$     |
| $M_0$                           | Initial level of imports (1980)              | bn USD   | $14.50$     |
| $A_0$                           | Initial level of TFP (1980)                  | index    | $0.203$     |
| $\varepsilon_x,\ \varepsilon_m$ | Exchange‐rate elasticities (exports/imports) | unitless | $1.5,\ 1.2$ |
| $\mu_x,\ \mu_m$                 | Income elasticities (exports/imports)        | unitless | $1.0,\ 1.0$ |


## Paths of exogenous variables

| Year | $\tilde e_t$ | $fdi\_ratio_t$ | $Y^*_t$ | $H_t$ |
| ---: | -----------: | -------------: | ------: | ----: |
| 1980 |         0.78 |          0.001 | 1000.00 |  1.58 |
| 1985 |         1.53 |          0.001 | 1159.27 |  1.77 |
| 1990 |         2.48 |           0.02 | 1343.92 |  1.80 |
| 1995 |         4.34 |           0.02 | 1557.97 |  2.02 |
| 2000 |         5.23 |           0.02 | 1806.11 |  2.24 |
| 2005 |         4.75 |           0.02 | 2093.78 |  2.43 |
| 2010 |         5.61 |           0.02 | 2427.26 |  2.61 |
| 2015 |         7.27 |           0.02 | 2813.86 |  2.60 |
| 2020 |         7.00 |           0.02 | 3262.04 |  6.71 |
| 2025 |         6.41 |           0.02 | 3781.60 |  6.49 |

- **Exchange-rate policy**
- $$  e_t   $$

- **Saving-rate policy**
  $$s\in[0.0,1.0]$$

## Model Equations

- **Production:**
  $$Y_t = A_t\,K_t^{\alpha}\,(L_t\,H_t)^{1-\alpha}$$

- **Capital accumulation:**
  $$K_{t+1} = (1-\delta)\,K_t + I_t$$
  $$K_0 \text{ given}$$

- **Labor force:**
  $$L_{t+1} = (1+n) L_t$$

- **TFP:**
  $$A_{t+1} = A_t (1 + g + \theta\,openness_t + \phi\,fdi\_ratio_t)$$

- **Exports:**

  $$
    X_t = X_0\Bigl(\tfrac{e_t}{e_{1980}}\Bigr)^{\varepsilon_x}
      \Bigl(\tfrac{Y^*_t}{Y^*_{1980}}\Bigr)^{\mu_x}
  $$

- **Imports:**

  $$
    M_t = M_0\Bigl(\tfrac{e_t}{e_{1980}}\Bigr)^{-\varepsilon_m}
      \Bigl(\tfrac{Y_t}{Y_{1980}}\Bigr)^{\mu_m}
  $$

- **Net exports:**

  $$
    NX_t = X_t - M_t
  $$

  - **Saving:**
    $$S_t = Y_t - C_t = I_t + NX_t$$

- **Consumption:**
  $$C_t = (1-s)\,Y_t$$

- **Investment:**
  $$I_t = s\,Y_t - NX_t$$

- **Openness ratio:**
  $$openness_t \;=\; \frac{X_t + M_t}{Y_t}$$

## Round-by-Round Calendar

Rounds/Periods $t=0, 1, \dots$ correspond to 1980, 1985, ..., (five year intervals).
Rounds/Periods correspond to (t = 0) → 1980, (t = 1) → 1985, ... . For example:

- K*{t+1}=K_1=K*{1985} when t=0
- K*{t+1}=K_2=K*{1990} when t=1

## Computation Steps for Each Round

### Read values

1. Read values of $s_t$ entered by player.
2. Read values of exogenous variables $fdi\_ratio_t$, $Y^*_t$, $H_t$ from table `Paths of exogenous variables'.
3. Read values for $K_t$, $L_t$, $A_t$:

- For first round (1980), $K_0$, $L_0$, $A_0$ given by parameter values
- For second and later rounds (1985, 1990, ...), $K_t$, $L_t$, $A_t$ determined in the previous round

### Compute current period variable values

4. Compute output/production:
   $$ Y_t = A_t K_t^{\alpha} (L_t\,H_t)^{1-\alpha} $$
5. Compute nominal exchange rate:
   $$ e_t = x_t \tilde e_t $$
6. Compute exports:
   $$
     X_t = X_0\Bigl(\tfrac{e_t}{e_{0}}\Bigr)^{\varepsilon_x}
       \Bigl(\tfrac{Y^*_t}{Y^*_{0}}\Bigr)^{\mu_x}.
   $$
7. Compute imports:
   $$
     M_t = M_0\Bigl(\tfrac{e_t}{e_{0}}\Bigr)^{-\varepsilon_m}
       \Bigl(\tfrac{Y_t}{Y_{0}}\Bigr)^{\mu_m}.
   $$
8. Compute net exports:
   $$ NX_t = X_t - M_t $$
9. Compute openness ratio:
   $$
     openness_t = \frac{X_t + M_t}{Y_t}
   $$
10. Compute consumption:
    $$ C_t = (1-s) Y_t $$

11. Compute investment:
    $$ I_t = s Y_t - NX_t $$

### Compute next period's variable values

12. Compute next period's labor force:
    $$ L\_{t+1} = (1+n) L_t $$

13. Compute next period's capital:
    $$ K\_{t+1} = (1-\delta) K_t + I_t $$

14. Compute next period's TFP:
    $$
      A_{t+1} = A_t
        (1 + g
          + \theta\,openness_t
          + \phi\,fdi\_ratio_t
        )
    $$

## China: National Accounts, Population & Labor Force (1960–2024)

This self-contained document presents a combined annual dataset for China covering:

* **Economic series** (expenditure components of GDP) in **current US\$ (billions)**:

  * C: Private Consumption
  * I: Gross Capital Formation (Investment)
  * G: Government Final Consumption
  * NX: Net Exports (Exports – Imports)
  * Y: GDP (Expenditure approach)

* **Demographic series** in **millions of persons**:

  * Population: Mid-year total population
  * Labor Force: Total economically active population ages 15+ (ILO-modeled estimate)

All values have been rounded to two decimals for economic series and no decimals for demographic series. “N/A” indicates data not available for that year.

---

## Combined Data Table

| Year | C (billions) | I (billions) | G (billions) | NX (billions) | Y (billions) | Population (million) | Labor Force (million) | K (billions) |
| ---: | -----------: | -----------: | -----------: | ------------: | -----------: | -------------------: | --------------------: | -----------: |
| 1960 |        30.13 |        23.64 |         7.43 |          0.18 |        61.38 |                  667 |                   N/A |        56.21 |
| 1961 |        33.17 |        11.42 |         3.57 |          1.95 |        50.10 |                  660 |                   N/A |        57.33 |
| 1962 |        34.07 |         7.43 |         3.61 |          1.35 |        46.46 |                  666 |                   N/A |        50.10 |
| 1963 |        34.29 |        11.03 |         3.89 |          1.07 |        50.28 |                  682 |                   N/A |        47.54 |
| 1964 |        36.14 |        14.54 |         4.43 |          3.50 |        58.61 |                  698 |                   N/A |        51.21 |
| 1965 |        38.65 |        19.09 |         5.08 |          6.87 |        69.71 |                  715 |                   N/A |        57.27 |
| 1966 |        41.48 |        23.36 |         5.78 |          5.27 |        75.88 |                  735 |                   N/A |        64.82 |
| 1967 |        43.93 |        17.39 |         5.43 |          5.31 |        72.06 |                  755 |                   N/A |        63.13 |
| 1968 |        43.73 |        17.67 |         5.27 |          3.32 |        69.99 |                  774 |                   N/A |        64.97 |
| 1969 |        45.81 |        19.93 |         6.36 |          6.62 |        78.72 |                  796 |                   N/A |        69.02 |
| 1970 |        49.02 |        30.44 |         7.10 |          4.94 |        91.51 |                  818 |                   N/A |        76.82 |
| 1971 |        51.26 |        33.57 |         7.91 |          5.80 |        98.56 |                  841 |                   N/A |        82.17 |
| 1972 |        59.43 |        35.53 |         8.93 |          8.25 |       112.16 |                  862 |                   N/A |       104.28 |
| 1973 |        72.01 |        45.66 |        10.41 |          8.70 |       136.77 |                  882 |                   N/A |       126.21 |
| 1974 |        74.80 |        47.88 |        10.63 |          8.99 |       142.25 |                  900 |                   N/A |       139.13 |
| 1975 |        82.19 |        57.26 |        12.11 |          9.51 |       161.16 |                  916 |                   N/A |       164.81 |
| 1976 |        81.82 |        51.14 |        11.40 |          7.36 |       151.63 |                  931 |                   N/A |       174.24 |
| 1977 |        88.70 |        59.27 |        12.94 |         11.44 |       172.35 |                  943 |                   N/A |       198.38 |
| 1978 |        71.51 |        56.23 |        19.51 |          0.93 |       148.18 |                  956 |                   N/A |       235.98 |
| 1979 |        87.56 |        64.71 |        27.05 |         -2.69 |       176.63 |                  969 |                   N/A |       295.59 |
| 1980 |        97.37 |        66.15 |        28.20 |         -2.32 |       189.40 |                  981 |                   N/A |       366.00 |
| 1981 |       104.27 |        64.54 |        29.11 |         -3.82 |       194.11 |                  994 |                   N/A |       368.19 |
| 1982 |       109.43 |        65.59 |        30.99 |          4.74 |       203.18 |                 1009 |                   N/A |       362.94 |
| 1983 |       123.41 |        73.65 |        34.30 |          2.48 |       228.46 |                 1023 |                   N/A |       393.30 |
| 1984 |       131.77 |        89.37 |        39.44 |         -0.03 |       260.55 |                 1037 |                   N/A |       393.58 |
| 1985 |       157.40 |       120.90 |        44.18 |        -15.80 |       306.67 |                 1051 |                   N/A |       379.60 |
| 1986 |       153.43 |       113.47 |        44.05 |         -9.68 |       297.83 |                 1067 |                   N/A |       394.03 |
| 1987 |       135.60 |       101.88 |        40.56 |          0.00 |       277.99 |                 1084 |                   N/A |       448.76 |
| 1988 |       154.98 |       122.06 |        47.60 |        -14.12 |       310.52 |                 1102 |                   N/A |       581.32 |
| 1989 |       177.69 |       129.40 |        50.47 |        -13.60 |       343.97 |                 1119 |                   N/A |       689.92 |
| 1990 |       180.40 |       123.26 |        58.56 |          5.72 |       356.94 |                 1135 |                   503 |       641.79 |
| 1991 |       183.70 |       135.09 |        65.98 |         11.60 |       396.36 |                 1151 |                   516 |       681.12 |
| 1992 |       193.28 |       166.80 |        68.42 |          4.99 |       433.49 |                 1165 |                   531 |       816.40 |
| 1993 |       195.67 |       192.49 |        85.84 |        -33.45 |       440.50 |                 1178 |                   547 |      1065.80 |
| 1994 |       248.80 |       226.01 |       100.33 |          7.61 |       582.75 |                 1192 |                   562 |      1023.18 |
| 1995 |       336.09 |       285.28 |       119.84 |         14.14 |       755.36 |                 1205 |                   576 |      1354.49 |
| 1996 |       404.66 |       336.09 |       135.34 |         15.87 |       891.96 |                 1218 |                   591 |      1663.03 |
| 1997 |       441.34 |       356.94 |       149.28 |         42.82 |       989.39 |                 1230 |                   605 |      1972.88 |
| 1998 |       468.27 |       379.47 |       165.69 |         43.84 |      1057.27 |                 1242 |                   619 |      2349.91 |
| 1999 |       505.49 |       422.66 |       189.18 |         30.64 |      1147.97 |                 1253 |                   631 |      2613.20 |
| 2000 |       566.09 |       440.50 |       213.42 |         28.87 |      1248.88 |                 1263 |                   644 |      2955.90 |
| 2001 |       609.69 |       559.22 |       231.00 |         28.08 |      1428.99 |                 1272 |                   655 |      3310.75 |
| 2002 |       660.47 |       728.01 |       249.06 |         37.38 |      1674.92 |                 1280 |                   663 |      3673.51 |
| 2003 |       709.07 |       856.08 |       280.29 |         35.82 |      1881.26 |                 1288 |                   671 |      4294.51 |
| 2004 |       794.08 |       952.65 |       324.68 |         51.17 |      2122.58 |                 1296 |                   679 |      5148.32 |
| 2005 |       904.94 |      1019.46 |       377.74 |        124.63 |      2426.77 |                 1304 |                   688 |      6085.95 |
| 2006 |      1038.98 |      1083.28 |       462.58 |        208.92 |      2793.76 |                 1311 |                   695 |      7584.56 |
| 2007 |      1291.24 |      1198.48 |       586.01 |        308.04 |      3383.77 |                 1318 |                   703 |     10039.93 |
| 2008 |      1621.23 |      1324.81 |       586.01 |        348.83 |      3880.88 |                 1325 |                   711 |     13768.62 |
| 2009 |      1802.29 |      1453.83 |       637.58 |        220.13 |      4113.82 |                 1331 |                   719 |     16203.94 |
| 2010 |      2089.50 |      1640.96 |       709.87 |        223.02 |      4663.35 |                 1338 |                   726 |     19555.97 |
| 2011 |      2637.02 |      1931.64 |       794.76 |        181.90 |      5545.32 |                 1345 |                   734 |     25530.23 |
| 2012 |      3019.26 |      2235.91 |       848.99 |        231.85 |      6336.01 |                 1354 |                   742 |     30140.55 |
| 2013 |      3429.38 |      2657.88 |       910.16 |        235.38 |      7232.80 |                 1363 |                   748 |     36887.70 |
| 2014 |      3845.40 |      3382.26 |       956.95 |        221.30 |      8405.91 |                 1372 |                   755 |     42915.23 |
| 2015 |      4178.28 |      4327.00 |      1024.95 |        357.87 |      9888.10 |                 1380 |                   761 |     48266.06 |
| 2016 |      4344.47 |      4326.99 |      1021.17 |        255.74 |      9948.37 |                 1388 |                   766 |     52169.73 |
| 2017 |      4744.77 |      4713.28 |      1087.65 |        217.01 |     10762.71 |                 1396 |                   771 |     59061.49 |
| 2018 |      5352.55 |      6085.02 |      2297.62 |         87.91 |     13823.10 |                 1403 |                   777 |     72849.18 |
| 2019 |      5604.60 |      6176.23 |      2394.82 |        131.84 |     14307.49 |                 1408 |                   780 |     79612.86 |
| 2020 |      5610.60 |      6410.86 |      2460.73 |        369.67 |     14851.86 |                 1411 |                   780 |     91901.15 |
| 2021 |          N/A |          N/A |          N/A |           N/A |          N/A |                 1412 |                   778 |     90033.22 |
| 2022 |          N/A |          N/A |          N/A |           N/A |          N/A |                 1426 |                   775 |     86679.39 |
| 2023 |          N/A |          N/A |          N/A |           N/A |          N/A |                 1426 |                   772 |     83120.08 |
| 2024 |          N/A |          N/A |          N/A |           N/A |          N/A |                 1424 |                   773 |     80387.11 |

---

## Data Sources & Reproduction

**Primary Sources:**
- World Bank World Development Indicators (WDI) via API or website, which consolidates UN population data and ILO-modeled labor force estimates.
- Penn World Table (PWT) 10.01 for capital stock data, with projections for recent years.

**Indicator Codes:**

|      Series | Source & Code   | Units       |
| ----------: | :-------------- | :---------- |
| Consumption | WDI: NE.CON.PRVT.CD | Current USD |
|  Investment | WDI: NE.GDI.TOTL.CD | Current USD |
|  Government | WDI: NE.CON.GOVT.CD | Current USD |
|     Exports | WDI: NE.EXP.GNFS.CD | Current USD |
|     Imports | WDI: NE.IMP.GNFS.CD | Current USD |
|         GDP | WDI: NY.GDP.MKTP.CD | Current USD |
|  Population | WDI: SP.POP.TOTL    | Persons     |
| Labor Force | WDI: SL.TLF.TOTL.IN | Persons     |
| Capital Stock | PWT 10.01: rkna, pl_gdpo | Current USD |

### Python Reproduction Example

```python
#!/usr/bin/env python3
"""
china_data_generator.py - Comprehensive data generator for China economic model
-----------------------------------------------------------------
Requires: pandas, requests, pandas_datareader, openpyxl
-----------------------------------------------------------------
Outputs:
    1. china_full_dataset.csv - All economic and demographic data
    2. china_full_dataset.md - Markdown version of the dataset
    3. china_capstock_nominal_1960_2024.csv - Capital stock data only
-----------------------------------------------------------------
"""

import pandas_datareader.wb as wb
import pandas as pd
import requests, tempfile, os, io, sys

def get_wdi_data():
    """Download and process World Bank Development Indicators data for China"""
    print("Downloading World Bank Development Indicators data...")

    # Define WDI indicators
    i_codes = {
        'C': 'NE.CON.PRVT.CD',        # Private consumption
        'I': 'NE.GDI.TOTL.CD',         # Investment (gross capital formation)
        'G': 'NE.CON.GOVT.CD',         # Government consumption
        'Exports': 'NE.EXP.GNFS.CD',   # Exports of goods and services
        'Imports': 'NE.IMP.GNFS.CD',   # Imports of goods and services
        'Y': 'NY.GDP.MKTP.CD',         # GDP (current USD)
        'Population': 'SP.POP.TOTL',   # Total population
        'LaborForce': 'SL.TLF.TOTL.IN' # Total labor force
    }

    # Download data 1960–2024
    try:
        raw = wb.download(
            country='CN',
            indicator=list(i_codes.values()),
            start=1960,
            end=2024
        )
        raw.rename(columns=dict(zip(i_codes.values(), i_codes.keys())), inplace=True)
    except Exception as e:
        print(f"Error downloading WDI data: {e}")
        sys.exit(1)

    # Calculate net exports
    raw['NX'] = raw['Exports'] - raw['Imports']

    # Convert economic series to billions USD and round
    econ = raw[['C','I','G','NX','Y']].div(1e9).round(2)

    # Convert demographic series to millions and round
    demo = raw[['Population','LaborForce']].div(1e6).round(0)

    # Mark pre-1990 labor force as N/A
    demo['LaborForce'] = demo['LaborForce'].where(demo.index >= 1990, pd.NA)

    return econ, demo, raw

def get_capital_stock_data():
    """Calculate and project capital stock data for China using PWT and investment data"""
    print("Calculating capital stock data...")

    # Download PWT 10.01 Excel file
    excel_url = "https://dataverse.nl/api/access/datafile/354095"  # Excel file from PWT 10.01
    try:
        print("Downloading Penn World Table 10.01 data...")
        response = requests.get(excel_url)
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Read PWT data
        pwt = pd.read_excel(tmp_path, sheet_name="Data")
        os.unlink(tmp_path)  # Delete the temporary file
    except Exception as e:
        print(f"Error downloading or processing PWT data: {e}")
        sys.exit(1)

    # Get China's data
    chn = pwt[pwt.countrycode == "CHN"].copy()

    # Calculate capital stock in billions USD
    # First, get China's nominal GDP in 2017 (in millions of USD)
    chn_2017 = chn[chn.year == 2017]
    gdp_2017 = chn_2017.cgdpo.values[0]  # This is in millions of USD

    # Get the capital-output ratio for China in 2017
    # This is typically around 3 for developing countries
    capital_output_ratio = 3.0

    # Calculate the nominal capital stock in 2017 (millions USD)
    capital_stock_2017 = gdp_2017 * capital_output_ratio

    # Now create a series with capital stock for all years
    chn = chn.set_index("year")
    chn["K_nominal_USD_bn"] = 0.0  # Initialize with zeros

    # For years with rkna data, calculate the nominal capital stock
    mask = ~chn.rkna.isna()
    chn.loc[mask, "K_nominal_USD_bn"] = (
        (chn.loc[mask, "rkna"] / chn.loc[2017, "rkna"]) *  # Scale by 2017 value
        capital_stock_2017 *  # Base value in 2017
        (chn.loc[mask, "pl_gdpo"] / chn.loc[2017, "pl_gdpo"])  # Adjust for price level changes
    ).div(1000).round(2)  # Convert from millions to billions and round

    # Project for 2020-2024 using investment data
    try:
        print("Projecting capital stock for recent years...")
        inv = wb.download("CN","NE.GDI.TOTL.CD",2020,2024)["NE.GDI.TOTL.CD"].div(1e9)
        inv.index = inv.index.get_level_values("year").astype(int)

        # Drop NaN values (years where data is not available)
        inv = inv.dropna()

        δ = 0.05   # 5% depreciation rate
        last = chn.loc[2019,"K_nominal_USD_bn"]

        proj = {}
        # Process years with available data
        available_years = inv.index.tolist()
        for y in available_years:
            if not pd.isna(inv.loc[y]):
                last = last*(1-δ) + inv.loc[y]
                proj[y] = round(last,2)

        # For 2024 (or any missing year), project based on previous years
        last_available_year = max(available_years)

        # Calculate average growth rate from the last 3 years
        if last_available_year >= 2022:  # We have at least 2021, 2022, 2023
            years_for_growth = [y for y in range(last_available_year-2, last_available_year+1) if y in proj]
            if len(years_for_growth) >= 2:
                growth_rates = [(proj[y] / proj[y-1] - 1) for y in years_for_growth if y-1 in proj]
                avg_growth = sum(growth_rates) / len(growth_rates)
            else:
                avg_growth = 0.03  # Default 3% growth
        else:
            avg_growth = 0.03  # Default 3% growth

        # Project for 2024 if needed
        if 2024 not in proj and last_available_year < 2024:
            for y in range(last_available_year + 1, 2025):
                proj[y] = round(proj[y-1] * (1 + avg_growth), 2)

        # Add projections
        proj_df = pd.Series(proj, name="K_nominal_USD_bn")
        cap_nom = pd.concat([chn["K_nominal_USD_bn"], proj_df])
    except Exception as e:
        print(f"Warning: Error in projections: {e}")
        cap_nom = chn["K_nominal_USD_bn"]

    # Keep only 1960-2024
    cap_nom = cap_nom.loc[1960:2024]

    return cap_nom

def main():
    """Main function to generate all data files"""
    # Get WDI data
    econ, demo, raw_wdi = get_wdi_data()

    # Get capital stock data
    cap_nom = get_capital_stock_data()

    # Save capital stock data to CSV
    cap_nom.to_csv("china_capstock_nominal_1960_2024.csv", header=True)
    print("✓ Wrote china_capstock_nominal_1960_2024.csv")

    # Combine all data
    df = pd.concat([econ, demo, cap_nom.rename('K')], axis=1)

    # Save combined data to CSV
    df.to_csv("china_full_dataset.csv")
    print("✓ Wrote china_full_dataset.csv")

    # Export to markdown
    with open('china_full_dataset.md', 'w') as f:
        f.write("# China: Economic and Demographic Data (1960-2024)\n\n")
        f.write("## Combined Data Table\n\n")
        f.write(df.to_markdown(tablefmt='github'))
        f.write("\n\n*Generated using data from World Bank WDI and Penn World Table 10.01*\n")
    print("✓ Wrote china_full_dataset.md")

    print("\nAll data files generated successfully!")
    return df

if __name__ == "__main__":
    main()
```

### Running the Script

To run this script:

1. Save the code above to a file named `china_data_generator.py`
2. Make sure you have the required packages installed:
   ```bash
   pip install pandas pandas-datareader requests openpyxl tabulate
   ```
3. Run the script:
   ```bash
   python china_data_generator.py
   ```

The script will generate three files:
- `china_full_dataset.csv` - Complete dataset with all variables
- `china_full_dataset.md` - Markdown version of the dataset
- `china_capstock_nominal_1960_2024.csv` - Capital stock data only

### Manual Download Steps

1. Navigate to [https://data.worldbank.org/indicator/](https://data.worldbank.org/indicator/) for each WDI indicator code above.
2. Select **Country:** China, **Years:** 1960–2024, **Format:** CSV.
3. Import into a spreadsheet, divide economic values by 1e9, demographic values by 1e6.
4. Compute `NX = Exports – Imports` and verify `C + I + G + NX = Y` for each year.
5. For years before 1990, record `N/A` under Labor Force if missing.
6. For capital stock data:
   - Download Penn World Table 10.01 from [https://www.rug.nl/ggdc/productivity/pwt/](https://www.rug.nl/ggdc/productivity/pwt/)
   - Extract China's data for `rkna` (capital stock index) and `pl_gdpo` (price level)
   - Calculate nominal capital stock using the methodology described in the Python code
   - For years after 2019, use investment data and a depreciation model to project capital stock

*End of self-contained dataset and reproduction guide.*
