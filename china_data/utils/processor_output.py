import pandas as pd
from jinja2 import Template
from datetime import datetime


def format_data_for_output(data_df):
    formatted_df = data_df.copy()
    for col_name in formatted_df.columns:
        vals = []
        for val in formatted_df[col_name]:
            if pd.isna(val):
                vals.append('nan')
            elif isinstance(val, float):
                if col_name in ['FDI (% of GDP)', 'TFP', 'Human Capital', 'Openness Ratio', 'Saving Rate']:
                    vals.append(f"{val:.4f}".rstrip('0').rstrip('.'))
                elif col_name in ['GDP', 'Consumption', 'Government', 'Investment', 'Exports', 'Imports', 'Net Exports', 'Physical Capital', 'Tax Revenue (bn USD)', 'Saving (bn USD)', 'Private Saving (bn USD)', 'Public Saving (bn USD)']:
                    vals.append(f"{val:.4f}".rstrip('0').rstrip('.'))
                elif col_name in ['Population', 'Labor Force']:
                    vals.append(f"{val:.2f}".rstrip('0').rstrip('.'))
                else:
                    vals.append(f"{val:.2f}".rstrip('0').rstrip('.'))
            elif isinstance(val, int) and col_name == 'Year':
                vals.append(str(val))
            elif col_name in ['Population', 'Labor Force'] and isinstance(val, (int, float)):
                vals.append(f"{val:.2f}".rstrip('0').rstrip('.'))
            else:
                vals.append(str(val))
        formatted_df[col_name] = vals
    return formatted_df


def create_markdown_table(data, output_path, extrapolation_info, alpha=1/3, capital_output_ratio=3.0, input_file="china_data_raw.md", end_year=2025):
    column_mapping = {
        'Year': 'year', 'GDP': 'GDP_USD_bn', 'Consumption': 'C_USD_bn', 'Government': 'G_USD_bn', 'Investment': 'I_USD_bn', 'Exports': 'X_USD_bn', 'Imports': 'M_USD_bn', 'Net Exports': 'NX_USD_bn', 'Population': 'POP_mn', 'Labor Force': 'LF_mn', 'Physical Capital': 'K_USD_bn', 'TFP': 'TFP', 'FDI (% of GDP)': 'FDI_pct_GDP', 'Human Capital': 'hc', 'Tax Revenue (bn USD)': 'T_USD_bn', 'Openness Ratio': 'Openness_Ratio', 'Saving (bn USD)': 'S_USD_bn', 'Private Saving (bn USD)': 'S_priv_USD_bn', 'Public Saving (bn USD)': 'S_pub_USD_bn', 'Saving Rate': 'Saving_Rate'
    }
    headers = list(data.columns)
    rows = data.values.tolist()
    notes = []
    for var, info in extrapolation_info.items():
        if not info['years']:
            continue
        display_name = var
        for disp, internal in column_mapping.items():
            if internal == var:
                display_name = disp
                break
        years = info['years']
        if len(years) == 1:
            years_str = f"{years[0]}"
        else:
            years_str = f"{years[0]}-{years[-1]}"
        notes.append(f"- {display_name}: {info['method']} ({years_str})")

    # Group extrapolation methods for detailed notes
    extrapolation_methods = {
        'ARIMA(1,1,1)': [],
        'Average growth rate': [],
        'Linear regression': [],
        'Investment-based projection': [],
        'IMF projections': [],
        'Extrapolated': []
    }

    for var, info in extrapolation_info.items():
        if not info['years']:
            continue
        display_name = var
        for disp, internal in column_mapping.items():
            if internal == var:
                display_name = disp
                break

        method = info['method']
        years_str = f"{info['years'][0]}-{info['years'][-1]}" if len(info['years']) > 1 else f"{info['years'][0]}"

        if 'ARIMA' in method:
            extrapolation_methods['ARIMA(1,1,1)'].append(f"{display_name} ({years_str})")
        elif 'growth rate' in method:
            extrapolation_methods['Average growth rate'].append(f"{display_name} ({years_str})")
        elif 'regression' in method:
            extrapolation_methods['Linear regression'].append(f"{display_name} ({years_str})")
        elif 'Investment' in method or 'investment' in method:
            extrapolation_methods['Investment-based projection'].append(f"{display_name} ({years_str})")
        elif 'IMF' in method:
            extrapolation_methods['IMF projections'].append(f"{display_name} ({years_str})")
        else:
            extrapolation_methods['Extrapolated'].append(f"{display_name} ({years_str})")

    today = datetime.today().strftime('%Y-%m-%d')
    tmpl = Template('''# Processed China Economic Data

|{% for h in headers %} {{ h }} |{% endfor %}
|{% for h in headers %}---|{% endfor %}
{% for row in rows %}|{% for cell in row %} {{ cell }} |{% endfor %}
{% endfor %}

# Notes on Computation

## Data Sources
The raw data in `{{ input_file }}` comes from the following sources:

- **World Bank World Development Indicators (WDI)** for GDP components, FDI, population, and labor force
- **Penn World Table (PWT) version 10.01** for human capital index and capital stock related variables
- **International Monetary Fund (IMF) Fiscal Monitor** for tax revenue data

This processed dataset was created by applying the following transformations to the raw data:

## Unit Conversions
- GDP and its components (Consumption, Government, Investment, Exports, Imports) were converted from USD to billions USD
- Population and Labor Force were converted from people to millions of people

## Derived Variables
### Net Exports
Calculated as Exports - Imports (in billions USD)
```
Net Exports = Exports - Imports
```

### Physical Capital
Calculated using PWT data with the following formula:
```
K_t = (rkna_t / rkna_2017) * K_2017 * (pl_gdpo_t / pl_gdpo_2017)
```
Where:
- K_t is the capital stock in year t (billions USD)
- rkna_t is the real capital stock index in year t (from PWT)
- rkna_2017 is the real capital stock index in 2017 (from PWT)
- K_2017 is the nominal capital stock in 2017, estimated as GDP_2017 * {{ capital_output_ratio }} (capital-output ratio)
- pl_gdpo_t is the price level of GDP in year t (from PWT)
- pl_gdpo_2017 is the price level of GDP in 2017 (from PWT)

### TFP (Total Factor Productivity)
Calculated using the Cobb-Douglas production function:
```
TFP_t = Y_t / (K_t^alpha * (L_t * H_t)^(1-alpha))
```
Where:
- Y_t is GDP in year t (billions USD)
- K_t is Physical Capital in year t (billions USD)
- L_t is Labor Force in year t (millions of people)
- H_t is Human Capital index in year t
- alpha = {{ alpha }} (capital share parameter)

## Extrapolation to {{ end_year }}
Each series was extrapolated using the following methods:

### ARIMA(1,1,1) model
{% for var in extrapolation_methods['ARIMA(1,1,1)'] %}
- {{ var }}{% endfor %}

### Average growth rate of historical data
{% for var in extrapolation_methods['Average growth rate'] %}
- {{ var }}{% endfor %}

### Linear regression
{% for var in extrapolation_methods['Linear regression'] %}
- {{ var }}{% endfor %}

{% if extrapolation_methods['IMF projections'] %}
### IMF projections
{% for var in extrapolation_methods['IMF projections'] %}
- {{ var }}: Projected using official IMF Fiscal Monitor projections{% endfor %}
{% endif %}

{% if extrapolation_methods['Investment-based projection'] %}
### Investment-based projection
{% for var in extrapolation_methods['Investment-based projection'] %}
- {{ var }}: Projected using the formula K_t = K_{t-1} * (1-delta) + I_t, where delta = 0.05 (5% depreciation rate) and I_t is investment in year t{% endfor %}
{% endif %}

{% if extrapolation_methods['Extrapolated'] %}
### Other methods
{% for var in extrapolation_methods['Extrapolated'] %}
- {{ var }}{% endfor %}
{% endif %}

Data processed with alpha={{ alpha }}, K/Y= {{ capital_output_ratio }}, source file={{ input_file }}, end year={{ end_year }}. Generated {{ today }}.''')
    with open(output_path, 'w') as f:
        f.write(tmpl.render(headers=headers, rows=rows, notes=notes, extrapolation_methods=extrapolation_methods, alpha=alpha, capital_output_ratio=capital_output_ratio, input_file=input_file, end_year=end_year, today=today))
