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
    today = datetime.today().strftime('%Y-%m-%d')
    tmpl = Template('''# Processed China Economic Data\n\n|{% for h in headers %} {{ h }} |{% endfor %}\n|{% for h in headers %}---|{% endfor %}\n{% for row in rows %}|{% for cell in row %} {{ cell }} |{% endfor %}\n{% endfor %}\n\n**Extrapolation Notes**\n{% for n in notes %}{{ n }}\n{% endfor %}\n\nData processed with alpha={{ alpha }}, K/Y= {{ capital_output_ratio }}, source file={{ input_file }}, end year={{ end_year }}. Generated {{ today }}.''')
    with open(output_path, 'w') as f:
        f.write(tmpl.render(headers=headers, rows=rows, notes=notes, alpha=alpha, capital_output_ratio=capital_output_ratio, input_file=input_file, end_year=end_year, today=today))
