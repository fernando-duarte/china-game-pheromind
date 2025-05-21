from datetime import datetime
from jinja2 import Template
import pandas as pd


def render_markdown_table(merged_data):
    display_data = merged_data.copy()
    column_mapping = {
        'year': 'Year',
        'GDP_USD': 'GDP (USD)',
        'C_USD': 'Consumption (USD)',
        'G_USD': 'Government (USD)',
        'I_USD': 'Investment (USD)',
        'X_USD': 'Exports (USD)',
        'M_USD': 'Imports (USD)',
        'FDI_pct_GDP': 'FDI (% of GDP)',
        'TAX_pct_GDP': 'Tax Revenue (% of GDP)',
        'POP': 'Population',
        'LF': 'Labor Force',
        'rgdpo': 'PWT rgdpo',
        'rkna': 'PWT rkna',
        'pl_gdpo': 'PWT pl_gdpo',
        'cgdpo': 'PWT cgdpo',
        'hc': 'PWT hc'
    }
    display_data = display_data.rename(columns=column_mapping)

    for col in display_data.columns:
        if col == 'Year':
            display_data[col] = display_data[col].astype(int)
        elif col in ['Population', 'Labor Force']:
            display_data[col] = display_data[col].apply(
                lambda x: f"{x:,.0f}" if not pd.isna(x) else 'N/A')
        else:
            display_data[col] = display_data[col].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else 'N/A')

    headers = list(display_data.columns)
    rows = display_data.values.tolist()
    today = datetime.today().strftime('%Y-%m-%d')
    template = Template('''# China Economic Data

Data sources:
- World Bank World Development Indicators (WDI)
- Penn World Table (PWT) version 10.01
- International Monetary Fund. Fiscal Monitor (FM)

## Economic Data (1960-present)

|{% for h in headers %} {{ h }} |{% endfor %}
|{% for h in headers %} --- |{% endfor %}
{% for row in rows %}|{% for cell in row %} {{ cell }} |{% endfor %}
{% endfor %}

**Notes:**
- GDP and its components (Consumption, Government, Investment, Exports, Imports) are in current US dollars
- FDI is shown as a percentage of GDP (net inflows)
- Tax Revenue is shown as a percentage of GDP
- Population and Labor Force are in number of people
- PWT rgdpo: Output-side real GDP at chained PPPs (in millions of 2017 USD)
- PWT rkna: Capital stock at constant 2017 national prices (index: 2017=1)
- PWT pl_gdpo: Price level of GDP (price level of USA GDPo in 2017=1)
- PWT cgdpo: Output-side real GDP at current PPPs (in millions of USD)
- PWT hc: Human capital index, based on years of schooling and returns to education

Sources:
- World Bank WDI data: World Development Indicators, The World Bank. Available at https://databank.worldbank.org/source/world-development-indicators.
- PWT data: Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "The Next Generation of the Penn World Table" American Economic Review, 105(10), 3150-3182. Available at https://www.ggdc.net/pwt.
- International Monetary Fund. Fiscal Monitor (FM),  https://data.imf.org/en/datasets/IMF.FAD:FM. Accessed on {{ today }}.
''')
    return template.render(headers=headers, rows=rows, today=today)
