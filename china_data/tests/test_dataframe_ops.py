import pandas as pd
from china_data.utils.processor_dataframe import merge_dataframe_column, merge_projections, merge_tax_data
from china_data.utils.processor_dataframe import prepare_final_dataframe


def test_merge_dataframe_column_adds_values():
    tgt = pd.DataFrame({'year':[2000,2001]})
    src = pd.DataFrame({'year':[2000,2001],'col':[1,2]})
    out, count = merge_dataframe_column(tgt, src, 'col', 'test')
    assert count == 2
    assert list(out['col']) == [1,2]


def test_merge_dataframe_column_missing_source():
    tgt = pd.DataFrame({'year':[2000,2001]})
    src = pd.DataFrame({'year':[2000,2001]})
    out, count = merge_dataframe_column(tgt, src, 'col', 'test')
    assert count == 0
    assert out['col'].isna().all()


def test_merge_projections_applies_projection():
    tgt = pd.DataFrame({'year':[2020,2021],'val':[1.0,None]})
    proj = pd.DataFrame({'year':[2021],'val':[2.0]})
    out, meta = merge_projections(tgt, proj, 'val', 'method', 'val')
    assert out.loc[out['year']==2021,'val'].iloc[0] == 2.0
    assert meta == {'method':'method','years':[2021]}


def test_merge_tax_data_simple():
    tgt = pd.DataFrame({'year':[2000,2001],'TAX_pct_GDP':[None,None]})
    tax = pd.DataFrame({'year':[2000],'TAX_pct_GDP':[10.0]})
    out, meta = merge_tax_data(tgt, tax)
    assert out.loc[out['year']==2000,'TAX_pct_GDP'].iloc[0] == 10.0
    assert meta is None


def test_prepare_final_dataframe_drops_duplicates():
    df = pd.DataFrame({'year':[2000,2000,2001],'a':[1,2,3],'b':[4,5,6]})
    column_map = {'year':'Year','a':'A'}
    out = prepare_final_dataframe(df, column_map)
    assert list(out.columns) == ['Year','A']
    assert len(out) == 2

