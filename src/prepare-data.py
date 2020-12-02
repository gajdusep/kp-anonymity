"""
This is a file with helping methods for preparing the data.
The data were downloaded from this page: https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=BRITANNIA.csv
It contains 51 companies and their stocks prices in approximately 20 years.

Methods in this file must prepare them to one single file and remove the columns that won't be used in anonymization.
"""

import pandas as pd
import numpy as np

def get_companies_names():
    metadata_df = pd.read_csv('data/stock_metadata.csv')
    return list(metadata_df['Symbol'])


def load_all_stock_data(names):
    table_df = None
    for name in names:
        company_df = pd.read_csv('data/companies/' + name + '.csv')
        for column in company_df.columns:
            if column == 'Date' or column == 'VWAP':
                continue
            company_df = company_df.drop(column, 1)
        
        company_df = company_df.rename(columns={'VWAP':name})
        
        if table_df is None:
            table_df = company_df
        else:        
            table_df = table_df.merge(company_df, how='outer', left_on='Date', right_on='Date')
        
    table_df = table_df.sort_values(by='Date')
    table_df = table_df.reset_index(drop=True)

    return table_df        


def write_table_df(table_df, path):
    table_df.to_csv(path, index=False)


names = get_companies_names()
table_df = load_all_stock_data(names)
write_table_df(table_df, 'data/table.csv')
