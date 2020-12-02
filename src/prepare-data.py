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
    print(metadata_df)


get_companies_names()
