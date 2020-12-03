
import pandas as pd
import matplotlib.pyplot as plt

from visualize import visualize_all_companies


def load_data_from_file(path_to_file: str):
    return pd.read_csv(path_to_file, index_col=0)
    

def remove_rows_with_nan(dataframe):
    return dataframe.dropna()


def remove_outliers(dataframe, max_stock_value=10000):
    outliers = []
    for col_name in dataframe.columns:
        df_column = dataframe[col_name]
        if df_column[df_column > max_stock_value].count() > 0:
            outliers.append(col_name)
    return dataframe.drop(outliers, axis=1)



df = load_data_from_file('data/table.csv')
df = remove_rows_with_nan(df)
visualize_all_companies(df)

df = remove_outliers(df, 5000)
visualize_all_companies(df)



plt.show()
