
import pandas as pd


def load_data_from_file(path_to_file: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_file, index_col=0)
    df += 1
    return df


def remove_rows_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def remove_outliers(df: pd.DataFrame, max_stock_value=10000) -> pd.DataFrame:
    outliers = []
    for col_name in df.columns:
        df_column = df[col_name]
        if df_column[df_column > max_stock_value].count() > 0:
            outliers.append(col_name)
    return df.drop(outliers, axis=1)


def reduce_dataframe(df: pd.DataFrame, companies_count=10, attributes_count=20) -> pd.DataFrame:
    every_nth = 35
    return df.iloc[0:attributes_count*every_nth:every_nth, 0:companies_count]


def reduce_dataframe_short(df: pd.DataFrame, companies_count=10, attributes_count=20) -> pd.DataFrame:
    return df.iloc[0:attributes_count, 0:companies_count]
