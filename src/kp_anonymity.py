import numpy as np
import matplotlib.pyplot as plt

from group import *
from node import *
from load_data import *
from visualize import *


def compute_ncp(rows: np.array, min_max_diff: np.array) -> float:
    """
    tuples: e.g. np.array([[1,2,5,2],[3,2,5,4],[2,2,0,5]])
        It means: 3 tuples, each of them 4 attributes
        Will be generalized into:
        [(1,3), (2,2), (0,5), (2,5)]
    min_max_diff: e.g. np.array([20,4,10,5])
        Therefore ncp: 
        3*((3-1)/20 + (2-2)/4 + (5-0)/10 + (5-2)/5) = 3*(0.1+0+0.5+0.6) = 3.6
    """

    z = np.max(rows, axis=0)
    y = np.min(rows, axis=0)
    zy_diff = z - y
    ncp = np.sum(zy_diff / min_max_diff)
    return rows.shape[0] * ncp


def do_kp_anonymity():
    # load the data from the file
    df = load_data_from_file('data/table.csv')

    # do some preprocessing with the data
    df = remove_rows_with_nan(df)
    visualize_all_companies(df)
    df = remove_outliers(df, max_stock_value=5000)
    visualize_all_companies(df)

    # for testing purposes, let's reduce the number of companies and attributes
    df = reduce_dataframe(df)
    visualize_all_companies(df)

    # uncomment if you want to see the graphs
    # plt.show()

    table_group = create_group_from_pandas_df(df)
    table_min_max_diff = table_group.get_min_max()


if __name__ == "__main__":
    do_kp_anonymity()
