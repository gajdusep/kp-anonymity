import numpy as np
import matplotlib.pyplot as plt

from group import *
from node import *
from load_data import *
from visualize import *


def compute_ncp(rows: np.array, min_max_diff: np.array) -> float:
    """
    rows: e.g. np.array([[1,2,5,2],[3,2,5,4],[2,2,0,5]])
        It means: 3 rows, each of them 4 attributes
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

    # UNCOMMENT IF YOU WANT TO SEE THE GRAPHS
    # plt.show()

    # -----------------------------------------------
    # examples of usage of group methods
    # - for the k anonymity, you can use all the methods
    # - please, check the dimensions and how the group works
    # - if you don't understand something, just ask me
    # - (delete the lines that you don't need)
    # -----------------------------------------------
    print('---group operations examples---')
    table_group = create_group_from_pandas_df(df)
    print('table created from out data:', table_group.shape())
    table_min_max_diff = table_group.get_min_max()
    row = table_group.get_row_at_index(3)
    print('a row is a vector (numpy array with one dimension)', row.shape)
    table_group.add_row_to_group(row)
    print('a row was added to the table', table_group.shape())

    # -----------------------------------------------
    # example of ncp computing
    # - will be needed in k anonymity
    # - (delete this, when you start coding k-anonymity)
    # -----------------------------------------------
    print('---ncp computing---')
    new_table = create_empty_group()
    print('empty group created:', new_table.shape())
    row_1 = table_group.get_row_at_index(2)
    row_2 = table_group.get_row_at_index(3)
    new_table.add_row_to_group(row_1)
    new_table.add_row_to_group(row_2)
    print('new rows added:', new_table.shape())
    new_table_ncp = compute_ncp(new_table.group_table, min_max_diff=table_min_max_diff)
    print('ncp computed:', new_table_ncp)


if __name__ == "__main__":
    do_kp_anonymity()
