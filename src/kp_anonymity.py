import numpy as np
import matplotlib.pyplot as plt
import random 

from group import *
from node import *
from load_data import *
from visualize import *

from typing import List


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


def get_init_tuples_uv(G: Group) -> Tuple[int, int]:
    """
    Returns the best rows to start the k-anonymity with
    :param G: Group to search
    :return: indexes of best u,v
    """

    size = G.size()
    index_u, u_max = G.get_random_row()

    for _ in range(3):
        max_ncp = 0
        for i in range(size):
            v = G.get_row_at_index(i)
            tmp = create_empty_group()
            tmp.add_row_to_group(u_max)
            tmp.add_row_to_group(v)
            tmp_min_max_diff = tmp.get_min_max_diff()
            ncp = compute_ncp(tmp.group_table, tmp_min_max_diff)
            if ncp > max_ncp:
                max_ncp = ncp
                v_max = v
                index_v = i

        max_ncp = 0
        for i in range(size):
            u = G.get_row_at_index(i)
            tmp = create_empty_group()
            tmp.add_row_to_group(v_max)
            tmp.add_row_to_group(u)
            tmp_min_max_diff = tmp.get_min_max_diff()
            ncp = compute_ncp(tmp.group_table, tmp_min_max_diff)
            if ncp > max_ncp:
                max_ncp = ncp
                u_max = u
                index_u = i

    return index_u, index_v


def group_partition(G: Group):
    size = G.size()
    Gu = create_empty_group()
    Gv = create_empty_group()

    (index_u, index_v) = get_init_tuples_uv(G)
    u_max = G.get_row_at_index(index_u)
    v_max = G.get_row_at_index(index_v)

    Gu.add_row_to_group(u_max)
    Gv.add_row_to_group(v_max)

    for i in random.sample(range(size), size):
        if i == index_u or i == index_v:
            continue
        else: 
            w = G.get_row_at_index(i)

            Gu.add_row_to_group(w)
            ncp_Gu = compute_ncp(Gu.group_table, Gu.get_min_max_diff())
            Gu.delete_last_added_row()

            Gv.add_row_to_group(w)
            ncp_Gv = compute_ncp(Gv.group_table, Gv.get_min_max_diff())
            Gv.delete_last_added_row()

            if ncp_Gu < ncp_Gv:
                Gu.add_row_to_group(w)
            else:
                Gv.add_row_to_group(w)

    return [Gu, Gv]


def k_anonymity_top_down(table_group: Group, k: int) -> List[Group]:
    if table_group.size() <= k:
        return [table_group]

    k_anonymized_groups = []
    groups_to_anonymize = [table_group]

    while len(groups_to_anonymize) > 0:
        group_to_anonymize = groups_to_anonymize.pop(0)
        group_list = group_partition(group_to_anonymize)
        for group in group_list:
            if group.size() > k:
                groups_to_anonymize.append(group)
            else:
                k_anonymized_groups.append(group)

    """
    TODO Gabriele: some of the groups might be smaller than k. There is some merging I think..
    It must be implemented here
    """

    return k_anonymized_groups

#minimize ncp
def find_group_to_be_merged(G: Group, list_of_groups: List[Group]) -> Group:
    group_with_min_ncp = list_of_groups.pop(0)
    merged_groups = G.append(group_with_min_ncp)
    min_ncp = compute_ncp(merged_groups, merged_groups.min_max_diff)

    for group in list_of_groups:
        tmp_merged_groups = G.append(group)
        tmp_ncp = compute_ncp(tmp_merged_groups, tmp_merged_groups.min_max_diff)
        if tmp_ncp < min_ncp:
            min_ncp = tmp_ncp
            group_with_min_ncp = group

    return group_with_min_ncp        

#k-anonymity bottom-up method
def k_anonymity_bottom_up(table_group: Group, k: int) -> List[Group]: 
    group_for_each_tuple = []
    total_number_of_tuple = len(group_for_each_tuple)

    #create a group for each tuple
    for single_tuple in table_group:
        group_for_each_tuple.append(single_tuple)
    group_anonymizing = group_for_each_tuple #

    #scan all group to find the one that minimizes ncp
    while len(group_anonymizing) >= total_number_of_tuple/k: #
        for group in group_anonymizing:
            if group.size < k:
                #merge the group with min ncp
                group.append(find_group_to_be_merged(group, group_anonymizing)) 
                
            if group.size >= k*2:
                return
                #split group into two parts   


    return group_anonymizing


def do_kp_anonymity(path_to_file: str, k: int):
    # load the data from the file
    df = load_data_from_file(path_to_file)

    # do some preprocessing with the data
    df = remove_rows_with_nan(df)
    visualize_all_companies(df)
    df = remove_outliers(df, max_stock_value=5000)
    visualize_all_companies(df)

    # for testing purposes, let's reduce the number of companies and attributes
    df = reduce_dataframe(df)
    visualize_all_companies(df)

    # UNCOMMENT IF YOU WANT TO SEE THE GRAPHS
    plt.show()

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
    table_min_max_diff = table_group.get_min_max_diff()

    anonymized_groups = k_anonymity_top_down(table_group, k)
    for ag in anonymized_groups:
        print(ag.shape())

    visualize_intervals(anonymized_groups)

    # row = table_group.get_row_at_index(3)
    # print('a row is a vector (numpy array with one dimension)', row.shape)
    # print('   - last row:', table_group.get_row_at_index(table_group.size() - 1))
    # table_group.add_row_to_group(row)
    # print('a row was added to the table', table_group.shape())
    # print('   - last row:', table_group.get_row_at_index(table_group.size() - 1))
    # table_group.delete_last_added_row()
    # print('a row was deleted from the table', table_group.shape())
    # print('   - last row:', table_group.get_row_at_index(table_group.size() - 1))


if __name__ == "__main__":
    do_kp_anonymity(path_to_file='data/table.csv', k=3)
