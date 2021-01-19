import numpy as np
import matplotlib.pyplot as plt
import random
import math 

import copy

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
        max_ncp = -math.inf
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

        max_ncp = -math.inf
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


def group_partition(G: Group, k: int):
    size = G.size()
    if size <= k:
        return [G]
    Gu = create_empty_group()
    Gv = create_empty_group()

    (index_u, index_v) = get_init_tuples_uv(G)
    u_max = G.get_row_at_index(index_u)
    u_max_id = G.get_row_id_at_index(index_u)
    v_max = G.get_row_at_index(index_v)
    v_max_id = G.get_row_id_at_index(index_v)

    Gu.add_row_to_group(u_max, u_max_id)
    Gv.add_row_to_group(v_max, v_max_id)

    for i in random.sample(range(size), size):
        if i == index_u or i == index_v:
            continue
        else: 
            w = G.get_row_at_index(i)
            w_id = G.get_row_id_at_index(i)

            Gu.add_row_to_group(w)
            ncp_Gu = compute_ncp(Gu.group_table, Gu.get_min_max_diff())
            Gu.delete_last_added_row()

            Gv.add_row_to_group(w)
            ncp_Gv = compute_ncp(Gv.group_table, Gv.get_min_max_diff())
            Gv.delete_last_added_row()

            if ncp_Gu < ncp_Gv:
                Gu.add_row_to_group(w, w_id)
            else:
                Gv.add_row_to_group(w, w_id)

    return [Gu, Gv]


def k_anonymity_top_down(table_group: Group, k: int) -> List[Group]:
    if table_group.size() <= k:
        return [table_group]

    groups_to_anonymize = [table_group]
    less_than_k_anonymized_groups = []
    k_anonymized_groups = []

    while len(groups_to_anonymize) > 0:
        group_to_anonymize = groups_to_anonymize.pop(0)
        group_list = group_partition(group_to_anonymize, k)
        for group in group_list:
            if group.size() > k:
                groups_to_anonymize.append(group)
            elif group.size() == k:
                k_anonymized_groups.append(group)
            else:
                less_than_k_anonymized_groups.append(group)

    """
    # postprocessing
    groups_to_anonymize = less_than_k_anonymized_groups
    while len(groups_to_anonymize) > 0:
        group_to_anonymize = groups_to_anonymize.pop(0)
        merging_group = group_to_be_merged(group_to_anonymize, groups_to_anonymize)
        index_of_merging_group = group_to_anonymize.index(merging_group)
        del groups_to_anonymize[index_of_merging_group]
        group_to_anonymize.merge_group(merging_group)
        if group_to_anonymize.size() >= k:
            k_anonymized_groups.append(group_to_anonymize)
        else:
            groups_to_anonymize.append(group_to_anonymize)
    """

    return k_anonymized_groups


# minimize ncp with index
def find_index_of_group_to_be_merged(G: Group, list_of_groups: List[Group]) -> int:
    # index_of_group_G = list_of_groups.index(G.group_table)
    min_ncp = math.inf
    index_of_group_with_min_ncp = 0
    for i in range(len(list_of_groups)):
        # if i != index_of_group_G:
            tmp = create_empty_group()

            for j in range(G.size()): 
                tmp_row = G.get_row_at_index(j)
                tmp.add_row_to_group(tmp_row)

            for k in range(list_of_groups[i].size()):
                tmp_row = list_of_groups[i].get_row_at_index(k)
                tmp.add_row_to_group(tmp_row)    
            tmp_ncp = compute_ncp(tmp.group_table, tmp.get_min_max_diff())
            
            if tmp_ncp < min_ncp and tmp_ncp != 0 and tmp_ncp != math.nan:
                min_ncp = tmp_ncp
                index_of_group_with_min_ncp = i
                print('index: ', i, 'ncp: ', min_ncp)

    return index_of_group_with_min_ncp


# find the smallest group among those in list
def find_smallest_group(list_of_groups: List[Group]) -> Group:
    return min(list_of_groups, key=lambda group: group.size())


# k-anonymity bottom-up method
def k_anonymity_bottom_up(table_group: Group, k: int) -> List[Group]:
    list_of_groups = []
    
    # create a group for each tuple
    for i in range(table_group.size()):
        group_with_single_tuple = create_empty_group()
        row = table_group.get_row_at_index(i)
        row_id = table_group.get_row_id_at_index(i)
        group_with_single_tuple.add_row_to_group(row, row_id)
        list_of_groups.append(group_with_single_tuple)

    print('List of initial groups: ')
    for i in range(len(list_of_groups)):
        print(list_of_groups[i].group_table, list_of_groups[i].ids)   
    # updated_list_of_groups = copy.deepcopy(list_of_groups)

    # do k-anonymity on groups
    while find_smallest_group(list_of_groups).size() < k:
        print('Size of smallest group: ', find_smallest_group(list_of_groups).size()) 
    
        for i in range(len(list_of_groups)):
            if i < len(list_of_groups) and list_of_groups[i].size() < k:

                # merge the group with min ncp
                print('Round ', i, ': merging groups with min ncp ...')

                index_of_merging_group = find_index_of_group_to_be_merged(list_of_groups[i], list_of_groups)
                merged_groups = create_empty_group()
                for j in range(list_of_groups[i].size()):
                    merged_groups.add_row_to_group(list_of_groups[i].get_row_at_index(j),
                                                   list_of_groups[i].get_row_id_at_index(j))
                for z in range(list_of_groups[index_of_merging_group].size()):
                    merged_groups.add_row_to_group(list_of_groups[index_of_merging_group].get_row_at_index(z),
                                                   list_of_groups[index_of_merging_group].get_row_id_at_index(z))
                list_of_groups.append(merged_groups)
                list_of_groups.pop(i)
                if i < index_of_merging_group:
                    list_of_groups.pop(index_of_merging_group - 1)
                else:
                    list_of_groups.pop(index_of_merging_group)

                print('List updated: ')
                for g in range(len(list_of_groups)):
                    print(list_of_groups[g].group_table)
                print('Size of updated list: ', len(list_of_groups))

            if i < len(list_of_groups) and list_of_groups[i].size() >= k*2:
                # split group into two parts
                print('Round ', i, ': splitting group with dim > 2k ...')
                new_group = create_empty_group()
                h = 0
                while h < k:
                    # TODO: implement pop(?)
                    row_to_separate = list_of_groups[i].pop(h)  # add method to eliminate a row and return it
                    id_row_to_separate = list_of_groups[i].get_row_id_at_index(h)
                    new_group.add_row_to_group(row_to_separate, id_row_to_separate)
                    i += 1
                list_of_groups.append(new_group)

    return list_of_groups


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

    table_group = create_group_from_pandas_df(df)
    print('table created from out data:', table_group.shape())
    print('   --- ', table_group.ids)

    '''
    list_of_groups = []
    #create a group for each tuple
    for i in  range(table_group.size()):
        group_with_single_tuple = create_empty_group()
        row = table_group.get_row_at_index(i)
        #print('row to be added: ',row)
        group_with_single_tuple.add_row_to_group(row)
        list_of_groups.append(group_with_single_tuple)
        #print('Creating a group for each tuple: ', group_with_single_tuple.group_table) 
    '''
    # anonymized_groups = k_anonymity_top_down(table_group, k)
    anonymized_groups = k_anonymity_bottom_up(table_group, k)

    '''
    #small test group
    group = create_empty_group()
    group.add_row_to_group(table_group.get_row_at_index(3))
    print(group.group_table, group)
    
    #i = find_index_of_group_to_be_merged(group, list_of_groups)

    row1 = table_group.get_row_at_index(1)
    row2 = table_group.get_row_at_index(2)
    row3 = table_group.get_row_at_index(3)
    group1 = create_empty_group()
    group2 = create_empty_group()
    group3 = create_empty_group()
    group1.add_row_to_group(row1)
    group1.add_row_to_group(row2)
    group2.add_row_to_group(row2)
    group2.add_row_to_group(row3)
    group3.add_row_to_group(row3)


    list_of_groups_2 = []
    list_of_groups_2.append(group1)
    list_of_groups_2.append(group2)
    list_of_groups_2.append(group3)
    group_with_smallest_size = find_smallest_group(list_of_groups_2)
    #print(list_of_groups_2)
    #print('group with min size: ', group_with_smallest_size)


    i = find_index_of_group_to_be_merged(group1, list_of_groups_2) 
    #print(i) 

    groups = create_empty_group()
    groups.add_row_to_group(row1)
    groups.add_row_to_group(row2)
    groups. add_row_to_group(row3)
    #anonymized_groups = k_anonymity_bottom_up(groups, k)

    '''
    
    for ag in anonymized_groups:
        print('shape:', ag.shape(), '; company codes:', ag.ids)

    visualize_intervals(anonymized_groups)

    #row = table_group.get_row_at_index(3)
    #print('a row is a vector (numpy array with one dimension)', row.shape)
    #print(row)
    # print('   - last row:', table_group.get_row_at_index(table_group.size() - 1))
    # table_group.add_row_to_group(row)
    # print('a row was added to the table', table_group.shape())
    # print('   - last row:', table_group.get_row_at_index(table_group.size() - 1))
    # table_group.delete_last_added_row()
    # print('a row was deleted from the table', table_group.shape())
    # print('   - last row:', table_group.get_row_at_index(table_group.size() - 1))


if __name__ == "__main__":
    do_kp_anonymity(path_to_file='data/table.csv', k=3)
