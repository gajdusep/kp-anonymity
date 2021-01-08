from typing import List, Tuple
import random
import numpy as np
import matplotlib.pyplot as plt

from group import *
from node import *
from load_data import *
from visualize import *

def pattern_similarity(N1, N2):
    """
    Calculate the similairty between two pattern representations as a value between 0 and 1.
    The difference is here defined as the average of the normalized distances (normalized over the number of levels) between the two characters in the same position of the two PRs.
    The similarity is 1 - difference.
    """
    diff = None
    if N1.level == N2.level:
        for i in range(len(N1.PR)):
            diff += abs(ord(N1.PR[i]) - ord(N2.PR[i])) / N1.level
        diff = diff / len(N1.PR)
        return 1 - diff

def p_anonimity_naive(group: Group, p: int, max_level: int, PR_len: int) -> List[Node]:
    """
    The algorithm is implemented in a non-recursive way because keeping the entire tree structure is not needed as we only use the leaf nodes.
    The nodes_to_process is the list of nodes which have not already been processed.
    As nodes are labeled as good or bad leaves, they are added to the good_leaves or bad_leaves lists, respectively.
    The new_nodes_to_process flag indicates wether during the current cycle are new nodes are created by splits.
    If there are new nodes, the nodes_to_process list is updated and those new nodes are processed.
    In the end, the list of processed nodes is returned. Those can then be used to rebuild the table rows.
    """
    # Initialize nodes list with the starting node, corresponding to group
    nodes_to_process = [create_node_from_group(group, PR_len)]
    new_nodes_to_process = True
    good_leaves = []
    bad_leaves = []

    # Node splitting
    while new_nodes_to_process:
        new_nodes_to_process = False
        for N in nodes_to_process:
            if N.size < p:
                bad_leaves.append(N)
            elif N.level == max_level:
                good_leaves.append(N)
            elif N.size < 2*p:
                good_leaves.append(N)
                N.maximize_level(max_level)
            else:
                child_nodes = N.split()
                # Split not possible
                if len(child_nodes) < 2 or max(child.size for child in child_nodes) < p:
                    good_leaves.append(N)
                # Split possible
                else:
                    new_nodes_to_process = True
                    TG_nodes = []
                    TB_nodes = []
                    total_TB_size = 0
                    for child in child_nodes:
                        if child.size < p:
                            TB_nodes.append(child)
                            total_TB_size += child.size
                        else:
                            TG_nodes.append(child)
                    
                    nodes_to_process = TG_nodes

                    if total_TB_size >= p:
                        child_merge = merge_nodes(TB_nodes)
                        nodes_to_process.append(child_merge)
                    else:
                        nodes_to_process.append(TB_nodes)
    
    bad_leaves.sort(key = lambda node: node.size)
    for bad in bad_leaves:
        max_similarity = None
        most_similar_good = None
        for good in good_leaves:
            similarity = pattern_similarity(bad, good)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_good = good
            elif similarity == max_similarity and good.size < most_similar_good:
                most_similar_good = good
        most_similar_good.members.extend(bad.members)

    return good_leaves

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
