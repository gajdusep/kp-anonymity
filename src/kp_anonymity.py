import numpy as np
import matplotlib.pyplot as plt
import random
import math 

import copy
import math
import argparse
from enum import Enum

from node import *
from load_data import *
from visualize import *


def compute_pattern_similarity(N1: Node, N2: Node) -> float:
    """
    Calculate the similairty between two pattern representations as a value between 0 and 1.
    The difference is here defined as the average of the normalized distances (normalized over the number of levels) between the two characters in the same position of the two PRs.
    The similarity is 1 - difference.
    """
    diff = 0
    if N1.level == N2.level:
        for i in range(len(N1.PR)):
            diff += abs(ord(N1.PR[i]) - ord(N2.PR[i])) / N1.level
        diff = diff / len(N1.PR)
        return 1 - diff
    else:
        # if the two PRs are of different levels, their values are normalized separately
        for i in range(len(N1.PR)):
            diff += abs((ord(N1.PR[i]) - 97) / (N1.level - 1) - (ord(N2.PR[i]) - 97) / (N2.level - 1))
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
    good_leaves: List[Node] = []
    bad_leaves: List[Node] = []

    # Node splitting
    while new_nodes_to_process:
        new_nodes_to_process = False
        for N in nodes_to_process:
            N_size = N.size()
            if N_size < p:
                bad_leaves.append(N)
            elif N.level == max_level:
                good_leaves.append(N)
            elif N_size < 2*p:
                good_leaves.append(N)
                N.maximize_level(max_level)
            else:
                child_nodes = N.split()
                # Split not possible
                if len(child_nodes) < 2 or max(child.size() for child in child_nodes) < p:
                    good_leaves.append(N)
                # Split possible
                else:
                    new_nodes_to_process = True
                    TG_nodes: List[Node] = []
                    TB_nodes: List[Node] = []
                    total_TB_size = 0
                    for child in child_nodes:
                        if child.size() < p:
                            TB_nodes.append(child)
                            total_TB_size += child.size()
                        else:
                            TG_nodes.append(child)
                    
                    nodes_to_process = TG_nodes

                    if total_TB_size >= p:
                        child_merge = merge_nodes(TB_nodes)
                        nodes_to_process.append(child_merge)
                    else:
                        nodes_to_process.extend(TB_nodes)
    
    bad_leaves.sort(key=lambda node: node.size())
    for bad in bad_leaves:
        max_similarity = None
        most_similar_good = None
        for good in good_leaves:
            similarity = compute_pattern_similarity(bad, good)
            if max_similarity is None or similarity > max_similarity:
                max_similarity = similarity
                most_similar_good = good
            elif similarity == max_similarity and good.size() < most_similar_good:
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
        both_have_less = True
        for group in group_list:
            if group.size() > k:
                both_have_less = False
        if not both_have_less:
            for group in group_list:
                if group.size() > k:
                    groups_to_anonymize.append(group)
                elif group.size() == k:
                    k_anonymized_groups.append(group)
                else:
                    less_than_k_anonymized_groups.append(group)
        else:
            k_anonymized_groups.append(group_to_anonymize)

    for ag in less_than_k_anonymized_groups:
        print('less than k:', ag.shape(), '; company codes:', ag.ids)

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
    min_max_diff = G.get_min_max_diff()
    print('min max diff :', min_max_diff )
    for i in range(len(list_of_groups)):
        #if list_of_groups[i].group_table != G.group_table:    
            tmp = create_empty_group()

            for j in range(G.size()): 
                tmp_row = G.get_row_at_index(j)
                tmp.add_row_to_group(tmp_row)

            
            for k in range(list_of_groups[i].size()):
                tmp_row_2 = list_of_groups[i].get_row_at_index(k)
                tmp.add_row_to_group(tmp_row_2)

            #min_max_diff = tmp.get_min_max_diff()
            tmp_ncp = compute_ncp(tmp.group_table, min_max_diff)
            #print('tmp group: ', tmp.group_table, 'ncp: ', tmp_ncp)   

            
            if tmp_ncp < min_ncp:
                min_ncp = tmp_ncp
                index_of_group_with_min_ncp = i
                
    #print('index of group to be merged: ', index_of_group_with_min_ncp, ', ncp: ', min_ncp)
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
                if i == index_of_merging_group:
                    print('MERGING GROUP WITH HIMSELF!')
                else: 
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
                    if i > index_of_merging_group:
                        list_of_groups.pop(index_of_merging_group)
                    if i == index_of_merging_group:
                        print('merging group with himself!!!!!!')
                    '''
                    print('List updated: ')
                    for g in range(len(list_of_groups)):
                        print(list_of_groups[g].group_table, list_of_groups[g].ids)
                    print('Size of updated list: ', len(list_of_groups))

                    '''
                    

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


class KPAlgorithm(str, Enum):
    TOPDOWN = 'top-down'
    BOTTOMUP = 'bottom-up'
    KAPRA = 'kapra'


def kp_anonymity_classic(table_group: Group, k: int, p: int, kp_algorithm: str):
    if kp_algorithm == KPAlgorithm.TOPDOWN:
        anonymized_groups = k_anonymity_top_down(table_group, k)
    else:
        anonymized_groups = k_anonymity_bottom_up(table_group, k)

    for ag in anonymized_groups:
        print('shape:', ag.shape(), '; company codes:', ag.ids)
    visualize_intervals(anonymized_groups)

    # TODO: p-anonymity here


def kp_anonymity_kapra(table_group: Group, k: int, p: int):
    # TODO: p-anonymity here
    # TODO: k-anonymity
    pass


def do_kp_anonymity(path_to_file: str, k: int, p: int, kp_algorithm: str):
    # load the data from the file
    df = load_data_from_file(path_to_file)

    # do some preprocessing with the data
    df = remove_rows_with_nan(df)
    visualize_all_companies(df)
    df = remove_outliers(df, max_stock_value=5000)
    visualize_all_companies(df)

    # for testing purposes, let's reduce the number of companies and attributes
    df = reduce_dataframe(df, companies_count=20)
    visualize_all_companies(df)

    # UNCOMMENT IF YOU WANT TO SEE THE GRAPHS
    plt.show()

    table_group = create_group_from_pandas_df(df)

    print('table created from out data:', table_group.shape(), table_group.ids)

    # TODO: kp_anonymity_classic and also kp_anonymity_kapra should return something to be for example saved to the file
    if kp_algorithm == KPAlgorithm.TOPDOWN or kp_algorithm == KPAlgorithm.BOTTOMUP:
        kp_anonymity_classic(table_group, k, p, kp_algorithm)
    else:
        kp_anonymity_kapra(table_group, k, p)
    
    # TODO: finish the QI and SD


def parse_arguments():
    # TODO: add parameter - visualize graphs?

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k-anonymity', required=True, type=int)
    parser.add_argument('-p', '--p-anonymity', required=True, type=int)
    parser.add_argument('-a', '--algorithm', required=False, default='top-down')
    parser.add_argument('-i', '--input-file', required=False)
    parser.add_argument('-o', '--output-file', required=False)
    args = vars(parser.parse_args())

    k = args['k_anonymity']
    p = args['p_anonymity']

    algo_str = args['algorithm']
    if algo_str == 'top-down':
        algo = KPAlgorithm.TOPDOWN
    elif algo_str == 'bottom-up':
        algo = KPAlgorithm.BOTTOMUP
    elif algo_str == 'kapra':
        algo = KPAlgorithm.KAPRA
    else:
        print('The algorithm should be one of the following: ' + ', '.join([e.value for e in KPAlgorithm]) +
              '\nChoosing the default one: top-down\n')
        algo = KPAlgorithm.TOPDOWN

    input_path = args['input_file']
    output_path = args['output_file']

    return k, p, algo, input_path, output_path


if __name__ == "__main__":
    k, p, algo, input_path, output_path = parse_arguments()
    print('kp-anonymity with the following parameters: k={}, p={}, algo={}, input_path={}, output_path={}'.format(
        k, p, algo.value, input_path, output_path))

    do_kp_anonymity(path_to_file='data/table.csv', k=k, p=p, kp_algorithm=algo)
