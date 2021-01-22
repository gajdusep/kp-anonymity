import argparse
from enum import Enum
from typing import List

from load_data import *
from visualize import visualize_intervals
from group import Group, create_group_from_pandas_df
from node import Node
from k_anonymity import k_anonymity_top_down
from p_anonymity import p_anonymity_naive, p_anonymity_kapra


class KPAlgorithm(str, Enum):
    TOPDOWN = 'top-down'
    BOTTOMUP = 'bottom-up'
    KAPRA = 'kapra'

# Global max_level constant
max_level = 4

def kp_anonymity_classic(table_group: Group, k: int, p: int, PR_len: int, kp_algorithm: str):
    if kp_algorithm == KPAlgorithm.TOPDOWN:
        anonymized_groups = k_anonymity_top_down(table_group, k)
    elif kp_algorithm == KPAlgorithm.BOTTOMUP:
        pass  # TODO: k_anonymity_bottom_up

    for ag in anonymized_groups:
        print('after the k anonymization:', ag.shape(), '; company codes:', ag.ids)
    visualize_intervals(anonymized_groups)
    print('--- final k-anonymized groups:', len(anonymized_groups))

    final_nodes: List[Node] = []
    for ag in anonymized_groups:
        final_nodes.extend(p_anonymity_naive(group=ag, p=p, max_level=max_level, PR_len=PR_len))
    print('--- final nodes:', len(final_nodes))
    for node in final_nodes:
        print(node.ids(), node.PR, node.group.ids)


def kp_anonymity_kapra(table_group: Group, k: int, p: int, PR_len: int):
    p_anonymized_nodes: List[Node] = p_anonymity_kapra(group=table_group, p=p, max_level=max_level, PR_len=PR_len)
    p_anonymized_groups: List[Group] = [node.to_group() for node in p_anonymized_nodes]
    print('all groups given as a parameter:', p_anonymized_groups)

    final_group_list: List[Group] = []

    # every group bigger than 2*p must be split
    for group in p_anonymized_groups:
        if group.size() > 2*p:
            # split it by top-down clustering
            continue

    # if any group is already bigger than k, add it to the final group list
    indices_bigger_than_k = [i for i, group in enumerate(p_anonymized_groups) if group.size() >= k]
    for i in sorted(indices_bigger_than_k, reverse=True):
        final_group_list.append(p_anonymized_groups.pop(i))

    print('----- after bigger than k check -----')
    print('p_anonymized', p_anonymized_groups)
    print('final_group_list', final_group_list)

    # while the total number of rows in p_anonymized_groups >= k
    while sum([g.size() for g in p_anonymized_groups]) >= k:
        index_of_min_vl = min(range(len(p_anonymized_groups)),
                              key=lambda i: p_anonymized_groups[i].instant_value_loss())
        group_to_grow = p_anonymized_groups.pop(index_of_min_vl)

        while group_to_grow.size() < k:
            index_of_other_group = min(
                range(len(p_anonymized_groups)),
                key=lambda i: Group.merge_two_groups(group_to_grow, p_anonymized_groups[i]).instant_value_loss())
            group_to_grow.merge_group(p_anonymized_groups.pop(index_of_other_group))

        final_group_list.append(group_to_grow)

    print('----- after merging -----')
    print('p_anonymized', p_anonymized_groups)
    print('final_group_list', final_group_list)

    # add the remaining p_anonymized_groups that were not merged yet into a groups that minimize the instant_value_loss
    for group in p_anonymized_groups:
        best_group_to_merge_in_index = min(
            range(len(final_group_list)),
            key=lambda i: Group.merge_two_groups(group, final_group_list[i]).instant_value_loss())
        final_group_list[best_group_to_merge_in_index].merge_group(group)

    print('----- after removing the last items -----')
    print('p_anonymized', p_anonymized_groups)
    print('final_group_list', final_group_list)


def do_kp_anonymity(path_to_file: str, k: int, p: int, PR_len: int, kp_algorithm: str):
    # load the data from the file
    df = load_data_from_file(path_to_file)

    # do some preprocessing with the data
    df = remove_rows_with_nan(df)
    # visualize_all_companies(df)
    df = remove_outliers(df, max_stock_value=5000)
    # visualize_all_companies(df)

    # for testing purposes, let's reduce the number of companies and attributes
    df = reduce_dataframe(df, companies_count=30)
    # visualize_all_companies(df)

    # UNCOMMENT IF YOU WANT TO SEE THE GRAPHS
    # plt.show()

    table_group = create_group_from_pandas_df(df)
    print('table created from out data:', table_group.shape(), table_group.ids)

    # TODO: kp_anonymity_classic and also kp_anonymity_kapra should return something to be for example saved to the file
    if kp_algorithm == KPAlgorithm.TOPDOWN or kp_algorithm == KPAlgorithm.BOTTOMUP:
        kp_anonymity_classic(table_group, k, p, PR_len, kp_algorithm)
    else:
        kp_anonymity_kapra(table_group, k, p, PR_len)
    
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
    k, p, PR_len, algo, input_path, output_path = parse_arguments()
    print('kp-anonymity with the following parameters: k={}, p={}, PR_len={}, algo={}, input_path={}, output_path={}'.format(
        k, p, PR_len, algo.value, input_path, output_path))

    do_kp_anonymity(path_to_file='data/table.csv', k=k, p=p, PR_len=PR_len, kp_algorithm=algo)
