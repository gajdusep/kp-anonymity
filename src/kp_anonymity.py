import argparse
from enum import Enum
from typing import DefaultDict, List, Dict, final

from load_data import *
from visualize import visualize_envelopes, visualize_p_anonymized_nodes
from group import Group, create_group_from_pandas_df
from node import Node
from k_anonymity import k_anonymity_top_down, kapra_group_formation, k_anonymity_bottom_up
from p_anonymity import p_anonymity_naive, p_anonymity_kapra
from verbose import setverbose, unsetverbose, getverbose, verbose

class KPAlgorithm(str, Enum):
    TOPDOWN = 'top-down'
    BOTTOMUP = 'bottom-up'
    KAPRA = 'kapra'

show_plots = False

def kp_anonymity_classic(table_group: Group, k: int, p: int, PR_len: int, max_level: int, kp_algorithm: str):
    if kp_algorithm == KPAlgorithm.TOPDOWN:
        anonymized_groups = k_anonymity_top_down(table_group, k)
    elif kp_algorithm == KPAlgorithm.BOTTOMUP:
        anonymized_groups = k_anonymity_bottom_up(table_group, k)

    for ag in anonymized_groups:
        print('after the k anonymization:', ag.shape(), '; company codes:', ag.ids)
    
    if show_plots:
        visualize_envelopes(anonymized_groups)

    print('--- final k-anonymized groups:', len(anonymized_groups))

    final_nodes: Dict[Group, List[Node]] = {}
    for ag in anonymized_groups:
        final_nodes[ag] = p_anonymity_naive(group=ag, p=p, max_level=max_level, PR_len=PR_len)
    print('--- final nodes:', sum(len(final_nodes[ag]) for ag in final_nodes))
    for i, ag in enumerate(final_nodes):
        print("Group {} nodes:".format(i))
        for node in final_nodes[ag]:
            print('Node {}: PR "{}", IDs {}'.format(node.id, node.pr, node.row_ids))
    
    nodes_list = []
    for g in final_nodes:
        nodes_list.extend(final_nodes[g])
    
    if show_plots:
        visualize_p_anonymized_nodes(nodes_list)


def kp_anonymity_kapra(table_group: Group, k: int, p: int, PR_len: int, max_level: int):
    p_anonymized_nodes: List[Node] = p_anonymity_kapra(group=table_group, p=p, max_level=max_level, PR_len=PR_len)
    verbose("P-anonymized nodes:")
    for node in p_anonymized_nodes:
        verbose('Node {}: size {}, PR "{}", IDs: {}'.format(node.id, node.size(), node.pr, node.row_ids))
    if show_plots:
        visualize_p_anonymized_nodes(p_anonymized_nodes)
    p_anonymized_groups: List[Group] = [node.to_group() for node in p_anonymized_nodes]
    final_group_list = kapra_group_formation(p_anonymized_groups, k, p)
    if show_plots:
        visualize_envelopes(final_group_list)
    return final_group_list


def do_kp_anonymity(path_to_file: str, k: int, p: int, PR_len: int, max_level: int, kp_algorithm: str):
    df = load_data_from_file(path_to_file)
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
        kp_anonymity_classic(table_group, k, p, PR_len, max_level, kp_algorithm)
    else:
        kp_anonymity_kapra(table_group, k, p, PR_len, max_level)

    # TODO: finish the QI and SD


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k-anonymity', required=True, type=int)
    parser.add_argument('-p', '--p-anonymity', required=True, type=int)
    parser.add_argument('-l', '--PR-length', required=False, type=int, default=4)
    parser.add_argument('-m', '--max-level', required=False, type=int, default=3)
    parser.add_argument('-s', '--show-plots', required=False, action='store_true')
    parser.add_argument('-i', '--input-file', required=False)
    parser.add_argument('-o', '--output-file', required=False)
    parser.add_argument('-a', '--algorithm', required=False, default='top-down')
    parser.add_argument('-v', '--verbose', required=False, action='store_true')
    args = vars(parser.parse_args())

    k = args['k_anonymity']
    p = args['p_anonymity']
    PR_len = args['PR_length']
    max_level = args['max_level']
    
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
    if args['verbose']:
        setverbose()
    else:
        unsetverbose()
    global show_plots
    show_plots = args['show_plots']

    return k, p, PR_len, max_level, algo, input_path, output_path


if __name__ == "__main__":
    k, p, PR_len, max_level, algo, input_path, output_path = parse_arguments()
    print("p-anonymity with the following parameters: k={}, p={}, PR_len={}, max_level={}, algo={}, input_path={},\
        output_path={}, verbose={}".format(
        k, p, PR_len, max_level, algo.value, input_path, output_path, getverbose()))
    if k < p:
        print("ERROR: k must be larger than P")
        exit()
    if k < 2 * p:
        print("WARNING: k should be at least 2*P in order to obtain meaningful results")
    verbose("Verbose output enabled")
    do_kp_anonymity(path_to_file='data/table.csv', k=k, p=p, PR_len=PR_len, max_level=max_level, kp_algorithm=algo)
