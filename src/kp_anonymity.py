import time
import argparse
from enum import Enum
from typing import List, Dict

from load_data import *
from save_anonymized_table import save_anonymized_table
from visualize import visualize_envelopes, visualize_p_anonymized_nodes, visualize_all_companies
from group import Group, create_group_from_pandas_df
from node import Node
from k_anonymity import k_anonymity_top_down, kapra_group_formation, k_anonymity_bottom_up
from p_anonymity import p_anonymity_naive, p_anonymity_kapra
from verbose import *


class KPAlgorithm(str, Enum):
    TOPDOWN = 'top-down'
    BOTTOMUP = 'bottom-up'
    KAPRA = 'kapra'


show_plots = False


def kp_anonymity_classic(table_group: Group, k: int, p: int, PR_len: int, max_level: int,
                         kp_algorithm: str) -> List[Group]:

    if kp_algorithm == KPAlgorithm.TOPDOWN:
        anonymized_groups = k_anonymity_top_down(table_group, k)
    else:  # kp_algorithm == KPAlgorithm.BOTTOMUP - no other option should get here
        anonymized_groups = k_anonymity_bottom_up(table_group, k)

    verbose('--- ' + kp_algorithm + ' is finished. Obtained' + str(len(anonymized_groups)) + 'groups')
    for ag in anonymized_groups:
        verbose('   -- shape:' + str(ag.shape()) + str('; company codes:') + str(ag.ids))
    
    if show_plots:
        visualize_envelopes(anonymized_groups)

    final_nodes: Dict[Group, List[Node]] = {}
    for ag in anonymized_groups:
        final_nodes[ag] = p_anonymity_naive(group=ag, p=p, max_level=max_level, PR_len=PR_len)

    verbose('--- number of final nodes: {}'.format(sum(len(final_nodes[ag]) for ag in final_nodes)))
    for i, ag in enumerate(final_nodes):
        verbose("Group {} nodes: {} {}".format(i, ag.pr_values, ag.ids))
        id_to_pr_value_dict = {}
        for node in final_nodes[ag]:
            for row_id in node.row_ids:
                id_to_pr_value_dict[row_id] = (node.pr, node.level)
            verbose('   ' + str(node))
        for j, group_row_id in enumerate(ag.ids):
            ag.pr_values[j] = id_to_pr_value_dict[group_row_id]
        verbose("  -- Group pr values added: {}".format(ag.pr_values))
    
    nodes_list = []
    for g in final_nodes:
        nodes_list.extend(final_nodes[g])
    
    if show_plots:
        visualize_p_anonymized_nodes(nodes_list)

    return anonymized_groups


def kp_anonymity_kapra(table_group: Group, k: int, p: int, PR_len: int, max_level: int):
    p_anonymized_nodes: List[Node] = p_anonymity_kapra(group=table_group, p=p, max_level=max_level, PR_len=PR_len)

    verbose("--- kapra: P-anonymized nodes:")
    for node in p_anonymized_nodes:
        verbose('  {}'.format(node))
    if show_plots:
        visualize_p_anonymized_nodes(p_anonymized_nodes)

    p_anonymized_groups: List[Group] = [node.to_group() for node in p_anonymized_nodes]

    final_group_list = kapra_group_formation(p_anonymized_groups, k, p)
    verbose("--- kapra: k-anonymized groups:")
    for ag in final_group_list:
        verbose('  {}, {}'.format(ag.ids, ag.pr_values))

    if show_plots:
        visualize_envelopes(final_group_list)

    return final_group_list


def do_kp_anonymity(path_to_file: str, output_path: str, k: int, p: int, PR_len: int, max_level: int,
                    kp_algorithm: str):
    df = load_data_from_file(path_to_file)

    # visualize_all_companies(df)
    # df = remove_outliers(df, max_stock_value=5000)
    # visualize_all_companies(df)

    # UNCOMMENT IF YOU WANT TO SEE THE GRAPHS
    # plt.show()

    table_group, sd_dict, col_labels = create_group_from_pandas_df(df)
    print('Table created: {} {}\n-----------------'.format(table_group.shape(), table_group.ids))

    if kp_algorithm == KPAlgorithm.TOPDOWN or kp_algorithm == KPAlgorithm.BOTTOMUP:
        ag = kp_anonymity_classic(table_group, k, p, PR_len, max_level, kp_algorithm)
    else:
        ag = kp_anonymity_kapra(table_group, k, p, PR_len, max_level)

    # TODO: call some method to write into the output file
    save_anonymized_table(ag, sd_dict)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k-anonymity', required=True, type=int)
    parser.add_argument('-p', '--p-anonymity', required=True, type=int)
    parser.add_argument('-l', '--PR-length', required=False, type=int, default=5)
    parser.add_argument('-m', '--max-level', required=False, type=int, default=5)
    parser.add_argument('-s', '--show-plots', required=False, action='store_true')
    parser.add_argument('-i', '--input-file', required=True)
    parser.add_argument('-o', '--output-file', required=False)
    parser.add_argument('-a', '--algorithm', required=False, default='top-down')
    parser.add_argument('-v', '--verbose', required=False, action='store_true')
    parser.add_argument('-d', '--debug', required=False, action='store_true')
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
    
    if args['debug']:
        setdebug()
        setverbose()
    else:
        unsetdebug()
    
    global show_plots
    show_plots = args['show_plots']

    return k, p, PR_len, max_level, algo, input_path, output_path


if __name__ == "__main__":
    start_time = time.time()

    k, p, PR_len, max_level, algo, input_path, output_path = parse_arguments()

    print("\n-----------------\nkp-anonymity with the following parameters: \nk={} p={}\nPR_len={}\n"
          "max_level={}\nalgo={}\ninput_path={}\noutput_path={}\nverbose={}\n-----------------".format(
        k, p, PR_len, max_level, algo.value, input_path, output_path, getverbose()))
    if max_level > 19:
        print("ERROR: maximum supported PR level is 19 (saxpy library limitation)")
        exit()
    if k < p:
        print("ERROR: k must be larger than P")
        exit()
    if k < 2 * p:
        print("WARNING: k should be at least 2*P in order to obtain meaningful results")
    verbose("Verbose output enabled")
    debug("Debug output enabled")

    do_kp_anonymity(path_to_file=input_path, output_path=output_path,
                    k=k, p=p, PR_len=PR_len, max_level=max_level, kp_algorithm=algo)

    end_time = time.time() - start_time
    print("The program ran for: {} seconds".format(end_time))
