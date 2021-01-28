import time
import math
import numpy as np
import pandas as pd
import sys

from typing import List, Tuple

from saxpy.paa import paa
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize

from group import Group, create_group_from_pandas_df
from kp_anonymity import kp_anonymity_kapra, kp_anonymity_classic, KPAlgorithm
from load_data import *
from visualize import visualize_performance_pattern_loss, visualize_envelopes
from node import SAX
from p_anonymity import compute_pattern_similarity, distance
from verbose import setverbose, unsetverbose


def instant_value_loss(groups: List[Group]):
    return sum(group.instant_value_loss() for group in groups)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0

    return np.dot(a, b)/(norm_a*norm_b)


def gaussian(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * math.pow(x, 2))


def row_pattern_loss(row: np.ndarray, pr: Tuple[str, int]):
    pattern = []
    cuts = cuts_for_asize(pr[1] + 1)[1:]
    for c in pr[0]:
        n = ord(c) - 97
        pattern.append(cuts[n])
    if len(pattern) != len(row):
        normalized_row = paa(znorm(row), len(pattern))
    else:
        normalized_row = znorm(row)
    return distance(normalized_row, pattern)


def table_pattern_loss(table: np.ndarray, pr_list: List[Tuple[str, int]]):
    return sum(row_pattern_loss(row, pr_list[i]) for i, row in enumerate(table))


def pattern_loss(groups: List[Group]):
    return sum(table_pattern_loss(group.group_table, group.pr_values) for group in groups)


def table_pattern_diff(table: np.ndarray, pr_list: List[Tuple[str, int]], max_level: int):
    pattern_diff = 0
    for i, row in enumerate(table):
        pr = SAX(row, max_level, pr_list[i])
        pattern_diff += 1 - compute_pattern_similarity(pr, pr_list[i])
    return pattern_diff


def run_all_tests(path: str):

    # k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    # k_values = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    # k_values = [5]
    # k_values = [7, 8, 9, 10]
    k_values = [15, 20]

    # p_values = [2, 3, 4, 5]
    # p_values = [2, 3, 4, 5]
    # p_values = [2, 3, 4, 5]
    p_values = [5]
    pr_len = 4
    max_level = 3

    df = load_data_from_file(path)

    df = remove_outliers(df, max_stock_value=5000)

    group, _, _ = create_group_from_pandas_df(df)

    kapra_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    topdown_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    bottomup_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    kapra_pl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    topdown_pl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    bottomup_pl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)

    times = pd.DataFrame(columns=k_values, index=p_values)
    # setverbose()
    unsetverbose()
    for k in k_values:
        for p in p_values:
            if k < p:
                kapra_ivl_result_dataframe[k][p] = float("NaN")
                topdown_ivl_result_dataframe[k][p] = float("NaN")
                bottomup_ivl_result_dataframe[k][p] = float("NaN")
                kapra_pl_result_dataframe[k][p] = float("NaN")
                topdown_pl_result_dataframe[k][p] = float("NaN")
                bottomup_pl_result_dataframe[k][p] = float("NaN")
                continue

            print('--- {},{}-anonymity:'.format(k, p))

            kapra_time_s = time.time()
            anonymized_kapra = kp_anonymity_kapra(group, k, p, pr_len, max_level)
            kapra_time_e = time.time() - kapra_time_s
            kapra_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_kapra)
            kapra_pl_result_dataframe[k][p] = pattern_loss(anonymized_kapra)

            print('------------------------------------------------------')

            topdown_time_s = time.time()
            anonymized_topdown = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.TOPDOWN)
            topdown_time_e = time.time() - topdown_time_s
            topdown_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_topdown)
            topdown_pl_result_dataframe[k][p] = pattern_loss(anonymized_topdown)

            print('------------------------------------------------------')

            bottomup_time_s = time.time()
            anonymized_bottomup = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.BOTTOMUP)
            bottomup_time_e = time.time() - bottomup_time_s
            bottomup_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_bottomup)
            bottomup_pl_result_dataframe[k][p] = pattern_loss(anonymized_bottomup)

            print('------------------------------------------------------')

            times[k][p] = (round(kapra_time_e, 2), round(topdown_time_e, 2), round(bottomup_time_e, 2))
            # times[k][p] = (round(kapra_time_e, 2))
            # times[k][p] = (round(kapra_time_e, 2), round(topdown_time_e, 2))

    print('\n----------- Instant Value Loss -----------')
    print('\n----------- kapra -----------')
    print(kapra_ivl_result_dataframe)
    print('\n----------- top-down -----------')
    print(topdown_ivl_result_dataframe)
    print('\n----------- bottom-up -----------')
    print(bottomup_ivl_result_dataframe)
    print('\n----------- Pattern Loss -----------')
    print('\n----------- kapra -----------')
    print(kapra_pl_result_dataframe)
    print('\n----------- top-down -----------')
    print(topdown_pl_result_dataframe)
    print('\n----------- bottom-up -----------')
    print(bottomup_pl_result_dataframe)
    print('\n----------- Times -----------')
    print(times)

    # visualize_performance_pattern_loss(kapra_ivl_result_dataframe, bottomup_ivl_result_dataframe, topdown_ivl_result_dataframe, k=None, p=4)
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == "__main__":
    run_all_tests(sys.argv[1])
