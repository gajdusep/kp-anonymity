from operator import index
import time
import math
import sys

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from saxpy.paa import paa
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize

from group import Group, create_group_from_pandas_df
from kp_anonymity import kp_anonymity_kapra, kp_anonymity_classic, KPAlgorithm
from load_data import *
from p_anonymity import distance
from verbose import setverbose, setdebug
from visualize import visualize_performance
from save_anonymized_table import save_anonymized_table


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


def run_all_tests(path: str):

    # k_values = list(range(4,5))
    # p_values = list(range(2,3))

    # k_values = list(range(4,11))
    # p_values = list(range(2,6))

    k_values = [4, 5, 7, 10, 15, 22, 35, 50]
    p_values = [2, 3, 4, 6, 8, 10]

    k_values.sort()
    p_values.sort()
    pr_len = 5
    max_level = 7

    df = load_data_from_file(path)

    df = remove_outliers(df, max_stock_value=5000)

    group, sd_dict, col_labels = create_group_from_pandas_df(df)

    algorithms = [KPAlgorithm.KAPRA, KPAlgorithm.TOPDOWN, KPAlgorithm.BOTTOMUP]
    ivl_results: Dict[KPAlgorithm, pd.DataFrame] = {}
    pl_results: Dict[KPAlgorithm, pd.DataFrame] = {}
    times: Dict[KPAlgorithm, pd.DataFrame] = {}
    for a in algorithms:
        ivl_results[a] = pd.DataFrame(columns=k_values, index=p_values)
        pl_results[a] = pd.DataFrame(columns=k_values, index=p_values)
        times[a] = pd.DataFrame(columns=k_values, index=p_values)

    # setverose()
    # setdebug()
    for a in algorithms:
        print('----------- {} algorithm -----------'.format(a))
        for k in k_values:
            for p in p_values:
                if k < p:
                    ivl_results[a][k][p] = float("NaN")
                    pl_results[a][k][p] = float("NaN")
                    times[a][k][p] = float("NaN")
                    continue

                print('--- {},{}-anonymity:'.format(k, p))

                time_s = time.time()
                if a == KPAlgorithm.KAPRA:
                    anonymized = kp_anonymity_kapra(group, k, p, pr_len, max_level)
                elif a == KPAlgorithm.TOPDOWN:
                    anonymized = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.TOPDOWN)
                elif a == KPAlgorithm.BOTTOMUP:
                    anonymized = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.BOTTOMUP)
                else:
                    ivl_results[a][k][p] = float("NaN")
                    pl_results[a][k][p] = float("NaN")
                    times[a][k][p] = float("NaN")
                    continue
                save_anonymized_table("data/anonymized_{}-{}_{}".format(k, p, a), anonymized, sd_dict, col_labels)
                time_e = time.time() - time_s
                ivl_results[a][k][p] = instant_value_loss(anonymized)
                pl_results[a][k][p] = pattern_loss(anonymized)
                times[a][k][p] = time_e
    
    print('\n----------- Instant Value Loss -----------')
    for a in algorithms:
        print('\n----------- {} -----------'.format(a))
        print(ivl_results[a])
    print('\n----------- Pattern Loss -----------')
    for a in algorithms:
        print('\n----------- {} -----------'.format(a))
        print(pl_results[a])
    print('\n----------- Times (in seconds) -----------')
    for a in algorithms:
        print('\n----------- {} -----------'.format(a))
        print(times[a])

    ivl_k_a = pd.DataFrame(columns=k_values, index=algorithms)
    ivl_p_a = pd.DataFrame(columns=p_values, index=algorithms)
    pl_k_a = pd.DataFrame(columns=k_values, index=algorithms)
    pl_p_a = pd.DataFrame(columns=p_values, index=algorithms)
    times_k_a = pd.DataFrame(columns=k_values, index=algorithms)
    times_p_a = pd.DataFrame(columns=p_values, index=algorithms)
    
    # Choose the largest p value that is not greater than any k if it exists,
    # choose smallest p value otherwise
    set_p = p_values[0]
    for p in reversed(p_values):
        if p <= k_values[0]:
            set_p = p
    # Choose the largest k value for plots
    set_k = k_values[-1]

    for a in algorithms:
        for k in k_values:
            ivl_k_a[k][a] = ivl_results[a][k][set_p]
            pl_k_a[k][a] = pl_results[a][k][set_p]
            times_k_a[k][a] = times[a][k][set_p]
    for a in algorithms:
        for p in p_values:
            ivl_p_a[p][a] = ivl_results[a][set_k][p]
            pl_p_a[p][a] = pl_results[a][set_k][p]
            times_p_a[p][a] = times[a][set_k][p]
    
    plots_set_p = {
        "Instant Value Loss (P = {})".format(set_p): ivl_k_a,
        "Pattern Loss (P = {})".format(set_p): pl_k_a,
        "Execution Times (P = {})".format(set_p): times_k_a
    }
    for title, dataframe in plots_set_p.items():
        labels = [l.value for l in dataframe.index]
        visualize_performance(dataframe, title, x="k", labels=labels)

    plots_set_k = {
        "Instant Value Loss (k = {})".format(set_p): ivl_p_a,
        "Pattern Loss (k = {})".format(set_p): pl_p_a,
        "Execution Time (k = {})".format(set_p): times_p_a
    }
    for title, dataframe in plots_set_k.items():
        labels = [l.value for l in dataframe.index]
        visualize_performance(dataframe, title, x="P", labels=labels)
    
    for a in algorithms:
        labels = ["p = {}".format(l) for l in times[a].index]
        visualize_performance(times[a], "{} Algorithm Execution Time (in seconds)".format(a), x="k", labels=labels, colormap="winter")
    
    return


if __name__ == "__main__":
    run_all_tests(sys.argv[1])
