import time

from typing import List
import pandas as pd

from group import Group, create_group_from_pandas_df
from kp_anonymity import kp_anonymity_kapra, kp_anonymity_classic, KPAlgorithm
from load_data import *


def instant_value_loss(groups: List[Group]):
    return sum(group.instant_value_loss() for group in groups)


def run_all_tests():

    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    # k_values = [9, 10]
    p_values = [3, 4, 5]
    # p_values = [2, 3, 4, 5]
    # p_values = [3, 4]
    pr_len = 4
    max_level = 3
    path_to_file = "data/table.csv"

    df = load_data_from_file(path_to_file)
    df = remove_rows_with_nan(df)
    df = remove_outliers(df, max_stock_value=5000)
    # df = reduce_dataframe(df, companies_count=30)

    group = create_group_from_pandas_df(df)

    kapra_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    topdown_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    bottomup_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)

    times = pd.DataFrame(columns=k_values, index=p_values)

    for k in k_values:
        for p in p_values:
            # TODO: what exactly is: 2 rows were suppressed: they could not be merged.
            #         TODO: -can it be removed / put to verbose..?
            # TODO: if line 24 is commented, it crashes.. why..?
            if k < p:
                kapra_ivl_result_dataframe[k][p] = -1
                topdown_ivl_result_dataframe[k][p] = -1
                bottomup_ivl_result_dataframe[k][p] = -1
                continue

            print('--- {},{}-anonymity:'.format(k, p))

            kapra_time_s = time.time()
            # anonymized_kapra = kp_anonymity_kapra(group, k, p, pr_len, max_level)
            # kapra_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_kapra)
            kapra_time_e = time.time() - kapra_time_s

            topdown_time_s = time.time()
            # anonymized_topdown = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.TOPDOWN)
            # topdown_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_topdown)
            topdown_time_e = time.time() - topdown_time_s

            bottomup_time_s = time.time()
            anonymized_bottomup = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.BOTTOMUP)
            bottomup_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_bottomup)
            bottomup_time_e = time.time() - bottomup_time_s

            times[k][p] = (round(kapra_time_e, 2), round(topdown_time_e, 2), round(bottomup_time_e, 2))

    print('\n----------- kapra -----------')
    print(kapra_ivl_result_dataframe)
    print('\n----------- top-down -----------')
    print(topdown_ivl_result_dataframe)
    print('\n----------- bottom-up -----------')
    print(bottomup_ivl_result_dataframe)
    print('\n----------- times -----------')
    print(times)


if __name__ == "__main__":
    run_all_tests()
