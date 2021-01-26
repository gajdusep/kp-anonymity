from typing import List
import pandas as pd

from group import Group, create_group_from_pandas_df
from kp_anonymity import kp_anonymity_kapra, kp_anonymity_classic, KPAlgorithm
from load_data import *


def instant_value_loss(groups: List[Group]):
    return sum(group.instant_value_loss() for group in groups)


def run_all_tests():

    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    p_values = [2, 3, 4, 5]
    pr_len = 4
    max_level = 3
    path_to_file = "data/table.csv"

    df = load_data_from_file(path_to_file)
    df = remove_rows_with_nan(df)
    df = remove_outliers(df, max_stock_value=5000)
    df = reduce_dataframe(df, companies_count=30)

    group = create_group_from_pandas_df(df)

    kapra_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    topdown_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)
    bottomup_ivl_result_dataframe = pd.DataFrame(columns=k_values, index=p_values)

    for k in k_values:
        for p in p_values:
            # TODO: discuss: if this condition is not here, it crashes...
            # TODO: what exactly is: 2 rows were suppressed: they could not be merged.
            #         TODO: -can it be removed / put to verbose..?
            if k < p:
                kapra_ivl_result_dataframe[k][p] = -1
                topdown_ivl_result_dataframe[k][p] = -1
                bottomup_ivl_result_dataframe[k][p] = -1
                continue

            print('--- {},{}-anonymity:'.format(k, p))
            anonymized_kapra = kp_anonymity_kapra(group, k, p, pr_len, max_level)
            kapra_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_kapra)

            anonymized_topdown = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.TOPDOWN)
            topdown_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_topdown)

            anonymized_bottomup = kp_anonymity_classic(group, k, p, pr_len, max_level, KPAlgorithm.BOTTOMUP)
            bottomup_ivl_result_dataframe[k][p] = instant_value_loss(anonymized_bottomup)

    print('----------- kapra -----------')
    print(kapra_ivl_result_dataframe)
    print('----------- top-down -----------')
    print(topdown_ivl_result_dataframe)
    print('----------- bottom-up -----------')
    print(bottomup_ivl_result_dataframe)


if __name__ == "__main__":
    run_all_tests()
