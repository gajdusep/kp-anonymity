import math
import numpy as np
import random

from itertools import combinations
from typing import Tuple, List

from group import Group, create_empty_group


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
    min_max_diff_g = G.get_min_max_diff()

    for _ in range(3):
        max_ncp = -math.inf
        for i in range(size):
            if i == index_u:
                continue
            v = G.get_row_at_index(i)
            tmp = create_empty_group()
            tmp.add_row_to_group(u_max)
            tmp.add_row_to_group(v)
            ncp = compute_ncp(tmp.group_table, min_max_diff_g)
            if ncp > max_ncp:
                max_ncp = ncp
                v_max = v
                index_v = i

        max_ncp = -math.inf
        for i in range(size):
            if i == index_v:
                continue
            u = G.get_row_at_index(i)
            tmp = create_empty_group()
            tmp.add_row_to_group(v_max)
            tmp.add_row_to_group(u)
            ncp = compute_ncp(tmp.group_table, min_max_diff_g)
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

    min_max_diff_g = G.get_min_max_diff()

    (index_u, index_v) = get_init_tuples_uv(G)
    u_max = G.get_row_at_index(index_u)
    u_max_id = G.get_row_id_at_index(index_u)
    v_max = G.get_row_at_index(index_v)
    v_max_id = G.get_row_id_at_index(index_v)

    Gu.add_row_to_group(u_max, u_max_id)
    Gv.add_row_to_group(v_max, v_max_id)

    for i in random.sample(range(size), size):
        if i != index_u and i != index_v:
            w = G.get_row_at_index(i)
            w_id = G.get_row_id_at_index(i)

            Gu.add_row_to_group(w)
            ncp_Gu = compute_ncp(Gu.group_table, min_max_diff_g)
            Gu.delete_last_added_row()

            Gv.add_row_to_group(w)
            ncp_Gv = compute_ncp(Gv.group_table, min_max_diff_g)
            Gv.delete_last_added_row()

            if ncp_Gu < ncp_Gv:
                Gu.add_row_to_group(w, w_id)
            else:
                Gv.add_row_to_group(w, w_id)

    return [Gu, Gv]


def k_anonymity_top_down_postprocessing(less_than_k_anonymized_groups, k_or_more_anonymized_groups, min_max_diff_g, k):
    while len(less_than_k_anonymized_groups) > 0:
        group_to_anonymize = less_than_k_anonymized_groups.pop(0)
        size = group_to_anonymize.size()

        min_ncp_small_groups = math.inf
        min_ncp_small_groups_object = None
        min_ncp_small_groups_index = -1
        for i, group in enumerate(less_than_k_anonymized_groups):
            trial_group = Group.merge_two_groups(group_to_anonymize, group)
            ncp = compute_ncp(trial_group.group_table, min_max_diff_g)
            if ncp < min_ncp_small_groups:
                min_ncp_small_groups = ncp
                min_ncp_small_groups_object = group
                min_ncp_small_groups_index = i

        min_ncp_more_than_k_groups = math.inf
        min_ncp_more_than_k_groups_object = None
        for i, group in enumerate(k_or_more_anonymized_groups):
            trial_group = Group.merge_two_groups(group_to_anonymize, group)
            ncp = compute_ncp(trial_group.group_table, min_max_diff_g)
            if ncp < min_ncp_more_than_k_groups:
                min_ncp_more_than_k_groups = ncp
                min_ncp_more_than_k_groups_object = group

        min_ncp_was_in_less_than_k = False
        min_ncp_group = None
        min_ncp = math.inf
        if min_ncp_small_groups < min_ncp_more_than_k_groups:
            min_ncp_was_in_less_than_k = True
            min_ncp = min_ncp_small_groups
            min_ncp_group = min_ncp_small_groups_object
        else:
            min_ncp = min_ncp_more_than_k_groups
            min_ncp_group = min_ncp_more_than_k_groups_object

        # ----------------------------------------------------
        # ----- ALTERNATIVE 1 (first, we can find a set G' blablabla)
        # ----------------------------------------------------
        # find all group with more than 2k-size tuple and put them in "groups with more than 2k-size"
        groups_with_more_than_2k_minus_size = [g for g in k_or_more_anonymized_groups if g.size() >= 2 * k - size]
        min_ncp_alt1 = math.inf
        min_ncp_alt1_group_index = -1
        min_subgroup_indexes = None
        if len(groups_with_more_than_2k_minus_size) > 0:
            for index_chosen_group in range(len(groups_with_more_than_2k_minus_size)):
                chosen_group = groups_with_more_than_2k_minus_size[index_chosen_group]

                indexes = list(range(chosen_group.size()))
                # combination of index without repetition in order to get all possible k-size subsets of group
                subgroup_combinations = list(combinations(indexes, k - size))

                for combination in subgroup_combinations:
                    tmp = create_empty_group()
                    for j in range(k - size):
                        r = chosen_group.get_row_at_index(combination[j])
                        tmp.add_row_to_group(r)

                    trial_group = Group.merge_two_groups(group_to_anonymize, tmp)
                    ncp = compute_ncp(trial_group.group_table, min_max_diff_g)

                    if ncp < min_ncp_alt1:
                        min_ncp_alt1_group_index = index_chosen_group
                        min_ncp_alt1 = ncp
                        min_subgroup_indexes = combination

        if min_ncp_alt1 < min_ncp:
            print('   -- group:', group_to_anonymize.ids, ' - alternative 1 happened', min_subgroup_indexes)
            # THE ALTERNATIVE 1 IS BETTER

            # merge the subgroup that we obtained in the alternative 1
            group_to_pop_tuples = groups_with_more_than_2k_minus_size[min_ncp_alt1_group_index]
            for combination_index in min_subgroup_indexes:
                row, row_id = group_to_pop_tuples.pop(combination_index)
                group_to_anonymize.add_row_to_group(row, row_id)
            k_or_more_anonymized_groups.append(group_to_anonymize)
        else:
            # THE ALTERNATIVE 2 IS BETTER

            if min_ncp_was_in_less_than_k:
                print('   -- group:', group_to_anonymize.ids, ' - merged with a small group')
                less_than_k_anonymized_groups[min_ncp_small_groups_index].merge_group(group_to_anonymize)
                if less_than_k_anonymized_groups[min_ncp_small_groups_index].size() >= k:
                    k_or_more_anonymized_groups.append(less_than_k_anonymized_groups.pop(min_ncp_small_groups_index))
            else:
                print('   -- group:', group_to_anonymize.ids, ' - merged with a bigger')
                min_ncp_group.merge_group(group_to_anonymize)


def k_anonymity_top_down(table_group: Group, k: int) -> List[Group]:
    print('table group:', table_group.ids)

    if table_group.size() <= k:
        return [table_group]

    groups_to_anonymize = [table_group]
    less_than_k_anonymized_groups = []
    k_or_more_anonymized_groups = []
    min_max_diff_g = table_group.get_min_max_diff()

    while len(groups_to_anonymize) > 0:
        group_to_anonymize = groups_to_anonymize.pop(0)
        group_list = group_partition(group_to_anonymize, k)

        both_have_less = True
        for group in group_list:
            if group.size() >= k:
                both_have_less = False
        if not both_have_less:
            for group in group_list:
                if group.size() > k:
                    groups_to_anonymize.append(group)
                elif group.size() == k:
                    k_or_more_anonymized_groups.append(group)
                else:
                    less_than_k_anonymized_groups.append(group)
        else:
            k_or_more_anonymized_groups.append(group_to_anonymize)

    for ag in k_or_more_anonymized_groups:
        print('k or more:', ag.shape(), '; company codes:', ag.ids)
    for ag in less_than_k_anonymized_groups:
        print('less than k:', ag.shape(), '; company codes:', ag.ids)

    k_anonymity_top_down_postprocessing(less_than_k_anonymized_groups, k_or_more_anonymized_groups, min_max_diff_g, k)

    return k_or_more_anonymized_groups
