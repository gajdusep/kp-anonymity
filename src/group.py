import numpy as np
import pandas as pd
import random
import copy

from typing import Union, Tuple, List, Dict

from verbose import debug


class Group:

    def __init__(self, group_table: Union[np.ndarray, None], ids: List[str], pr_values: List[Tuple[str, int]] = []):
        self.group_table = group_table
        self.ids = ids
        self.pr_values = pr_values
    
    def add_row_to_group(self, row: np.ndarray, row_id: str = "no_id", pr_value: Tuple[str, int] = ("no_pr", 0)):
        self.ids.append(row_id)
        self.pr_values.append(pr_value)
        if self.group_table is None:
            self.group_table = row.reshape(1, row.shape[0])
        else:
            self.group_table = np.vstack([self.group_table, row])

    def delete_last_added_row(self):
        if self.size() > 0:
            self.ids.pop()
            self.pr_values.pop()
            self.group_table = np.delete(self.group_table, -1, axis=0)

    def pop_row(self, index) -> Union[Tuple[np.ndarray, str, Tuple[str, int]], None]:
        if self.size() > 0:
            popped_id = self.ids.pop(index)
            popped_pr_value = self.pr_values.pop(index)
            popped_row = self.group_table[index]
            self.group_table = np.delete(self.group_table, index, axis=0)

            return popped_row, popped_id, popped_pr_value
        return None

    def merge_group(self, group: 'Group'):
        """
        Adds the group to merge to this group.
        :param group: group that you want to merge
        """
        self.group_table = np.concatenate((self.group_table, group.group_table), axis=0)
        self.ids.extend(group.ids)
        self.pr_values.extend(group.pr_values)

    @staticmethod
    def merge_two_groups(group1: 'Group', group2: 'Group') -> 'Group':
        """
        Merges two Groups into a new Group
        :param group1: first group
        :param group2: second group
        :return: merged group
        """
        group_table_copy = copy.deepcopy(group1.group_table)
        group_ids_copy = copy.deepcopy(group1.ids)
        group_pr_copy = copy.deepcopy(group1.pr_values)
        new_group = Group(group_table_copy, group_ids_copy, group_pr_copy)
        new_group.merge_group(group2)
        return new_group

    def get_row_at_index(self, index: int) -> np.ndarray:
        return self.group_table[index]

    def get_row_id_at_index(self, index: int) -> str:
        return self.ids[index]

    def get_pr_value_at_index(self, index: int) -> Tuple[str, int]:
        return self.pr_values[index]

    def get_all_attrs_at_index(self, index: int) -> Tuple[np.ndarray, str, Tuple[str, int]]:
        return self.group_table[index], self.ids[index], self.pr_values[index]

    def get_random_row(self) -> Tuple[int, np.ndarray]:
        """
        :return: index of the row, row
        """
        i = random.randint(0, self.size() - 1)
        return i, self.get_row_at_index(i)

    def get_maxes(self) -> np.ndarray:
        return np.max(self.group_table, axis=0)

    def get_mins(self) -> np.ndarray:
        return np.min(self.group_table, axis=0)

    def get_min_max_diff(self) -> np.ndarray:
        table_maxs = self.get_maxes()
        table_mins = self.get_mins()
        return table_maxs - table_mins

    def instant_value_loss(self) -> float:
        return np.sqrt(np.sum(self.get_min_max_diff()) / self.shape()[1])

    def get_group_intervals(self):
        table_maxs = self.get_maxes()
        table_mins = self.get_mins()
        return list(zip(table_mins, table_maxs))

    def size(self):
        if self.group_table is None:
            return 0
        return self.group_table.shape[0]

    def shape(self) -> Tuple[int, int]:
        if self.group_table is None:
            return 0, 0
        return self.group_table.shape

    def __str__(self):
        return str(self.ids)

    def __repr__(self):
        return str(self)


def create_empty_group() -> Group:
    return Group(group_table=None, ids=[], pr_values=[])


def create_group_from_pandas_df(df: pd.DataFrame) -> Tuple[Group, Dict[str, float], List[str]]:
    group_table = df.iloc[:, :-1].to_numpy()
    col_labels = list(df.columns.values)
    ids = list(df.index.values)
    sd = df.iloc[:, -1].tolist()

    debug("cols: " + str(col_labels))
    debug(len(col_labels))
    debug("ids: " + str(ids))
    debug(len(ids))
    debug("SD:" + str(sd))
    debug(len(sd))
    debug("table:\n" + str(group_table))
    debug(group_table.shape)
    
    sd_dict = {}
    for i, id in enumerate(ids):
        sd_dict[id] = sd[i]

    return Group(group_table=group_table, ids=ids, pr_values=[("no_pr", 0)]*len(ids)), sd_dict, col_labels
