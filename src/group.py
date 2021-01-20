import numpy as np
import pandas as pd
import random
from typing import Union, Tuple, List


class Group:

    def __init__(self, group_table: Union[np.ndarray, None], ids: List[str]):
        self.group_table = group_table
        self.ids = ids
    
    def add_row_to_group(self, row: np.ndarray, row_id: str = "no_id"):
        self.ids.append(row_id)
        if self.group_table is None:
            self.group_table = row.reshape(1, row.shape[0])
        else:
            self.group_table = np.vstack([self.group_table, row])

    def delete_last_added_row(self):
        if self.size() > 0:
            self.ids.pop()
            self.group_table = np.delete(self.group_table, -1, axis=0)

    def merge_group(self, group: 'Group'):
        """
        Adds the group to merge to this group.
        :param group: group that you want to merge
        """
        self.group_table = np.concatenate((self.group_table, group.group_table), axis=0)
        self.ids.extend(group.ids)

    @staticmethod
    def merge_two_groups(group1: 'Group', group2: 'Group') -> 'Group':
        """
        Merges two Groups into a new Group
        :param group1: first group
        :param group2: second group
        :return: merged group
        """
        new_group = Group(group1.group_table, group1.ids)
        new_group.merge_group(group2)
        return new_group

    def get_row_at_index(self, index: int) -> np.ndarray:
        return self.group_table[index]

    def get_row_id_at_index(self, index: int) -> str:
        return self.ids[index]

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


def create_empty_group() -> Group:
    return Group(group_table=None, ids=[])


def create_group_from_pandas_df(df: pd.DataFrame) -> Group:
    ids = list(df.columns)
    df = df.transpose()
    return Group(group_table=df.values, ids=ids)
