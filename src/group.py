
import numpy as np
import pandas as pd

from typing import Union, Tuple
from node import Node


class Group:

    def __init__(self, group_table: Union[np.ndarray, None]):
        self.group_table = group_table
    
    def add_row_to_group(self, row: np.ndarray):
        if self.group_table is None:
            self.group_table = row.reshape(1, row.shape[0])
        else:
            self.group_table = np.vstack([self.group_table, row])

    def get_row_at_index(self, index: int) -> np.ndarray:
        return self.group_table[index]

    def get_random_row(self) -> Tuple[int, np.ndarray]:
        pass

    def get_min_max(self) -> np.ndarray:
        table_maxs = np.max(self.group_table, axis=0)
        table_mins = np.min(self.group_table, axis=0)
        return table_maxs - table_mins

    def size(self):
        if self.group_table is None:
            return 0
        return self.group_table.shape[0]

    def shape(self) -> Tuple[int, int]:
        if self.group_table is None:
            return 0,0
        return self.group_table.shape

    def to_node(self) -> Node:
        # TODO: this Method should be implemented by Mattia, when he has the Node implementation
        pass


def create_empty_group() -> Group:
    return Group(group_table=None)


def create_group_from_pandas_df(df: pd.DataFrame) -> Group:
    df = df.transpose()
    return Group(group_table=df.values)
