
import numpy as np
import pandas as pd

class Group:

    def __init__(self, group_df: pd.DataFrame):
        self.group_df = group_df
        self.table_np = 
    
    def prepare_table_np():
        return group_df.to_numpy()


def compute_ncp(tuples: np.array, min_max_diff: np.array) -> float:
    """
    tuples: e.g. np.array([[1,2,5,2],[3,2,5,4],[2,2,0,5]])
        It means: 4 attributes, 3 tuples.
        Will be generalized into:
        [(1,3), (2,2), (0,5), (2,5)]
    min_max_diff: e.g. np.array([20,4,10,5])
        Therefore ncp: 
        3*((3-1)/20 + (2-2)/4 + (5-0)/10 + (5-2)/5) = 3*(0.1+0+0.5+0.6) = 3.6
    """

    z = np.max(tuples, axis=0)
    y = np.min(tuples, axis=0)
    zy_diff = z - y
    ncp = np.sum(zy_diff / min_max_diff)
    return tuples.shape[0] * ncp


tuples = np.array([[1,2,5,2],[3,2,5,4],[2,2,0,5]])
min_max_diff = np.array([20,4,10,5])
print(compute_ncp(tuples, min_max_diff))
