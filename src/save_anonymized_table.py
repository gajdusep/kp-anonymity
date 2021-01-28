import numpy as np
from group import Group
from typing import List, Dict

def save_anonymized_table(ag: List[Group], sd_dict: Dict[str,float]):
    anonymized_table = []
    for group in ag:
        group_intervals = group.get_group_intervals()
        for i in range(group.size()):
            tuple = np.array([i+1 , group_intervals, group.pr_values[i][0], sd_dict[group.ids[i]]])
            anonymized_table.append(tuple)
        
    print(anonymized_table)

