import numpy as np
from pandas.core.frame import DataFrame
from group import Group
from typing import List, Dict
import pandas as pd
import csv 

def save_anonymized_table(ag: List[Group], sd_dict: Dict[str,float], col_labels: List[str]):
    anonymized_table = []
    id = 1
    for group in ag:
        group_intervals = group.get_group_intervals()
        for i in range(group.size()):
            tuple = [id]
            tuple.extend(group_intervals)
            tuple.append(group.pr_values[i][0])
            tuple.append(sd_dict[group.ids[i]])
            anonymized_table.append(tuple)
            id += 1

    df = pd.DataFrame(anonymized_table)
    col_labels.insert(0, 'Time')
    col_labels.pop()
    col_labels.append('Pr_value')
    col_labels.append('SD')
    df.to_csv('data/anonymized_table.csv', index = False, header = col_labels, quotechar=' ')       
        



    print(anonymized_table)

