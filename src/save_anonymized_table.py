from group import Group
from typing import List, Dict
import pandas as pd
import csv


def save_anonymized_table(output_path: str, ag: List[Group], sd_dict: Dict[str, float], col_labels: List[str]):
    anonymized_table = []
    id = 1
    for group in ag:
        group_intervals = group.get_group_intervals()
        for i in range(group.size()):
            tuple = [id]
            tuple.extend("({};{})".format(gi[0], gi[1]) for gi in group_intervals)
            tuple.append(group.pr_values[i][0])
            tuple.append(sd_dict[group.ids[i]])
            anonymized_table.append(tuple)
            id += 1

    df = pd.DataFrame(anonymized_table)
    col_labels_copy = col_labels.copy()
    col_labels_copy.insert(0, 'Time')
    SD_time = col_labels_copy.pop()
    col_labels_copy.append('Pr_value')
    col_labels_copy.append(SD_time)
    df.to_csv(output_path, index=False, header=col_labels_copy, sep=',', quoting=csv.QUOTE_NONE)

    # print(anonymized_table)
