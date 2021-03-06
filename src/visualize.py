import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import List, Dict
import numpy as np
from saxpy.paa import paa


from group import Group
from node import Node


def visualize_all_companies(dataframe: pd.DataFrame):
    dataframe.plot()
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.yscale("log")
    plt.legend(loc='upper left', ncol=3, prop=fontP)
    return


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def visualize_envelopes(group_list: List[Group], algorithm_str: str):
    # https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
    # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    cmap = get_cmap(len(group_list) + 1)
    for group_i, group in enumerate(group_list):
        color = cmap(group_i)
        n = group.shape()[1]
        for i in range(group.size()):
            plt.plot(range(n), group.get_row_at_index(i), color=color)
        plt.fill_between(range(group.shape()[1]), group.get_maxes(), group.get_mins(), alpha=0.1, color=color)
    plt.title("k-group envelopes ({})".format(algorithm_str))
    plt.yscale("symlog")
    plt.show()
    return


def visualize_p_anonymized_nodes(nodes_list: List[Node]):
    # https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
    # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    
    pr_dict: Dict[str, int] = {}
    number_of_pr: int = 0
    for node in nodes_list:
        if node.pr not in pr_dict and node.pr != "a" * node.pr_len():
            pr_dict[node.pr] = number_of_pr
            number_of_pr += 1
    
    pr_cmap = get_cmap(len(pr_dict) + 1)

    n = len(nodes_list[0].table[0])
    if nodes_list[0].pr_len() != n:
        paa_linespace = np.linspace(0, n-1, (2 * nodes_list[0].pr_len() + 1))
        paa_positions = paa_linespace[1::2]
        for node in nodes_list:
            if node.pr != "a" * node.pr_len():
                node_color = pr_cmap(pr_dict[node.pr])
                marker_alpha = 1
                line_alpha = 0.4
            else:
                node_color = "grey"
                marker_alpha = 0.5
                line_alpha = 0.2
            for row in node.table:
                plt.plot(range(n), row, color=node_color, label=node.pr, alpha=line_alpha)
                plt.plot(paa_positions, paa(row, node.pr_len()), color=node_color, label="", linestyle='', marker="_", markeredgewidth=2, markersize=10, alpha=marker_alpha)
                plt.plot(paa_positions, paa(row, node.pr_len()), color=node_color, label="", linestyle=':', alpha=0.5)
    else:
        for node in nodes_list:
            if node.pr != "a" * node.pr_len():
                node_color = pr_cmap(pr_dict[node.pr])
                line_alpha = 1
            else:
                node_color = "grey"
                line_alpha = 0.5
            for row in node.table:
                plt.plot(range(n), row, color=node_color, label=node.pr, alpha=line_alpha)

    fontP = FontProperties()
    fontP.set_size('xx-small')

    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 'upper left', ncol = 3, prop = fontP)
    
    plt.title("p-anonymization")
    plt.yscale("symlog")
    plt.show()
    return


def visualize_performance(values: pd.DataFrame, title: str = "", x: str= "", y: str = "", labels=None, colormap="hsv"):
    if labels is None:
        labels = values.index
    cmap = get_cmap(len(values.index) + 1, colormap)
    for label, row in values.iterrows():
        i = values.index.get_loc(label)
        color = cmap(i)
        plt.plot(row, color=color, label=labels[i], marker="o", linestyle="-")
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(loc = 'upper left', prop = fontP)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    return
