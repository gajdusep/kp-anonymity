import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from typing import List, Dict
import numpy as np

from group import Group
from node import Node


def visualize_all_companies(dataframe: pd.DataFrame):
    dataframe.plot()
    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    plt.legend(loc='upper left', ncol=3, prop=fontP)
    return


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def visualize_envelopes(group_list: List[Group]):
    # https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
    # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    cmap = get_cmap(len(group_list) + 1)
    for group_i, group in enumerate(group_list):
        color = cmap(group_i)
        n = group.shape()[1]
        for i in range(group.size()):
            plt.plot(range(n), group.get_row_at_index(i), color=color)
        plt.fill_between(range(group.shape()[1]), group.get_maxes(), group.get_mins(), alpha=0.1, color=color)
    plt.title("k-group envelopes")
    plt.yscale("log")
    plt.show()


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
    for node in nodes_list:
        if node.pr != "a" * node.pr_len():
            node_color = pr_cmap(pr_dict[node.pr])
        else:
            node_color = "grey"
        for row in node.table:
            plt.plot(range(n), row, color=node_color, label=node.pr)
    
    fontP = FontProperties()
    fontP.set_size('xx-small')

    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 'upper left', ncol = 3, prop = fontP)
    
    plt.title("p-anonymization")
    plt.yscale("log")
    plt.show()

    """
    def visualize_p_anonymized_nodes_old(nodes_dict: Dict[Group, List[Node]]):
    # https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
    # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    
    pr_dict: Dict[str, int] = {}
    number_of_pr: int = 0
    for group in nodes_dict:
        for node in nodes_dict[group]:
            if node.pr not in pr_dict and node.pr != "a" * node.pr_len():
                pr_dict[node.pr] = number_of_pr
                number_of_pr += 1
    
    # group_cmap = plt.cm.get_cmap("hsv", len(nodes_dict) + 1)
    pr_cmap = plt.cm.get_cmap("hsv", len(pr_dict) + 1)
    for i, group in enumerate(nodes_dict):
        n = group.shape()[1]
        for node in nodes_dict[group]:
            if node.pr != "a" * node.pr_len():
                node_color = pr_cmap(pr_dict[node.pr])
            else:
                node_color = "grey"
            for row in node.table:
                plt.plot(range(n), row, color=node_color, label=node.pr)
        
        # group_color = group_cmap(i)
        # plt.fill_between(range(n), group.get_maxes(), group.get_mins(), alpha=0.1, color=group_color)
    
    fontP = FontProperties()
    fontP.set_size('xx-small')

    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 'upper left', ncol = 3, prop = fontP)
    plt.title("p-anonymization")
    plt.yscale("log")
    plt.show()
    """


def visualize_performance_times(times_df):
    pass


def visualize_performance_ivl(instant_value_losses):
    pass


def visualize_performance_pattern_loss(kapra_pl: pd.DataFrame, bottom_up_pl: pd.DataFrame, top_down_pl: pd.DataFrame,
                                       k=None, p=None):
    if p is None and k is None:
        return

    names = ["kapra", "bottom-up", "top-down"]

    if k is not None:
        kapra_l = list(kapra_pl[k])
        bottom_up_l = list(bottom_up_pl[k])
        top_down_l = list(top_down_pl[k])
        plt.xticks(kapra_pl.index)
        plt.plot(np.array([kapra_l, bottom_up_l, top_down_l]).transpose())

    if p is not None:
        kapra_l = list(kapra_pl.loc[p])
        bottom_up_l = list(bottom_up_pl.loc[p])
        top_down_l = list(top_down_pl.loc[p])

        plt.title("Instant value loss, p={}".format(p))
        plt.xticks(range(len(kapra_pl.columns)), kapra_pl.columns)
        plt.xlabel("k")
        plt.ylabel("instant value loss")
        plt.plot(kapra_l, label=names[0])
        plt.plot(bottom_up_l, label=names[1])
        plt.plot(top_down_l, label=names[2])
        plt.legend()


def visualize_performance_results():
    pass
