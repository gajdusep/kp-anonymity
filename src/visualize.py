import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from group import *


def visualize_all_companies(dataframe: pd.DataFrame):
    dataframe.plot()
    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    plt.legend(loc='upper left', ncol=3, prop=fontP)


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def visualize_intervals(group_list: List[Group]):
    # https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
    # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    cmap = get_cmap(len(group_list) + 1)
    for group_i, group in enumerate(group_list):
        color = cmap(group_i)
        n = group.shape()[1]
        for i in range(group.size()):
            plt.plot(range(n), group.get_row_at_index(i), color=color)
        plt.fill_between(range(group.shape()[1]), group.get_maxes(), group.get_mins(), alpha=0.1, color=color)
    plt.show()
