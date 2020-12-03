import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def visualize_all_companies(dataframe: pd.DataFrame):
    dataframe.plot()
    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    plt.legend(loc='upper left', ncol=3, prop=fontP)    


# for the anonymized result with intervals, we might use this:
# https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
