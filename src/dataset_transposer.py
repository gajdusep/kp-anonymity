import sys
import pandas as pd
from pandas.core.frame import DataFrame

def transpose_csv(path_to_file: str):
    df: DataFrame = pd.read_csv(path_to_file, index_col = False)
    df.T.to_csv(path_to_file + "_transposed.csv", header = False)

if __name__ == "__main__":
    transpose_csv(sys.argv[1])