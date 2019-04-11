import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def hm(data: pd.DataFrame, grp: str):
    cols_mean = list(map(lambda s: s + "_" + grp, cols_base))
    d1 = data[cols_mean]
    sns.set(style="ticks", color_codes=True)
    plt.figure(figsize=(10, 9))
    sns.heatmap(d1.astype(float).corr(), linewidths=0.1, square=True, linecolor="white", annot=True)
    plt.show()


_data = pd.read_csv("../../data/bc-data.csv", header=0)

col_groups = ["mean", "se", "worst"]

cols_base = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
             "concavity", "concave points", "symmetry", "fractal_dimension"]

for grp in col_groups:
    hm(_data, grp)