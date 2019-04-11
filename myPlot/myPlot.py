import math
from typing import Iterable, Tuple
import pandas as pd
import matplotlib.pyplot as plt

_data = pd.read_csv("../../data/bc-data.csv", header=0)

_col_groups = ["mean", "se", "worst"]

_diagnosis_name = "diagnosis"

_diagnosis_B = "B"
_diagnosis_M = "M"

_cols_base = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
              "concavity", "concave points", "symmetry", "fractal_dimension"]

_grps = ["mean", "worst", "se"]


def ij(size: int, cols: int) -> Iterable[Tuple[int, int, bool]]:
    rows = math.ceil(size / cols)

    def nj(n: int) -> Tuple[int, int, bool]:
        return math.floor(n / cols), n % cols, n < size

    return [nj(i) for i in range(0, rows * cols)]


def extract_lm(d: pd.DataFrame, col_nam: str) -> pd.DataFrame:
    d1 = d[[_diagnosis_name, col_nam]]
    return d1.pivot(columns=_diagnosis_name, values=col_nam)


def boxes(d: pd.DataFrame, grp: str, cols_base: Iterable[str]):
    col_nams = list(map(lambda s: "{}_{}".format(s, grp), cols_base))
    fs = 10  # fontsize
    _n = len(col_nams)
    col = 4
    row = math.ceil(_n / col)
    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(10, 8))
    _ij = ij(row, col)
    k = 0
    for _i, _j, vis in ij(_n, col):
        if vis:
            cnam = col_nams[k]
            data1 = extract_lm(d, cnam)
            db = data1[[_diagnosis_B]]
            dm = data1[[_diagnosis_M]]
            data2 = [db[db[_diagnosis_B].notna()][_diagnosis_B].values,
                     dm[dm[_diagnosis_M].notna()][_diagnosis_M].values]
            axes[_i, _j].boxplot(data2, labels=[_diagnosis_B, _diagnosis_M])
            axes[_i, _j].set_title(cnam, fontsize=fs)
        else:
            axes[_i, _j].axis('off')
        k += 1
    plt.show()


for grp in _grps:
    boxes(_data, grp, _cols_base)
