from typing import Iterable, Tuple
import math
import pandas as pd
import matplotlib.pyplot as plt

C_B = "B"
C_M = "M"


def ij(size: int, cols: int) -> Iterable[Tuple[int, int, bool]]:
    rows = int(math.ceil(size / cols))

    def nj(n: int) -> Tuple[int, int, bool]:
        return int(math.floor(n / cols)), n % cols, n < size

    return [nj(i) for i in range(0, rows * cols)]


def extract_lm(d: pd.DataFrame, col_nam: str) -> pd.DataFrame:
    d1 = d[[_diagnosis_nam, col_nam]]
    return d1.pivot(columns=_diagnosis_nam, values=col_nam)


def boxes(d: pd.DataFrame, grp: str, cols_base: Iterable[str]):
    col_nams = list(map(lambda s: "{}_{}".format(s, grp), cols_base))
    col_len = len(col_nams)
    col = 4
    row = int(math.ceil(col_len / col))
    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(12, 10))
    k = 0
    for _i, _j, vis in ij(col_len, col):
        if vis:
            cnam = col_nams[k]
            data1 = extract_lm(d, cnam)
            db = data1[[C_B]]
            dm = data1[[C_M]]
            data2 = [db[db[C_B].notna()][C_B].values,
                     dm[dm[C_M].notna()][C_M].values]
            axes[_i, _j].boxplot(data2, labels=[C_B, C_M])
            axes[_i, _j].set_title(cnam)
        else:
            axes[_i, _j].axis('off')
        k += 1
    plt.show()


_values = ["id",
           "diagnosis"]

_id_nam = _values[0]
_diagnosis_nam = _values[1]

_data_groups = ["mean",
                "se",
                "worst"]

_data = pd.read_csv('../data/bc-data.csv', header=0)

_features_base = ["radius",
                  "texture",
                  "perimeter",
                  "area",
                  "smoothness",
                  "compactness",
                  "concavity",
                  "concave_points",
                  "symmetry",
                  "fractal_dimension"]

for _grp in _data_groups:
    boxes(_data, _grp, _features_base)
