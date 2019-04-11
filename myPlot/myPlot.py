import math
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ij(size: int, cols: int) -> Iterable[Tuple[int, int, bool]]:
    rows = math.ceil(size / cols)

    def nj(n: int) -> Tuple[int, int, bool]:
        return math.floor(n / cols), n % cols, n < size

    return [nj(i) for i in range(0, rows * cols)]


np.random.seed(19680801)
data = np.random.normal(size=(1000, 4))

labels = list('ABCD')
fs = 10  # fontsize

_n = 10
col = 3
row = math.ceil(_n / col)

fig, axes = plt.subplots(nrows=row, ncols=col)
_ij = ij(row, col)

for i, j, vis in ij(_n, col):
    if vis:
        axes[i, j].boxplot(data, labels=labels)
        axes[i, j].set_title('Default', fontsize=fs)
    else:
        axes[i, j].axis('off')

fig.subplots_adjust(hspace=0.4)

plt.show()
