import math
from typing import Iterable, Tuple


def ij(size: int, cols: int) -> Iterable[Tuple[int, int, bool]]:
    rows = math.ceil(size / cols)

    def nj(n: int) -> Tuple[int, int, bool]:
        return math.floor(n / cols), n % cols, n < size

    return [nj(i) for i in range(0, rows * cols)]


_iter = ij(5, 2)

print(list(_iter))
