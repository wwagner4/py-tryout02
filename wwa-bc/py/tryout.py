from itertools import combinations
from typing import List


def named_tuples():
    from collections import namedtuple
    H = namedtuple("H", "a b c")
    x = H(1, 2, True)
    print(x.a)
    print(x.b)
    print(x.c)


def all_combinations():
    def all_combis(li: List[str]):
        for l in range(len(li)):
            for i in combinations(li, l + 1):
                yield list(i)

    # Print the obtained combinations
    _li = ["A", "B", "C"]
    ac = all_combis(_li)
    for c in ac:
        print(c)


all_combinations()

# named_tuples()
