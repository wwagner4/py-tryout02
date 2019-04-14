import itertools as it
from typing import List


def named_tuples():
    from collections import namedtuple
    H = namedtuple("H", "a b c")
    x = H(1, 2, True)
    print(x.a)
    print(x.b)
    print(x.c)


def all_combinations():
    def all_combis(li: List):
        for l in range(len(li)):
            for i in it.combinations(li, l + 1):
                yield list(i)

    # Print the obtained combinations
    _li = ["A", "B", "C"]
    ac = all_combis(_li)
    for c in ac:
        print(c)


f = "_".join(["A", "B", "C"])
print(f)

# all_combinations()

# named_tuples()
