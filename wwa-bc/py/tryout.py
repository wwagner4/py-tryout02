import itertools as it
from typing import List, Iterable
from sklearn import preprocessing as pp
import pandas as pd


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


def norm():

    def extract_features(d: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        return d[features]

    data = pd.read_csv('../data/bc-data.csv', header=0)
    cols = ["radius_mean", "radius_se"]
    d = extract_features(data, cols)

    d1 = pp.normalize(d)
    print(d1)
    x = pd.DataFrame(d1, columns=cols)
    print(x)


norm()

# all_combinations()

# named_tuples()
