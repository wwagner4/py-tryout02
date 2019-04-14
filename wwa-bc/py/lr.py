import itertools as it
from typing import Dict, List, Iterable, Tuple

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
# from sklearn.lda import LDA # for older versions
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
import sklearn.model_selection as ms


def all_combis(li: List) -> Iterable[List]:
    for l in range(len(li)):
        for i in it.combinations(li, l + 1):
            yield i


def extract_features(d: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    return d[features]


def extract_labels(d: pd.DataFrame):
    dia = d[['diagnosis']]
    return dia.values.ravel()


def train_fs(data: pd.DataFrame, features_key: str, features: Iterable[str], clf_key, clf):
    x = extract_features(data, features)
    y = extract_labels(data)
    x_tr, x_te, y_tr, y_te = ms.train_test_split(x, y)
    clf: LinearClassifierMixin = clf().fit(x_tr, y_tr)
    s = clf.score(x_te, y_te)
    print("{:10} {:20} {:10.3f}".format(clf_key, features_key, s))


def feature_names(grps: List[str], bases: List[str]) -> List[str]:
    re = []
    for grp in grps:
        for base in bases:
            re.append("{}_{}".format(base, grp))
    return re


def feature_keys(grps: Iterable[str], base: str, all_grps: bool) -> str:
    if base == "all" and all_grps:
        return "all"
    if all_grps:
        return "all_{}".format(base)
    return "{}_{}".format('_'.join(grps), base)


def fs(grp_combis: Iterable[List[str]], base_sel: Dict[str, Iterable[str]], len_grps_base) -> Dict[str, Iterable[str]]:
    feature_sets = {}
    for grps in grp_combis:
        for sel_key in base_sel.keys():
            key = feature_keys(grps, sel_key, len(grps) == len_grps_base)
            val = feature_names(grps, feature_selections[sel_key])
            feature_sets[key] = val
    return feature_sets


clfs = {
    "lr": lambda: LogisticRegression(solver='lbfgs', max_iter=4000),
    "lda": lambda: Lda(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
                       tol=0.0001),
}

feature_groups = ["mean", "se", "worst"]

feature_selections: Dict[str, List[str]] = {
    "all": ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points",
            "symmetry", "fractal_dimension"],
    "most_rel": ["radius", "texture", "compactness", "concavity", "concave_points"],
    "two_rel": ["radius", "concavity"],
    "one_rel": ["radius"],
}

_data = pd.read_csv('../data/bc-data.csv', header=0)

feature_group_combis = all_combis(feature_groups)
_feature_sets = fs(feature_group_combis, feature_selections, len(feature_groups))

for _clf_key in clfs:
    for _features_key in _feature_sets.keys():
        train_fs(_data, _features_key, _feature_sets[_features_key], _clf_key, clfs[_clf_key])
