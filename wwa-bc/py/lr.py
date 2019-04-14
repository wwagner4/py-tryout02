from typing import Dict, List

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
# from sklearn.lda import LDA # for older versions
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
import pandas as pd
import itertools as it


# feature_names = list(map(lambda b: b + "_" + grp, features_base))


def extract_features(d: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return d[features]


def all_combis(li: List):
    for l in range(len(li)):
        for i in it.combinations(li, l + 1):
            yield list(i)


def extract_labels(d: pd.DataFrame):
    dia = d[['diagnosis']]
    return dia.values.ravel()


def train_fs(data: pd.DataFrame, features_key: str, features: List[str], clf_key, clf):
    rows = data.shape[0]
    ntest = int(rows * 0.2)
    data_te = data[:ntest]
    data_tr = data[ntest:]
    x_tr = extract_features(data_tr, features)
    x_te = extract_features(data_te, features)
    y_tr = extract_labels(data_tr)
    y_te = extract_labels(data_te)
    clf: LinearClassifierMixin = clf().fit(x_tr, y_tr)
    s = clf.score(x_te, y_te)
    print("{:10} {:20} {:10.3f}".format(clf_key, features_key, s))


clfs = {
    "lr": lambda: LogisticRegression(solver='lbfgs', max_iter=1500),
    "lda": lambda: Lda(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
                       tol=0.0001),
}

feature_groups_base = ["mean", "se", "worst"]
feature_groups = all_combis(feature_groups_base)

feature_base_selection: Dict[str, List[str]] = {
    "all": ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points",
            "symmetry", "fractal_dimension"],
    "most_rel": ["radius", "texture", "compactness", "concavity", "concave_points"],
    "two_rel": ["radius", "concavity"],
    "one_rel": ["radius"],
}


def feature_names(grps: List[str], bases: List[str]) -> List[str]:
    re = []
    for grp in grps:
        for base in bases:
            re.append("{}_{}".format(base, grp))
    return re


def feature_keys(grps: List[str], base: str, all_grps: bool) -> str:
    if base == "all" and all_grps:
        return "all"
    if all_grps:
        return "all_{}".format(sel_key)
    return "{}_{}".format('_'.join(grps), sel_key)


feature_sets = {}
for _grps in feature_groups:
    for sel_key in feature_base_selection.keys():
        key = feature_keys(_grps, sel_key, len(_grps) == len(feature_groups_base))
        val = feature_names(_grps, feature_base_selection[sel_key])
        feature_sets[key] = val

_data = pd.read_csv('../data/bc-data.csv', header=0)

for _clf_key in clfs:
    for _features_key in feature_sets.keys():
        train_fs(_data, _features_key, feature_sets[_features_key], _clf_key, clfs[_clf_key])
