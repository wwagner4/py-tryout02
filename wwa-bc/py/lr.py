from typing import Dict, List

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
import pandas as pd

feature_sets: Dict[str, List[str]] = {
    "all": ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave_points", "symmetry", "fractal_dimension"],
    "most_rel": ["radius", "texture", "compactness", "concavity", "concave_points"],
    "two_rel": ["radius", "concavity"],
    "radius": ["radius"],
}

feature_groups = ["mean", "se", "worst"]


def extract_features(d: pd.DataFrame, features_base: List[str], grp: str) -> pd.DataFrame:
    feature_names = list(map(lambda b: b + "_" + grp, features_base))
    return d[feature_names]


def extract_labels(d: pd.DataFrame):
    dia = d[['diagnosis']]
    return dia.values.ravel()


def train_fs(features_key: str, features_base: List[str], features_grp: str, clf_key, clf):
    """
    Trainiert einen Classifier für ein feature set
    """
    X_tr = extract_features(_data_tr, features_base, features_grp)
    X_te = extract_features(_data_te, features_base, features_grp)
    y_tr = extract_labels(_data_tr)
    y_te = extract_labels(_data_te)
    clf: LinearClassifierMixin = clf().fit(X_tr, y_tr)
    s = clf.score(X_te, y_te)
    print("clf:{:10} grp:{:10} fs:{:10} score:{:10}".format(clf_key, features_grp, features_key, s))


_data = pd.read_csv('../data/bc-data.csv', header=0)
rows = _data.shape[0]
ntest = int(rows * 0.2)

_data_te = _data[:ntest]
_data_tr = _data[ntest:]

clfs = {
    "lr1": lambda: LogisticRegression(solver='lbfgs', max_iter=500),
    "lr2": lambda: LogisticRegression(solver='lbfgs', max_iter=500),
    "lr3": lambda: LogisticRegression(solver='lbfgs', max_iter=500),
    "lr4": lambda: LogisticRegression(solver='lbfgs', max_iter=500),
}

for clf_key in clfs:
    for fg in feature_groups:
        for fs_key in feature_sets.keys():
            train_fs(fs_key, feature_sets[fs_key], fg, clf_key, clfs[clf_key])
