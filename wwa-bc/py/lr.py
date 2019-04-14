from typing import Dict, List

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
# from sklearn.lda import LDA # for older versions
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
import pandas as pd


# feature_names = list(map(lambda b: b + "_" + grp, features_base))


def extract_features(d: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return d[features]


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
    "lda_1": lambda: Lda(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
                         tol=0.0001),
    "lda_2": lambda: Lda(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
                         tol=0.001),
    "lda_3": lambda: Lda(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
                         tol=0.01),
}

feature_groups = ["mean", "se", "worst"]

feature_base_selection: Dict[str, List[str]] = {
    "all": ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points",
            "symmetry", "fractal_dimension"],
    "most_rel": ["radius", "texture", "compactness", "concavity", "concave_points"],
    "two_rel": ["radius", "concavity"],
    "one_rel": ["radius"],
}

feature_sets = {}
for grp in feature_groups:
    for sel_key in feature_base_selection.keys():
        key = "{}_{}".format(grp, sel_key)
        val = list(map(lambda b: b + "_" + grp, feature_base_selection[sel_key]))
        feature_sets[key] = val

fall = []
ba = feature_base_selection["all"]
for grp in feature_groups:
    for f in ba:
        fall.append("{}_{}".format(f, grp))

feature_sets["all"] = fall

_data = pd.read_csv('../data/bc-data.csv', header=0)

for _clf_key in clfs:
    for _features_key in feature_sets.keys():
        train_fs(_data, _features_key, feature_sets[_features_key], _clf_key, clfs[_clf_key])
