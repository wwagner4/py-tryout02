import itertools as it
from typing import Dict, List, Iterable

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
# from sklearn.lda import LDA # for older versions
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
import sklearn.model_selection as ms
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing as pp
from collections import namedtuple
import xgboost as xgb

ClfConf = namedtuple("ClfConf", "id clf normalized")


def all_combis(li: List) -> Iterable[List]:
    for l in range(len(li)):
        for i in it.combinations(li, l + 1):
            yield i


def create_featuresets(
        grp_combis: Iterable[List[str]],
        base_sel: Dict[str, Iterable[str]],
        len_grps_base: int) -> Dict[str, Iterable[str]]:
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

    feature_sets = {}
    for _grps in grp_combis:
        for sel_key in base_sel.keys():
            key = feature_keys(_grps, sel_key, len(_grps) == len_grps_base)
            val = feature_names(_grps, feature_selections[sel_key])
            feature_sets[key] = val
    return feature_sets


def train(data: pd.DataFrame, features_key: str, features: Iterable[str], clf_conf: ClfConf):
    def extract_features(d: pd.DataFrame, normalize: bool) -> pd.DataFrame:
        if normalize:
            vals = pp.normalize(d[features])
            return pd.DataFrame(vals, columns=features)
        return d[features]

    def extract_labels(d: pd.DataFrame):
        dia = d[['diagnosis']]
        return dia.values.ravel()

    x = extract_features(data, clf_conf.normalized)
    y = extract_labels(data)
    x_tr, x_te, y_tr, y_te = ms.train_test_split(x, y)

    clf: LinearClassifierMixin = clf_conf.clf().fit(x_tr, y_tr)

    s = clf.score(x_te, y_te)
    print("{:10} {:20} {:10.3f}".format(clf_conf.id, features_key, s))


clfs = [
    ClfConf(id="lr",
            clf=lambda: LogisticRegression(solver='lbfgs', max_iter=4000),
            normalized=False
            ),
    ClfConf(id="lda",
            clf=lambda: Lda(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False,
                            tol=0.0001),
            normalized=False
            ),
    ClfConf(id="svm_lin",
            clf=lambda: LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                                  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                                  verbose=0),
            normalized=True
            ),
    ClfConf(id="svm",
            clf=lambda: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                            decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                            max_iter=-1, probability=False, random_state=None, shrinking=True,
                            tol=0.001, verbose=False),
            normalized=True
            ),
    ClfConf(id="knn",
            clf=lambda: KNeighborsClassifier(n_neighbors=3),
            normalized=False
            ),
    ClfConf(id="nm_g",
            clf=lambda: GaussianNB(),
            normalized=False
            ),
    ClfConf(id="rf",
            clf=lambda: RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
            normalized=False
            ),
    ClfConf(id="xgb",
            clf=lambda: xgb.XGBClassifier(),
            normalized=False
            ),
    ClfConf(id="gb",
            clf=lambda: GradientBoostingClassifier(
                loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None,
                verbose=0, max_leaf_nodes=None),
            normalized=False
            ),
]

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
_feature_sets = create_featuresets(feature_group_combis, feature_selections, len(feature_groups))

for _clf in clfs:
    for _features_key in _feature_sets.keys():
        train(_data, _features_key, _feature_sets[_features_key], _clf)
