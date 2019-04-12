from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
import pandas as pd



feature_sets = {
    "all": ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave_points", "symmetry", "fractal_dimension"],
    "most_rel": ["radius", "texture", "compactness", "concavity", "concave_points"],
    "two_rel": ["radius", "concavity"],
    "radius": ["radius"],
 }


def extract_features(d: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    features_base = feature_sets[feature_set]
    feature_names = list(map(lambda b: b + "_mean", features_base))
    return d[feature_names]


def extract_labels(d: pd.DataFrame):
    dia = d[['diagnosis']]
    return dia.values.ravel()


def train_fs(fs: str, clfFac):
    """
    Trainiert einen Classifier für ein feature set
    :param fs: ID des Feature set
    :param clfFac: Classifier Factory. Aufruf erzeugt einen neuen Classifier
    """
    X_tr = extract_features(_data_tr, fs)
    X_te = extract_features(_data_te, fs)
    y_tr = extract_labels(_data_tr)
    y_te = extract_labels(_data_te)
    clf: LinearClassifierMixin = clfFac().fit(X_tr, y_tr)
    _iter = clf.n_iter_
    s = clf.score(X_te, y_te)
    print("fs: {:10} iter:{:5} score:{:10}".format(fs, _iter[0], s))


_data = pd.read_csv('../data/bc-data.csv', header=0)
rows = _data.shape[0]
ntest = int(rows * 0.2)

_data_te = _data[:ntest]
_data_tr = _data[ntest:]

clf = lambda :LogisticRegression(solver='lbfgs')

for fs in feature_sets.keys():
    train_fs(fs, clf)
