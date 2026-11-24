from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C

HEADS = {
    "logistic": (lambda: [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))],
                 {"clf__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]}),
    "gbm": (lambda: [("clf", HistGradientBoostingClassifier(random_state=C.RANDOM_SEED, l2_regularization=1.0))],
            {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1], "clf__max_iter": [100, 300]}),
}


class BlockPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_tab=4, n_components=16):
        self.n_tab = n_tab
        self.n_components = n_components

    def fit(self, X, y=None):
        img = X[:, self.n_tab :]
        self.scaler_ = StandardScaler().fit(img)
        k = min(self.n_components, img.shape[1], max(1, img.shape[0] - 1))
        self.pca_ = PCA(n_components=k, random_state=C.RANDOM_SEED).fit(self.scaler_.transform(img))
        return self

    def transform(self, X):
        return np.hstack([X[:, : self.n_tab], self.pca_.transform(self.scaler_.transform(X[:, self.n_tab :]))])


def make_search(head, groups, n_tab=None, n_pca=0):
    if head not in HEADS:
        raise SystemExit(f"unknown head {head}")
    steps, grid = HEADS[head]
    pca = [("pca", BlockPCA(n_tab, n_pca))] if n_pca and n_tab is not None else []
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    return GridSearchCV(Pipeline(pca + steps()), grid, scoring="neg_log_loss", cv=cv, n_jobs=-1)


def searcher(head, **kw):
    return partial(make_search, head, **kw)


def logistic_start(X, y, groups):
    s = make_search("logistic", groups).fit(X, y, groups=groups)
    sc, clf = s.best_estimator_.named_steps["scale"], s.best_estimator_.named_steps["clf"]
    w = clf.coef_[0] / sc.scale_
    return w.astype(np.float32), np.float32(clf.intercept_[0] - (w * sc.mean_).sum())
