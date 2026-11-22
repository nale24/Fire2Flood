import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import config as C

LOGISTIC_GRID = {"clf__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]}
GBM_GRID = {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1], "clf__max_iter": [100, 300]}


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
    steps = []
    if n_pca and n_tab is not None:
        steps.append(("pca", BlockPCA(n_tab, n_pca)))
    if head == "logistic":
        steps += [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]
        grid = LOGISTIC_GRID
    elif head == "gbm":
        steps += [("clf", HistGradientBoostingClassifier(random_state=C.RANDOM_SEED, l2_regularization=1.0))]
        grid = GBM_GRID
    else:
        raise SystemExit(f"unknown head {head}")
    n_inner = min(5, len(np.unique(groups)))
    return GridSearchCV(Pipeline(steps), grid, scoring="neg_log_loss", cv=GroupKFold(n_splits=n_inner), n_jobs=-1)
