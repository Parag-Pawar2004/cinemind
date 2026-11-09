"""Collaborative filtering on MovieLens (100k real ratings, free).

Implicit-feedback matrix factorisation via truncated SVD on a BM25-weighted
user x item matrix. New users are handled by *fold-in*: a profile is the
weighted mean of liked item factors, so no retraining is needed.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from . import config


def load_ratings() -> pd.DataFrame | None:
    p = config.RAW_DIR / "ml-latest-small" / "ratings.csv"
    return pd.read_csv(p) if p.exists() else None


def bm25_weight(X: sparse.csr_matrix, K1=1.2, B=0.75) -> sparse.csr_matrix:
    X = X.tocoo().astype(np.float32)
    N = X.shape[0]
    idf = np.log(N) - np.log1p(np.bincount(X.col, minlength=X.shape[1]))
    row_sums = np.ravel(X.sum(axis=1))
    avg = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / avg
    data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return sparse.csr_matrix((data, (X.row, X.col)), shape=X.shape)


class ItemFactorModel:
    def __init__(self, k: int = 64):
        self.k = k
        self.item_ids: np.ndarray | None = None  # MovieLens ids
        self.factors: np.ndarray | None = None
        self.pop: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame, min_rating: float = 3.5) -> "ItemFactorModel":
        from sklearn.decomposition import TruncatedSVD
        pos = ratings[ratings.rating >= min_rating]
        users = pos.userId.astype("category")
        items = pos.movieId.astype("category")
        X = sparse.csr_matrix((pos.rating.values - min_rating + 1.0,
                               (users.cat.codes.values, items.cat.codes.values)))
        Xw = bm25_weight(X)
        k = min(self.k, min(Xw.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=0).fit(Xw)
        F = (svd.components_.T * np.sqrt(svd.singular_values_)).astype(np.float32)
        self.factors = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        self.item_ids = np.asarray(items.cat.categories)
        self.pop = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
        self._index = {int(m): i for i, m in enumerate(self.item_ids)}
        return self

    def index_of(self, ml_id: int) -> int | None:
        return self._index.get(int(ml_id))

    def user_vector(self, liked_ml_ids: list[int], weights: list[float] | None = None) -> np.ndarray | None:
        idx = [self.index_of(m) for m in liked_ml_ids]
        pairs = [(i, (weights[j] if weights else 1.0)) for j, i in enumerate(idx) if i is not None]
        if not pairs:
            return None
        v = sum(w * self.factors[i] for i, w in pairs)
        return v / (np.linalg.norm(v) + 1e-9)

    def score_all(self, uvec: np.ndarray) -> np.ndarray:
        return self.factors @ uvec

    def save(self):
        joblib.dump({"k": self.k, "ids": self.item_ids, "F": self.factors, "pop": self.pop}, config.CF_FILE)

    @classmethod
    def load(cls) -> "ItemFactorModel | None":
        if not config.CF_FILE.exists():
            return None
        d = joblib.load(config.CF_FILE)
        m = cls(d["k"])
        m.item_ids, m.factors, m.pop = d["ids"], d["F"], d["pop"]
        m._index = {int(x): i for i, x in enumerate(m.item_ids)}
        return m
