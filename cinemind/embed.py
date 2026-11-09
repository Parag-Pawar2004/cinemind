"""Text embeddings with graceful fallback.

1. sentence-transformers (all-MiniLM-L6-v2, free, local CPU) if installed
2. otherwise TF-IDF + truncated SVD (latent semantic analysis) - zero downloads
"""
from __future__ import annotations

import math
import re
from collections import Counter

import joblib
import numpy as np

from . import config


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Embedder:
    def __init__(self):
        self.backend = None
        self._st = None
        self._lsa = None

    # ---------- fit / load
    def fit(self, texts: list[str], prefer_st: bool = True) -> "Embedder":
        if prefer_st and self._try_st():
            return self
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english",
                              max_features=60000)
        n_comp = max(2, min(256, len(texts) - 1, 4000))
        X = vec.fit_transform(texts)
        n_comp = min(n_comp, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        svd.fit(X)
        self._lsa = make_pipeline(vec, svd)
        self.backend = "lsa"
        return self

    def _try_st(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(config.get("EMBEDDING_MODEL"))
            self.backend = "sentence-transformers"
            return True
        except Exception:  # noqa: BLE001
            return False

    def save(self):
        joblib.dump({"backend": self.backend, "lsa": self._lsa}, config.EMBEDDER_FILE)

    @classmethod
    def load(cls) -> "Embedder":
        e = cls()
        d = joblib.load(config.EMBEDDER_FILE)
        e.backend = d["backend"]
        e._lsa = d["lsa"]
        if e.backend == "sentence-transformers" and not e._try_st():
            raise RuntimeError("sentence-transformers missing; rebuild index")
        return e

    # ---------- encode
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return _norm(self._st.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 500))
        return _norm(self._lsa.transform(texts))


# ---------------------------------------------------------------- BM25 (lexical)
_TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an the of and or to in on for with is are was were be by at as from that this it its "
           "i me my want something like show shows movie movies film films series drama dramas "
           "recommend please some watch anything something good great best nice need looking get give "
           "suggest suggestions".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in toks]
        self.len = np.array([len(t) for t in toks], dtype=np.float32)
        self.avg = float(self.len.mean() or 1)
        df = Counter(w for t in toks for w in set(t))
        N = len(docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        # inverted index for speed
        self.inv: dict[str, list[int]] = {}
        for i, t in enumerate(self.tf):
            for w in t:
                self.inv.setdefault(w, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            idf = self.idf[w]
            for i in self.inv[w]:
                f = self.tf[i][w]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s
