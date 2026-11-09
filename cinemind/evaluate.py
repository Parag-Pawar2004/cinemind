"""Offline evaluation.

1. Ranking accuracy on MovieLens (temporal hold-out, last 20% of each user's likes):
   Popularity vs Content (embeddings) vs CF (SVD fold-in) vs Hybrid  -> HR@10, NDCG@10, Recall@10, coverage
2. Agent quality on a fixed suite of natural-language queries:
   filter precision (results respect requested industry/kind), intra-list diversity, novelty.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import config
from .agents import Orchestrator
from .engine import Engine, cosine_ild, novelty

QUERY_SUITE = [
    ("dark mind-bending thriller with a big twist", {}),
    ("feel-good Korean drama about friendship and healing", {"industry": "Korean", "kind": "tv"}),
    ("epic Telugu action movie with mythology", {"industry": "Tollywood", "kind": "movie"}),
    ("lighthearted Bollywood romcom", {"industry": "Bollywood"}),
    ("Chinese xianxia fantasy romance series", {"industry": "Chinese", "kind": "tv"}),
    ("something like Inception but from Bollywood", {"industry": "Bollywood"}),
    ("emotional father daughter story that will make me cry", {}),
    ("slow burn revenge kdrama", {"industry": "Korean", "kind": "tv"}),
]


def ranking_eval(engine: Engine, n_users: int = 300, k: int = 10, seed: int = 0) -> dict | None:
    from .cf import ItemFactorModel, load_ratings
    R = load_ratings()
    if R is None or engine.cf_to_cat is None:
        return None
    R = R[R.rating >= 3.5].sort_values("timestamp")
    rng = np.random.default_rng(seed)
    users = R.userId.value_counts()
    users = users[users >= 15].index.values
    users = rng.choice(users, size=min(n_users, len(users)), replace=False)
    test_users = set(users.tolist())
    train_parts, test = [], {}
    for u, g in R.groupby("userId"):
        if u in test_users:
            cut = int(len(g) * 0.8)
            train_parts.append(g.iloc[:cut])
            test[u] = (g.movieId.iloc[:cut].tolist(), set(g.movieId.iloc[cut:].tolist()))
        else:
            train_parts.append(g)
    import pandas as pd
    model = ItemFactorModel(64).fit(pd.concat(train_parts))  # no leakage: retrain without test items
    ml_ids = model.item_ids
    cat_idx = np.array([engine.cf_to_cat[engine.cf.index_of(m)] if engine.cf.index_of(m) is not None else -1
                        for m in ml_ids])
    has_cat = cat_idx >= 0
    E_ml = np.zeros((len(ml_ids), engine.E.shape[1]), dtype=np.float32)
    E_ml[has_cat] = engine.E[cat_idx[has_cat]]
    pop = model.pop / model.pop.max()

    def metrics(score_fn):
        hr = nd = rec = 0.0
        recommended = set()
        for u, (train, held) in test.items():
            s = score_fn(train)
            seen = [model.index_of(m) for m in train]
            s[[i for i in seen if i is not None]] = -np.inf
            top = np.argsort(-s)[:k]
            ids = [int(ml_ids[i]) for i in top]
            recommended.update(ids)
            hits = [1 if m in held else 0 for m in ids]
            hr += 1 if any(hits) else 0
            rec += sum(hits) / min(len(held), k)
            dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(held), k)))
            nd += dcg / idcg
        n = len(test)
        return {"HR@10": round(hr / n, 4), "NDCG@10": round(nd / n, 4), "Recall@10": round(rec / n, 4),
                "coverage": round(len(recommended) / len(ml_ids), 4)}

    def pop_fn(train):
        return pop.copy()

    def content_fn(train):
        idx = [model.index_of(m) for m in train[-30:]]
        idx = [i for i in idx if i is not None and has_cat[i]]
        if not idx:
            return pop.copy()
        v = E_ml[idx].mean(0)
        return E_ml @ (v / (np.linalg.norm(v) + 1e-9)) + 0.05 * pop

    def cf_fn(train):
        u = model.user_vector(train)
        return model.score_all(u) if u is not None else pop.copy()

    def hybrid_fn(train):
        a, b = cf_fn(train), content_fn(train)
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9)  # noqa: E731
        return 0.75 * z(a) + 0.25 * z(b)

    return {"users": len(test), "Popularity": metrics(pop_fn), "Content": metrics(content_fn),
            "CF (SVD fold-in)": metrics(cf_fn), "Hybrid": metrics(hybrid_fn)}


def agent_eval(engine: Engine) -> dict:
    orch = Orchestrator(engine, recall=False)
    rows = []
    for q, expect in QUERY_SUITE:
        out = orch.run(q)
        res = out["results"]
        idx = [engine.find_title(r["title"]) for r in res]
        idx = [i for i in idx if i is not None]
        ok = [((not expect.get("industry") or r["industry"] == expect["industry"]) and
               (not expect.get("kind") or r["kind"] == expect["kind"])) for r in res]
        rows.append({"query": q, "n": len(res), "filter_precision": round(float(np.mean(ok)) if ok else 0, 3),
                     "diversity_ILD": round(cosine_ild(engine.E, idx), 3),
                     "novelty": round(novelty(np.clip(engine.pop, 1e-3, 1), idx), 3),
                     "top3": [r["title"] for r in res[:3]], "intent_mode": out["trace"][0]["mode"]})
    return {"queries": rows,
            "mean_filter_precision": round(float(np.mean([r["filter_precision"] for r in rows])), 3),
            "mean_diversity": round(float(np.mean([r["diversity_ILD"] for r in rows])), 3)}


def run_all(engine: Engine | None = None, log=print) -> dict:
    engine = engine or Engine()
    report = {"agent_suite": agent_eval(engine)}
    log("Agent suite done")
    try:
        rk = ranking_eval(engine)
        report["movielens_ranking"] = rk or "skipped (MovieLens not available)"
    except Exception as e:  # noqa: BLE001
        report["movielens_ranking"] = f"failed: {e}"
    config.EVAL_FILE.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
