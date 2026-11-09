"""CineMind retrieval engine.

Score(item) = w_dense*semantic + w_lex*BM25 + w_ref*similarity-to-reference-titles
            + w_vibe*vibe-match + w_cf*collaborative + w_pop*popularity + w_q*quality
then hard filters (industry/kind/year/genre) and MMR re-ranking for diversity.

Novel pieces:
  * Cross-Industry Bridge  - analogical vector arithmetic: "the Tollywood version of Inception"
        q = e(src) - a*centroid(src_industry) + a*centroid(target_industry)
  * Vibe steering           - interpretable 6-axis sliders blended into ranking
  * Serendipity knob        - trades relevance for novelty (popularity penalty) + diversity
"""
from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import config
from .build import norm_title
from .cf import ItemFactorModel
from .embed import BM25, Embedder, tokenize
from .vibes import AXES


@dataclass
class Intent:
    query_text: str = ""
    industries: list[str] = field(default_factory=list)
    kind: str | None = None                    # "movie" | "tv" | None
    genres_include: list[str] = field(default_factory=list)
    genres_exclude: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    reference_titles: list[str] = field(default_factory=list)
    exclude_titles: list[str] = field(default_factory=list)
    vibes: dict = field(default_factory=dict)  # axis -> 0..1
    min_rating: float | None = None
    serendipity: float = 0.2
    n: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "Intent":
        ok = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__ and v is not None}
        it = cls(**ok)
        it.vibes = {k: float(v) for k, v in (it.vibes or {}).items() if k in AXES}
        it.n = int(max(1, min(50, it.n)))
        return it

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class Engine:
    def __init__(self):
        self.items: list[dict] = []
        with open(config.CATALOG_FILE, encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.E = np.load(config.EMB_FILE)
        self.embedder = Embedder.load()
        from .build import item_text
        self.bm25 = BM25([item_text(r) for r in self.items])
        self.V = np.array([r["vibe"] for r in self.items], dtype=np.float32)
        pop = np.array([r.get("popularity") or 0 for r in self.items], dtype=np.float32)
        self._pop_den = float(np.log1p(pop).max() + 1e-9)
        self.pop = np.log1p(pop) / self._pop_den
        self.rating = np.array([(r.get("rating") or 0) for r in self.items], dtype=np.float32) / 10.0
        self.industry = np.array([r["industry"] for r in self.items])
        self.kind = np.array([r["kind"] for r in self.items])
        self.year = np.array([r.get("year") or 0 for r in self.items])
        self.title_index: dict[str, list[int]] = {}
        for i, r in enumerate(self.items):
            self.title_index.setdefault(norm_title(r["title"]), []).append(i)
        self.cf = ItemFactorModel.load()
        self.cf_to_cat = None
        if self.cf is not None:
            m = {r["ml_id"]: i for i, r in enumerate(self.items) if r.get("ml_id") is not None}
            self.cf_to_cat = np.array([m.get(int(x), -1) for x in self.cf.item_ids])
        self.centroids = {ind: self._centroid(self.industry == ind) for ind in set(self.industry)}
        # MovieLens titles have only genres; down-weight them for text matching
        self.sparse_text = np.array([len(r.get("overview") or "") < 30 for r in self.items])
        self._raw_sorted = None
        self._load_learned()

    # ------------------------------------------------------------------ live-learned titles
    def _raw_cols(self):
        if self._raw_sorted is None:
            from .vibes import raw_vibe
            raw = np.stack([raw_vibe(r) for r in self.items])
            self._raw_sorted = [np.sort(raw[:, j]) for j in range(raw.shape[1])]
        return self._raw_sorted

    def _load_learned(self):
        if config.LEARNED_FILE.exists():
            recs = [json.loads(line) for line in open(config.LEARNED_FILE, encoding="utf-8") if line.strip()]
            if recs:
                self.add_items(recs, persist=False)

    def add_items(self, records: list[dict], persist: bool = True) -> list[int]:
        """Add titles at runtime (embedding + vibe computed on the fly). Returns catalog index per record;
        titles already present are not duplicated."""
        from .build import item_text
        from .vibes import raw_vibe
        out: list = []
        fresh: list[dict] = []
        for r in records:
            hit = next((i for i in self.title_index.get(norm_title(r["title"]), [])
                        if not r.get("year") or not self.year[i] or abs(int(self.year[i]) - int(r["year"])) <= 1), None)
            out.append(hit)
            if hit is None:
                fresh.append(r)
        if fresh:
            base = len(self.items)
            cols = self._raw_cols()
            for k, r in enumerate(fresh):
                r.setdefault("sources", [r.get("source")])
                r["id"] = f"lr{base + k:06d}"
                v = raw_vibe(r)
                r["vibe"] = [round(min(1.0, float((np.searchsorted(c, x, side="left") +
                                                    np.searchsorted(c, x, side="right")) / 2 / max(1, len(c) - 1))), 3)
                             for c, x in zip(cols, v)]
            self.E = np.vstack([self.E, self.embedder.encode([item_text(r) for r in fresh])])
            self.V = np.vstack([self.V, np.array([r["vibe"] for r in fresh], dtype=np.float32)])
            pop = np.log1p(np.array([r.get("popularity") or 0 for r in fresh], dtype=np.float32)) / self._pop_den
            self.pop = np.concatenate([self.pop, np.clip(pop, 0, 1)])
            self.rating = np.concatenate([self.rating, np.array([(r.get("rating") or 0) for r in fresh],
                                                                dtype=np.float32) / 10.0])
            self.industry = np.concatenate([self.industry, np.array([r["industry"] for r in fresh])])
            self.kind = np.concatenate([self.kind, np.array([r["kind"] for r in fresh])])
            self.year = np.concatenate([self.year, np.array([r.get("year") or 0 for r in fresh])])
            self.sparse_text = np.concatenate([self.sparse_text, np.array(
                [len(r.get("overview") or "") < 30 for r in fresh])])
            self.items.extend(fresh)
            for k, r in enumerate(fresh):
                self.title_index.setdefault(norm_title(r["title"]), []).append(base + k)
            if persist:
                with open(config.LEARNED_FILE, "a", encoding="utf-8") as f:
                    for r in fresh:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            new_idx = iter(range(base, base + len(fresh)))
            out = [o if o is not None else next(new_idx) for o in out]
        return out

    # ------------------------------------------------------------------ utils
    def _centroid(self, mask) -> np.ndarray:
        c = self.E[mask].mean(axis=0)
        return c / (np.linalg.norm(c) + 1e-9)

    @property
    def industries(self) -> list[str]:
        return sorted(set(self.industry), key=lambda x: (x == "Global", x))

    def find_title(self, name: str, cutoff: float = 0.82) -> int | None:
        key = norm_title(name)
        if key in self.title_index:
            idxs = self.title_index[key]
            return max(idxs, key=lambda i: self.items[i].get("popularity") or 0)
        close = difflib.get_close_matches(key, self.title_index.keys(), n=1, cutoff=cutoff)
        if close:
            idxs = self.title_index[close[0]]
            return max(idxs, key=lambda i: self.items[i].get("popularity") or 0)
        return None

    def _mask(self, it: Intent) -> np.ndarray:
        m = np.ones(len(self.items), dtype=bool)
        if it.industries:
            m &= np.isin(self.industry, it.industries)
        if it.kind in ("movie", "tv"):
            m &= self.kind == it.kind
        if it.year_min:
            m &= self.year >= it.year_min
        if it.year_max:
            m &= (self.year <= it.year_max) & (self.year > 0)
        if it.min_rating:
            m &= (self.rating * 10 >= it.min_rating) | (self.rating == 0)
        inc = [g.lower() for g in it.genres_include]
        exc = [g.lower() for g in it.genres_exclude]
        if inc or exc:
            gm = np.array([
                (not inc or any(any(x in g.lower() for g in r.get("genres", [])) for x in inc)) and
                not any(any(x in g.lower() for g in r.get("genres", [])) for x in exc)
                for r in self.items])
            m &= gm
        for t in it.reference_titles + it.exclude_titles:
            i = self.find_title(t)
            if i is not None:
                m[i] = False
        return m

    @staticmethod
    def _mm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return x
        lo, hi = x[mask].min(), x[mask].max()
        return (x - lo) / (hi - lo + 1e-9)

    # ------------------------------------------------------------------ main search
    def search(self, it: Intent, profile_vec: np.ndarray | None = None, cf_scores: np.ndarray | None = None,
               k_candidates: int = 60, exclude_idx: list[int] | None = None,
               boost_idx: list[int] | None = None) -> tuple[list[dict], dict]:
        mask = self._mask(it)
        for i in exclude_idx or []:
            mask[i] = False
        comps: dict[str, np.ndarray] = {}
        weights: dict[str, float] = {}
        n = len(self.items)

        q = it.query_text.strip()
        if q and tokenize(q):  # skip empty / filler-only queries
            qv = self.embedder.encode([q])[0]
            dense = self.E @ qv
            dense = np.where(self.sparse_text, dense * 0.8, dense)
            comps["semantic"] = self._mm(dense, mask)
            weights["semantic"] = 0.45
            lex = self.bm25.scores(q)
            if len(lex) < n:  # titles learned at runtime are not in the BM25 index yet
                lex = np.pad(lex, (0, n - len(lex)))
            if lex.max() > 0:
                comps["keywords"] = lex / lex.max()
                weights["keywords"] = 0.20

        ref_idx = [i for i in (self.find_title(t) for t in it.reference_titles) if i is not None]
        if ref_idx:
            sims = self.E @ self.E[ref_idx].T  # n x refs
            comps["like_your_titles"] = self._mm(sims.mean(axis=1) * 0.5 + sims.max(axis=1) * 0.5, mask)
            weights["like_your_titles"] = 0.45
            # vibe of references becomes a soft target if user gave none
            if not it.vibes:
                it.vibes = {a: round(float(v), 2) for a, v in zip(AXES, self.V[ref_idx].mean(axis=0))}
                weights["_ref_vibe"] = 1

        if it.vibes:
            ax = [AXES.index(a) for a in it.vibes]
            target = np.array(list(it.vibes.values()), dtype=np.float32)
            comps["vibe_match"] = 1 - np.abs(self.V[:, ax] - target).mean(axis=1)
            weights["vibe_match"] = 0.12 if weights.pop("_ref_vibe", 0) else 0.25

        if profile_vec is not None:
            comps["taste_profile"] = self._mm(self.E @ profile_vec, mask)
            weights["taste_profile"] = 0.35
        if cf_scores is not None:
            comps["people_like_you"] = cf_scores
            weights["people_like_you"] = 0.30

        if boost_idx:
            b = np.zeros(n, dtype=np.float32)
            b[list(boost_idx)] = 1.0
            comps["llm_recall"] = b
            weights["llm_recall"] = 0.45
        if not weights:  # nothing specified -> popular & well rated in filter
            weights["popularity"] = 0.6
            weights["quality"] = 0.4

        s = it.serendipity
        comps.setdefault("popularity", self.pop)
        comps.setdefault("quality", self.rating)
        weights["popularity"] = weights.get("popularity", 0.10) * (1 - 2 * s)  # negative => novelty boost
        weights["quality"] = weights.get("quality", 0.06)

        total = np.zeros(n, dtype=np.float32)
        for name, w in weights.items():
            total += w * comps[name]
        total[~mask] = -np.inf
        k = min(k_candidates, int(mask.sum()))
        if k == 0:
            return [], {"n_filtered": 0, "weights": weights}
        top = np.argpartition(-total, k - 1)[:k]
        top = top[np.argsort(-total[top])]
        cands = []
        for i in top:
            c = {name: round(float(comps[name][i]), 3) for name in weights if name in comps}
            cands.append({"idx": int(i), "score": float(total[i]), "components": c})
        return cands, {"n_filtered": int(mask.sum()), "weights": {k: round(v, 3) for k, v in weights.items()},
                       "ref_found": [self.items[i]["title"] for i in ref_idx]}

    # ------------------------------------------------------------------ diversity
    def mmr(self, cands: list[dict], n: int, lam: float = 0.75) -> list[dict]:
        if not cands:
            return []
        scores = np.array([c["score"] for c in cands])
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        idx = [c["idx"] for c in cands]
        S = self.E[idx] @ self.E[idx].T
        chosen = [0]
        while len(chosen) < min(n, len(cands)):
            rest = [j for j in range(len(cands)) if j not in chosen]
            mm = [lam * scores[j] - (1 - lam) * S[j, chosen].max() for j in rest]
            chosen.append(rest[int(np.argmax(mm))])
        return [cands[j] for j in chosen]

    # ------------------------------------------------------------------ bridge
    def bridge(self, title: str, target_industry: str, n: int = 10, alpha: float = 0.8,
               kind: str | None = None) -> tuple[dict | None, list[dict]]:
        i = self.find_title(title)
        if i is None:
            return None, []
        src = self.items[i]
        e = self.E[i] - alpha * self.centroids[src["industry"]] + alpha * self.centroids[target_industry]
        e /= np.linalg.norm(e) + 1e-9
        sims = self.E @ e
        vibe = 1 - np.abs(self.V - self.V[i]).mean(axis=1)
        g0 = {g.lower() for g in src.get("genres", [])}
        genre = np.array([len(g0 & {g.lower() for g in r.get("genres", [])}) / (len(g0 | {g.lower() for g in r.get("genres", [])}) or 1)
                          for r in self.items], dtype=np.float32)
        score = 0.6 * sims + 0.2 * vibe + 0.15 * genre + 0.05 * self.pop
        mask = (self.industry == target_industry)
        if kind:
            mask &= self.kind == kind
        mask[i] = False
        score[~mask] = -np.inf
        top = np.argsort(-score)[: n * 3]
        cands = [{"idx": int(j), "score": float(score[j]),
                  "components": {"analogy": round(float(sims[j]), 3), "vibe_match": round(float(vibe[j]), 3),
                                 "genre_overlap": round(float(genre[j]), 3)}}
                 for j in top if np.isfinite(score[j])]
        return src, self.mmr(cands, n, lam=0.85)

    # ------------------------------------------------------------------ taste DNA
    def taste(self, liked: list[int], disliked: list[int] | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not liked:
            return None, None
        v = self.E[liked].mean(axis=0)
        if disliked:
            v = v - 0.5 * self.E[disliked].mean(axis=0)
        v /= np.linalg.norm(v) + 1e-9
        cf_scores = None
        if self.cf is not None:
            ml_ids = [self.items[i]["ml_id"] for i in liked if self.items[i].get("ml_id") is not None]
            u = self.cf.user_vector(ml_ids) if ml_ids else None
            if u is not None:
                raw = self.cf.score_all(u)
                cf_scores = np.zeros(len(self.items), dtype=np.float32)
                ok = self.cf_to_cat >= 0
                cf_scores[self.cf_to_cat[ok]] = np.clip(raw[ok], 0, None)
                cf_scores /= cf_scores.max() + 1e-9
        return v, cf_scores

    def taste_dna(self, liked: list[int]) -> dict:
        if not liked:
            return {}
        vib = self.V[liked].mean(axis=0)
        inds, gens = {}, {}
        for i in liked:
            r = self.items[i]
            inds[r["industry"]] = inds.get(r["industry"], 0) + 1
            for g in r.get("genres", []):
                gens[g] = gens.get(g, 0) + 1
        return {"vibe": dict(zip(AXES, map(float, vib))), "industries": inds,
                "genres": dict(sorted(gens.items(), key=lambda x: -x[1])[:8])}

    # ------------------------------------------------------------------ explanation helpers
    def evidence(self, idx: int, it: Intent | None, comps: dict) -> list[str]:
        r = self.items[idx]
        ev = []
        if it and it.query_text:
            qt = set(tokenize(it.query_text))
            hit = [t for t in r.get("tags", []) + r.get("genres", []) if set(tokenize(t)) & qt]
            if hit:
                ev.append("matches " + ", ".join(hit[:3]))
        if comps.get("like_your_titles", 0) > 0.6:
            ev.append("close in story/theme to your reference titles")
        if comps.get("people_like_you", 0) > 0.4:
            ev.append("loved by MovieLens users with similar taste")
        if comps.get("taste_profile", 0) > 0.6:
            ev.append("fits your Taste DNA")
        if it and it.vibes:
            v = dict(zip(AXES, self.V[idx]))
            good = [a for a, t in it.vibes.items() if abs(v[a] - t) < 0.2]
            if good:
                ev.append("vibe fits on " + ", ".join(good))
        if r.get("rating"):
            ev.append(f"rated {r['rating']:.1f}/10")
        return ev

    def card(self, c: dict) -> dict:
        r = self.items[c["idx"]]
        return {**{k: r.get(k) for k in ("id", "title", "year", "kind", "industry", "language",
                                         "genres", "overview", "poster", "rating", "tags")},
                "vibe": dict(zip(AXES, r["vibe"])), "score": round(c["score"], 3),
                "components": c.get("components", {}), "why": c.get("why", "")}


def cosine_ild(E: np.ndarray, idx: list[int]) -> float:
    """Intra-list diversity: 1 - mean pairwise cosine."""
    if len(idx) < 2:
        return 0.0
    S = E[idx] @ E[idx].T
    n = len(idx)
    return float(1 - (S.sum() - n) / (n * (n - 1)))


def novelty(pop: np.ndarray, idx: list[int]) -> float:
    return float(np.mean([-math.log2(max(1e-3, pop[i])) for i in idx])) if idx else 0.0
