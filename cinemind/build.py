"""Build pipeline: ingest every free source -> merge/dedupe -> embeddings -> vibes -> CF.

    python -m cinemind.build            # everything available
    python -m cinemind.build --offline  # seed catalog only, no network
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata

import numpy as np

from . import config, sources
from .data_seed import seed_records
from .embed import Embedder
from .vibes import vibe_matrix


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return re.sub(r"[^a-z0-9]+", "", t)


def _merge(a: dict, b: dict) -> dict:
    """Merge b into a, keeping the richest information from both."""
    if len(b.get("overview") or "") > len(a.get("overview") or ""):
        a["overview"] = b["overview"]
    for f in ("poster", "rating", "ml_id", "tmdb_id"):
        if a.get(f) in (None, "") and b.get(f) not in (None, ""):
            a[f] = b[f]
    if a.get("industry") == "Global" and b.get("industry") != "Global":
        a["industry"], a["language"], a["kind"] = b["industry"], b["language"], b["kind"]
    a["genres"] = list(dict.fromkeys((a.get("genres") or []) + (b.get("genres") or [])))[:6]
    a["tags"] = list(dict.fromkeys((a.get("tags") or []) + (b.get("tags") or [])))[:12]
    a["popularity"] = max(a.get("popularity") or 0, b.get("popularity") or 0)
    a["sources"] = sorted(set(a.get("sources", [a.get("source")]) + [b.get("source")]))
    return a


def merge_records(groups: list[list[dict]]) -> list[dict]:
    by_key: dict[str, dict] = {}
    by_title: dict[str, list[str]] = {}
    for recs in groups:
        for r in recs:
            if not r.get("title"):
                continue
            nt = norm_title(r["title"])
            key = f"{nt}|{r.get('year')}"
            if key not in by_key:
                # tolerate +-1 year mismatches across sources
                for y in ((r.get("year") or 0) - 1, (r.get("year") or 0) + 1):
                    alt = f"{nt}|{y}"
                    if alt in by_key and by_key[alt].get("kind") == r.get("kind"):
                        key = alt
                        break
            if key in by_key:
                _merge(by_key[key], r)
            else:
                r = dict(r)
                r["sources"] = [r.get("source")]
                by_key[key] = r
                by_title.setdefault(nt, []).append(key)
    out = list(by_key.values())
    for i, r in enumerate(out):
        r["id"] = f"cm{i:06d}"
    return out


def item_text(r: dict) -> str:
    kind = "TV series drama show" if r["kind"] == "tv" else "film movie"
    return (f"{r['title']}. {r['industry']} {kind} ({r.get('year') or ''}). "
            f"Genres: {', '.join(r.get('genres', []))}. Themes: {', '.join(r.get('tags', []))}. "
            f"{r.get('overview', '')}")


def build(offline: bool = False, log=print) -> dict:
    t0 = time.time()
    status: dict = {"sources": {}}
    groups = [seed_records()]
    status["sources"]["seed"] = len(groups[0])

    def attempt(name, fn):
        if offline:
            status["sources"][name] = "skipped (offline)"
            return
        try:
            recs = fn()
            groups.append(recs)
            status["sources"][name] = len(recs)
        except Exception as e:  # noqa: BLE001
            status["sources"][name] = f"failed: {type(e).__name__}"
            log(f"[warn] {name} unavailable -> continuing without it ({type(e).__name__})")

    attempt("tmdb", lambda: sources.tmdb_records(int(config.get("TMDB_PAGES")), log))
    attempt("tvmaze", lambda: sources.tvmaze_records(int(config.get("TVMAZE_PAGES")), log))
    attempt("wikidata", lambda: sources.wikidata_records(int(config.get("WIKIDATA_LIMIT")), log))
    ml_ok = False
    if not offline:
        try:
            ml_ok = sources.fetch_movielens(log)
        except Exception as e:  # noqa: BLE001
            log(f"[warn] MovieLens unavailable ({type(e).__name__}) -> CF disabled")
    ml = sources.movielens_records()
    if ml:
        if config.get("ADD_MOVIELENS_TITLES") == "1":
            groups.append(ml)
        else:  # only use to attach ml_id to known titles
            groups.append([dict(m, _ml_only=True) for m in ml])
    status["sources"]["movielens"] = len(ml)

    catalog = merge_records(groups)
    catalog = [r for r in catalog if not r.pop("_ml_only", False) or r.get("sources") != ["movielens"]]
    for i, r in enumerate(catalog):
        r["id"] = f"cm{i:06d}"
    if not offline:
        try:
            sources.omdb_enrich(catalog, log=log)
        except Exception as e:  # noqa: BLE001
            log(f"[warn] OMDb skipped ({e})")
    log(f"Catalog: {len(catalog)} unique titles")

    # embeddings
    texts = [item_text(r) for r in catalog]
    prefer_st = config.get("USE_SENTENCE_TRANSFORMERS", "1") == "1"
    emb = Embedder().fit(texts, prefer_st=prefer_st)
    log(f"Embedding backend: {emb.backend}")
    E = emb.encode(texts)
    emb.save()
    np.save(config.EMB_FILE, E)

    V = vibe_matrix(catalog)
    for r, v in zip(catalog, V):
        r["vibe"] = [round(float(x), 3) for x in v]
    with open(config.CATALOG_FILE, "w", encoding="utf-8") as f:
        for r in catalog:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # collaborative filtering
    from .cf import ItemFactorModel, load_ratings
    ratings = load_ratings()
    if ratings is not None:
        ItemFactorModel(64).fit(ratings).save()
        status["cf"] = "trained on MovieLens"
        log("CF model trained")
    else:
        status["cf"] = "unavailable (no MovieLens)"

    by_ind: dict[str, int] = {}
    for r in catalog:
        by_ind[r["industry"]] = by_ind.get(r["industry"], 0) + 1
    status.update({"n_titles": len(catalog), "by_industry": by_ind, "embedding_backend": emb.backend,
                   "seconds": round(time.time() - t0, 1), "built_at": time.strftime("%Y-%m-%d %H:%M"),
                   "ml_downloaded": ml_ok})
    config.BUILD_LOG.write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build(offline=a.offline), indent=2))
