"""Free data-source connectors. Each one fails soft: if a source is unreachable or a key
is missing, it returns [] and the build continues (self-reliant by design).

| Source     | Key needed? | What it adds                                             |
|------------|-------------|----------------------------------------------------------|
| MovieLens  | no          | 100k real user ratings -> collaborative filtering + eval |
| TVMaze     | no          | K-dramas / C-dramas / Indian TV with summaries           |
| Wikidata   | no          | Telugu / Hindi / Korean / Chinese films (+director)      |
| TMDB       | free key    | Posters, overviews, ratings for every industry           |
| OMDb       | free key    | IMDb ratings enrichment                                  |
"""
from __future__ import annotations

import html
import io
import re
import time
import zipfile

import requests

from . import config

UA = {"User-Agent": "CineMind/1.0 (open-source educational recommender)"}

# ISO language -> industry label
LANG_INDUSTRY = {
    "en": "Hollywood", "hi": "Bollywood", "te": "Tollywood", "ko": "Korean",
    "zh": "Chinese", "cn": "Chinese", "ta": "Kollywood", "ml": "Mollywood",
    "kn": "Sandalwood", "ja": "Japanese",
}
TVMAZE_LANG = {"Korean": "ko", "Chinese": "zh", "Hindi": "hi", "Telugu": "te",
               "Tamil": "ta", "Japanese": "ja", "Malayalam": "ml"}


def _get(url, params=None, headers=None, timeout=30, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 + 2 * i)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(1 + i)
    return None


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


# ---------------------------------------------------------------- MovieLens
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def fetch_movielens(log=print) -> bool:
    dest = config.RAW_DIR / "ml-latest-small"
    if (dest / "ratings.csv").exists():
        log("MovieLens: already downloaded")
        return True
    log("MovieLens: downloading ml-latest-small (~1 MB)...")
    r = _get(ML_URL, timeout=120)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(config.RAW_DIR)
    log("MovieLens: done")
    return True


def movielens_records() -> list[dict]:
    import pandas as pd
    p = config.RAW_DIR / "ml-latest-small" / "movies.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    out = []
    for _, row in df.iterrows():
        m = re.match(r"^(.*)\s+\((\d{4})\)\s*$", str(row["title"]))
        title, year = (m.group(1), int(m.group(2))) if m else (row["title"], None)
        # "Matrix, The" -> "The Matrix"
        title = re.sub(r"^(.*), (The|A|An)$", r"\2 \1", title)
        genres = [g for g in str(row["genres"]).split("|") if g != "(no genres listed)"]
        out.append({"title": title, "year": year, "kind": "movie", "industry": "Global",
                    "language": "unknown", "genres": genres, "overview": "", "tags": [],
                    "popularity": 0.0, "rating": None, "poster": None, "source": "movielens",
                    "ml_id": int(row["movieId"])})
    return out


# ---------------------------------------------------------------- TVMaze (no key)
def tvmaze_records(pages: int = 60, log=print) -> list[dict]:
    out = []
    for page in range(pages):
        try:
            r = requests.get(f"https://api.tvmaze.com/shows?page={page}", headers=UA, timeout=30)
            if r.status_code == 404:
                break
            if r.status_code == 429:
                time.sleep(3)
                continue
            shows = r.json()
        except Exception as e:  # noqa: BLE001
            log(f"TVMaze page {page} failed: {type(e).__name__} (check internet/firewall)")
            break
        for s in shows:
            lang = TVMAZE_LANG.get(s.get("language") or "")
            if not lang:
                continue
            prem = s.get("premiered") or ""
            ind = LANG_INDUSTRY[lang]
            out.append({
                "title": s.get("name"), "year": int(prem[:4]) if prem[:4].isdigit() else None,
                "kind": "tv", "industry": ind, "language": lang,
                "genres": s.get("genres") or [], "overview": _clean(s.get("summary"))[:600],
                "tags": [s.get("type", "")] if s.get("type") else [],
                "popularity": float(s.get("weight") or 0),
                "rating": (s.get("rating") or {}).get("average"),
                "poster": (s.get("image") or {}).get("medium"), "source": "tvmaze",
            })
        time.sleep(0.25)
    log(f"TVMaze: {len(out)} Asian/Indian shows")
    return out


# ---------------------------------------------------------------- Wikidata (no key)
WD_LANGS = {"te": "Q8097", "hi": "Q1568", "ko": "Q9176", "zh": "Q7850", "ta": "Q5885", "ml": "Q36236"}
WD_QUERY = """
SELECT ?film ?filmLabel (YEAR(MIN(?date)) AS ?year)
       (GROUP_CONCAT(DISTINCT ?genreLabel; separator="|") AS ?genres)
       (SAMPLE(?dirLabel) AS ?director) (COUNT(DISTINCT ?sitelink) AS ?links) WHERE {
  ?film wdt:P31 wd:Q11424; wdt:P364 wd:%s; wdt:P577 ?date.
  OPTIONAL { ?film wdt:P136 ?g. ?g rdfs:label ?genreLabel. FILTER(LANG(?genreLabel)="en") }
  OPTIONAL { ?film wdt:P57 ?d. ?d rdfs:label ?dirLabel. FILTER(LANG(?dirLabel)="en") }
  OPTIONAL { ?sitelink schema:about ?film. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} GROUP BY ?film ?filmLabel ORDER BY DESC(?links) LIMIT %d
"""


def wikidata_records(limit: int = 800, log=print) -> list[dict]:
    out = []
    for lang, qid in WD_LANGS.items():
        try:
            r = _get("https://query.wikidata.org/sparql",
                     params={"query": WD_QUERY % (qid, limit), "format": "json"},
                     headers={"Accept": "application/sparql-results+json"}, timeout=90)
            rows = r.json()["results"]["bindings"]
        except Exception as e:  # noqa: BLE001
            log(f"Wikidata {lang} failed: {type(e).__name__}")
            continue
        for b in rows:
            title = b["filmLabel"]["value"]
            if re.match(r"^Q\d+$", title):
                continue
            genres = [g.replace(" film", "").title() for g in b.get("genres", {}).get("value", "").split("|") if g]
            director = b.get("director", {}).get("value")
            ind = LANG_INDUSTRY[lang]
            ov = f"{ind} film" + (f" directed by {director}" if director else "")
            if genres:
                ov += f"; {', '.join(genres[:4]).lower()}."
            out.append({"title": title, "year": int(b["year"]["value"]) if "year" in b else None,
                        "kind": "movie", "industry": ind, "language": lang, "genres": genres[:5],
                        "overview": ov, "tags": [f"dir:{director}"] if director else [],
                        "popularity": float(b.get("links", {}).get("value", 0)), "rating": None,
                        "poster": None, "source": "wikidata"})
        log(f"Wikidata {lang}: {len(rows)} films")
        time.sleep(1)
    return out


# ---------------------------------------------------------------- TMDB (free key)
TMDB = "https://api.themoviedb.org/3"


def _tmdb_auth():
    key = config.get("TMDB_API_KEY")
    if not key:
        return None, None
    if len(key) > 40:  # v4 read-access token
        return {}, {"Authorization": f"Bearer {key}"}
    return {"api_key": key}, {}


def tmdb_records(pages: int = 15, log=print) -> list[dict]:
    params, headers = _tmdb_auth()
    if params is None:
        log("TMDB: no key, skipped")
        return []
    gmap = {}
    for kind in ("movie", "tv"):
        r = _get(f"{TMDB}/genre/{kind}/list", params={**params, "language": "en-US"}, headers=headers)
        gmap.update({g["id"]: g["name"] for g in r.json()["genres"]})
    out = []
    for lang, ind in LANG_INDUSTRY.items():
        if lang == "cn":
            continue
        for kind in ("movie", "tv"):
            n_pages = pages if lang in ("en", "hi", "te", "ko", "zh") else max(3, pages // 3)
            for page in range(1, n_pages + 1):
                try:
                    r = _get(f"{TMDB}/discover/{kind}", headers=headers, params={
                        **params, "with_original_language": lang, "sort_by": "vote_count.desc",
                        "page": page, "language": "en-US", "include_adult": "false"})
                    res = r.json().get("results", [])
                except Exception as e:  # noqa: BLE001
                    log(f"TMDB {lang}/{kind} p{page} failed: {type(e).__name__}")
                    break
                if not res:
                    break
                for m in res:
                    date = m.get("release_date") or m.get("first_air_date") or ""
                    out.append({
                        "title": m.get("title") or m.get("name"),
                        "year": int(date[:4]) if date[:4].isdigit() else None,
                        "kind": kind, "industry": ind, "language": lang,
                        "genres": [gmap.get(g, "") for g in m.get("genre_ids", []) if g in gmap],
                        "overview": m.get("overview") or "", "tags": [],
                        "popularity": float(m.get("popularity") or 0),
                        "rating": m.get("vote_average"),
                        "poster": f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get("poster_path") else None,
                        "source": "tmdb", "tmdb_id": m.get("id"),
                    })
                time.sleep(0.05)
        log(f"TMDB {lang}: total so far {len(out)}")
    return out


# ---------------------------------------------------------------- OMDb (free key)
def omdb_enrich(records: list[dict], max_calls: int = 300, log=print) -> int:
    key = config.get("OMDB_API_KEY")
    if not key:
        return 0
    todo = [r for r in records if r.get("rating") is None and r["industry"] != "Global"]
    todo.sort(key=lambda r: -r.get("popularity", 0))
    n = 0
    for r in todo[:max_calls]:
        try:
            j = _get("https://www.omdbapi.com/", params={"apikey": key, "t": r["title"], "y": r.get("year") or ""}).json()
        except Exception:  # noqa: BLE001
            continue
        if j.get("Response") == "True":
            try:
                r["rating"] = float(j.get("imdbRating"))
            except (TypeError, ValueError):
                pass
            if not r.get("poster") and j.get("Poster", "N/A") != "N/A":
                r["poster"] = j["Poster"]
            if len(r.get("overview", "")) < 40 and j.get("Plot", "N/A") != "N/A":
                r["overview"] = j["Plot"]
            n += 1
    log(f"OMDb: enriched {n}")
    return n


# ---------------------------------------------------------------- live single-title lookup
_GENRES: dict = {}


def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    from .build import norm_title
    return SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def _year_ok(want, got, tol=2) -> bool:
    return not want or not got or abs(int(want) - int(got)) <= tol


def _tmdb_lookup(title, year, kind):
    params, headers = _tmdb_auth()
    if params is None:
        return None
    if not _GENRES:
        for k in ("movie", "tv"):
            r = _get(f"{TMDB}/genre/{k}/list", params={**params, "language": "en-US"}, headers=headers)
            _GENRES.update({g["id"]: g["name"] for g in r.json()["genres"]})
    best = None
    for k in ([kind] if kind in ("movie", "tv") else ["tv", "movie"]):
        for use_year in (True, False):
            if not use_year and best is not None:
                break
            q = {**params, "query": title, "include_adult": "false", "language": "en-US"}
            if use_year and year:
                q["first_air_date_year" if k == "tv" else "year"] = year
            elif not use_year and not year:
                break
            res = _get(f"{TMDB}/search/{k}", params=q, headers=headers).json().get("results", [])
            for m in res[:6]:
                name = m.get("title") or m.get("name") or ""
                orig = m.get("original_title") or m.get("original_name") or ""
                score = max(_ratio(title, name), _ratio(title, orig))
                date = m.get("release_date") or m.get("first_air_date") or ""
                y = int(date[:4]) if date[:4].isdigit() else None
                if score >= 0.8 and _year_ok(year, y):
                    key = (score, m.get("popularity") or 0)
                    if best is None or key > best[0]:
                        best = (key, m, k, y)
    if not best:
        return None
    _, m, k, y = best
    lang = m.get("original_language") or "unknown"
    return {"title": m.get("title") or m.get("name"), "year": y, "kind": k,
            "industry": LANG_INDUSTRY.get(lang, "Global"), "language": lang,
            "genres": [_GENRES[g] for g in m.get("genre_ids", []) if g in _GENRES],
            "overview": m.get("overview") or "", "tags": [], "popularity": float(m.get("popularity") or 0),
            "rating": m.get("vote_average") or None,
            "poster": f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get("poster_path") else None,
            "source": "tmdb-lookup", "tmdb_id": m.get("id")}


def _tvmaze_lookup(title, year):
    r = requests.get("https://api.tvmaze.com/search/shows", params={"q": title}, headers=UA, timeout=20)
    r.raise_for_status()
    best = None
    for item in r.json()[:6]:
        s = item["show"]
        prem = s.get("premiered") or ""
        y = int(prem[:4]) if prem[:4].isdigit() else None
        score = _ratio(title, s.get("name") or "")
        if score >= 0.8 and _year_ok(year, y):
            if best is None or (score, item.get("score", 0)) > best[0]:
                best = ((score, item.get("score", 0)), s, y)
    if not best:
        return None
    _, s, y = best
    lang = {**TVMAZE_LANG, "English": "en"}.get(s.get("language") or "", "unknown")
    return {"title": s["name"], "year": y, "kind": "tv", "industry": LANG_INDUSTRY.get(lang, "Global"),
            "language": lang, "genres": s.get("genres") or [], "overview": _clean(s.get("summary"))[:600],
            "tags": [], "popularity": float(s.get("weight") or 0), "rating": (s.get("rating") or {}).get("average"),
            "poster": (s.get("image") or {}).get("medium"), "source": "tvmaze-lookup"}


def lookup_title(title: str, year: int | None = None, kind: str | None = None, log=print) -> dict | None:
    """Verify a title exists and fetch its metadata. TMDB first (needs free key), then TVMaze (no key, TV only).
    Returns a catalog record or None. Never raises."""
    for fn in (lambda: _tmdb_lookup(title, year, kind),
               lambda: _tvmaze_lookup(title, year) if kind != "movie" else None):
        try:
            rec = fn()
            if rec:
                return rec
        except Exception as e:  # noqa: BLE001
            log(f"[lookup] {title}: {type(e).__name__}")
    return None
