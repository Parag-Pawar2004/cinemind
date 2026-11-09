"""Multi-agent recommendation pipeline.

  User text --> [IntentAgent] --> structured Intent (filters, refs, vibes)
            --> [RetrievalAgent] hybrid semantic+lexical+CF+vibe search (tool: Engine)
            --> [CriticAgent]  LLM-as-judge re-rank for fit + diversity (fallback: MMR)
            --> [ExplainerAgent] grounded one-line "why" per pick
Every agent works with a free LLM *or* without one (deterministic fallback), and each
step is logged to a trace shown in the dashboard.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time

from . import config, llm, sources
from .engine import Engine, Intent
from .vibes import AXES

THIS_YEAR = dt.date.today().year

INDUSTRY_WORDS = {
    "Korean": ["kdrama", "k-drama", "k drama", "korean", "korea", "hallyu"],
    "Chinese": ["cdrama", "c-drama", "c drama", "chinese", "china", "mandarin", "wuxia", "xianxia", "hong kong"],
    "Bollywood": ["bollywood", "hindi"],
    "Tollywood": ["tollywood", "telugu"],
    "Hollywood": ["hollywood", "american", "english"],
    "Kollywood": ["kollywood", "tamil"],
    "Mollywood": ["mollywood", "malayalam"],
    "Japanese": ["japanese", "j-drama", "jdrama"],
}
GENRE_WORDS = {
    "Horror": ["horror", "scary", "spooky"], "Thriller": ["thriller", "suspense"],
    "Comedy": ["comedy", "comedies", "funny", "hilarious", "romcom", "rom-com"],
    "Romance": ["romance", "romantic", "love story", "romcom", "rom-com"],
    "Action": ["action", "fight", "martial arts"], "Science Fiction": ["sci-fi", "scifi", "science fiction", "space"],
    "Fantasy": ["fantasy", "magic", "mythology", "mythological", "xianxia"], "Mystery": ["mystery", "whodunit", "detective"],
    "Crime": ["crime", "gangster", "heist", "mafia"], "Animation": ["animation", "animated", "anime", "cartoon"],
    "Family": ["family", "kids"], "War": ["war movie", "war film", "war drama", "war genre"], "History": ["historical", "history", "period", "sageuk"],
    "Documentary": ["documentary"], "Music": ["musical", "music"], "Sport": ["sport", "sports", "cricket", "football"],
    "Biography": ["biopic", "biography", "true story"],
}
VIBE_WORDS = {
    ("darkness", .85): ["dark", "gritty", "disturbing", "bleak", "twisted"],
    ("darkness", .15): ["light", "lighthearted", "light-hearted", "feel-good", "feel good", "wholesome", "cozy", "comfort"],
    ("pace", .85): ["fast", "fast-paced", "thrilling", "adrenaline", "edge of my seat", "gripping", "binge"],
    ("pace", .2): ["slow burn", "slow-burn", "slow paced", "calm"],
    ("humor", .85): ["funny", "hilarious", "laugh", "comedy"],
    ("humor", .15): ["serious"],
    ("complexity", .9): ["mind-bending", "mind bending", "complex", "twist", "twisty", "cerebral", "confusing", "puzzle"],
    ("complexity", .15): ["easy watch", "mindless", "simple", "no brainer"],
    ("feels", .9): ["cry", "tearjerker", "emotional", "heartbreaking", "sad", "touching"],
    ("feels", .2): ["chill", "relaxing", "background"],
    ("romance", .85): ["romantic", "love story", "swoon"],
}


def _has(q: str, w: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", q) is not None


# =============================================================== Intent agent
class IntentAgent:
    SYSTEM = (
        "You convert a viewer's request into a JSON search intent for a movie/TV recommender. "
        "Return ONLY JSON with keys: query_text (string: rewrite of the request as a rich plot/theme "
        "description, no filler), industries (subset of {inds}), kind ('movie'|'tv'|null; K-dramas/C-dramas "
        "are 'tv'), genres_include (list), genres_exclude (list), year_min (int|null), year_max (int|null), "
        "reference_titles (titles the user says they liked / wants similar to), exclude_titles (already "
        "seen), vibes (object with any of {axes} -> 0..1; pace 1=fast, darkness 1=dark, romance 1=very "
        "romantic, humor 1=very funny, complexity 1=mind-bending, feels 1=very emotional; only include axes "
        "the user implied), min_rating (0-10|null), n (int, default 10). Current year: {year}."
    )

    def __init__(self, engine: Engine):
        self.engine = engine

    def rules(self, q: str) -> dict:
        ql = " " + q.lower() + " "
        d: dict = {"query_text": q}
        d["industries"] = [ind for ind, ws in INDUSTRY_WORDS.items() if any(_has(ql, w) for w in ws)]
        if re.search(r"\b(series|shows?|tv|k-?drama|c-?drama|j-?drama|episodes?|binge)\b", ql):
            d["kind"] = "tv"
        elif re.search(r"\b(movies?|films?|cinema)\b", ql):
            d["kind"] = "movie"
        excl, incl = [], []
        for g, ws in GENRE_WORDS.items():
            for w in ws:
                if re.search(rf"\b(no|not|without|avoid|hate|except)\s+(any\s+)?{re.escape(w)}", ql):
                    excl.append(g)
                elif _has(ql, w):
                    incl.append(g)
        d["genres_exclude"] = sorted(set(excl))
        # genres from vibe words (e.g. "funny") are soft: only hard-filter on explicit genre nouns
        hard = {"Horror", "Animation", "Documentary", "Sport", "Biography", "War"}
        d["genres_include"] = sorted({g for g in incl if g in hard} - set(excl))
        m = re.search(r"\b(?:19|20)?(\d0)'?s\b", ql)
        if m:
            dec = int(m.group(1))
            base = 1900 + dec if dec >= 30 else 2000 + dec
            d["year_min"], d["year_max"] = base, base + 9
        if (m := re.search(r"\b(?:after|since|post|newer than)\s+((?:19|20)\d\d)", ql)):
            d["year_min"] = int(m.group(1))
        if (m := re.search(r"\b(?:before|pre|older than)\s+((?:19|20)\d\d)", ql)):
            d["year_max"] = int(m.group(1))
        if re.search(r"\b(recent|new|latest)\b", ql):
            d["year_min"] = THIS_YEAR - 5
        if re.search(r"\b(classic|old school|retro)\b", ql):
            d["year_max"] = 2005
        vibes = {}
        for (ax, val), ws in VIBE_WORDS.items():
            if any(_has(ql, w) for w in ws):
                vibes[ax] = val
        if re.search(r"\b(no|without|not)\s+(much\s+)?(romance|love)", ql):
            vibes["romance"] = 0.1
        d["vibes"] = vibes
        # reference titles: "like X", "similar to X", "loved X"
        refs = []
        for m in re.finditer(r"(?:like|similar to|loved|enjoyed|fan of|fans of|vibes? of|such as)\s+"
                             r"([\"']?[A-Za-z0-9][^,.;!?\"]*?)(?=\s+(?:but|and|with|in|from|without|set|that|which)\b|[,.;!?\"]|$)",
                             q, flags=re.I):
            cand = m.group(1).strip(" '\"")
            words = cand.split()
            for L in range(len(words), 0, -1):  # longest prefix that is a real title
                i = self.engine.find_title(" ".join(words[:L]), cutoff=0.88)
                if i is not None:
                    refs.append(self.engine.items[i]["title"])
                    break
        refs += [self.engine.items[i]["title"] for i in
                 (self.engine.find_title(x, 0.88) for x in re.findall(r"\"([^\"]+)\"", q)) if i is not None]
        d["reference_titles"] = list(dict.fromkeys(refs))
        if (m := re.search(r"\b(?:top|give me|show me)\s+(\d{1,2})\b", ql)):
            d["n"] = int(m.group(1))
        return d

    def run(self, q: str, prev: dict | None = None) -> tuple[Intent, str]:
        sys = self.SYSTEM.format(inds=self.engine.industries, axes=AXES, year=THIS_YEAR)
        msgs = [{"role": "system", "content": sys}]
        if prev:
            msgs.append({"role": "user", "content": "Previous intent: " + json.dumps(prev) +
                         "\nThe user now refines it. Update the previous intent (keep what still applies)."})
        msgs.append({"role": "user", "content": q})
        out = llm.chat_json(msgs)
        if isinstance(out, dict):
            it = Intent.from_dict(out)
            it.industries = [i for i in it.industries if i in self.engine.industries]
            if not it.query_text:
                it.query_text = q
            return it, "llm"
        d = self.rules(q)
        if prev:  # rule-based refinement: new constraints override, others persist
            merged = dict(prev)
            for k, v in d.items():
                if v not in (None, [], {}, ""):
                    merged[k] = ({**prev.get("vibes", {}), **v} if k == "vibes" else v)
            merged["query_text"] = (prev.get("query_text", "") + " " + q).strip()
            d = merged
        return Intent.from_dict(d), "rules"


# =============================================================== Critic agent
class CriticAgent:
    SYSTEM = ("You are a film critic curating recommendations. Given the viewer request and candidate "
              "titles, choose the {n} best matches: they must truly fit the request, avoid near-duplicates, "
              "and include at least one less-obvious gem if possible. Return JSON: "
              "{{\"picks\":[{{\"id\":\"...\",\"why\":\"one specific sentence tied to the request\"}}], "
              "\"summary\":\"one friendly sentence\"}}. Use only ids from the list.")

    def __init__(self, engine: Engine):
        self.engine = engine

    def run(self, request: str, it: Intent, cands: list[dict]) -> tuple[list[dict], str, str]:
        n = it.n
        pool = cands[: max(25, n * 2)]
        lines = []
        for c in pool:
            r = self.engine.items[c["idx"]]
            lines.append(f"{r['id']} | {r['title']} ({r.get('year')}) | {r['industry']} {r['kind']} | "
                         f"{', '.join(r.get('genres', [])[:3])} | {(r.get('overview') or '')[:160]}")
        out = llm.chat_json([
            {"role": "system", "content": self.SYSTEM.format(n=n)},
            {"role": "user", "content": f"Request: {request}\n\nCandidates:\n" + "\n".join(lines)}])
        if isinstance(out, dict) and isinstance(out.get("picks"), list):
            byid = {self.engine.items[c["idx"]]["id"]: c for c in pool}
            picks, seen = [], set()
            for p in out["picks"]:
                c = byid.get(str(p.get("id")))
                if c and c["idx"] not in seen:
                    seen.add(c["idx"])
                    picks.append({**c, "why": str(p.get("why", ""))[:300]})
            if len(picks) >= max(1, n // 2):
                for c in self.engine.mmr(pool, n * 2):  # top-up if the LLM returned too few
                    if len(picks) >= n:
                        break
                    if c["idx"] not in seen:
                        picks.append(c)
                        seen.add(c["idx"])
                return picks[:n], str(out.get("summary", "")), "llm"
        lam = 0.85 - 0.35 * it.serendipity
        return self.engine.mmr(cands, n, lam=lam), "", "mmr"


# =============================================================== Explainer agent
class ExplainerAgent:
    def __init__(self, engine: Engine):
        self.engine = engine

    def run(self, it: Intent, picks: list[dict]) -> list[dict]:
        for p in picks:
            if not p.get("why"):
                ev = self.engine.evidence(p["idx"], it, p.get("components", {}))
                r = self.engine.items[p["idx"]]
                p["why"] = (f"{r['industry']} {('series' if r['kind'] == 'tv' else 'film')}"
                            + (": " + "; ".join(ev) if ev else " that matches your description") + ".")
        return picks


# =============================================================== Recall agent
class RecallAgent:
    """LLM proposes real titles from its own knowledge; free APIs verify each one exists and fetch metadata;
    verified titles missing from the catalog are added on the fly. Grounded, so hallucinated titles are dropped."""

    SYSTEM = ("You are a film and TV encyclopedia. Given a viewer's request, name up to 8 REAL titles "
              "(movies or TV series, any country) that best match the specific plot, theme or setting described. "
              "Only include titles you are confident exist; prefer the official English or romanised title. "
              "Return ONLY JSON: {\"titles\":[{\"title\":\"...\",\"year\":2022,\"kind\":\"tv\"}]} "
              "where kind is \"tv\" or \"movie\".")

    def __init__(self, engine: Engine):
        self.engine = engine

    def run(self, request: str, it: Intent) -> tuple[list[int], dict]:
        info = {"suggested": [], "in_catalog": [], "added": [], "unresolved": []}
        if not llm.active_provider() or config.get("LLM_RECALL") == "0":
            return [], {**info, "skipped": "no LLM configured or recall disabled"}
        hint = f" Region hint: {', '.join(it.industries)}." if it.industries else ""
        out = llm.chat_json([{"role": "system", "content": self.SYSTEM},
                             {"role": "user", "content": request + hint}])
        titles = out.get("titles") if isinstance(out, dict) else None
        if not isinstance(titles, list):
            return [], {**info, "skipped": "LLM returned no titles"}
        idxs = []
        for t in titles[:8]:
            if not isinstance(t, dict) or not str(t.get("title") or "").strip():
                continue
            name = str(t["title"]).strip()
            try:
                year = int(t.get("year"))
            except (TypeError, ValueError):
                year = None
            info["suggested"].append(f"{name} ({year})" if year else name)
            i = self.engine.find_title(name, cutoff=0.9)
            if i is not None and (not year or not self.engine.year[i] or abs(int(self.engine.year[i]) - year) <= 1):
                info["in_catalog"].append(self.engine.items[i]["title"])
                idxs.append(i)
                continue
            rec = sources.lookup_title(name, year, t.get("kind"))
            if not rec:
                info["unresolved"].append(name)
                continue
            n_before = len(self.engine.items)
            j = self.engine.add_items([rec])[0]
            idxs.append(j)
            info["added" if j >= n_before else "in_catalog"].append(self.engine.items[j]["title"])
        return list(dict.fromkeys(idxs)), info


# =============================================================== Orchestrator
class Orchestrator:
    def __init__(self, engine: Engine, recall: bool = True):
        self.engine = engine
        self.recall_enabled = recall
        self.recall = RecallAgent(engine)
        self.intent_agent = IntentAgent(engine)
        self.critic = CriticAgent(engine)
        self.explainer = ExplainerAgent(engine)

    def run(self, query: str, prev_intent: dict | None = None, overrides: dict | None = None,
            liked: list[int] | None = None) -> dict:
        trace = []
        t = time.time()
        it, mode = self.intent_agent.run(query, prev_intent)
        for k, v in (overrides or {}).items():
            if v not in (None, [], {}):
                setattr(it, k, ({**it.vibes, **v} if k == "vibes" else v))
        trace.append({"agent": "IntentAgent", "mode": mode, "ms": int((time.time() - t) * 1000),
                      "output": {k: v for k, v in it.to_dict().items() if v not in (None, [], {}, "")}})

        boost, learned = [], []
        if self.recall_enabled:
            t = time.time()
            boost, rinfo = self.recall.run(query, it)
            learned = rinfo.get("added", [])
            trace.append({"agent": "RecallAgent", "mode": "skipped" if "skipped" in rinfo else "llm+verify",
                          "ms": int((time.time() - t) * 1000), "output": rinfo})

        t = time.time()
        prof, cf_scores = self.engine.taste(liked or [])
        cands, info = self.engine.search(it, profile_vec=prof, cf_scores=cf_scores, exclude_idx=liked,
                                         boost_idx=boost)
        if not cands and (it.genres_include or it.year_min or it.year_max or it.kind):
            relaxed = Intent.from_dict({**it.to_dict(), "genres_include": [], "year_min": None,
                                        "year_max": None, "kind": None})
            cands, info = self.engine.search(relaxed, profile_vec=prof, cf_scores=cf_scores, exclude_idx=liked,
                                             boost_idx=boost)
            info["relaxed"] = True
        trace.append({"agent": "RetrievalAgent", "mode": "hybrid", "ms": int((time.time() - t) * 1000),
                      "output": {**info, "top_candidates": [self.engine.items[c["idx"]]["title"] for c in cands[:8]]}})

        t = time.time()
        picks, summary, cmode = self.critic.run(query, it, cands)
        trace.append({"agent": "CriticAgent", "mode": cmode, "ms": int((time.time() - t) * 1000),
                      "output": {"picked": [self.engine.items[p["idx"]]["title"] for p in picks]}})

        t = time.time()
        picks = self.explainer.run(it, picks)
        trace.append({"agent": "ExplainerAgent", "mode": "llm" if cmode == "llm" else "grounded-template",
                      "ms": int((time.time() - t) * 1000), "output": {"n": len(picks)}})
        return {"intent": it.to_dict(), "results": [self.engine.card(p) for p in picks],
                "summary": summary, "trace": trace, "learned": learned, "provider": llm.active_provider() or "none (rule-based)",
                "llm_error": llm.last_error()}
