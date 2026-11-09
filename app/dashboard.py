"""CineMind dashboard.  Run:  streamlit run app/dashboard.py"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cinemind import config, llm  # noqa: E402
from cinemind.vibes import AXES, AXIS_LABELS  # noqa: E402

st.set_page_config(page_title="CineMind", page_icon="🎬", layout="wide")

CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1400px}
h1.cm-title{font-family:Georgia,serif;font-size:2.6rem;letter-spacing:-.02em;margin:0}
.cm-sub{color:#9aa0a6;margin:.2rem 0 1rem 0}
.cm-card{border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;background:#15171c;height:100%}
.cm-poster{height:230px;background-size:cover;background-position:center;display:flex;align-items:flex-end}
.cm-ph{height:230px;display:flex;align-items:center;justify-content:center;font:700 2.2rem Georgia,serif;color:#fff}
.cm-body{padding:.7rem .8rem .9rem .8rem}
.cm-t{font-weight:700;font-size:1rem;line-height:1.2;color:#f5f5f5}
.cm-m{font-size:.75rem;color:#9aa0a6;margin:.2rem 0 .4rem 0}
.cm-why{font-size:.8rem;color:#d7d7d7;line-height:1.35}
.cm-chip{display:inline-block;font-size:.66rem;padding:.08rem .45rem;border-radius:999px;margin:0 .2rem .2rem 0;
         background:rgba(232,179,57,.14);color:#e8b339;border:1px solid rgba(232,179,57,.35)}
.cm-bar{height:4px;background:#2a2d34;border-radius:2px;margin-top:2px}
.cm-bar>div{height:4px;border-radius:2px;background:#e8b339}
.cm-vl{font-size:.62rem;color:#80868b;display:flex;justify-content:space-between}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

IND_COLORS = {"Hollywood": "#4C78A8", "Bollywood": "#F58518", "Tollywood": "#E45756", "Korean": "#72B7B2",
              "Chinese": "#B279A2", "Kollywood": "#54A24B", "Mollywood": "#EECA3B", "Japanese": "#FF9DA6",
              "Sandalwood": "#9D755D", "Global": "#79706E"}


# ------------------------------------------------------------------ loading
def index_ready() -> bool:
    return config.CATALOG_FILE.exists() and config.EMB_FILE.exists() and config.EMBEDDER_FILE.exists()


@st.cache_resource(show_spinner="Loading CineMind index…")
def load_engine(stamp: float):
    from cinemind.agents import Orchestrator
    from cinemind.engine import Engine
    e = Engine()
    return e, Orchestrator(e)


def run_build(offline: bool):
    cmd = [sys.executable, "-m", "cinemind.build"] + (["--offline"] if offline else [])
    with st.status("Building catalog & index (this can take several minutes online)…", expanded=True) as s:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        box = st.empty()
        lines = []
        for line in p.stdout:
            lines.append(line.rstrip())
            box.code("\n".join(lines[-15:]))
        p.wait()
        s.update(label="Build finished" if p.returncode == 0 else "Build failed", state="complete" if p.returncode == 0 else "error")
    st.cache_resource.clear()


if not index_ready():
    st.markdown("<h1 class='cm-title'>🎬 CineMind</h1>", unsafe_allow_html=True)
    st.warning("No index yet. Build it now (offline seed works instantly; online pulls TMDB/TVMaze/Wikidata/MovieLens).")
    c1, c2 = st.columns(2)
    if c1.button("⚡ Quick offline build"):
        run_build(True)
        st.rerun()
    if c2.button("🌐 Full online build"):
        run_build(False)
        st.rerun()
    st.stop()

engine, orch = load_engine(config.CATALOG_FILE.stat().st_mtime)
build_info = json.loads(config.BUILD_LOG.read_text()) if config.BUILD_LOG.exists() else {}


# ------------------------------------------------------------------ helpers
def vibe_bars(v: dict) -> str:
    out = []
    for a in AXES:
        lo, hi = AXIS_LABELS[a]
        out.append(f"<div class='cm-vl'><span>{lo}</span><span>{hi}</span></div>"
                   f"<div class='cm-bar'><div style='width:{int(v[a] * 100)}%'></div></div>")
    return "".join(out)


def render_cards(cards: list[dict], cols: int = 5, show_vibes: bool = False):
    if not cards:
        st.info("No matches. Try loosening filters.")
        return
    for row in range(0, len(cards), cols):
        cs = st.columns(cols)
        for c, r in zip(cs, cards[row: row + cols]):
            color = IND_COLORS.get(r["industry"], "#555")
            if r.get("poster"):
                top = f"<div class='cm-poster' style=\"background-image:url('{html.escape(r['poster'])}')\"></div>"
            else:
                init = "".join(w[0] for w in r["title"].split()[:2]).upper()
                top = (f"<div class='cm-ph' style='background:linear-gradient(135deg,{color},#111)'>"
                       f"{html.escape(init)}</div>")
            meta = f"{r['industry']} · {'Series' if r['kind'] == 'tv' else 'Film'} · {r.get('year') or '—'}"
            if r.get("rating"):
                meta += f" · ★ {r['rating']:.1f}"
            chips = "".join(f"<span class='cm-chip'>{html.escape(g)}</span>" for g in (r.get("genres") or [])[:3])
            vb = vibe_bars(r["vibe"]) if show_vibes else ""
            c.markdown(f"<div class='cm-card'>{top}<div class='cm-body'><div class='cm-t'>{html.escape(r['title'])}</div>"
                       f"<div class='cm-m'>{meta}</div>{chips}<div class='cm-why'>{html.escape(r.get('why') or '')}</div>"
                       f"{vb}</div></div>", unsafe_allow_html=True)
            with c.expander("Details"):
                st.caption(r.get("overview") or "No overview available.")
                st.json(r.get("components", {}), expanded=False)


def title_picker(label: str, key: str, multi: bool = False, default: list[str] | None = None):
    q = st.text_input(f"Search titles for “{label}”", key=key + "_q", placeholder="type a few letters…")
    pool = [r["title"] + f" ({r.get('year') or '?'})" for r in engine.items
            if not q or q.lower() in r["title"].lower()]
    pool = sorted(set(pool), key=lambda t: (len(t), t))[:300]
    if multi:
        return st.multiselect(label, sorted(set(pool + (default or []))), default=default or [], key=key)
    return st.selectbox(label, pool, key=key) if pool else None


def strip_year(t: str) -> str:
    return t.rsplit(" (", 1)[0] if t else t


def radar(vibe: dict, name="You"):
    labels = [AXIS_LABELS[a][1] for a in AXES]
    vals = [vibe[a] for a in AXES]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself", name=name,
                                    line_color="#e8b339"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), showlegend=False,
                      height=330, margin=dict(l=30, r=30, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------ header
prov = llm.active_provider()
st.markdown("<h1 class='cm-title'>🎬 CineMind</h1>", unsafe_allow_html=True)
st.markdown(f"<div class='cm-sub'>Multi-agent, cross-industry recommender · {len(engine.items):,} titles · "
            f"embeddings: <b>{engine.embedder.backend}</b> · LLM: <b>{prov or 'none (rule-based agents)'}</b> · "
            f"CF: <b>{'on' if engine.cf is not None else 'off'}</b></div>", unsafe_allow_html=True)

tabs = st.tabs(["🔎 Describe & Discover", "🌉 Cross-Industry Bridge", "🧬 Taste DNA", "📊 Catalog Explorer",
                "🧪 Evaluation", "⚙️ Settings & API keys"])

# ================================================================== Discover
with tabs[0]:
    left, right = st.columns([3, 1])
    with right:
        st.markdown("**Steer the agents**")
        inds = st.multiselect("Industries", engine.industries, key="d_inds")
        kind = st.radio("Type", ["Any", "Movies", "Series"], horizontal=True, key="d_kind")
        yr = st.slider("Year range", 1950, 2026, (1950, 2026), key="d_year")
        ser = st.slider("Serendipity (popular → hidden gems)", 0.0, 1.0, 0.2, 0.05, key="d_ser")
        n = st.slider("How many", 5, 30, 10, key="d_n")
        orch.recall_enabled = st.toggle("LLM title recall + live lookup", value=True, key="d_recall",
                                        help="LLM names titles matching your description; they are verified via TMDB/TVMaze and added if missing.")
        use_v = st.toggle("Set vibe manually", key="d_usev")
        vib = {}
        if use_v:
            for a in AXES:
                lo, hi = AXIS_LABELS[a]
                vib[a] = st.slider(f"{lo} ↔ {hi}", 0.0, 1.0, 0.5, 0.05, key=f"d_v_{a}")
    with left:
        examples = ["A dark, twisty revenge K-drama that's a slow burn",
                    "Something like Inception but from Bollywood",
                    "Feel-good Telugu romcom for a lazy Sunday",
                    "Chinese xianxia fantasy romance with a tragic love story",
                    "Emotional father-daughter story that will make me cry, no horror"]
        ex = st.selectbox("Try an example (or write your own below)", ["—"] + examples, key="d_ex")
        q = st.text_area("Describe what you're in the mood for", value="" if ex == "—" else ex, height=90, key="d_q")
        c1, c2, _ = st.columns([1, 1, 3])
        go_btn = c1.button("✨ Recommend", type="primary", width="stretch")
        refine = c2.toggle("Refine previous", key="d_ref", help="Treat this as a follow-up (e.g. 'lighter', 'only series')")
        if go_btn and q.strip():
            ov = {"industries": inds, "kind": {"Movies": "movie", "Series": "tv"}.get(kind),
                  "serendipity": ser, "n": n, "vibes": vib}
            if yr != (1950, 2026):
                ov["year_min"], ov["year_max"] = yr
            prev = st.session_state.get("last_intent") if refine else None
            with st.spinner("Agents at work…"):
                out = orch.run(q, prev_intent=prev, overrides=ov)
            st.session_state["last_out"] = out
            st.session_state["last_intent"] = out["intent"]
        out = st.session_state.get("last_out")
        if out:
            if out.get("summary"):
                st.success(out["summary"])
            if out.get("learned"):
                st.info("➕ Found via live lookup and added to your catalog: " + ", ".join(out["learned"]))
            render_cards(out["results"], cols=5)
            with st.expander("🧠 Agent trace (how this was decided)"):
                for t in out["trace"]:
                    st.markdown(f"**{t['agent']}** · mode `{t['mode']}` · {t['ms']} ms")
                    st.json(t["output"], expanded=False)
                if out.get("llm_error"):
                    st.caption(f"Last LLM error (fell back gracefully): {out['llm_error']}")

# ================================================================== Bridge
with tabs[1]:
    st.markdown("Find the **equivalent** of a title in another film industry using analogical vector arithmetic: "
                "`e(title) − centroid(source industry) + centroid(target industry)`.")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        src = title_picker("Source title", "b_src")
    tgt = c2.selectbox("Target industry", [i for i in engine.industries if i != "Global"], key="b_tgt")
    bk = c3.radio("Type", ["Any", "Movies", "Series"], horizontal=True, key="b_kind")
    alpha = st.slider("Bridge strength α (0 = plain similarity, 1 = full industry shift)", 0.0, 1.2, 0.8, 0.05)
    if st.button("🌉 Build the bridge", type="primary") and src:
        s, res = engine.bridge(strip_year(src), tgt, n=10, alpha=alpha,
                               kind={"Movies": "movie", "Series": "tv"}.get(bk))
        if s is None:
            st.error("Title not found.")
        else:
            st.markdown(f"**{s['title']}** ({s['industry']}) → **{tgt}**")
            cards = []
            for c in res:
                c["why"] = (f"Analogy {c['components']['analogy']:.2f} · vibe match {c['components']['vibe_match']:.2f}"
                            f" · genre overlap {c['components']['genre_overlap']:.2f}")
                cards.append(engine.card(c))
            render_cards(cards, cols=5, show_vibes=True)

# ================================================================== Taste DNA
with tabs[2]:
    st.markdown("Pick titles you love (any industry). CineMind builds your **Taste DNA** from content embeddings, "
                "vibe axes and — for titles in MovieLens — real collaborative signals from 600+ users.")
    c1, c2 = st.columns(2)
    with c1:
        liked = title_picker("Titles you loved", "t_like", multi=True)
    with c2:
        disliked = title_picker("Titles you disliked (optional)", "t_dis", multi=True)
    tk = st.multiselect("Recommend from industries (empty = all)", engine.industries, key="t_inds")
    if st.button("🧬 Analyse my taste", type="primary") and liked:
        li = [i for i in (engine.find_title(strip_year(t)) for t in liked) if i is not None]
        di = [i for i in (engine.find_title(strip_year(t)) for t in disliked) if i is not None]
        dna = engine.taste_dna(li)
        a, b = st.columns([1, 2])
        a.plotly_chart(radar(dna["vibe"]), width="stretch")
        with b:
            st.markdown("**Your genre fingerprint**")
            st.bar_chart(dna["genres"])
            st.caption("Industries: " + ", ".join(f"{k} ×{v}" for k, v in dna["industries"].items()))
        from cinemind.engine import Intent
        prof, cfs = engine.taste(li, di)
        cands, _ = engine.search(Intent(industries=tk, n=15), profile_vec=prof, cf_scores=cfs, exclude_idx=li + di)
        picks = orch.explainer.run(Intent(), engine.mmr(cands, 15, lam=0.8))
        st.markdown("### Picked for you")
        render_cards([engine.card(p) for p in picks], cols=5)

# ================================================================== Explorer
with tabs[3]:
    import pandas as pd
    df = pd.DataFrame([{k: r.get(k) for k in ("title", "year", "kind", "industry", "rating", "popularity")}
                       for r in engine.items])
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(df.groupby(["industry", "kind"]).size().reset_index(name="titles"), x="industry",
                           y="titles", color="kind", title="Titles by industry", template="plotly_dark"),
                    width="stretch")
    c2.plotly_chart(px.histogram(df[df.year > 1950], x="year", color="industry", nbins=60, title="Release years",
                                 template="plotly_dark", color_discrete_map=IND_COLORS), width="stretch")
    st.markdown("**Semantic map** — every dot is a title; nearby dots are similar in story & theme (PCA of embeddings).")
    hide_global = st.checkbox("Hide MovieLens-only 'Global' titles", value=True)
    m = (df.industry != "Global") if hide_global else np.ones(len(df), bool)
    idx = np.where(m)[0]
    if len(idx) > 4000:
        idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
    from sklearn.decomposition import PCA
    xy = PCA(2, random_state=0).fit_transform(engine.E[idx])
    sub = df.iloc[idx].assign(x=xy[:, 0], y=xy[:, 1])
    st.plotly_chart(px.scatter(sub, x="x", y="y", color="industry", hover_name="title", hover_data=["year", "kind"],
                               template="plotly_dark", height=560, color_discrete_map=IND_COLORS, opacity=.75),
                    width="stretch")
    st.json(build_info, expanded=False)

# ================================================================== Evaluation
with tabs[4]:
    if st.button("▶ Run evaluation now"):
        with st.spinner("Evaluating…"):
            from cinemind.evaluate import run_all
            run_all(engine)
    if config.EVAL_FILE.exists():
        rep = json.loads(config.EVAL_FILE.read_text())
        rk = rep.get("movielens_ranking")
        if isinstance(rk, dict):
            st.markdown(f"**Ranking accuracy on MovieLens** (temporal 80/20 hold-out, {rk['users']} users)")
            rows = [{"model": k, **v} for k, v in rk.items() if isinstance(v, dict)]
            rdf = pd.DataFrame(rows)
            st.dataframe(rdf, hide_index=True, width="stretch")
            st.plotly_chart(px.bar(rdf.melt(id_vars="model", value_vars=["HR@10", "NDCG@10", "Recall@10"]),
                                   x="variable", y="value", color="model", barmode="group", template="plotly_dark"),
                            width="stretch")
        else:
            st.info(f"MovieLens ranking: {rk}")
        ag = rep["agent_suite"]
        st.markdown(f"**Agent suite** · filter precision **{ag['mean_filter_precision']}** · "
                    f"intra-list diversity **{ag['mean_diversity']}**")
        st.dataframe(pd.DataFrame(ag["queries"]), hide_index=True, width="stretch")
    else:
        st.info("No evaluation yet — click the button.")

# ================================================================== Settings
with tabs[5]:
    st.markdown("All keys are **free** and optional. They're saved only to your local `.env` file.")
    status = config.key_status()
    with st.form("keys"):
        vals = {}
        for k, desc in config.API_KEYS.items():
            vals[k] = st.text_input(f"{k} {'✅' if status[k] else '—'}", type="password", help=desc,
                                    placeholder="(set)" if status[k] else desc)
        provider = st.selectbox("LLM provider", ["auto", "groq", "gemini", "openrouter", "ollama", "none"],
                                index=["auto", "groq", "gemini", "openrouter", "ollama", "none"].index(config.get("LLM_PROVIDER")))
        if st.form_submit_button("💾 Save"):
            upd = {k: v for k, v in vals.items() if v.strip()}
            upd["LLM_PROVIDER"] = provider
            config.save_env(upd)
            st.success("Saved. LLM changes apply immediately; data keys (TMDB/OMDb) need a rebuild.")
    st.markdown(f"Detected LLM providers: `{llm.available_providers() or 'none'}` → active: `{llm.active_provider()}`")
    if st.button("🔌 Test LLM"):
        r = llm.chat([{"role": "user", "content": "Reply with exactly: CineMind online"}])
        st.write(r or f"No response. {llm.last_error() or 'No provider configured.'}")
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🌐 Rebuild with live data"):
        run_build(False)
        st.rerun()
    if c2.button("⚡ Rebuild offline"):
        run_build(True)
        st.rerun()
