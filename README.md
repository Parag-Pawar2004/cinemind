# 🎬 CineMind
### A multi-agent, LLM-augmented, cross-industry recommender for movies & dramas

Hollywood · Bollywood · Tollywood · K-dramas & Korean films · C-dramas & Chinese films (plus Kollywood, Mollywood and Japanese titles when you pull live data)

Describe what you want in plain language, e.g. *"a dark, twisty revenge K-drama that's a slow burn"* or *"something like Inception but from Bollywood"*. A team of agents turns that into a structured search, finds candidates with hybrid retrieval, has an LLM critic re-rank them, and explains each pick.

Everything is **free**. With no keys and no internet it still runs from a bundled catalog and rule-based agents.

---

## 🚀 Quick start (one command)

**Option A: launcher notebook (unzips and does everything)**
Put `cinemind.zip` and `CineMind_Launcher.ipynb` in the same folder, then run:
```bash
pip install jupyter
jupyter nbconvert --to notebook --execute --inplace CineMind_Launcher.ipynb
```
It unzips the project, installs dependencies, loads your keys, ingests data, builds the index, runs the evaluation and a smoke test, then opens the dashboard at **http://localhost:8501**.

**Option B: from inside the unzipped folder (creates a venv and asks for keys in the terminal)**
```bash
bash run.sh        # Linux / macOS
run.bat            # Windows
```

**Option C: manual**
```bash
pip install -r requirements.txt            # + requirements-full.txt for better embeddings
python scripts/configure.py                # optional free keys
python -m cinemind.build                   # or --offline
streamlit run app/dashboard.py
```

| Switch | Effect |
|---|---|
| `CINEMIND_LITE=1` | Skip the ~200 MB local embedding model and use TF-IDF+LSA instead |
| `CINEMIND_OFFLINE=1` | No network; seed catalog only |
| `CINEMIND_REBUILD=1` | Force a rebuild of the catalog and index |
| `CINEMIND_PORT=8600` | Use a different dashboard port |

---

## 🔑 Free API keys (all optional)

You can enter keys in the terminal prompt, in the dashboard's **⚙️ Settings** tab, or in a `.env` file (see `.env.example`). They are stored only on your machine.

| Key | Where (free, no card) | What it unlocks |
|---|---|---|
| `TMDB_API_KEY` | themoviedb.org → Settings → API | Thousands of titles in every industry, with posters and ratings (**most impactful**) |
| `GROQ_API_KEY` | console.groq.com/keys | Fast LLM for the intent, critic and explainer agents |
| `GEMINI_API_KEY` | aistudio.google.com/apikey | Alternative free LLM |
| `OPENROUTER_API_KEY` | openrouter.ai/keys | Free `:free` models |
| `OMDB_API_KEY` | omdbapi.com/apikey.aspx | IMDb ratings and plot enrichment |
| *(no key)* Ollama | ollama.com, then `ollama pull llama3.2` | Fully local LLM, detected automatically |

**Keyless sources used by default:** MovieLens (100k real ratings), TVMaze (K-, C- and Indian dramas), and Wikidata SPARQL (Telugu, Hindi, Korean, Chinese, Tamil and Malayalam films).

> Free-tier model names change over time. If an LLM call fails, change `GROQ_MODEL`, `GEMINI_MODEL` or `OPENROUTER_MODEL` in `.env`. The agents automatically fall back to rule-based logic, so the app keeps working in the meantime.

---

## 🧠 Architecture

```
 "something like Parasite but Bollywood, not too dark"
            │
   ┌────────▼────────┐  LLM → JSON intent (fallback: regex/lexicon parser)
   │  IntentAgent    │  industries, kind, years, genres±, reference titles, vibe targets
   └────────┬────────┘
   ┌────────▼────────┐  semantic (MiniLM / LSA) + BM25 + similarity-to-references
   │ RetrievalAgent  │  + vibe match + MovieLens CF + popularity/quality, hard filters
   └────────┬────────┘
   ┌────────▼────────┐  LLM-as-judge re-rank for fit & diversity (fallback: MMR)
   │  CriticAgent    │
   └────────┬────────┘
   ┌────────▼────────┐  grounded one-line "why" per pick (LLM or evidence template)
   │ ExplainerAgent  │
   └────────┬────────┘
            ▼   cards + full agent trace in the dashboard
```

### What makes it novel
1. **Cross-Industry Bridge (analogical retrieval):** `e(title) − centroid(source industry) + centroid(target industry)` answers questions like *"What's the Tollywood Inception?"* and *"What's the C-drama Squid Game?"*. It's word2vec-style analogy applied to film industries.
1b. **Recall-then-verify agent (v1.1):** when an LLM is configured, it names real titles matching your description (e.g. *Under the Queen's Umbrella* for "queen vs queen dowager political war"). Each is verified against the catalog, then TMDB or TVMaze, and missing ones are embedded and added on the fly (saved in `data/processed/learned.jsonl`). Hallucinated titles fail verification and are dropped.
2. **Interpretable vibe vectors:** every title gets a 0–1 score on six axes (pace, darkness, romance, humor, complexity, feels). Users steer recommendations with sliders, and the Taste DNA radar shows their profile.
3. **Hybrid cold-start bridging:** collaborative filtering (MovieLens SVD with fold-in, so no retraining for new users) is fused with content embeddings. Asian titles with no ratings can still be recommended to a user whose taste was learned from Hollywood ratings.
4. **Graceful-degradation agent design:** each agent has a deterministic fallback, so the system works with any free LLM or with none at all.
5. **Serendipity control and MMR diversity:** an explicit trade-off between relevance and novelty.
6. **Conversational refinement:** follow-ups like *"lighter"* or *"only series"* update the previous intent.
7. **Built-in evaluation:** HR@10, NDCG@10, Recall@10 and coverage on a temporal hold-out (Popularity vs Content vs CF vs Hybrid), plus filter precision, intra-list diversity and novelty for the agent suite.

### Dashboard tabs
🔎 Describe & Discover · 🌉 Cross-Industry Bridge · 🧬 Taste DNA · 📊 Catalog Explorer (semantic map) · 🧪 Evaluation · ⚙️ Settings & API keys

---

## 📁 Layout
```
cinemind/
├── CineMind_Launcher.ipynb   # one-click: unzip → install → keys → build → eval → dashboard
├── run.sh / run.bat          # venv + terminal key prompt + execute launcher
├── app/dashboard.py          # Streamlit UI
├── cinemind/
│   ├── config.py             # paths, .env handling
│   ├── data_seed.py          # 110+ curated offline titles across 5 industries
│   ├── sources.py            # TMDB, TVMaze, Wikidata, MovieLens, OMDb connectors
│   ├── build.py              # ingest → merge/dedupe → embeddings → vibes → CF
│   ├── embed.py              # sentence-transformers or TF-IDF+LSA; BM25
│   ├── vibes.py              # 6-axis interpretable vibe vectors
│   ├── cf.py                 # implicit SVD collaborative filtering + fold-in
│   ├── engine.py             # hybrid scoring, MMR, bridge, taste profile
│   ├── llm.py                # Groq / Gemini / OpenRouter / Ollama via plain HTTP
│   ├── agents.py             # Intent, Recall, Retrieval, Critic, Explainer agents + orchestrator
│   └── evaluate.py           # offline metrics
├── scripts/                  # configure.py (keys), make_launcher.py
└── tests/test_smoke.py
```

---

## Ideas to extend
- Swap in a bigger embedding model, or add posters with CLIP for visual "vibe" search.
- Add a LangGraph or tool-calling version of the orchestrator.
- Add user accounts and log feedback for online learning (bandits).
- Deploy free on Streamlit Community Cloud or Hugging Face Spaces.
