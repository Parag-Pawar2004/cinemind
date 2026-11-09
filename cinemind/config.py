"""Central configuration. All secrets come from `.env` (never committed)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
ENV_FILE = ROOT / ".env"

for _d in (RAW_DIR, PROC_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CATALOG_FILE = PROC_DIR / "catalog.jsonl"
EMB_FILE = PROC_DIR / "item_embeddings.npy"
EMBEDDER_FILE = PROC_DIR / "embedder.joblib"
CF_FILE = PROC_DIR / "cf_model.joblib"
EVAL_FILE = REPORT_DIR / "evaluation.json"
LEARNED_FILE = PROC_DIR / "learned.jsonl"
BUILD_LOG = REPORT_DIR / "build_log.json"

# Every key below is optional and free. The system runs fully offline without any of them.
API_KEYS = {
    "TMDB_API_KEY": "TMDB v3 API key or v4 read token (free) - themoviedb.org/settings/api",
    "OMDB_API_KEY": "OMDb key (free tier, 1000 req/day) - omdbapi.com/apikey.aspx",
    "GROQ_API_KEY": "Groq LLM key (free tier) - console.groq.com/keys",
    "GEMINI_API_KEY": "Google Gemini key (free tier) - aistudio.google.com/apikey",
    "OPENROUTER_API_KEY": "OpenRouter key (use ':free' models) - openrouter.ai/keys",
}

SETTINGS_DEFAULTS = {
    "LLM_PROVIDER": "auto",  # auto | groq | gemini | openrouter | ollama | none
    "GROQ_MODEL": "llama-3.1-8b-instant",
    "GEMINI_MODEL": "gemini-flash-latest",
    "OPENROUTER_MODEL": "meta-llama/llama-3.3-70b-instruct:free",
    "OLLAMA_MODEL": "llama3.2",
    "OLLAMA_URL": "http://localhost:11434",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "TMDB_PAGES": "15",        # pages per language x kind (20 titles/page)
    "TVMAZE_PAGES": "60",      # TVMaze index pages (250 shows/page)
    "WIKIDATA_LIMIT": "800",   # films per language
    "ADD_MOVIELENS_TITLES": "1",
    "LLM_RECALL": "1",        # LLM proposes titles, free APIs verify them (recall-then-verify)
}


def _parse_env(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_env() -> None:
    for k, v in _parse_env(ENV_FILE).items():
        if v and not os.environ.get(k):
            os.environ[k] = v


def get(key: str, default: str | None = None) -> str | None:
    load_env()
    val = os.environ.get(key)
    if val:
        return val
    return SETTINGS_DEFAULTS.get(key, default)


def save_env(updates: dict) -> None:
    """Merge updates into .env (empty value removes the key)."""
    cur = _parse_env(ENV_FILE)
    for k, v in updates.items():
        if v is None or str(v).strip() == "":
            cur.pop(k, None)
            os.environ.pop(k, None)
        else:
            cur[k] = str(v).strip()
            os.environ[k] = str(v).strip()
    lines = ["# CineMind secrets - keep private, never commit"]
    lines += [f"{k}={v}" for k, v in cur.items()]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def key_status() -> dict:
    load_env()
    return {k: bool(os.environ.get(k)) for k in API_KEYS}


load_env()
