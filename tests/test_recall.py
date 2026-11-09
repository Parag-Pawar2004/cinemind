"""Mocked test of the recall-then-verify path (no network / LLM needed):  python tests/test_recall.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cinemind import agents, config, llm, sources  # noqa: E402
from cinemind.agents import Orchestrator  # noqa: E402
from cinemind.build import build  # noqa: E402
from cinemind.engine import Engine  # noqa: E402

NEW = {"title": "Under the Queen's Umbrella", "year": 2022, "kind": "tv", "industry": "Korean", "language": "ko",
       "genres": ["Drama", "History"], "popularity": 20.0, "rating": 8.2, "poster": None, "tags": [],
       "overview": "In Joseon, a queen raising unruly princes clashes in a fierce political war "
                   "with the queen dowager inside the palace.", "source": "tmdb-lookup"}


def test_recall_adds_and_ranks_first():
    if not config.CATALOG_FILE.exists():
        build(offline=True)
    config.LEARNED_FILE = Path(tempfile.mkdtemp()) / "learned.jsonl"
    e = Engine()
    n0 = len(e.items)
    saved = (llm.active_provider, llm.chat_json, sources.lookup_title)
    llm.active_provider = lambda: "groq"
    llm.chat_json = lambda msgs: ({"titles": [{"title": "Under the Queen's Umbrella", "year": 2022, "kind": "tv"},
                                              {"title": "Totally Made Up Show", "year": 2020, "kind": "tv"}]}
                                  if "encyclopedia" in msgs[0]["content"] else None)
    sources.lookup_title = lambda name, year=None, kind=None, log=print: (
        dict(NEW) if name.startswith("Under the Queen") else None)
    try:
        out = Orchestrator(e).run("joseon palace political war between the queen and the queen dowager")
    finally:
        llm.active_provider, llm.chat_json, sources.lookup_title = saved
    rec = next(t for t in out["trace"] if t["agent"] == "RecallAgent")["output"]
    assert rec["added"] == ["Under the Queen's Umbrella"] and rec["unresolved"] == ["Totally Made Up Show"], rec
    assert len(e.items) == n0 + 1 and out["learned"] == ["Under the Queen's Umbrella"]
    assert out["results"][0]["title"] == "Under the Queen's Umbrella", [r["title"] for r in out["results"][:3]]
    # persisted + restored on next start, and not duplicated
    e2 = Engine()
    assert len(e2.items) == n0 + 1 and e2.find_title("Under the Queen's Umbrella") is not None
    assert e2.add_items([dict(NEW)])[0] == n0 and len(e2.items) == n0 + 1
    # search still works with runtime-added items (BM25 padding, arrays aligned)
    assert Orchestrator(e2, recall=False).run("korean palace intrigue")["results"]
    assert len(e2.E) == len(e2.V) == len(e2.pop) == len(e2.industry) == len(e2.items)


if __name__ == "__main__":
    agents.THIS_YEAR  # noqa: B018
    test_recall_adds_and_ranks_first()
    print("recall test passed")
