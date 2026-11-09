"""Offline smoke tests:  python -m pytest -q   (or: python tests/test_smoke.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cinemind import config  # noqa: E402
from cinemind.agents import Orchestrator  # noqa: E402
from cinemind.build import build  # noqa: E402
from cinemind.engine import Engine  # noqa: E402


def _engine():
    if not config.CATALOG_FILE.exists():
        build(offline=True)
    return Engine()


def test_pipeline():
    e = _engine()
    o = Orchestrator(e)
    out = o.run("feel-good Korean drama about friendship")
    assert out["results"], "no results"
    assert all(r["industry"] == "Korean" for r in out["results"])
    assert [t["agent"] for t in out["trace"]] == ["IntentAgent", "RecallAgent", "RetrievalAgent", "CriticAgent", "ExplainerAgent"]


def test_bridge():
    e = _engine()
    src, res = e.bridge("Inception", "Tollywood", n=5)
    assert src and res and all(e.items[c["idx"]]["industry"] == "Tollywood" for c in res)


def test_reference_parsing():
    e = _engine()
    out = Orchestrator(e).run("something like Parasite but from Bollywood")
    assert "Parasite" in out["intent"]["reference_titles"]
    assert out["intent"]["industries"] == ["Bollywood"]


if __name__ == "__main__":
    test_pipeline(); test_bridge(); test_reference_parsing(); print("all smoke tests passed")
