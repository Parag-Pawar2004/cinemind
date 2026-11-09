"""'Vibe vectors': every title gets interpretable 0-1 scores on six axes, so users can
steer with sliders ("dark but not slow", "funny and light"). Scores combine genre priors
with keyword evidence from overviews/tags, then are percentile-normalised per catalog.
"""
from __future__ import annotations

import numpy as np

AXES = ["pace", "darkness", "romance", "humor", "complexity", "feels"]
AXIS_LABELS = {
    "pace": ("Slow burn", "Fast-paced"), "darkness": ("Light", "Dark"),
    "romance": ("No romance", "Romantic"), "humor": ("Serious", "Funny"),
    "complexity": ("Easy watch", "Mind-bending"), "feels": ("Chill", "Emotional"),
}

GENRE_PRIOR = {
    "action": {"pace": .8, "darkness": .2}, "adventure": {"pace": .6},
    "thriller": {"pace": .6, "darkness": .6, "complexity": .3}, "horror": {"darkness": .9, "pace": .4},
    "crime": {"darkness": .6, "complexity": .3}, "mystery": {"complexity": .6, "darkness": .3},
    "romance": {"romance": 1.0, "feels": .4}, "comedy": {"humor": 1.0, "darkness": -.4},
    "drama": {"feels": .5, "pace": -.3}, "family": {"darkness": -.6, "humor": .3, "feels": .3},
    "animation": {"darkness": -.4, "humor": .4}, "science fiction": {"complexity": .6},
    "sci-fi": {"complexity": .6}, "fantasy": {"complexity": .2}, "war": {"darkness": .7, "feels": .5},
    "history": {"pace": -.3, "complexity": .2}, "documentary": {"pace": -.5},
    "music": {"feels": .3}, "musical": {"feels": .3, "humor": .2}, "sport": {"feels": .5, "pace": .2},
    "biography": {"pace": -.2, "feels": .3}, "legal": {"complexity": .3},
}
KEYWORDS = {
    "pace": {"fast-paced": 1, "chase": .8, "action": .5, "spectacle": .5, "survival": .5,
             "slow burn": -1, "quiet": -.8, "slice of life": -.7, "healing": -.4, "contemplative": -.8},
    "darkness": {"dark": 1, "murder": .7, "revenge": .6, "killer": .7, "disturbing": 1, "gritty": .8,
                 "zombie": .7, "bullying": .6, "feel-good": -1, "heartwarming": -.8, "light": -.8,
                 "sweet": -.7, "whimsical": -.6, "fun": -.5},
    "romance": {"love": .6, "romance": .8, "romcom": 1, "marriage": .5, "crush": .8, "letters": .3,
                "enemies to lovers": 1, "bromance": -.2},
    "humor": {"comedy": .8, "witty": .8, "quirky": .7, "satire": .7, "romcom": .7, "absurd": .7,
              "fun": .6, "dark comedy": .5, "tragic": -.6, "tearjerker": -.5},
    "complexity": {"mind-bending": 1, "nonlinear": .9, "twist": .6, "cerebral": 1, "time travel": .7,
                   "multiverse": .9, "political intrigue": .7, "conspiracy": .6, "simulation": .7,
                   "feel-good": -.4, "light": -.6, "romcom": -.5},
    "feels": {"tearjerker": 1, "emotional": .9, "bittersweet": .8, "tragic": .8, "heartwarming": .6,
              "healing": .6, "melancholic": .7, "father": .4, "mother": .4, "grief": .9, "heartbreak": .8},
}


def raw_vibe(rec: dict) -> np.ndarray:
    v = np.zeros(len(AXES), dtype=np.float32)
    for g in rec.get("genres", []):
        for ax, w in GENRE_PRIOR.get(g.lower(), {}).items():
            v[AXES.index(ax)] += w
    text = " ".join([rec.get("overview", ""), " ".join(rec.get("tags", []))]).lower()
    for ax, kws in KEYWORDS.items():
        for kw, w in kws.items():
            if kw in text:
                v[AXES.index(ax)] += w
    return v


def vibe_matrix(records: list[dict]) -> np.ndarray:
    """Percentile-rank each axis so 0.5 == catalog median."""
    raw = np.stack([raw_vibe(r) for r in records]) if records else np.zeros((0, len(AXES)))
    from scipy.stats import rankdata
    out = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        out[:, j] = (rankdata(raw[:, j], method="average") - 1) / max(1, len(raw) - 1)
    return out.astype(np.float32)
