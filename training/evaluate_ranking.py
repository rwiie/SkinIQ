"""Sanity evaluation of the TF-IDF ranking (proposal 3.6.4 / 3.8).

The product dataset's afterUse tags (e.g. "Acne Fighting") come from an external
ingredient checker and are NOT used by the TF-IDF similarity, so they serve as an
independent, weak relevance label. For single-concern users, precision@k = share of
the top-k products (treatments and moisturizers) carrying the matching tag, compared
with the share among all eligible products (what a random pick would give).

The rules also read these tags, so only the TF-IDF similarity is evaluated here.

    python training/evaluate_ranking.py      # -> training/results/ranking/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import SKIN_CONDITIONS  # noqa: E402
from skiniq.fusion import ImageAnalysis, build_usce  # noqa: E402
from skiniq.recommender import default_index, query_terms  # noqa: E402

BASE = {
    "age_group": "25_34", "pregnant": "no", "known_conditions": ["none"], "sensitivity": "not",
    "reacts_to_products": "no", "tightness": "rarely", "oiliness": "balanced", "oil_location": "nowhere",
    "flakiness": "no", "breakouts": "rarely", "goals": [], "avoid_ingredients": ["none"],
}
# profile name -> (photo concern or None, answer overrides, relevant tags)
PROFILES = {
    "acne": ("acne", {}, {"Acne Fighting"}),
    "hyperpigmentation": ("hyperpigmentation", {}, {"Brightening", "Dark Spots"}),
    "redness": ("redness", {}, {"Redness Reducing"}),
    "wrinkles": ("wrinkles", {}, {"Anti-Aging"}),
    "enlarged pores": ("enlarged_pores", {}, {"Reduces Large Pores"}),
    "blackheads": ("blackheads", {}, {"Reduces Large Pores"}),
    "dry skin": (None, {"oiliness": "dry", "tightness": "often", "flakiness": "yes"}, {"Hydrating"}),
    "oily skin": (None, {"oiliness": "very_oily", "oil_location": "all_over", "tightness": "never"},
                  {"Good For Oily Skin"}),
}
STEPS = ("treatment", "moisturizer")
KS = (5, 10, 20)


def evaluate() -> pd.DataFrame:
    index = default_index()
    pool = [i for i, p in enumerate(index.products) if p.step in STEPS]
    rows = []
    for name, (concern, overrides, tags) in PROFILES.items():
        image = None
        if concern:
            image = ImageAnalysis({c: 0.9 if c == concern else 0.05 for c in SKIN_CONDITIONS}, independent=True)
        usce = build_usce({**BASE, **overrides}, image)
        sims = index.similarity(query_terms(usce))
        order = sorted(pool, key=lambda i: (-sims[i], i))
        relevant = np.array([bool(tags & index.products[i].tags) for i in order])
        row = {"profile": name, "relevant tags": ", ".join(sorted(tags)),
               "base rate": round(relevant.mean(), 3)}
        for k in KS:
            row[f"precision@{k}"] = round(relevant[:k].mean(), 3)
        row[f"lift@{KS[1]}"] = round(row[f"precision@{KS[1]}"] / row["base rate"], 2) if row["base rate"] else None
        rows.append(row)
    df = pd.DataFrame(rows)
    mean = {"profile": "mean", "relevant tags": "", **{c: round(df[c].mean(), 3) for c in df.columns[2:]}}
    return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)


def main() -> None:
    out = Path(__file__).resolve().parent / "results" / "ranking"
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate()
    df.to_csv(out / "precision_at_k.csv", index=False)
    (out / "precision_at_k.json").write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
