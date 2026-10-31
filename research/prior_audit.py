# -*- coding: utf-8 -*-
"""How much future information is in the league priors?

features.compute_priors is called ONCE, on BURN_IN + TRAIN_SEASONS, and the
resulting shrinkage targets are baked into every row of the dataset - including
rows from seasons the walk-forward later scores as out-of-sample.  The leakage
gate cannot see this: it computes priors on the full history and passes the
same object into the truncated rebuild, so priors are held fixed by design.

This measures the size of the effect two ways:

1. How far do the priors move when they are computed only on seasons that
   precede the first walk-forward season?
2. What is the implied effect on a feature - the EB shrinkage target is a
   weighted average, so a prior that is X% high moves a low-exposure player's
   rate by roughly X% times the prior's weight.
"""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import pandas as pd
import config
import features

players = pd.read_parquet(config.BASE_TABLE_PATH)
teams = pd.read_parquet(config.MATCH_TABLE_PATH)

shipped_seasons = config.BURN_IN_SEASONS + config.TRAIN_SEASONS
first_wf = config.WALK_FORWARD_SEASONS[0]
honest_seasons = [s for s in shipped_seasons if s < first_wf]

print(f"shipped priors fitted on : {shipped_seasons[0]} .. {shipped_seasons[-1]} "
      f"({len(shipped_seasons)} seasons)")
print(f"honest priors for {first_wf}: {honest_seasons[0]} .. {honest_seasons[-1]} "
      f"({len(honest_seasons)} seasons)")
print()

a = features.compute_priors(players, teams, shipped_seasons)
b = features.compute_priors(players, teams, honest_seasons)

da, db = asdict(a), asdict(b)
rows = []
for k in da:
    shipped, honest = da[k], db[k]
    rel = (shipped - honest) / honest if honest else float("nan")
    rows.append({"prior": k, "shipped": shipped, "honest": honest,
                 "abs_diff": shipped - honest, "rel_diff_pct": 100 * rel})
t = pd.DataFrame(rows).sort_values("rel_diff_pct", key=abs, ascending=False)
print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

worst = t.iloc[0]
print()
print(f"largest relative move: {worst['prior']} at {worst['rel_diff_pct']:+.2f}%")
print()
print("Reading: the prior is a shrinkage TARGET, so a player with k pseudo-matches")
print("of prior weight and n real ones carries it with weight k/(n+k).  At the")
print("configured EB_PRIOR_MATCHES = "
      f"{config.EB_PRIOR_MATCHES}, a player with 8 appearances is 50% prior, and")
print("a debutant is 100% prior; a 150-appearance regular is about 5%.")
