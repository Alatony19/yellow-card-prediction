"""Understat team-style features for the card model.

``fetch_understat.py`` writes one row per match with each side's xG, PPDA and
deep completions.  This turns that into *pre-match* team state - rolling means
over each club's previous ``WINDOW`` league matches, shifted so a match never
sees itself - and attaches it to player rows for the team and the opponent.

Why these could matter for bookings, and why only as candidates until the
ablation says otherwise:

* ``us_team_ppda_roll``      how hard this side presses; pressing is tackling
* ``us_opp_ppda_roll``       how hard the opposition presses this side
* ``us_team_deep_allowed_roll`` how often this side is pinned back - defenders
                             under siege foul more
* ``us_team_xga_roll`` / ``us_team_xg_roll``   chance quality against / for

Club names are mapped to FBref's automatically: two names are the same club
when they play on the same dates.  No hand-kept alias table to drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

TEAM_MATCH_PATH = config.ARTIFACT_DIR / "understat" / "team_match.csv"
WINDOW = 10

US_FEATURES = [
    "us_team_ppda_roll",
    "us_opp_ppda_roll",
    "us_team_deep_allowed_roll",
    "us_team_xga_roll",
    "us_team_xg_roll",
]


def load_team_rows(path: Path = TEAM_MATCH_PATH) -> pd.DataFrame:
    """Two rows per match: one per side, with what it did and what it allowed."""
    m = pd.read_csv(path, parse_dates=["date"])
    sides = []
    for me, them in (("home", "away"), ("away", "home")):
        sides.append(pd.DataFrame({
            "league": m["league_key"],
            "date": m["date"],
            "us_team": m[f"{me}_team"],
            "us_opp": m[f"{them}_team"],
            "ppda": m[f"{me}_ppda"],
            "deep_allowed": m[f"{them}_deep_completions"],
            "xg": m[f"{me}_np_xg"],
            "xga": m[f"{them}_np_xg"],
        }))
    rows = pd.concat(sides, ignore_index=True)
    # PPDA is a ratio; a side that made no defensive actions reads as infinite.
    rows["ppda"] = rows["ppda"].replace([np.inf, -np.inf], np.nan)
    return rows.sort_values(["league", "us_team", "date"]).reset_index(drop=True)


def team_state(rows: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Rolling means as they stood *before* each match."""
    out = rows[["league", "us_team", "date"]].copy()
    g = rows.groupby(["league", "us_team"], sort=False)
    for col in ("ppda", "deep_allowed", "xg", "xga"):
        out[f"{col}_roll"] = g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    return out


def name_map(ds: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """(league, FBref name) -> Understat name, by shared match dates."""
    fb = ds[["league", "team", "kickoff"]].astype({"league": str}).drop_duplicates()
    fb = fb.assign(day=fb["kickoff"].dt.normalize())
    us = rows[["league", "us_team", "date"]].assign(day=rows["date"].dt.normalize())
    j = fb.merge(us, on=["league", "day"])
    # Every club that played that day is a candidate; the true partner is the
    # one that shares (nearly) all of this club's match days.
    counts = j.groupby(["league", "team", "us_team"]).size().rename("n").reset_index()
    best = counts.sort_values("n", ascending=False).drop_duplicates(["league", "team"])
    days = fb.groupby(["league", "team"])["day"].nunique().rename("days").reset_index()
    best = best.merge(days, on=["league", "team"])
    best["share"] = best["n"] / best["days"]
    weak = best[best["share"] < 0.9]
    if len(weak):
        print("  ! weak Understat name matches (share of match days shared):")
        print(weak.to_string(index=False))
    best = best[best["share"] >= 0.9]
    return {(r.league, r.team): r.us_team for r in best.itertuples()}


def attach(ds: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add US_FEATURES to player rows carrying league, team, opponent, kickoff."""
    rows = load_team_rows()
    state = team_state(rows)
    state["day"] = state["date"].dt.normalize()
    mapping = name_map(ds, rows)

    out = ds.copy()
    league = out["league"].astype(str)
    out["_day"] = out["kickoff"].dt.normalize()
    out["_us_team"] = [mapping.get(k) for k in zip(league, out["team"])]
    out["_us_opp"] = [mapping.get(k) for k in zip(league, out["opponent"])]

    st = state.drop(columns=["date"]).rename(columns={"league": "_league"})
    st = st.drop_duplicates(["_league", "us_team", "day"])
    out["_league"] = league

    mine = st.rename(columns={"us_team": "_us_team", "day": "_day",
                              "ppda_roll": "us_team_ppda_roll",
                              "deep_allowed_roll": "us_team_deep_allowed_roll",
                              "xg_roll": "us_team_xg_roll",
                              "xga_roll": "us_team_xga_roll"})
    before = len(out)
    out = out.merge(mine, on=["_league", "_us_team", "_day"], how="left")
    theirs = st[["_league", "us_team", "day", "ppda_roll"]].rename(
        columns={"us_team": "_us_opp", "day": "_day", "ppda_roll": "us_opp_ppda_roll"})
    out = out.merge(theirs, on=["_league", "_us_opp", "_day"], how="left")
    assert len(out) == before, "Understat join changed the row count"
    out = out.drop(columns=["_day", "_us_team", "_us_opp", "_league"])

    if verbose:
        cov = out.groupby([out["league"].astype(str), "season"])["us_team_ppda_roll"].apply(
            lambda s: s.notna().mean())
        low = cov[cov < 0.9]
        print(f"  Understat attached: {out['us_team_ppda_roll'].notna().mean():.1%} of rows")
        if len(low):
            print("  league-seasons under 90% (the first matches of a club's "
                  "history have no rolling window yet):")
            print(low.round(3).to_string())
    return out
