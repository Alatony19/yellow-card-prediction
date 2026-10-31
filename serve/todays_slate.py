"""Card predictions for a day's fixtures, from the archive plus a price.

What this does and does not know
--------------------------------
``predict.py`` scores a fixture given two confirmed XIs, a referee and a price.
For a match that has not kicked off, none of those three is in the archive, so
this module fills each one as honestly as it can and says which is a guess:

* **XIs.**  FBref publishes line-ups in the match report, which does not exist
  until after the match.  The projected XI here is each team's **most recent
  starting XI**, which is the best free proxy and is usually 8-10 of the 11.
  Pass ``--xi`` with a JSON file to override it once the real sheets are out;
  that is the only input that materially changes who tops the list.
* **Referee.**  Not published pre-match on FBref either.  An unknown official
  is not a hole: ``ref_cards_pm_eb`` is an empirical-Bayes rate, so an unseen
  name falls back to the league prior rather than going missing.  Supply
  ``--referee`` per fixture if you know it - a lenient or strict official is
  worth a few points of probability.
* **Price.**  The football-data archive is published after the fact, so today's
  price is not in it.  Supply the real one with ``--odds``; failing that this
  falls back to the project's own Dixon-Coles ratings, converted to fair
  decimal odds.  **That fallback is a model's opinion, not the market's**, and
  the whole point of the odds features is that the market knows things the
  model does not - injuries, rotation, motivation.  The fallback keeps the
  columns in a sane range instead of NaN; it does not deliver the edge.

Leagues
-------
Scored by the five-league ensemble (``ensemble.py``), the same model every
other entry point uses.  Each fixture may carry a ``"league"`` key -
premier_league, serie_a, la_liga, bundesliga or ligue_1 - and ``--league`` sets
the default for those that do not.  XIs, formations and the Dixon-Coles
fallback price all come from that league's own archive.

Usage
-----
    python serve/todays_slate.py --fixtures fx.json                       # Premier League
    python serve/todays_slate.py --fixtures fx.json --league serie_a
    python serve/todays_slate.py                          # today, fair-odds fallback
    python serve/todays_slate.py --date 2026-09-20
    python serve/todays_slate.py --odds odds.json         # real prices, recommended
    python serve/todays_slate.py --top 5

``odds.json`` maps "Home v Away" to the decimal prices::

    {"Bournemouth v Liverpool": {"h": 4.20, "d": 3.90, "a": 1.80,
                                 "over25": 1.75, "under25": 2.05},
     "Fulham v Manchester Utd": {"h": 2.90, "d": 3.40, "a": 2.45}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import config
import pooled
import predict
from match_model import config as mm_config
from match_model import ratings as mm_ratings
from match_model import score_matrix as mm_score


# --------------------------------------------------------------------------- #
# projected line-ups
# --------------------------------------------------------------------------- #
def projected_xi(players: pd.DataFrame, team: str, before: pd.Timestamp):
    """The team's most recent starting XI, goalkeeper first.

    Returns ``(xi, kickoff_of_that_match)`` so the caller can report how stale
    the projection is - an XI from three weeks ago is a weaker guess than one
    from last weekend.
    """
    hist = players[(players["team"] == team) & (players["kickoff"] < before)
                   & players["is_starter"]]
    if hist.empty:
        raise ValueError(f"no starting history for {team!r} before {before}")
    last_kick = hist["kickoff"].max()
    xi = hist[hist["kickoff"] == last_kick].copy()
    gk = xi[xi["pos_group_actual"] == "GK"]["player"]
    others = xi[xi["pos_group_actual"] != "GK"]["player"].tolist()
    if gk.empty:                       # shouldn't happen, but don't crash on it
        return others[:11], last_kick
    return [gk.iloc[0]] + others[:10], last_kick


def recent_formation(players: pd.DataFrame, team: str, before: pd.Timestamp) -> str:
    hist = players[(players["team"] == team) & (players["kickoff"] < before)]
    if hist.empty or "formation" not in hist.columns:
        return "UNK"
    val = hist.sort_values("kickoff")["formation"].dropna()
    return str(val.iloc[-1]) if len(val) else "UNK"


# --------------------------------------------------------------------------- #
# the price
# --------------------------------------------------------------------------- #
def fair_odds_from_ratings(match_tbl: pd.DataFrame, home: str, away: str,
                           rho: float = 0.0):
    """Dixon-Coles fair decimal odds for a fixture that has no market price.

    A stand-in, clearly not a market: see the module docstring.
    """
    model = mm_ratings.fit(match_tbl, xi=mm_config.DC_XI)
    lam_h, lam_a = model.rate(home, away)
    m = mm_score.score_matrix(lam_h, lam_a, rho=rho)
    p_h = mm_score.market_probability(m, "result", "H")
    p_d = mm_score.market_probability(m, "result", "D")
    p_a = mm_score.market_probability(m, "result", "A")
    p_o = mm_score.market_probability(m, "total_over", 2.5)
    return ((1 / p_h, 1 / p_d, 1 / p_a), (1 / p_o, 1 / (1 - p_o)),
            (lam_h, lam_a))


# --------------------------------------------------------------------------- #
def _league_inputs(league: str, date: pd.Timestamp) -> dict:
    """History, config and a Dixon-Coles match table for one league."""
    players, teams, cfg, _odds = pooled.league_history(league)
    if league == pooled.PL:
        # ratings.fit wants the MATCH-level frame (one row per match, with
        # home_goals / away_goals), not the two-rows-per-match team table.
        match_tbl = pd.read_parquet(mm_config.MATCH_DATASET_PATH)
    else:
        match_tbl = pooled.match_table(teams)
    if "kickoff" in match_tbl.columns:
        match_tbl = match_tbl[match_tbl["kickoff"] < date]
    return {"players": players, "teams": teams, "match_tbl": match_tbl}


def run(fixtures: list[dict], date: pd.Timestamp, top: int,
        odds_book: dict | None,
        default_league: str = pooled.PL) -> pd.DataFrame:
    model = predict.load_default_model()
    by_league: dict = {}

    all_rows = []
    for fx in fixtures:
        league = fx.get("league", default_league)
        if league not in by_league:
            by_league[league] = _league_inputs(league, date)
        players = by_league[league]["players"]
        teams = by_league[league]["teams"]
        match_tbl = by_league[league]["match_tbl"]
        home, away = fx["home"], fx["away"]
        kickoff = fx["kickoff"]
        key = f"{home} v {away}"
        print("=" * 78)
        print(f"{key}    {kickoff}    [{league}]")
        print("=" * 78)

        try:
            home_xi, home_when = projected_xi(players, home, kickoff)
            away_xi, away_when = projected_xi(players, away, kickoff)
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue
        print(f"  projected XIs from {home_when.date()} ({home}) and "
              f"{away_when.date()} ({away})")

        book = (odds_book or {}).get(key)
        if book:
            o = (book["h"], book["d"], book["a"])
            ou = ((book["over25"], book["under25"])
                  if "over25" in book and "under25" in book else None)
            print(f"  price: MARKET  {o}" + (f"  o/u {ou}" if ou else ""))
        else:
            o, ou, lam = fair_odds_from_ratings(match_tbl, home, away)
            o = tuple(round(v, 2) for v in o)
            ou = tuple(round(v, 2) for v in ou)
            print(f"  price: DIXON-COLES FALLBACK (not a market) {o}  o/u {ou}"
                  f"   lambda {lam[0]:.2f}/{lam[1]:.2f}")

        ref = fx.get("referee") or "Unknown"
        if ref == "Unknown":
            print("  referee: not announced - ref_cards_pm_eb falls back to the "
                  "league prior")

        try:
            slate = predict.predict_fixture(
                home_team=home, away_team=away, kickoff=kickoff, referee=ref,
                home_xi=home_xi, away_xi=away_xi,
                home_formation=recent_formation(players, home, kickoff),
                away_formation=recent_formation(players, away, kickoff),
                matchweek=fx.get("matchweek"),
                model=model, players=players, teams=teams,
                odds=o, odds_ou=ou, league=league,
            )
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        cols = ["player", "team", "pos_group", "card_probability"]
        cols += [c for c in ("bag_lo", "bag_hi") if c in slate.columns]
        print()
        print(slate.head(top)[cols].to_string(
            float_format=lambda v: f"{v:.3f}"))
        print(f"\n  expected booked starters in this match: "
              f"{slate['card_probability'].sum():.2f}")
        print()

        s = slate.copy()
        s["fixture"] = key
        s["league"] = league
        s["priced_from"] = "market" if book else "dixon-coles"
        all_rows.append(s)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", required=True,
                    help="JSON file: [{home, away, kickoff, matchweek, referee}]")
    ap.add_argument("--odds", default=None, help="JSON file of decimal prices")
    ap.add_argument("--date", default=None)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out", default=None, help="write the full slate to CSV")
    ap.add_argument("--league", default=pooled.PL, choices=pooled.ALL_LEAGUES,
                    help="league of fixtures that carry no \"league\" key")
    args = ap.parse_args()

    fixtures = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    for fx in fixtures:
        fx["kickoff"] = pd.to_datetime(fx["kickoff"])
    date = pd.to_datetime(args.date) if args.date else min(
        fx["kickoff"] for fx in fixtures)
    odds_book = (json.loads(Path(args.odds).read_text(encoding="utf-8"))
                 if args.odds else None)

    out = run(fixtures, date, args.top, odds_book, args.league)
    if args.out and len(out):
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
