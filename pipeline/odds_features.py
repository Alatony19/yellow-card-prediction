"""Closing-price features for the card model.

The match model has used the closing line since it was built; the card model
never has.  This module is the bridge: it takes the same de-vigged prices and
market-implied scoring rates that ``match_model/odds.py`` produces and turns
them into per-player-row columns.

The mechanism being tested
--------------------------
Bookings are not distributed evenly over a match.  Three regularities are well
enough established to be worth a feature each:

* **Tight matches are dirtier.**  A contest that stays in the balance keeps
  both sides pressing, and referees book more.  The closing line measures
  "expected to stay in the balance" better than any table-derived number,
  because it also prices injuries, rotation and motivation.
* **The side without the ball fouls.**  A team the market expects to be
  chasing the game spends more of it defending, tackling and tracking back,
  which is where cards come from.
* **The market knows things the Elo does not.**  ``ctx_match_difficulty`` is
  built from results only.  The gap between it and the market's own view of
  the same fixture is, by construction, the part of the team-strength picture
  that the pipeline has never seen.

The distinction that matters for ranking
----------------------------------------
The referee experiment established the rule the hard way: a feature that is
constant across all twenty players in a match shifts all twenty probabilities
together and cannot change who comes top.  Odds features split cleanly on that
line and the list below is deliberately ordered by it:

* ``odds_p_draw``, ``odds_closeness``, ``odds_total_lambda`` are match
  constants.  They can only help through *interactions* - a booster splitting
  on tightness and then on position group - never on their own.
* ``odds_team_expectancy``, ``odds_supremacy_for``, ``odds_lambda_against``,
  ``odds_is_underdog`` and ``odds_elo_surprise`` take opposite values for the
  two sides of a match, so they genuinely reorder eleven players against the
  other eleven.

That is the strongest a priori case any candidate group has had in this
project, which is why it was the last idea left standing.

Coverage
--------
Every Premier League season in ``ALL_SEASONS`` has a football-data file, so the
join is expected to be near-complete.  Rows that miss a price are left NaN
rather than filled; XGBoost routes missing values down a learned default branch
and the walk-forward will show the miss rate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import config
from match_model import config as mm_config
from match_model import odds as odds_mod

# Match-average expected goals over the nine seasons in hand, split into the
# typical home/away shares.  Used only as the fallback totals line in
# odds_row_from_prices when the caller has no over/under price to hand.
DEFAULT_LAMBDA_HOME = 1.55
DEFAULT_LAMBDA_AWAY = 1.30

# A real season label, "2024-2025".  predict.py stamps the synthetic fixture
# row with "UPCOMING" instead, which has no archive file and must not be passed
# to the loader.
SEASON_RE = re.compile(r"^\d{4}-\d{4}$")

# The columns this module adds.  config.ODDS_CANDIDATES mirrors this list; it
# is defined here so the module is the single source of truth for what exists.
ODDS_FEATURE_NAMES = [
    # --- match constants: tightness.  Interaction-only by construction. ----
    "odds_p_draw",           # the market's own probability the match is level
    "odds_closeness",        # 1 - |p_home - p_away|; 1 = a coin toss
    "odds_total_lambda",     # expected goals in the match
    # --- team-oriented: opposite signs for the two sides, so they reorder --
    "odds_team_expectancy",  # p(win) + 0.5 p(draw) for THIS player's team
    "odds_supremacy_for",    # expected goals for minus against
    "odds_lambda_against",   # how much this side is expected to be defending
    "odds_is_underdog",      # 1 when the market favours the opposition
    # --- what the market knows that the results-only Elo does not ----------
    "odds_elo_surprise",     # market expectancy minus Elo expectancy
]


def load_match_odds(seasons=None, odds_dir: Path | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """One row per match: de-vigged prices plus market-implied scoring rates.

    Labels that are not real seasons are dropped rather than passed through -
    ``predict.py`` stamps an upcoming fixture's rows with ``"UPCOMING"``, which
    has no archive file, and that row's price arrives through ``extra_odds``.
    """
    seasons = [s for s in (seasons or config.ALL_SEASONS) if SEASON_RE.match(str(s))]
    return odds_mod.load_odds(seasons=seasons or config.ALL_SEASONS,
                              odds_dir=odds_dir or mm_config.ODDS_DIR,
                              verbose=verbose)


def odds_row_from_prices(season: str, home_team: str, away_team: str,
                         home: float, draw: float, away: float,
                         over25: float | None = None,
                         under25: float | None = None) -> pd.DataFrame:
    """A one-match odds frame from the decimal prices on the screen.

    Needed because an upcoming fixture is not in the football-data archive: the
    archive is published after the fact.  The caller is looking at a price
    anyway - that is the whole point of the model - so the price is an input,
    not something to be looked up.

    ``over25`` / ``under25`` are optional but recommended: without them the
    market's expected goals cannot be inverted, and ``odds_supremacy_for`` and
    ``odds_lambda_against`` come back NaN while the 1X2-derived columns still
    work.  When they are missing the totals line is assumed to sit at the
    league-typical 2.85 expected goals, which is far better than a NaN the
    booster has no learned branch for.
    """
    ph, pdr, pa = odds_mod.devig_proportional(1 / float(home), 1 / float(draw),
                                              1 / float(away))
    out = pd.DataFrame({
        "season": [season], "home_team": [home_team], "away_team": [away_team],
        "p_home": ph, "p_draw": pdr, "p_away": pa,
    })
    if over25 is not None and under25 is not None:
        po, _pu = odds_mod.devig_proportional(1 / float(over25), 1 / float(under25))
        out["p_over25"] = po
    else:
        # Solve the totals line a league-average market would quote, so the
        # lambda inversion still has three targets instead of two.
        from match_model import score_matrix
        m = score_matrix.score_matrix(DEFAULT_LAMBDA_HOME, DEFAULT_LAMBDA_AWAY)
        out["p_over25"] = score_matrix.market_probability(m, "total_over", 2.5)

    lam_h, lam_a = odds_mod.implied_lambdas(out["p_home"], out["p_draw"],
                                            out["p_over25"])
    out["mkt_lambda_home"] = lam_h
    out["mkt_lambda_away"] = lam_a
    out["mkt_total_lambda"] = lam_h + lam_a
    return out


def attach_odds(df: pd.DataFrame, odds: pd.DataFrame | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """Add ODDS_FEATURE_NAMES to a frame of player rows.

    ``df`` must already carry ``season``, ``team``, ``opponent``, ``is_home``
    and - for ``odds_elo_surprise`` - ``ctx_match_difficulty``.  The join key is
    the same (season, home_team, away_team) triple the match model uses, so the
    two pipelines agree about which price belongs to which fixture.
    """
    if odds is None:
        odds = load_match_odds(seasons=sorted(df["season"].unique()),
                               verbose=verbose)

    out = df.copy()
    home = out["is_home"] == 1
    out["_home_team"] = np.where(home, out["team"], out["opponent"])
    out["_away_team"] = np.where(home, out["opponent"], out["team"])

    cols = ["season", "home_team", "away_team", "p_home", "p_draw", "p_away",
            "mkt_lambda_home", "mkt_lambda_away", "mkt_total_lambda"]
    od = odds[cols].rename(columns={"home_team": "_home_team",
                                    "away_team": "_away_team"})
    # One price per fixture.  A duplicated key would silently multiply rows.
    dup = od.duplicated(["season", "_home_team", "_away_team"]).sum()
    if dup and verbose:
        print(f"  ! {dup} duplicated odds keys; keeping the first")
    od = od.drop_duplicates(["season", "_home_team", "_away_team"])

    before = len(out)
    out = out.merge(od, on=["season", "_home_team", "_away_team"], how="left")
    assert len(out) == before, "odds join changed the row count"

    # ---- match constants ---------------------------------------------------- #
    out["odds_p_draw"] = out["p_draw"]
    out["odds_closeness"] = 1.0 - (out["p_home"] - out["p_away"]).abs()
    out["odds_total_lambda"] = out["mkt_total_lambda"]

    # ---- team-oriented ------------------------------------------------------ #
    p_for = np.where(home, out["p_home"], out["p_away"])
    p_against = np.where(home, out["p_away"], out["p_home"])
    lam_for = np.where(home, out["mkt_lambda_home"], out["mkt_lambda_away"])
    lam_against = np.where(home, out["mkt_lambda_away"], out["mkt_lambda_home"])

    # Elo expectancy counts a draw as half a win, so the market view has to be
    # put on the same scale before the two can be differenced.
    out["odds_team_expectancy"] = p_for + 0.5 * out["p_draw"]
    out["odds_supremacy_for"] = lam_for - lam_against
    out["odds_lambda_against"] = lam_against
    out["odds_is_underdog"] = np.where(
        np.isfinite(p_for) & np.isfinite(p_against),
        (p_for < p_against).astype(float), np.nan)

    # ---- market minus model ------------------------------------------------- #
    if "ctx_match_difficulty" in out.columns:
        elo_expectancy = 1.0 - out["ctx_match_difficulty"]
        out["odds_elo_surprise"] = out["odds_team_expectancy"] - elo_expectancy
    else:
        out["odds_elo_surprise"] = np.nan

    out = out.drop(columns=["_home_team", "_away_team"] + cols[3:])

    if verbose:
        miss = out["odds_team_expectancy"].isna().mean()
        print(f"  odds attached: {1 - miss:.2%} of {len(out):,} rows priced")
        if miss > 0:
            gaps = (out.loc[out["odds_team_expectancy"].isna(), "season"]
                    .value_counts().sort_index())
            for season, n in gaps.items():
                print(f"    unpriced {season}: {n:,} rows")
    return out


def main() -> None:
    """Report join coverage against the built model dataset."""
    ds = pd.read_parquet(config.MODEL_DATASET_PATH)
    print(f"model dataset: {len(ds):,} rows")
    out = attach_odds(ds)
    print()
    desc = out[ODDS_FEATURE_NAMES].describe().T
    print(desc[["count", "mean", "std", "min", "max"]].to_string())

    # Does the market's tightness read actually line up with cards?  A quick
    # unconditional check before any model sees the columns.
    print("\ncard rate by market closeness decile:")
    q = pd.qcut(out["odds_closeness"], 10, labels=False, duplicates="drop")
    tab = out.groupby(q)[config.TARGET_COL].agg(["mean", "size"])
    for k, row in tab.iterrows():
        print(f"  decile {int(k)}  rate {row['mean']:.4f}  n {int(row['size']):,}")


if __name__ == "__main__":
    main()
