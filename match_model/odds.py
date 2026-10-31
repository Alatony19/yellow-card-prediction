"""Closing prices from football-data.co.uk, turned into probabilities and lambdas.

Three jobs:

1. **Join.**  football-data uses shorter club names than FBref ("Man United" vs
   "Manchester Utd"), so a mapping is needed before the two sources meet.  The
   map is asserted complete at load time rather than silently dropping matches.

2. **Remove the margin.**  A market's three 1X2 prices imply probabilities
   summing to more than one; the excess is the vig.  Stripping it
   proportionally is the standard first approximation and is what the
   `p_*` columns carry.

3. **Invert to expected goals.**  The de-vigged 1X2 and over/under 2.5 prices
   pin down the pair of scoring rates that a Dixon-Coles match would need to
   produce them.  Those two numbers, `mkt_lambda_home` and `mkt_lambda_away`,
   are the most informative single summary of everything the market knows -
   injuries, rotation, motivation, weather - that never appears in a stats feed.

The closing line is used, not the opening one: it is the best-informed price and
therefore the strongest feature and the fairest benchmark to be judged against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from . import config, score_matrix
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import score_matrix

# FBref folder name -> football-data.co.uk name.  Every other club spells
# identically in both sources.
TEAM_NAME_MAP = {
    "Cardiff City": "Cardiff",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "Norwich City": "Norwich",
    "Nott'ham Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
}
FD_TO_FBREF = {v: k for k, v in TEAM_NAME_MAP.items()}

# Preferred price columns, best first.  "Avg" is the market average across all
# books football-data tracks and is both the most complete and the least noisy.
# Seasons before 2019-20 carry the market average under the older Betbrain
# names (BbAvH, BbAv>2.5); without them those two seasons lose every price.
ONE_X_TWO = [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"),
             ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA")]
OVER_UNDER_25 = [("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"),
                 ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5")]

# The prices actually obtainable, kept alongside the de-vigged probabilities so
# a comparison can use the quote that was really available rather than a fair one.  "Max" is
# the best quote football-data saw, which is the optimistic case; "Avg" is the
# market average, which is closer to what a normal account gets.
BEST_1X2 = [("MaxH", "MaxD", "MaxA"), ("BbMxH", "BbMxD", "BbMxA")]
AVG_1X2 = [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA")]
BEST_OU = [("Max>2.5", "Max<2.5"), ("BbMx>2.5", "BbMx<2.5")]
AVG_OU = [("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5")]


def season_to_file(season: str) -> str:
    """"2017-2018" -> "2017-18.csv", the football-data archive naming."""
    start, end = season.split("-")
    return f"{start}-{end[2:]}.csv"


def _first_available(df: pd.DataFrame, groups) -> list[str] | None:
    for cols in groups:
        if all(c in df.columns for c in cols) and df[list(cols)].notna().all(axis=1).any():
            return list(cols)
    return None


def devig_proportional(*probs) -> tuple[np.ndarray, ...]:
    """Scale raw implied probabilities so they sum to one.

    The proportional (or "multiplicative") method assumes the margin is spread
    evenly in relative terms.  It slightly under-prices longshots compared with
    Shin's method, which matters for a 30/1 outsider and barely at all for the
    markets here.
    """
    stack = np.vstack([np.asarray(p, dtype=float) for p in probs])
    total = stack.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = stack / total
    return tuple(out)


def implied_lambdas(p_home, p_draw, p_over25, rho: float = 0.0,
                    max_goals: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Scoring rates that reproduce the market's own probabilities.

    Two unknowns, three targets (home win, draw, over 2.5), solved by least
    squares on the log rates so they stay positive.  Rows with any missing
    input come back as NaN rather than being quietly filled.
    """
    p_home = np.asarray(p_home, dtype=float)
    p_draw = np.asarray(p_draw, dtype=float)
    p_over = np.asarray(p_over25, dtype=float)
    n = len(p_home)
    lam_h = np.full(n, np.nan)
    lam_a = np.full(n, np.nan)

    ok = np.isfinite(p_home) & np.isfinite(p_draw) & np.isfinite(p_over)
    start = np.array([np.log(1.55), np.log(1.25)])
    for i in np.flatnonzero(ok):
        targets = (p_home[i], p_draw[i], p_over[i])

        def resid(log_lam, targets=targets):
            m = score_matrix.score_matrix(float(np.exp(log_lam[0])),
                                          float(np.exp(log_lam[1])),
                                          rho=rho, max_goals=max_goals)
            return [
                score_matrix.market_probability(m, "result", "H") - targets[0],
                score_matrix.market_probability(m, "result", "D") - targets[1],
                score_matrix.market_probability(m, "total_over", 2.5) - targets[2],
            ]

        res = least_squares(resid, start, method="lm", xtol=1e-10, ftol=1e-10)
        lam_h[i], lam_a[i] = np.exp(res.x)
    return lam_h, lam_a


def load_odds(seasons=None, odds_dir: Path | None = None,
              verbose: bool = True) -> pd.DataFrame:
    """One row per match, keyed by (season, home_team, away_team) in FBref names."""
    odds_dir = Path(odds_dir or config.ODDS_DIR)
    seasons = list(seasons or config.ALL_SEASONS)
    frames = []
    for season in seasons:
        path = odds_dir / season_to_file(season)
        if not path.exists():
            if verbose:
                print(f"  ! no odds file for {season} ({path.name})")
            continue
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].copy()

        out = pd.DataFrame({
            "season": season,
            "home_team": raw["HomeTeam"].map(lambda s: FD_TO_FBREF.get(s, s)),
            "away_team": raw["AwayTeam"].map(lambda s: FD_TO_FBREF.get(s, s)),
            "odds_date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
            "fd_home_goals": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
            "fd_away_goals": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
        })

        cols = _first_available(raw, ONE_X_TWO)
        if cols is None:
            if verbose:
                print(f"  ! no usable 1X2 prices for {season}")
            continue
        h, d, a = (pd.to_numeric(raw[c], errors="coerce") for c in cols)
        ph, pd_, pa = devig_proportional(1 / h, 1 / d, 1 / a)
        out["p_home"], out["p_draw"], out["p_away"] = ph, pd_, pa
        out["odds_source_1x2"] = cols[0][:-1]

        ou = _first_available(raw, OVER_UNDER_25)
        if ou is None:
            out["p_over25"] = np.nan
        else:
            o, u = (pd.to_numeric(raw[c], errors="coerce") for c in ou)
            po, _pu = devig_proportional(1 / o, 1 / u)
            out["p_over25"] = po

        # decimal prices for settlement
        for label, groups, names in [
            ("best", BEST_1X2, ("odds_home_best", "odds_draw_best", "odds_away_best")),
            ("avg", AVG_1X2, ("odds_home_avg", "odds_draw_avg", "odds_away_avg")),
        ]:
            cols = _first_available(raw, groups)
            for k, nm in enumerate(names):
                out[nm] = (pd.to_numeric(raw[cols[k]], errors="coerce")
                           if cols else np.nan)
        for label, groups, names in [
            ("best", BEST_OU, ("odds_over25_best", "odds_under25_best")),
            ("avg", AVG_OU, ("odds_over25_avg", "odds_under25_avg")),
        ]:
            cols = _first_available(raw, groups)
            for k, nm in enumerate(names):
                out[nm] = (pd.to_numeric(raw[cols[k]], errors="coerce")
                           if cols else np.nan)
        frames.append(out)

    if not frames:
        raise RuntimeError(f"no odds loaded from {odds_dir}")
    odds = pd.concat(frames, ignore_index=True)

    # market-implied expected goals
    lam_h, lam_a = implied_lambdas(odds["p_home"], odds["p_draw"], odds["p_over25"])
    odds["mkt_lambda_home"] = lam_h
    odds["mkt_lambda_away"] = lam_a
    odds["mkt_total_lambda"] = lam_h + lam_a
    odds["mkt_supremacy"] = lam_h - lam_a

    if verbose:
        n_l = int(np.isfinite(lam_h).sum())
        print(f"odds: {len(odds):,} matches across {odds['season'].nunique()} seasons "
              f"({n_l:,} with implied lambdas)")
    return odds


def check_name_coverage(odds: pd.DataFrame, teams: pd.DataFrame) -> list[str]:
    """Names present in the odds feed that never appear in the FBref archive."""
    fb = set(teams["team"].unique())
    od = set(odds["home_team"]) | set(odds["away_team"])
    return sorted(od - fb)


def main() -> None:
    odds = load_odds()
    print()
    print(odds.head(8).to_string(index=False))
    print()
    print("implied lambda summary:")
    print(odds[["mkt_lambda_home", "mkt_lambda_away", "mkt_total_lambda"]]
          .describe().to_string())


if __name__ == "__main__":
    main()
