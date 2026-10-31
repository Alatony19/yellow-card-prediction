"""Quote every market for a fixture, with the closing line next to it.

Two modes:

    python predict.py --season 2024-2025 --matchweek 30
        Scores a historical matchweek with a model that only saw earlier
        matches, and prints what actually happened.  This is the mode to use
        when you want to judge whether to trust the numbers.

    python predict.py --home Arsenal --away Chelsea
        Rates an arbitrary fixture from the most recent state in the archive.

The `edge` column is the model probability minus the market's, in percentage
points.  It is a disagreement, not a gain: the closing line beats this model
on ranked probability score in every one of five seasons, so a large edge is an
invitation to look closer at that fixture, not a signal to act on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, modelling, ratings, score_matrix
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import modelling
    import ratings
    import score_matrix

PRINT_MARKETS = [
    ("home_win", "Home win"), ("draw", "Draw"), ("away_win", "Away win"),
    ("over_1.5", "Over 1.5 goals"), ("over_2.5", "Over 2.5 goals"),
    ("over_3.5", "Over 3.5 goals"), ("btts", "Both teams score"),
    ("home_multigol_1_3", "Home multigol 1-3"),
    ("home_multigol_1_2", "Home multigol 1-2"),
    ("away_multigol_1_3", "Away multigol 1-3"),
    ("total_multigol_2_3", "Total multigol 2-3"),
    ("total_multigol_2_4", "Total multigol 2-4"),
    ("home_over_0.5", "Home to score"), ("away_over_0.5", "Away to score"),
]


def _load():
    tm = pd.read_parquet(config.TEAM_MATCH_PATH)
    match = pd.read_parquet(config.MATCH_DATASET_PATH)
    dc_path = config.ARTIFACT_DIR / "dc_ratings.parquet"
    if dc_path.exists():
        dc = pd.read_parquet(dc_path)
    else:
        dc = ratings.rolling_ratings(match, xi=config.DC_XI)
        dc.to_parquet(dc_path, index=False)
    return tm, match, dc


class _NoOddsConfig:
    """`config` with the closing-line features switched off."""

    def __getattr__(self, k):
        return False if k == "USE_ODDS_FEATURES" else getattr(config, k)


def _fit_through(tm, match, dc, season):
    """Fit exactly as the walk-forward does: nothing from `season` is seen.

    Fixtures with no closing line are scored by a second fit that never had the
    odds features, because the main model has never seen them *missing*: the
    price feed is complete for every one of the nine finished seasons, so a NaN
    there routes down whatever default direction the trees happen to imply.
    Measured on the six unpriced matchweek-5 fixtures of 2026-27, that default
    pinned every one of them to a total lambda of 4.24 +/- 0.07 - 1.4 goals
    above the league mean, and identical across fixtures as different as
    Newcastle v Hull and Brentford v Chelsea.  The no-odds fit separates the
    same six across 2.14 to 2.96.  Same trap as the missing xG, same rule: a
    feature the model never saw absent must not be handed to it absent.
    """
    past = [s for s in config.ALL_SEASONS if s < season]
    valid_season = past[-1]
    train_seasons = [s for s in past if s != valid_season
                     and s not in config.BURN_IN_SEASONS]
    res = modelling.lambdas_for_split(tm, match, train_seasons, valid_season,
                                      season, dc, verbose=False)

    unpriced = res["match"]["mkt_lambda_home"].isna().to_numpy()
    if unpriced.any():
        alt = modelling.lambdas_for_split(tm, match, train_seasons, valid_season,
                                          season, dc, cfg=_NoOddsConfig(),
                                          verbose=False)
        res["lam_home"] = np.where(unpriced, alt["lam_home"], res["lam_home"])
        res["lam_away"] = np.where(unpriced, alt["lam_away"], res["lam_away"])
        print(f"  ! {int(unpriced.sum())} fixture(s) have no closing line yet "
              f"- scored without the odds features")
    res["unpriced"] = unpriced
    return res


def _confidence(p: float) -> str:
    """A directional label for one market's probability.

    Deliberately says which way the number leans.  A symmetric label keyed on
    |p - 0.5| would print "strong" against a 24% draw, which reads as strong
    confidence *in* the draw and means the opposite.
    """
    if p >= 0.85:
        return "very likely"
    if p >= 0.70:
        return "likely"
    if p >= 0.58:
        return "leans yes"
    if p > 0.42:
        return "toss-up"
    if p > 0.30:
        return "leans no"
    if p > 0.15:
        return "unlikely"
    return "very unlikely"


def _headline(probs: dict, home: str, away: str) -> str:
    """The single most likely result, and how decisive that call is."""
    order = [("home_win", f"{home} win"), ("draw", "draw"), ("away_win", f"{away} win")]
    name, label = max(order, key=lambda kv: probs[kv[0]])
    p = probs[name]
    second = sorted((probs[k] for k, _ in order), reverse=True)[1]
    gap = p - second
    if gap >= 0.25:
        strength = "clear favourite"
    elif gap >= 0.12:
        strength = "favoured"
    elif gap >= 0.05:
        strength = "slight edge"
    else:
        strength = "too close to call"
    return f"{label} at {p:.1%}  ({strength}; next best {second:.1%})"


def print_fixture(home: str, away: str, lam_h: float, lam_a: float, rho: float,
                  market_row: pd.Series | None = None,
                  actual: tuple[int, int] | None = None) -> None:
    m = score_matrix.score_matrix(lam_h, lam_a, rho=rho, max_goals=config.MAX_GOALS)
    probs = score_matrix.all_markets(m, config.MARKETS)
    summary = score_matrix.matrix_summary(m)

    print(f"\n{'=' * 74}")
    head = f"{home}  v  {away}"
    if actual is not None:
        head += f"      actual: {actual[0]}-{actual[1]}"
    print(head)
    print(f"{'=' * 74}")
    print(f"expected goals   {home}: {summary['exp_home_goals']:.2f}    "
          f"{away}: {summary['exp_away_goals']:.2f}    "
          f"total: {summary['exp_total_goals']:.2f}")
    print(f"most likely      {_headline(probs, home, away)}")

    has_mkt = market_row is not None and np.isfinite(market_row.get("p_home", np.nan))
    print()
    if has_mkt:
        print(f"{'market':<22}{'model':>8}{'line':>8}{'edge':>8}   confidence")
        print("-" * 74)
    else:
        print(f"{'market':<22}{'model':>8}   confidence")
        print("-" * 58)

    line_lookup = {}
    if has_mkt:
        lh = market_row.get("mkt_lambda_home", np.nan)
        la = market_row.get("mkt_lambda_away", np.nan)
        if np.isfinite(lh) and np.isfinite(la):
            mm = score_matrix.score_matrix(lh, la, rho=0.0, max_goals=config.MAX_GOALS)
            line_lookup = score_matrix.all_markets(mm, config.MARKETS)
        line_lookup["home_win"] = market_row["p_home"]
        line_lookup["draw"] = market_row["p_draw"]
        line_lookup["away_win"] = market_row["p_away"]
        if np.isfinite(market_row.get("p_over25", np.nan)):
            line_lookup["over_2.5"] = market_row["p_over25"]

    for key, label in PRINT_MARKETS:
        p = probs[key]
        if has_mkt and key in line_lookup:
            q = line_lookup[key]
            print(f"{label:<22}{p:>7.1%}{q:>8.1%}{(p - q) * 100:>+8.1f}   {_confidence(p)}")
        elif has_mkt:
            print(f"{label:<22}{p:>7.1%}{'-':>8}{'-':>8}   {_confidence(p)}")
        else:
            print(f"{label:<22}{p:>7.1%}   {_confidence(p)}")

    print()
    print("most likely scorelines:")
    for h, a, p in score_matrix.top_scores(m, config.TOP_SCORES):
        mark = "  <-- actual" if actual is not None and (h, a) == actual else ""
        print(f"   {h}-{a}   {p:6.1%}{mark}")


def run_matchweek(season: str, matchweek: int) -> None:
    tm, match, dc = _load()
    res = _fit_through(tm, match, dc, season)
    m_te = res["match"].copy()
    m_te["lam_home"] = res["lam_home"]
    m_te["lam_away"] = res["lam_away"]
    sel = m_te[m_te["matchday"] == matchweek]
    if sel.empty:
        print(f"no matches found for {season} matchweek {matchweek}")
        return
    print(f"\n{season}  matchweek {matchweek}   "
          f"(model trained only on earlier seasons; rho = {res['rho']:.4f}, "
          f"blend weight {res['weight']:.2f})")
    for _, row in sel.iterrows():
        print_fixture(row["home_team"], row["away_team"],
                      row["lam_home"], row["lam_away"], res["rho"],
                      market_row=row,
                      actual=(int(row["home_goals"]), int(row["away_goals"])))


def run_fixture(home: str, away: str) -> None:
    tm, match, dc = _load()
    known = set(match["home_team"]) | set(match["away_team"])
    for name in (home, away):
        if name not in known:
            close = [t for t in sorted(known) if name.lower() in t.lower()]
            print(f"unknown team {name!r}."
                  + (f"  Did you mean: {close}?" if close else ""))
            print(f"known teams: {sorted(known)}")
            return

    last_season = config.ALL_SEASONS[-1]
    res = _fit_through(tm, match, dc, last_season)
    dc_model = ratings.fit(match, xi=config.DC_XI)
    lam_h, lam_a = dc_model.rate(home, away)
    print(f"\nRating {home} v {away} from the state at the end of {last_season}.")
    print("Dixon-Coles ratings only - the boosted model needs this fixture's own "
          "rolling form row, which does not exist for a match that is not in the "
          "archive.  Treat this as an approximation.")
    print_fixture(home, away, lam_h, lam_a, res["rho"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Quote markets for a fixture.")
    ap.add_argument("--season", default=config.TEST_SEASONS[0])
    ap.add_argument("--matchweek", type=int, default=None)
    ap.add_argument("--home", default=None)
    ap.add_argument("--away", default=None)
    args = ap.parse_args()

    if args.home and args.away:
        run_fixture(args.home, args.away)
    else:
        run_matchweek(args.season, args.matchweek or 30)


if __name__ == "__main__":
    main()
