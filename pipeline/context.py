"""Match context: where each side stands, and how hard the fixture is.

Shared by both pipelines - the card model and the match-result model - because
"Arsenal, 2nd, away at Manchester City, matchweek 34, four points off the top"
is the same piece of information whether you are pricing a booking or a goal.

Everything here obeys the same rule as features.py: a value attached to match M
may only use matches that kicked off *strictly before* M.  League-table state is
built as a post-match cumulative and attached with a backward, non-exact
``merge_asof``; the standings are ranked from those same strictly-earlier states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "build_league_table_state", "build_standings", "build_elo",
    "build_fixture_state", "attach_match_context", "CONTEXT_FEATURES",
]

# Everything this module adds to a (match, team) row.
CONTEXT_FEATURES = [
    # --- where the two sides stand ---------------------------------------- #
    "ctx_points", "ctx_position", "ctx_goal_diff", "ctx_played",
    "ctx_opp_points", "ctx_opp_position", "ctx_opp_goal_diff",
    "ctx_position_gap", "ctx_points_gap",
    # --- strength, and how hard this particular fixture is ------------------ #
    "ctx_elo", "ctx_opp_elo", "ctx_elo_diff", "ctx_match_difficulty",
    # --- what is at stake --------------------------------------------------- #
    "ctx_season_progress", "ctx_pts_from_top", "ctx_pts_above_drop",
    "ctx_opp_pts_from_top", "ctx_opp_pts_above_drop", "ctx_stakes",
    # --- fixture history and congestion ------------------------------------ #
    "ctx_fixture_cards_pm_eb", "ctx_matches_last_14d",
]

ELO_START = 1500.0
ELO_START_PROMOTED = 1420.0   # a promoted side is not an average Premier League side
ELO_K = 20.0
ELO_HOME_ADV = 60.0           # rating points; roughly the long-run home edge
ELO_SEASON_CARRY = 0.85       # regression towards the mean between seasons


# --------------------------------------------------------------------------- #
# league table
# --------------------------------------------------------------------------- #
def build_league_table_state(teams: pd.DataFrame) -> pd.DataFrame:
    """Post-match league-table state: one row per (team, match)."""
    t = teams.sort_values(["season", "kickoff", "match_id"], kind="mergesort").copy()
    t["_gd"] = t["goals_for"] - t["goals_against"]
    g = t.groupby(["team", "season"], sort=False)
    out = pd.DataFrame({
        "team": t["team"],
        "season": t["season"],
        "kickoff": t["kickoff"],
        "tbl_points": g["points"].cumsum(),
        "tbl_goal_diff": g["_gd"].cumsum(),
        "tbl_goals_for": g["goals_for"].cumsum(),
        "tbl_played": g.cumcount() + 1.0,
    })
    return out.reset_index(drop=True)


def _table_key(points, goal_diff, goals_for):
    """Premier League ordering as one sortable number: points, then GD, then GF."""
    return points * 1e6 + (goal_diff + 500.0) * 1e3 + goals_for


def build_standings(table_state: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """The live table as it stood immediately before each kickoff.

    Ranking is cross-sectional - it needs every team's state at one instant -
    so it cannot be expressed as a plain as-of join on a single team.  Instead
    the twenty teams of a season are evaluated against every kickoff time in
    that season, each carrying a strictly-earlier state, and then ranked.
    """
    frames = []
    for season, grp in teams.groupby("season", sort=True):
        season_teams = np.sort(grp["team"].unique())
        times = np.sort(grp["kickoff"].unique())
        left = pd.DataFrame({
            "team": np.tile(season_teams, len(times)),
            "kickoff": np.repeat(times, len(season_teams)),
        })
        st = table_state[table_state["season"] == season].drop(columns=["season"])
        merged = pd.merge_asof(
            left.sort_values("kickoff", kind="mergesort"),
            st.sort_values("kickoff", kind="mergesort"),
            on="kickoff", by="team", direction="backward", allow_exact_matches=False,
        )
        cols = ["tbl_points", "tbl_goal_diff", "tbl_goals_for", "tbl_played"]
        merged[cols] = merged[cols].fillna(0.0)
        key = _table_key(merged["tbl_points"], merged["tbl_goal_diff"],
                         merged["tbl_goals_for"])
        merged["tbl_position"] = key.groupby(merged["kickoff"]).rank(
            ascending=False, method="min")
        by_time = merged.groupby("kickoff")["tbl_points"]
        merged["tbl_lead_points"] = by_time.transform("max")
        # 18th of 20 is the first relegation place; deriving the rank from the
        # season size keeps this correct if the league is ever a different size.
        drop_rank = max(len(season_teams) - 2, 1)
        merged["tbl_drop_points"] = by_time.transform(
            lambda s: np.sort(s.to_numpy())[::-1][min(drop_rank - 1, len(s) - 1)]
        )
        merged["season"] = season
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #
def build_elo(teams: pd.DataFrame, k: float = ELO_K, home_adv: float = ELO_HOME_ADV,
              carry: float = ELO_SEASON_CARRY) -> pd.DataFrame:
    """Sequential Elo, returned as the rating *after* each match.

    Elo is preferred to rolling points-per-game as the strength term because it
    is opponent-adjusted: beating the leaders and beating the bottom club move
    it by very different amounts, and a promoted side inherits a sensible
    starting level instead of an empty history.
    """
    m = teams[teams["side"] == "home"][
        ["match_id", "season", "kickoff", "team", "opponent",
         "goals_for", "goals_against"]
    ].sort_values(["kickoff", "match_id"], kind="mergesort")

    rating: dict[str, float] = {}
    rows = []
    prev_season = None
    for r in m.itertuples(index=False):
        if r.season != prev_season:
            if rating:
                mean = float(np.mean(list(rating.values())))
                rating = {t: mean + carry * (v - mean) for t, v in rating.items()}
            prev_season = r.season
        for t in (r.team, r.opponent):
            if t not in rating:
                rating[t] = ELO_START if not rating else ELO_START_PROMOTED

        eh, ea = rating[r.team], rating[r.opponent]
        exp_h = 1.0 / (1.0 + 10 ** ((ea - eh - home_adv) / 400.0))
        gd = r.goals_for - r.goals_against
        score_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        # a heavier win is stronger evidence, with diminishing returns
        mult = float(np.log1p(abs(gd))) + 1.0
        delta = k * mult * (score_h - exp_h)
        rating[r.team] = eh + delta
        rating[r.opponent] = ea - delta
        rows.append((r.team, r.kickoff, rating[r.team]))
        rows.append((r.opponent, r.kickoff, rating[r.opponent]))

    return pd.DataFrame(rows, columns=["team", "kickoff", "elo"])


# --------------------------------------------------------------------------- #
# fixture history
# --------------------------------------------------------------------------- #
def build_fixture_state(teams: pd.DataFrame) -> pd.DataFrame:
    """Cards per match in this exact pairing, across both venues.

    A data-driven stand-in for "is this a derby": some fixtures are reliably
    more ill-tempered than the two sides' individual records imply, and the
    pairing's own history says so without anyone having to curate a list.
    """
    t = teams.sort_values(["kickoff", "match_id"], kind="mergesort").copy()
    t["_pair"] = np.where(t["team"] < t["opponent"],
                          t["team"] + " v " + t["opponent"],
                          t["opponent"] + " v " + t["team"])
    t["_match_cards"] = (t["team_yellow"] + t["team_red"]
                         + t["opp_yellow"] + t["opp_red"])
    one = t.drop_duplicates(subset=["match_id"]).copy()
    g = one.groupby("_pair", sort=False)
    one["fx_cum_cards"] = g["_match_cards"].cumsum()
    one["fx_matches"] = g.cumcount() + 1.0
    return one[["_pair", "kickoff", "fx_cum_cards", "fx_matches"]].rename(
        columns={"_pair": "pair_key"}).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _asof(left: pd.DataFrame, state: pd.DataFrame, by: str) -> pd.DataFrame:
    return pd.merge_asof(
        left.sort_values("kickoff", kind="mergesort"),
        state.sort_values("kickoff", kind="mergesort"),
        on="kickoff", by=by, direction="backward", allow_exact_matches=False,
    )


def attach_match_context(cand: pd.DataFrame, teams: pd.DataFrame,
                         fixture_prior_cards: float = 3.6,
                         eb_matches: float = 6.0) -> pd.DataFrame:
    """Add every CONTEXT_FEATURES column to rows keyed by (kickoff, team, opponent).

    Works for a player-level candidate table and for a team-match table alike;
    the only requirement is those three columns plus ``season``.
    """
    need = {"kickoff", "team", "opponent", "season"}
    missing = need - set(cand.columns)
    if missing:
        raise KeyError(f"attach_match_context needs {sorted(missing)}")

    table_state = build_league_table_state(teams)
    standings = build_standings(table_state, teams)
    elo = build_elo(teams)

    # --- own and opponent standings --------------------------------------- #
    # `standings` is already an as-of snapshot keyed by kickoff, so this join is
    # an exact lookup rather than a second as-of.
    own = standings.rename(columns={
        "tbl_points": "ctx_points", "tbl_position": "ctx_position",
        "tbl_goal_diff": "ctx_goal_diff", "tbl_played": "ctx_played",
        "tbl_lead_points": "_lead", "tbl_drop_points": "_drop",
    })[["team", "kickoff", "season", "ctx_points", "ctx_position",
        "ctx_goal_diff", "ctx_played", "_lead", "_drop"]]
    out = cand.merge(own, on=["team", "kickoff", "season"], how="left")

    opp = own.rename(columns={
        "team": "opponent", "ctx_points": "ctx_opp_points",
        "ctx_position": "ctx_opp_position", "ctx_goal_diff": "ctx_opp_goal_diff",
        "ctx_played": "_opp_played", "_lead": "_opp_lead", "_drop": "_opp_drop",
    })
    out = out.merge(opp, on=["opponent", "kickoff", "season"], how="left")

    out["ctx_position_gap"] = out["ctx_opp_position"] - out["ctx_position"]
    out["ctx_points_gap"] = out["ctx_points"] - out["ctx_opp_points"]

    # --- Elo and match difficulty ------------------------------------------ #
    out = _asof(out, elo, by="team").rename(columns={"elo": "ctx_elo"})
    out = _asof(out, elo.rename(columns={"team": "opponent", "elo": "ctx_opp_elo"}),
                by="opponent")
    out["ctx_elo"] = out["ctx_elo"].fillna(ELO_START_PROMOTED)
    out["ctx_opp_elo"] = out["ctx_opp_elo"].fillna(ELO_START_PROMOTED)
    out["ctx_elo_diff"] = out["ctx_elo"] - out["ctx_opp_elo"]

    # Difficulty is the opponent's Elo win expectancy for this fixture, venue
    # included: 0 means a certain win, 0.5 a coin toss, 1 hopeless.
    if "is_home" in out.columns:
        venue_adj = np.where(out["is_home"] == 1, ELO_HOME_ADV, -ELO_HOME_ADV)
    else:
        venue_adj = 0.0
    out["ctx_match_difficulty"] = 1.0 / (
        1.0 + 10 ** ((out["ctx_elo"] + venue_adj - out["ctx_opp_elo"]) / 400.0)
    )

    # --- stakes -------------------------------------------------------------- #
    out["ctx_season_progress"] = (out["ctx_played"] / 38.0).clip(0, 1)
    out["ctx_pts_from_top"] = out["_lead"] - out["ctx_points"]
    out["ctx_pts_above_drop"] = out["ctx_points"] - out["_drop"]
    out["ctx_opp_pts_from_top"] = out["_opp_lead"] - out["ctx_opp_points"]
    out["ctx_opp_pts_above_drop"] = out["ctx_opp_points"] - out["_opp_drop"]
    # How much the table still matters to this side: close to the title or close
    # to the drop, late in the season, is the sharp end.  Mid-table in April is not.
    pressure = np.minimum(out["ctx_pts_from_top"].abs(),
                          out["ctx_pts_above_drop"].abs())
    out["ctx_stakes"] = out["ctx_season_progress"] / (1.0 + pressure / 6.0)

    # --- fixture history ------------------------------------------------------ #
    fx = build_fixture_state(teams)
    out["pair_key"] = np.where(out["team"] < out["opponent"],
                               out["team"] + " v " + out["opponent"],
                               out["opponent"] + " v " + out["team"])
    out = _asof(out, fx, by="pair_key")
    out["ctx_fixture_cards_pm_eb"] = (
        (out["fx_cum_cards"].fillna(0.0) + fixture_prior_cards * eb_matches)
        / (out["fx_matches"].fillna(0.0) + eb_matches)
    )

    # --- congestion ------------------------------------------------------------ #
    sched = teams[["team", "kickoff"]].drop_duplicates().sort_values("kickoff")
    counts = []
    for team, grp in sched.groupby("team", sort=False):
        ks = grp["kickoff"].to_numpy()
        prior = np.searchsorted(ks, ks, side="left")
        floor = np.searchsorted(ks, ks - np.timedelta64(14, "D"), side="left")
        counts.append(pd.DataFrame({
            "team": team, "kickoff": ks,
            "ctx_matches_last_14d": (prior - floor).astype(float),
        }))
    out = out.merge(pd.concat(counts, ignore_index=True),
                    on=["team", "kickoff"], how="left")

    drop = ["_lead", "_drop", "_opp_lead", "_opp_drop", "_opp_played",
            "pair_key", "fx_cum_cards", "fx_matches"]
    return out.drop(columns=[c for c in drop if c in out.columns])
