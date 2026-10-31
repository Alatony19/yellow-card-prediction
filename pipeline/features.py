"""Stage 2 - build the (small) model dataset from the raw base tables.

Design rule, enforced everywhere in this module
-----------------------------------------------
A feature for match M may only use information from matches that kicked off
*strictly before* M.  This is implemented with one consistent pattern:

  1. compute the **post-match cumulative / rolling state** of an entity
     (player, team, referee) on the rows of the matches it actually played;
  2. attach that state to the candidate rows with ``pd.merge_asof(...,
     direction="backward", allow_exact_matches=False)``.

Because the as-of join is strictly backward, the state that lands on a row is
by construction "everything known up to, and including, the entity's previous
match".  There is no ``shift`` juggling and no ``reset_index(drop=True)``, which
is what silently misaligned the v2 feature table.

Rates for thin histories are shrunk towards a league prior with a simple
empirical-Bayes rule, so a player with two appearances is not credited with a
0.0 or 9.0 fouls-per-90 rate.  Priors are estimated on the training seasons
only.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, context, odds_features
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import context
    import odds_features

warnings.filterwarnings("ignore", category=FutureWarning)

POS_GROUPS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "FW", "UNK"]
EWM_HALFLIFE_APPS = 30   # half-life, in appearances, for recency-weighted rates
REF_WINDOW = 25          # recent-form window for referee card rates


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def eb_rate(events, exposure, prior: float, k: float):
    """Empirical-Bayes shrunk rate: (events + prior*k) / (exposure + k).

    ``k`` is expressed in the same unit as ``exposure`` and acts as a number of
    pseudo-observations pulling a thin history towards the league ``prior``.
    With no history at all the result is exactly ``prior``.
    """
    events = np.asarray(events, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    events = np.nan_to_num(events, nan=0.0)
    exposure = np.nan_to_num(exposure, nan=0.0)
    return (events + prior * k) / (exposure + k)


def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    ok = np.isfinite(den) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


@dataclass
class LeaguePriors:
    """League-average rates, estimated on the training seasons only."""
    fouls_p90: float
    cards_p90: float
    cards_per_match: float
    fouls_per_tackle: float
    ref_cards_per_match: float
    ref_fouls_per_match: float
    cards_per_foul: float
    subbed_off_rate: float      # P(a starter is replaced)
    repl_minutes_per_start: float   # minutes a start hands to a replacement
    defacts_p90: float          # tackles won + interceptions, per 90
    fouls_per_defact: float     # how often a defensive action becomes a foul
    pens_p90: float             # penalties conceded per 90
    away_card_share: float      # share of all cards shown to the away side

    @property
    def ref_cards_per_foul(self) -> float:
        return self.ref_cards_per_match / self.ref_fouls_per_match


def compute_priors(players: pd.DataFrame, teams: pd.DataFrame,
                   seasons: list[str]) -> LeaguePriors:
    """Estimate league priors from a *fixed, early* set of seasons.

    Using only the training seasons keeps the validation and test seasons out of
    every quantity the model sees, including these six scalars.
    """
    p = players[players["season"].isin(seasons) & players["appeared"]]
    t = teams[teams["season"].isin(seasons)]
    minutes90 = p["minutes"].sum() / 90.0
    cards = (p["yellow"] + p["red"]).sum()
    ref_match_cards = t.groupby("match_id")[["team_yellow", "team_red"]].sum().sum(axis=1)
    ref_match_fouls = t.groupby("match_id")["team_fouls"].sum()
    starts = p[p["is_starter"]]
    return LeaguePriors(
        fouls_p90=p["fouls"].sum() / minutes90,
        cards_p90=cards / minutes90,
        cards_per_match=cards / len(p),
        fouls_per_tackle=p["fouls"].sum() / max(p["tackles"].sum(), 1.0),
        ref_cards_per_match=float(ref_match_cards.mean()),
        ref_fouls_per_match=float(ref_match_fouls.mean()),
        cards_per_foul=cards / max(p["fouls"].sum(), 1.0),
        subbed_off_rate=float(starts["was_replaced"].mean()),
        repl_minutes_per_start=float(starts["slot_repl_minutes"].mean()),
        defacts_p90=(p["tackles_won"].fillna(0).sum()
                     + p["interceptions"].fillna(0).sum()) / minutes90,
        fouls_per_defact=p["fouls"].sum() / max(
            p["tackles_won"].fillna(0).sum() + p["interceptions"].fillna(0).sum(), 1.0),
        pens_p90=p["pens_conceded"].fillna(0).sum() / minutes90,
        away_card_share=float(
            (t.loc[t["side"] != "home", "team_yellow"].sum()
             + t.loc[t["side"] != "home", "team_red"].sum())
            / max((t["team_yellow"].sum() + t["team_red"].sum()), 1.0)),
    )


def _rolling_sum(df: pd.DataFrame, group: str, col: str, window: int) -> pd.Series:
    """Rolling sum over the last ``window`` rows of each group, inclusive.

    ``df`` must already be sorted by (group, kickoff).  ``transform`` keeps the
    original index, so the result always aligns with ``df``.
    """
    return df.groupby(group, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=1).sum()
    )


# --------------------------------------------------------------------------- #
# entity state tables (post-match)
# --------------------------------------------------------------------------- #
def build_player_state(players: pd.DataFrame, window: int) -> pd.DataFrame:
    """Post-match cumulative + rolling state, one row per player appearance."""
    p = players[players["appeared"]].copy()
    p["cards"] = ((p["yellow"] >= 1) | (p["red"] >= 1)).astype(float)
    p["n_cards"] = p["yellow"] + p["red"]
    p["duels_lost"] = p["challenges_lost"] + p["aerials_lost"]
    # Substitution exposure: how often this player, when he starts, hands the
    # rest of the match to someone else.  That replacement can decide the slot
    # the starter himself would have lost.
    # Defensive workload that survives FBref's 2025-26 table removals: only
    # tackles *won* and interceptions are still published per player, so the
    # workload proxy is built from those two rather than from total tackles.
    p["_defacts"] = p["tackles_won"].fillna(0.0) + p["interceptions"].fillna(0.0)
    # Venue-split discipline: some players are demonstrably worse away.
    p["_home_min"] = np.where(p["is_home"] == 1, p["minutes"], 0.0)
    p["_away_min"] = np.where(p["is_home"] != 1, p["minutes"], 0.0)
    p["_home_cards"] = np.where(p["is_home"] == 1, p["n_cards"], 0.0)
    p["_away_cards"] = np.where(p["is_home"] != 1, p["n_cards"], 0.0)
    p["_start"] = p["is_starter"].astype(float)
    p["_subbed_off"] = (p["is_starter"] & (p["was_replaced"] == 1)).astype(float)
    p["_repl_min"] = np.where(p["is_starter"], p["slot_repl_minutes"], 0.0)
    p = p.sort_values(["player_key", "kickoff"]).reset_index(drop=True)

    g = p.groupby("player_key", sort=False)
    st = pd.DataFrame(index=p.index)
    st["player_key"] = p["player_key"]
    st["kickoff"] = p["kickoff"]
    st["season"] = p["season"]

    st["apps"] = g.cumcount() + 1
    for col, name in [("minutes", "cum_min"), ("fouls", "cum_fouls"),
                      ("n_cards", "cum_cards"), ("tackles", "cum_tackles"),
                      ("fouled", "cum_fouled"), ("_start", "cum_starts"),
                      ("_subbed_off", "cum_subbed_off"), ("_repl_min", "cum_repl_min"),
                      # --- recommendation 4: behaviour, not just rate --------
                      ("_defacts", "cum_defacts"), ("pens_conceded", "cum_pens"),
                      ("_home_min", "cum_home_min"), ("_away_min", "cum_away_min"),
                      ("_home_cards", "cum_home_cards"),
                      ("_away_cards", "cum_away_cards")]:
        st[name] = g[col].cumsum()

    for col, name in [("minutes", "roll_min"), ("fouls", "roll_fouls"),
                      ("cards", "roll_cards"), ("tackles", "roll_tackles"),
                      ("duels_lost", "roll_duels_lost"), ("fouled", "roll_fouled")]:
        st[name] = _rolling_sum(p, "player_key", col, window)
    for col, name in [("_start", "roll_starts"), ("_repl_min", "roll_repl_min")]:
        st[name] = _rolling_sum(p, "player_key", col, window)
    p["_one"] = 1.0
    st["roll_apps"] = _rolling_sum(p, "player_key", "_one", window)

    # Exponentially weighted per-appearance means.  A hard 10-match window
    # throws away everything older; a half-life keeps a long tail of evidence
    # while still favouring recent form.
    for col, name in [("minutes", "ewm_min"), ("fouls", "ewm_fouls"),
                      ("n_cards", "ewm_cards")]:
        st[name] = p.groupby("player_key", sort=False)[col].transform(
            lambda x: x.ewm(halflife=EWM_HALFLIFE_APPS, adjust=True).mean()
        )

    # appearances since the last card (0 on the match in which one was shown)
    st["matches_since_card"] = _matches_since_event(p["player_key"].values, p["cards"].values)

    # season-to-date cards (resets every season)
    st["cards_season_to_date"] = p.groupby(["player_key", "season"], sort=False)["cards"].cumsum()

    # rolling modal position, from one-hot counts over the same window
    pos_dummies = pd.get_dummies(
        pd.Categorical(p["pos_group_actual"], categories=POS_GROUPS), dtype=float
    )
    pos_dummies.columns = [f"_pos_{c}" for c in POS_GROUPS]
    p = pd.concat([p, pos_dummies], axis=1)
    rolled = np.column_stack([
        _rolling_sum(p, "player_key", f"_pos_{c}", window).to_numpy() for c in POS_GROUPS
    ])
    st["pos_group_prior"] = np.array(POS_GROUPS, dtype=object)[rolled.argmax(axis=1)]

    return st


def _matches_since_event(groups: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Appearances since the last event, evaluated *after* each match.

    0 means "the event happened in this match".  NaN until the first event.
    """
    out = np.full(len(groups), np.nan)
    counter = np.nan
    prev = None
    for i in range(len(groups)):
        if groups[i] != prev:
            counter = np.nan
            prev = groups[i]
        if event[i] > 0:
            counter = 0.0
        elif not np.isnan(counter):
            counter += 1.0
        out[i] = counter
    return out


def build_player_ref_state(players: pd.DataFrame) -> pd.DataFrame:
    """Career history of one player under one specific referee.

    Recommendation 4 asked for richer player context.  FBref will not give us
    fouls by zone or dissent counts, but it does give us every (player,
    referee) pairing ever played, and "this official has booked this player
    three times in seven matches" is exactly the kind of context a human
    handicapper uses.

    The pairing is thin by nature - a player meets a given referee a handful of
    times a season - so the rate is shrunk hard towards the player's own career
    rate rather than towards the league, which is done at the join site where
    both numbers are available.
    """
    p = players[players["appeared"]].copy()
    p["n_cards"] = p["yellow"] + p["red"]
    p["_pr_key"] = p["player_key"].astype(str) + "|" + p["referee"].astype(str)
    p = p.sort_values(["_pr_key", "kickoff"]).reset_index(drop=True)
    g = p.groupby("_pr_key", sort=False)
    st = pd.DataFrame(index=p.index)
    st["_pr_key"] = p["_pr_key"]
    st["kickoff"] = p["kickoff"]
    st["pr_apps"] = g.cumcount() + 1
    st["pr_cards"] = g["n_cards"].cumsum()
    st["pr_min"] = g["minutes"].cumsum()
    return st


# Premier League accumulation thresholds: five yellows before matchweek 19 is a
# one-match ban, ten before matchweek 32 is two, fifteen across the season is
# three.  A player one booking away from a live deadline has a real incentive to
# hold back, and nothing else in the feature set encodes that.
SUSPENSION_THRESHOLDS = [(5, 19), (10, 32), (15, 38)]


def suspension_pressure(cards_std, matchweek):
    """Cards needed to the next live ban threshold, and matches left to avoid it.

    Returns (cards_to_ban, matches_left).  Both NaN once every threshold for the
    season has either been passed or expired, which XGBoost handles natively as
    its own branch.
    """
    cards = np.asarray(cards_std, dtype=float)
    mw = np.asarray(matchweek, dtype=float)
    to_ban = np.full(cards.shape, np.nan)
    left = np.full(cards.shape, np.nan)
    # Walk the thresholds from the last to the first so the earliest still-live
    # one wins the assignment.
    for count, deadline in reversed(SUSPENSION_THRESHOLDS):
        live = (mw <= deadline) & (cards < count) & np.isfinite(cards)
        to_ban[live] = count - cards[live]
        left[live] = deadline - mw[live]
    return to_ban, left


def build_team_state(teams: pd.DataFrame, window: int) -> pd.DataFrame:
    """Post-match rolling state, one row per (match, team)."""
    t = teams.copy()
    t["team_cards"] = t["team_yellow"] + t["team_red"]
    t = t.sort_values(["team", "kickoff"]).reset_index(drop=True)
    t["_one"] = 1.0

    st = pd.DataFrame(index=t.index)
    st["team"] = t["team"]
    st["kickoff"] = t["kickoff"]

    n = _rolling_sum(t, "team", "_one", window)
    st["team_matches"] = n
    st["team_fouls_pm"] = _rolling_sum(t, "team", "team_fouls", window) / n
    st["team_fouled_pm"] = _rolling_sum(t, "team", "team_fouled", window) / n
    st["team_cards_pm"] = _rolling_sum(t, "team", "team_cards", window) / n
    st["team_ppg"] = _rolling_sum(t, "team", "points", window) / n
    st["team_poss_share"] = _rolling_sum(t, "team", "poss_share", window) / n
    return st


def build_referee_state(teams: pd.DataFrame) -> pd.DataFrame:
    """Post-match expanding state, one row per (match, referee).

    Referee tendencies are slow-moving, so an expanding window (career to date)
    beats a short rolling one; empirical-Bayes shrinkage handles the early days
    of a new official.
    """
    t = teams.copy()
    t["_cards"] = t["team_yellow"] + t["team_red"]
    t["_home_cards"] = np.where(t["side"] == "home", t["_cards"], 0.0)
    t["_away_cards"] = np.where(t["side"] != "home", t["_cards"], 0.0)
    m = t.groupby(["match_id", "referee", "kickoff", "season"], as_index=False).agg(
        match_cards=("team_yellow", "sum"),
        match_reds=("team_red", "sum"),
        match_fouls=("team_fouls", "sum"),
        home_cards=("_home_cards", "sum"),
        away_cards=("_away_cards", "sum"),
    )
    m["match_cards"] = m["match_cards"] + m["match_reds"]
    m = m.sort_values(["referee", "kickoff"]).reset_index(drop=True)
    g = m.groupby("referee", sort=False)

    st = pd.DataFrame(index=m.index)
    st["referee"] = m["referee"]
    st["kickoff"] = m["kickoff"]
    st["season"] = m["season"]
    st["ref_matches"] = g.cumcount() + 1
    st["ref_cum_cards"] = g["match_cards"].cumsum()
    st["ref_cum_fouls"] = g["match_fouls"].cumsum()
    n_recent = _rolling_sum(m.assign(_one=1.0), "referee", "_one", REF_WINDOW)
    st["ref_cards_pm_recent"] = _rolling_sum(m, "referee", "match_cards", REF_WINDOW) / n_recent

    # --- who he shows them to, rather than how many --------------------------
    # Every existing referee feature is a volume measure.  These two are about
    # distribution, which is the part that can reorder a field.
    st["ref_cum_home_cards"] = g["home_cards"].cumsum()
    st["ref_cum_away_cards"] = g["away_cards"].cumsum()

    # Season-to-date level, to catch mid-season directive changes - the league
    # card rate moves in steps when the competition issues new guidance, and a
    # career average is slow to notice.
    gs = m.groupby(["referee", "season"], sort=False)
    st["ref_season_matches"] = gs.cumcount() + 1
    st["ref_season_cards"] = gs["match_cards"].cumsum()
    return st


def build_referee_pos_state(players: pd.DataFrame) -> pd.DataFrame:
    """Cumulative cards a referee has shown to each position group.

    Some officials book the midfield battle and wave away defensive fouls;
    others do the reverse.  Expressed at the join site as a ratio to the league
    share for that position, so the feature says "this referee books centre-
    backs 1.2x as often as officials in general" rather than restating how many
    centre-backs get booked.
    """
    p = players[players["appeared"]].copy()
    p["n_cards"] = p["yellow"] + p["red"]
    p["_rp_key"] = (p["referee"].astype(str) + "|"
                    + p["pos_group_actual"].astype(str))
    p = p.sort_values(["_rp_key", "kickoff"]).reset_index(drop=True)
    # One row per (referee, position, match): the cards that group took.
    m = p.groupby(["_rp_key", "match_id", "kickoff"], as_index=False).agg(
        rp_cards=("n_cards", "sum"), rp_players=("n_cards", "size"))
    m = m.sort_values(["_rp_key", "kickoff"]).reset_index(drop=True)
    g = m.groupby("_rp_key", sort=False)
    st = pd.DataFrame(index=m.index)
    st["_rp_key"] = m["_rp_key"]
    st["kickoff"] = m["kickoff"]
    st["rp_cum_cards"] = g["rp_cards"].cumsum()
    st["rp_cum_players"] = g["rp_players"].cumsum()
    return st


# --------------------------------------------------------------------------- #
# as-of attachment
# --------------------------------------------------------------------------- #
def asof_attach(left: pd.DataFrame, state: pd.DataFrame, by: str,
                suffix: str = "") -> pd.DataFrame:
    """Attach the entity's last *strictly earlier* state to every candidate row.

    ``allow_exact_matches=False`` is what makes this leak-free: a row never sees
    the state produced by its own match.
    """
    left_sorted = left.sort_values("kickoff", kind="mergesort")
    state_sorted = state.sort_values("kickoff", kind="mergesort")
    state_cols = [c for c in state_sorted.columns if c not in {"kickoff", by}]
    if suffix:
        state_sorted = state_sorted.rename(columns={c: c + suffix for c in state_cols})
    merged = pd.merge_asof(
        left_sorted,
        state_sorted,
        on="kickoff",
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged


def attach_bench_strength(cand: pd.DataFrame, players: pd.DataFrame,
                          player_state: pd.DataFrame, priors: LeaguePriors,
                          cfg) -> pd.DataFrame:
    """Card propensity of the *named* outfield bench, per (match, team).

    The nine substitutes are known the moment the line-up sheet is published,
    so this is legitimately available at prediction time -- and it is the only
    read we have on who might come on and pick up the card that settles the slot.
    Goalkeepers are dropped using each player's *prior* modal position, because
    an unused substitute has no position recorded for the match itself.
    """
    bench = players.loc[~players["is_starter"], ["match_id", "team", "kickoff", "player_key"]]
    bench = bench.drop_duplicates()
    bench = asof_attach(bench, player_state.drop(columns=["season"]), by="player_key")
    k90 = cfg.EB_PRIOR_MINUTES / 90.0
    bench["_cards_p90"] = eb_rate(bench["cum_cards"], bench["cum_min"] / 90.0,
                                  priors.cards_p90, k90)
    bench = bench[bench["pos_group_prior"].fillna("UNK") != "GK"]
    agg = bench.groupby(["match_id", "team"], as_index=False).agg(
        bench_cards_p90_eb=("_cards_p90", "mean"),
        bench_named=("_cards_p90", "size"),
    )
    return cand.merge(agg, on=["match_id", "team"], how="left")


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def build_model_dataset(players: pd.DataFrame,
                        teams: pd.DataFrame,
                        cfg=config,
                        priors: LeaguePriors | None = None,
                        verbose: bool = True,
                        extra_odds: pd.DataFrame | None = None
                        ) -> tuple[pd.DataFrame, LeaguePriors]:
    """Build the model dataset.

    ``extra_odds`` is a one-or-more-row odds frame appended to the archive
    before the join, so a fixture that has not been published yet can still
    carry a price.  See ``odds_features.odds_row_from_prices``.
    """
    if priors is None:
        priors = compute_priors(players, teams, cfg.BURN_IN_SEASONS + cfg.TRAIN_SEASONS)
    if verbose:
        print("League priors (train seasons only):")
        for k, v in asdict(priors).items():
            print(f"  {k:24s} {v:.4f}")

    # ---- candidate rows --------------------------------------------------- #
    cand = players.copy()
    if not cfg.INCLUDE_BENCH:
        cand = cand[cand["is_starter"]]
    if cfg.EXCLUDE_GOALKEEPERS:
        cand = cand[cand["pos_group_actual"] != "GK"]
    cand = cand.reset_index(drop=True)

    keep = ["match_id", "season", "matchday", "matchweek", "kickoff", "team", "opponent",
            "player", "player_key", "is_starter", "is_home", "referee", "formation", "age_years",
            "pos_group_actual", "minutes", "y_yellow", "y_any_card",
            "y_slot_any_card", "y_slot_yellow", "was_replaced", "slot_repl_minutes",
            "repl_card"]
    cand = cand[keep]

    # ---- attach entity states -------------------------------------------- #
    player_state = build_player_state(players, cfg.PLAYER_WINDOW)
    team_state = build_team_state(teams, cfg.TEAM_WINDOW)
    ref_state = build_referee_state(teams)

    # player state carries a `season` column that would clash on the join
    cand = attach_bench_strength(cand, players, player_state, priors, cfg)
    # League position, Elo, fixture difficulty and what is at stake - the
    # things a human reads off the table before looking at any player.
    cand = context.attach_match_context(cand, teams)
    # Closing prices.  Attached here, after the context block, because
    # odds_elo_surprise is the market's view minus ctx_match_difficulty and
    # therefore needs the Elo column to already exist.
    if verbose:
        print("Attaching closing prices:")
    odds_tbl = odds_features.load_match_odds(
        seasons=sorted(cand["season"].dropna().unique()), verbose=verbose)
    # An upcoming fixture has no archive row; its price arrives via extra_odds.
    if extra_odds is not None and len(extra_odds):
        odds_tbl = pd.concat([odds_tbl, extra_odds], ignore_index=True)
    cand = odds_features.attach_odds(cand, odds=odds_tbl, verbose=verbose)

    df = asof_attach(cand, player_state.drop(columns=["season"]), by="player_key")
    df = asof_attach(df, team_state, by="team")

    opp_state = team_state.rename(columns={"team": "opponent"})
    opp_state = opp_state[["opponent", "kickoff", "team_fouled_pm", "team_ppg",
                           "team_fouls_pm", "team_poss_share"]]
    opp_state.columns = ["opponent", "kickoff", "opp_fouled_pm", "opp_ppg",
                         "opp_fouls_pm", "opp_poss_share"]
    df = asof_attach(df, opp_state, by="opponent")
    ref_state_j = ref_state.rename(columns={"season": "_ref_state_season"})
    df = asof_attach(df, ref_state_j, by="referee")

    # referee x position-group history.  The state is keyed on the position a
    # player *actually* filled in each past match, but the lookup must use his
    # rolling modal position - ``pos_group_actual`` is only known after kick-off,
    # so keying the lookup on it would train on something that is "UNK" at serve
    # time.  This is the same distinction the model makes everywhere else
    # between pos_group (prior) and pos_group_actual.
    ref_pos_state = build_referee_pos_state(players)
    df["_rp_key"] = (df["referee"].astype(str) + "|"
                     + df["pos_group_prior"].fillna("UNK").astype(str))
    df = asof_attach(df, ref_pos_state, by="_rp_key")

    # player x referee pairing history (recommendation 4)
    pr_state = build_player_ref_state(players)
    df["_pr_key"] = df["player_key"].astype(str) + "|" + df["referee"].astype(str)
    df = asof_attach(df, pr_state, by="_pr_key")

    # season-to-date card count must not carry across a season boundary
    sd = player_state[["player_key", "kickoff", "season", "cards_season_to_date"]].copy()
    sd = sd.rename(columns={"cards_season_to_date": "_cstd", "season": "_state_season"})
    df = asof_attach(df, sd, by="player_key")
    df["cards_season_to_date"] = np.where(
        df["_state_season"] == df["season"], df["_cstd"], 0.0
    )

    # ---- days of rest ------------------------------------------------------ #
    last_player_kick = player_state[["player_key", "kickoff"]].copy()
    last_player_kick["plr_last_kickoff"] = last_player_kick["kickoff"]
    df = asof_attach(df, last_player_kick, by="player_key")
    df["plr_days_rest"] = (df["kickoff"] - df["plr_last_kickoff"]).dt.total_seconds() / 86400.0

    last_team_kick = team_state[["team", "kickoff"]].copy()
    last_team_kick["team_last_kickoff"] = last_team_kick["kickoff"]
    df = asof_attach(df, last_team_kick, by="team")
    df["team_days_rest"] = (df["kickoff"] - df["team_last_kickoff"]).dt.total_seconds() / 86400.0

    # ---- derived features -------------------------------------------------- #
    k90 = cfg.EB_PRIOR_MINUTES / 90.0

    df["plr_apps_prior"] = df["apps"].fillna(0.0)
    df["plr_mins_roll"] = _safe_div(df["roll_min"], df["roll_apps"])
    df["plr_fouls_p90_eb"] = eb_rate(df["cum_fouls"], df["cum_min"] / 90.0,
                                     priors.fouls_p90, k90)
    df["plr_cards_p90_eb"] = eb_rate(df["cum_cards"], df["cum_min"] / 90.0,
                                     priors.cards_p90, k90)
    df["plr_fouls_p90_roll"] = _safe_div(df["roll_fouls"], df["roll_min"] / 90.0)
    df["plr_tackles_p90_roll"] = _safe_div(df["roll_tackles"], df["roll_min"] / 90.0)
    df["plr_duels_lost_p90_roll"] = _safe_div(df["roll_duels_lost"], df["roll_min"] / 90.0)
    df["plr_cards_pm_roll"] = _safe_div(df["roll_cards"], df["roll_apps"])
    df["plr_fouls_per_tackle_eb"] = eb_rate(df["cum_fouls"], df["cum_tackles"],
                                            priors.fouls_per_tackle, 20.0)
    # How readily this player's fouls turn into a card - a dimension that is
    # genuinely separate from how often they foul (reputation, timing, dissent).
    df["plr_cards_per_foul_eb"] = eb_rate(df["cum_cards"], df["cum_fouls"],
                                          priors.cards_per_foul, 40.0)
    df["plr_fouled_p90_roll"] = _safe_div(df["roll_fouled"], df["roll_min"] / 90.0)
    df["plr_fouls_p90_ewm"] = _safe_div(df["ewm_fouls"] * 90.0, df["ewm_min"])
    df["plr_cards_p90_ewm"] = _safe_div(df["ewm_cards"] * 90.0, df["ewm_min"])
    df["plr_matches_since_card"] = df["matches_since_card"]
    df["plr_cards_season_to_date"] = df["cards_season_to_date"]
    df["plr_age_years"] = df["age_years"]

    # --- the replacement leg of the slot ------------------------------- #
    # How often this starter is taken off, and how many minutes he typically
    # hands over.  Both are pure exposure terms for the second leg.
    df["plr_subbed_off_rate_eb"] = eb_rate(df["cum_subbed_off"], df["cum_starts"],
                                           priors.subbed_off_rate, cfg.EB_PRIOR_MATCHES)
    df["plr_exp_repl_mins"] = _safe_div(df["roll_repl_min"], df["roll_starts"])
    df["plr_exp_repl_mins"] = pd.Series(df["plr_exp_repl_mins"], index=df.index).fillna(
        priors.repl_minutes_per_start
    )
    df["bench_cards_p90_eb"] = df["bench_cards_p90_eb"].fillna(priors.cards_p90)

    # ---- recommendation 4: richer player context --------------------------- #
    # Defensive workload - the exposure term behind fouling.  A centre-back who
    # makes eight defensive actions a game has eight chances to mistime one.
    df["plr_defacts_p90_eb"] = eb_rate(df["cum_defacts"], df["cum_min"] / 90.0,
                                       priors.defacts_p90, k90)
    # ...and how clumsy that workload is.  Separates "does a lot of defending"
    # from "fouls while doing it", which the raw foul rate conflates.
    df["plr_fouls_per_defact_eb"] = eb_rate(df["cum_fouls"], df["cum_defacts"],
                                            priors.fouls_per_defact, 20.0)
    # Aggressor or victim: the share of this player's duels that end with him
    # committing the foul rather than drawing it.
    df["plr_aggression_ratio"] = _safe_div(
        df["cum_fouls"], df["cum_fouls"] + df["cum_fouled"])
    # Penalties conceded are rare, but a rash tackler in his own box is a
    # different animal from a tactical fouler in midfield.
    df["plr_pens_p90_eb"] = eb_rate(df["cum_pens"], df["cum_min"] / 90.0,
                                    priors.pens_p90, k90 * 3)

    # Venue-split discipline, and the gap between the two halves.
    df["plr_cards_p90_home_eb"] = eb_rate(df["cum_home_cards"],
                                          df["cum_home_min"] / 90.0,
                                          priors.cards_p90, k90)
    df["plr_cards_p90_away_eb"] = eb_rate(df["cum_away_cards"],
                                          df["cum_away_min"] / 90.0,
                                          priors.cards_p90, k90)
    df["plr_venue_card_gap"] = (df["plr_cards_p90_away_eb"]
                                - df["plr_cards_p90_home_eb"])
    # The one that is actually relevant to this fixture: his rate at THIS venue.
    df["plr_cards_p90_this_venue"] = np.where(
        df["is_home"] == 1, df["plr_cards_p90_home_eb"], df["plr_cards_p90_away_eb"])

    # This player under this referee, shrunk towards his own career rate.
    df["plr_ref_pair_apps"] = df["pr_apps"].fillna(0.0)
    df["plr_ref_pair_cards_p90"] = eb_rate(
        df["pr_cards"], df["pr_min"] / 90.0, priors.cards_p90, k90)
    # Expressed as a ratio to his own rate, so the feature is "this referee
    # books him more than average" rather than a second copy of his card rate.
    df["plr_ref_pair_ratio"] = _safe_div(df["plr_ref_pair_cards_p90"],
                                         df["plr_cards_p90_eb"])

    # Form divergence: recent fouling against the career baseline.
    df["plr_fouls_trend"] = _safe_div(df["plr_fouls_p90_ewm"], df["plr_fouls_p90_eb"])

    # Suspension pressure (see SUSPENSION_THRESHOLDS).
    to_ban, mw_left = suspension_pressure(df["cards_season_to_date"], df["matchweek"])
    df["plr_cards_to_ban"] = to_ban
    df["plr_ban_window_left"] = mw_left

    df["team_fouls_pm_roll"] = df["team_fouls_pm"]
    df["team_cards_pm_roll"] = df["team_cards_pm"]
    df["team_poss_share_roll"] = df["team_poss_share"]
    df["team_ppg_roll"] = df["team_ppg"]
    df["opp_fouls_drawn_pm_roll"] = df["opp_fouled_pm"]
    df["opp_ppg_roll"] = df["opp_ppg"]
    df["strength_diff"] = df["team_ppg"] - df["opp_ppg"]
    # Expected share of the ball.  The side that will chase the game without it
    # is the side that commits the fouls.
    df["poss_share_diff"] = df["team_poss_share"] - df["opp_poss_share"]

    df["ref_matches_prior"] = df["ref_matches"].fillna(0.0)
    df["ref_cards_pm_recent"] = df["ref_cards_pm_recent"]
    df["ref_cards_pm_eb"] = eb_rate(df["ref_cum_cards"], df["ref_matches"],
                                    priors.ref_cards_per_match * 2,  # both teams
                                    cfg.EB_PRIOR_REF_MATCHES)
    df["ref_cards_per_foul_eb"] = eb_rate(
        df["ref_cum_cards"], df["ref_cum_fouls"],
        priors.ref_cards_per_foul,
        cfg.EB_PRIOR_REF_MATCHES * priors.ref_fouls_per_match,
    )

    # --- referee: distribution, not volume ---------------------------------- #
    # Share of this referee's cards that go to the away side.  The league runs
    # above 0.5 (away teams are booked more), so the feature is the deviation
    # from that, shrunk for officials with short careers.
    ref_total_side_cards = df["ref_cum_home_cards"] + df["ref_cum_away_cards"]
    df["ref_away_card_share_eb"] = eb_rate(
        df["ref_cum_away_cards"], ref_total_side_cards,
        priors.away_card_share, cfg.EB_PRIOR_REF_MATCHES * 3.5)
    # The version that is actually about this row: his card rate for the side
    # this player is on.
    df["ref_venue_card_bias"] = np.where(
        df["is_home"] == 1,
        1.0 - df["ref_away_card_share_eb"],
        df["ref_away_card_share_eb"]) / 0.5

    # Season-to-date level against his own career level: catches a referee who
    # has changed how he calls the game this year, which a career average and a
    # 25-match rolling window both smooth away.
    ref_season_rate = _safe_div(df["ref_season_cards"], df["ref_season_matches"])
    ref_career_rate = _safe_div(df["ref_cum_cards"], df["ref_matches"])
    same_season = df["_ref_state_season"] == df["season"]
    df["ref_season_level_ratio"] = np.where(
        same_season & (df["ref_season_matches"] >= 3),
        _safe_div(ref_season_rate, ref_career_rate), np.nan)

    # Does this referee book THIS position group more than officials generally?
    rp_rate = _safe_div(df["rp_cum_cards"], df["rp_cum_players"])
    df["ref_pos_affinity"] = _safe_div(rp_rate, priors.cards_per_match)
    df["ref_pos_apps"] = df["rp_cum_players"].fillna(0.0)

    # analytic prior: expected cards = fouls the player is expected to commit
    # times how readily this referee turns a foul into a card.
    exp_mins = df["plr_mins_roll"].fillna(80.0).clip(1, 90)
    df["exp_cards_analytic"] = (
        df["plr_fouls_p90_eb"] * exp_mins / 90.0 * df["ref_cards_per_foul_eb"]
    )

    # Same idea for the replacement: expected minutes on the pitch times the
    # bench's card rate, scaled by how card-happy the referee is.
    ref_factor = _safe_div(df["ref_cards_pm_eb"], priors.ref_cards_per_match * 2.0)
    ref_factor = pd.Series(ref_factor, index=df.index).fillna(1.0)
    df["slot_exp_repl_cards"] = (
        df["plr_exp_repl_mins"] / 90.0 * df["bench_cards_p90_eb"] * ref_factor
    )

    # ---- categoricals ------------------------------------------------------ #
    df["pos_group"] = df["pos_group_prior"].fillna("UNK").astype(str)
    df.loc[df["plr_apps_prior"] == 0, "pos_group"] = "UNK"
    df["pos_group"] = pd.Categorical(df["pos_group"], categories=POS_GROUPS)
    df["formation"] = df["formation"].fillna("UNK").astype(str)
    df["formation"] = pd.Categorical(df["formation"])

    df["is_starter"] = df["is_starter"].astype(int)
    try:
        df[cfg.TARGET_COL] = {
            "any_card": df["y_any_card"],            # the named player only
            "yellow": df["y_yellow"],
            "slot_any_card": df["y_slot_any_card"],  # the player OR his substitute
            "slot_yellow": df["y_slot_yellow"],
        }[cfg.TARGET_MODE]
    except KeyError:
        raise ValueError(f"unknown TARGET_MODE {cfg.TARGET_MODE!r}") from None

    out_cols = cfg.ID_COLUMNS + cfg.FEATURES + list(
        getattr(cfg, "CANDIDATE_FEATURES", [])) + list(
        getattr(cfg, "REFEREE_CANDIDATES", [])) + list(
        getattr(cfg, "ODDS_CANDIDATES", [])) + [
        cfg.TARGET_COL, "y_yellow", "y_any_card", "y_slot_any_card", "y_slot_yellow",
        "was_replaced", "repl_card", "minutes", "pos_group_actual"]
    out_cols = list(dict.fromkeys(out_cols))  # matchweek is both an id and a feature
    out = df[out_cols].sort_values(["kickoff", "match_id", "team", "player"], kind="mergesort")
    out = out.reset_index(drop=True)

    if verbose:
        print(f"\nmodel dataset: {out.shape[0]:,} rows x {len(cfg.FEATURES)} features")
        print(f"  target ({cfg.TARGET_MODE}) base rate: {out[cfg.TARGET_COL].mean():.4f}")
        print(f"    player-only base rate    : {out['y_any_card'].mean():.4f}")
        print(f"    slot (player or sub)     : {out['y_slot_any_card'].mean():.4f}")
        miss = out[cfg.FEATURES].isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss):
            print("  features with missing values (XGBoost handles these natively):")
            for name, frac in miss.items():
                print(f"    {name:28s} {frac:.3%}")
    return out, priors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v3 model dataset.")
    parser.add_argument("--include-bench", action="store_true")
    args = parser.parse_args()
    if args.include_bench:
        config.INCLUDE_BENCH = True

    players = pd.read_parquet(config.BASE_TABLE_PATH)
    teams = pd.read_parquet(config.MATCH_TABLE_PATH)
    ds, _ = build_model_dataset(players, teams)
    ds.to_parquet(config.MODEL_DATASET_PATH, index=False)
    print(f"\nwrote {config.MODEL_DATASET_PATH}")


if __name__ == "__main__":
    main()
