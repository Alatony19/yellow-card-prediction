"""Features drawn from the five-league appearance history.

Three groups, each built from data the base tables already hold but no feature
had read:

* **geometry** - where on the pitch a player actually lines up.  FBref prints
  each team's player table attack-to-defence and, inside each line, left to
  right; read against the line-up's formation that gives every starter a slot
  ``(x, depth)``.  (Checked on every league: LB averages x = 0.12, RB 0.88, CB
  0.50; the lines land exactly where the formation puts them.)  The feature is
  the player's *recent* slot, never the one of the match being predicted - that
  is only known from the line-up diagram, which the serving input does not have.
* **matchup** - who he will be marking.  Each starter's recent slot is mirrored
  onto the opposition's, and the opponents in his zone are weighted by
  distance.  The feature is how often *those* players draw fouls.  It is a
  player x opponent interaction, so unlike the referee features it can reorder
  a field: the left-back facing a winger who is fouled four times a game is a
  different proposition from the right-back facing a target man.
* **cross-league** - a career that started in another of the five leagues.
  Every history in ``features.py`` lives inside one league, so a player who
  moves from Serie A to the Premier League arrives with nothing.  Players are
  linked across leagues by name + date of birth (FBref's ``Age`` is years-days
  at kick-off, so the birth date is exact), and his card and foul rates are
  carried over in units of each league's own average.

Every feature obeys the project's one rule: built from post-match states and
attached with a strictly-backward as-of join (``features.asof_attach``).

    python pipeline/history_features.py      # rebuild the appearance table
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import config
from features import asof_attach

GEOMETRY = ["pos_depth_prior", "pos_width_prior"]
MATCHUP = ["mu_opp_fouled_ratio", "mu_opp_fouled_rel", "mu_opp_fouls_ratio"]
CROSS_LEAGUE = ["plr_xl_m90", "plr_cards_ratio_xl", "plr_fouls_ratio_xl"]
# The cross-league career rates in this league's own units - drop-in
# replacements for features.py's in-league plr_cards_p90_eb / plr_fouls_p90_eb.
XL_RATES = {"plr_cards_p90_eb": "plr_cards_p90_xl", "plr_fouls_p90_eb": "plr_fouls_p90_xl"}
ALL_FEATURES = GEOMETRY + MATCHUP + CROSS_LEAGUE + list(XL_RATES.values())

SLOT_WINDOW = 10          # recent starts that define a player's usual slot
EB_K90 = config.EB_PRIOR_MINUTES / 90.0
MATCHUP_SIGMA = 0.2       # kernel width on the unit pitch
DOB_MERGE_DAYS = 3        # FBref ages drift by a day around leap years

HISTORY_PATH = config.POOLED_DIR / "history_appearances.parquet"


# --------------------------------------------------------------------------- #
# pitch slots
# --------------------------------------------------------------------------- #
def _formation_lines(formation) -> list[int] | None:
    """'4-2-3-1' -> [4, 2, 3, 1] (defence first); None if unreadable."""
    if not isinstance(formation, str):
        return None
    lines = [int(t) for t in re.findall(r"\d+", formation)]
    return lines if len(lines) >= 2 and all(n > 0 for n in lines) else None


def _slot_map(lines: list[int]) -> list[tuple[float, float]]:
    """(x, depth) for each outfield starter in FBref table order.

    The table runs attack to defence, left to right inside each line.  Depth is
    0 for the back line and 1 for the front one; x is 0 on the team's left.
    """
    n = len(lines)
    out = []
    for back_idx in range(n - 1, -1, -1):        # front line first
        size = lines[back_idx]
        depth = back_idx / (n - 1)
        out.extend(((j + 0.5) / size, depth) for j in range(size))
    return out


def slot_coordinates(players: pd.DataFrame) -> pd.DataFrame:
    """``x``, ``depth``, ``width`` for every outfield starter that played.

    NaN where the formation does not account for exactly the outfield starters
    listed (an unreadable formation string, a missing row).
    """
    st = players[players["is_starter"] & players["appeared"]
                 & (players["pos_group_actual"] != "GK")]
    st = st[["match_id", "team", "formation", "stat_order"]].sort_values(
        ["match_id", "team", "stat_order"], kind="mergesort")
    rank = st.groupby(["match_id", "team"], sort=False).cumcount().to_numpy()
    size = st.groupby(["match_id", "team"], sort=False)["stat_order"].transform("size").to_numpy()

    maps: dict = {}
    x = np.full(len(st), np.nan)
    d = np.full(len(st), np.nan)
    for i, (f, r, n) in enumerate(zip(st["formation"].to_numpy(), rank, size)):
        if f not in maps:
            lines = _formation_lines(f)
            maps[f] = _slot_map(lines) if lines else None
        m = maps[f]
        if m is None or len(m) != n:
            continue
        x[i], d[i] = m[r]
    out = pd.DataFrame({"x": x, "depth": d}, index=st.index)
    out["width"] = (out["x"] - 0.5).abs() * 2.0
    return out


# --------------------------------------------------------------------------- #
# identity across leagues
# --------------------------------------------------------------------------- #
def birth_dates(players: pd.DataFrame) -> pd.Series:
    """Date of birth from FBref's ``Age`` ('27-250' = 27 years 250 days at kick-off)."""
    a = players["Age"].astype(str).str.extract(r"^(\d+)-(\d+)$").astype(float)
    ok = a.notna().all(axis=1)
    out = pd.Series(pd.NaT, index=players.index, dtype="datetime64[ns]")
    if not ok.any():
        return out
    last_bday = (players.loc[ok, "kickoff"].dt.normalize()
                 - pd.to_timedelta(a.loc[ok, 1], unit="D"))
    leap = (last_bday.dt.month == 2) & (last_bday.dt.day == 29)
    out.loc[ok] = pd.to_datetime(dict(
        year=last_bday.dt.year - a.loc[ok, 0].astype(int),
        month=last_bday.dt.month,
        day=last_bday.dt.day.where(~leap, 28)), errors="coerce")
    return out


def _canonical_dobs(names: pd.Series, dobs: pd.Series) -> pd.Series:
    """Snap birth dates within DOB_MERGE_DAYS of each other (same name) together."""
    u = pd.DataFrame({"n": names, "d": dobs}).dropna().drop_duplicates()
    u = u.sort_values(["n", "d"])
    gap = u.groupby("n")["d"].diff().dt.days
    new = gap.isna() | (gap > DOB_MERGE_DAYS)
    u["canon"] = u["d"].where(new).ffill()
    lut = u.set_index(["n", "d"])["canon"]
    idx = pd.MultiIndex.from_arrays([names, dobs])
    return pd.Series(lut.reindex(idx).to_numpy(), index=names.index)


def _ident(names: pd.Series, dobs: pd.Series) -> pd.Series:
    s = names.astype(str) + "|" + dobs.dt.strftime("%Y-%m-%d")
    return s.where(dobs.notna())


# --------------------------------------------------------------------------- #
# the appearance table
# --------------------------------------------------------------------------- #
def appearance_table(players_by_league: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per appearance in any league, with slot, identity and counts."""
    frames = []
    for league, p in players_by_league.items():
        p = p[p["appeared"]].copy()
        slots = slot_coordinates(p)
        p = p.join(slots)
        frames.append(pd.DataFrame({
            "league": league, "match_id": p["match_id"], "team": p["team"],
            "player": p["player"], "player_key": p["player_key"],
            "kickoff": p["kickoff"], "season": p["season"],
            "dob": birth_dates(p), "minutes": p["minutes"].fillna(0.0),
            "n_cards": (p["yellow"].fillna(0) + p["red"].fillna(0)).astype(float),
            "fouls": p["fouls"].fillna(0.0), "fouled": p["fouled"].fillna(0.0),
            "is_starter": p["is_starter"].astype(bool),
            "x": p["x"], "depth": p["depth"], "width": p["width"],
        }))
    a = pd.concat(frames, ignore_index=True)
    a["dob"] = _canonical_dobs(a["player"], a["dob"])
    a["ident"] = _ident(a["player"], a["dob"])
    return a.sort_values(["kickoff", "league", "match_id"], kind="mergesort").reset_index(drop=True)


def league_priors(a: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Per-league cards / fouls / fouled per 90 on ``seasons`` (the training span)."""
    s = a[a["season"].isin(seasons)]
    g = s.groupby("league")[["n_cards", "fouls", "fouled", "minutes"]].sum()
    m90 = g["minutes"] / 90.0
    return pd.DataFrame({"cards": g["n_cards"] / m90, "fouls": g["fouls"] / m90,
                         "fouled": g["fouled"] / m90})


def default_priors(a: pd.DataFrame) -> pd.DataFrame:
    """The convention ``features.compute_priors`` uses: burn-in + train seasons."""
    return league_priors(a, config.BURN_IN_SEASONS + config.TRAIN_SEASONS)


# --------------------------------------------------------------------------- #
# post-match states
# --------------------------------------------------------------------------- #
def career_state(a: pd.DataFrame, priors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative career state by identity (all leagues) and by identity x league.

    Counts are stored in units of the league's own average, so a booking in La
    Liga (2.4 yellows a team-match) weighs less than one in the Premier League
    (1.75).
    """
    h = a[a["ident"].notna()].copy()
    h = h.sort_values(["ident", "kickoff"], kind="mergesort")
    h["m90"] = h["minutes"] / 90.0
    for col, prior in (("n_cards", "cards"), ("fouls", "fouls"), ("fouled", "fouled")):
        h[f"{col}_n"] = h[col] / h["league"].map(priors[prior]).to_numpy()
    g = h.groupby("ident", sort=False)
    all_st = pd.DataFrame({
        "ident": h["ident"], "kickoff": h["kickoff"],
        "xl_cum_m90": g["m90"].cumsum(), "xl_cum_cards_n": g["n_cards_n"].cumsum(),
        "xl_cum_fouls_n": g["fouls_n"].cumsum(), "xl_cum_fouled_n": g["fouled_n"].cumsum(),
    })
    h["_il"] = h["ident"] + "#" + h["league"]
    own_st = pd.DataFrame({"_il": h["_il"], "kickoff": h["kickoff"],
                           "own_cum_m90": h.groupby("_il", sort=False)["m90"].cumsum()})
    return all_st, own_st


def slot_state(a: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean slot over each identity's last SLOT_WINDOW starts with a slot."""
    s = a[a["ident"].notna() & a["depth"].notna()]
    s = s.sort_values(["ident", "kickoff"], kind="mergesort")
    g = s.groupby("ident", sort=False)
    roll = lambda col: g[col].transform(lambda v: v.rolling(SLOT_WINDOW, min_periods=1).mean())
    return pd.DataFrame({"ident": s["ident"], "kickoff": s["kickoff"],
                         "slot_x": roll("x"), "slot_depth": roll("depth"),
                         "slot_width": roll("width")})


# --------------------------------------------------------------------------- #
# candidate rows
# --------------------------------------------------------------------------- #
def resolve_idents(cand: pd.DataFrame, a: pd.DataFrame) -> pd.Series:
    """Identity for each candidate row.

    A played row carries its own ``Age``.  A synthetic (unplayed) row takes the
    birth date of the same player_key's most recent appearance in its league;
    failing that - a debut in this league - the one birth date that name has
    anywhere in the five leagues, if it is unambiguous.
    """
    dob = birth_dates(cand) if "Age" in cand else pd.Series(pd.NaT, index=cand.index)
    miss = dob.isna()
    if miss.any():
        hist = a[a["dob"].notna()].sort_values("kickoff")
        by_key = hist.groupby(["league", "player_key"])["dob"].last()
        k = pd.MultiIndex.from_arrays([cand.loc[miss, "league"].astype(str),
                                       cand.loc[miss, "player_key"]])
        dob.loc[miss] = by_key.reindex(k).to_numpy()
        miss = dob.isna()
    if miss.any():
        per_name = hist.groupby("player")["dob"].agg(["nunique", "last"])
        unique = per_name.loc[per_name["nunique"] == 1, "last"]
        dob.loc[miss] = cand.loc[miss, "player"].map(unique).to_numpy()
    dob = _canonical_dobs(pd.concat([a["player"], cand["player"]], ignore_index=True),
                          pd.concat([a["dob"], dob], ignore_index=True)).iloc[len(a):]
    dob.index = cand.index
    return _ident(cand["player"], pd.to_datetime(dob))


def attach(cand: pd.DataFrame, a: pd.DataFrame, priors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ALL_FEATURES to candidate rows.

    ``cand`` needs league, match_id, team, opponent, player, player_key, kickoff
    and (for played rows) ``Age``.  Returns a frame with the same index.
    """
    priors = default_priors(a) if priors is None else priors
    c = cand[["league", "match_id", "team", "opponent", "player", "player_key",
              "kickoff"]].copy()
    c["league"] = c["league"].astype(str)
    c["_row"] = np.arange(len(c))
    c["ident"] = resolve_idents(cand.assign(league=c["league"]), a).to_numpy()
    c["_il"] = c["ident"] + "#" + c["league"]

    all_st, own_st = career_state(a, priors)
    sl = slot_state(a)
    known = c[c["ident"].notna()]
    known = asof_attach(known, all_st, by="ident")
    known = asof_attach(known, own_st, by="_il")
    known = asof_attach(known, sl, by="ident")
    cols = ["xl_cum_m90", "xl_cum_cards_n", "xl_cum_fouls_n", "xl_cum_fouled_n",
            "own_cum_m90", "slot_x", "slot_depth", "slot_width"]
    c = c.merge(known[["_row"] + cols], on="_row", how="left").sort_values("_row")
    for col in cols:
        c[col] = c[col].astype(float)

    m90 = c["xl_cum_m90"].fillna(0.0)
    c["plr_xl_m90"] = m90 - c["own_cum_m90"].fillna(0.0)
    c["plr_cards_ratio_xl"] = (c["xl_cum_cards_n"].fillna(0.0) + EB_K90) / (m90 + EB_K90)
    c["plr_fouls_ratio_xl"] = (c["xl_cum_fouls_n"].fillna(0.0) + EB_K90) / (m90 + EB_K90)
    c["_fouled_ratio"] = (c["xl_cum_fouled_n"].fillna(0.0) + EB_K90) / (m90 + EB_K90)
    c["_fouls_ratio"] = c["plr_fouls_ratio_xl"]
    c["plr_cards_p90_xl"] = c["plr_cards_ratio_xl"] * c["league"].map(priors["cards"]).to_numpy()
    c["plr_fouls_p90_xl"] = c["plr_fouls_ratio_xl"] * c["league"].map(priors["fouls"]).to_numpy()
    c["pos_depth_prior"] = c["slot_depth"]
    c["pos_width_prior"] = c["slot_width"]

    c = c.join(_matchup(c), how="left")
    out = c.set_index(cand.index)[ALL_FEATURES]
    return out


def _matchup(c: pd.DataFrame) -> pd.DataFrame:
    """Distance-weighted read of the opponents in each starter's zone.

    My slot ``(x, d)`` meets an opponent at ``(1 - x', 1 - d')`` - his left is
    my right, his attack is my defence.
    """
    me = c[["_row", "match_id", "team", "opponent", "slot_x", "slot_depth"]].dropna(
        subset=["slot_x"])
    opp = c[["match_id", "team", "slot_x", "slot_depth", "_fouled_ratio",
             "_fouls_ratio"]].dropna(subset=["slot_x"])
    opp = opp.rename(columns={"team": "opponent", "slot_x": "ox", "slot_depth": "od"})
    pairs = me.merge(opp, on=["match_id", "opponent"], how="inner")
    dx = pairs["slot_x"] - (1.0 - pairs["ox"])
    dd = pairs["slot_depth"] - (1.0 - pairs["od"])
    pairs["w"] = np.exp(-(dx ** 2 + dd ** 2) / (2 * MATCHUP_SIGMA ** 2))
    pairs["wf"] = pairs["w"] * pairs["_fouled_ratio"]
    pairs["wl"] = pairs["w"] * pairs["_fouls_ratio"]
    agg = pairs.groupby("_row").agg(w=("w", "sum"), wf=("wf", "sum"), wl=("wl", "sum"),
                                    team_fouled=("_fouled_ratio", "mean"))
    out = pd.DataFrame(index=agg.index)
    ok = agg["w"] > 1e-9
    out["mu_opp_fouled_ratio"] = np.where(ok, agg["wf"] / agg["w"], np.nan)
    out["mu_opp_fouls_ratio"] = np.where(ok, agg["wl"] / agg["w"], np.nan)
    out["mu_opp_fouled_rel"] = out["mu_opp_fouled_ratio"] - agg["team_fouled"]
    # map back from _row to c's index
    pos = pd.Series(c.index, index=c["_row"].to_numpy())
    out.index = pos.reindex(out.index).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# the five shipped histories
# --------------------------------------------------------------------------- #
def base_tables() -> dict[str, pd.DataFrame]:
    """The base player tables of the shipped five-league build."""
    import pooled
    return pooled.history_tables(pooled.FINAL_FIRST_SEASON)


def _base_table_paths() -> list[Path]:
    import pooled
    d = pooled.dataset_dir(pooled.FINAL_FIRST_SEASON)
    return [config.BASE_TABLE_PATH] + [d / lg / "base_player_match.parquet"
                                       for lg in pooled.LEAGUES]


def load_history(rebuild: bool = False) -> pd.DataFrame:
    """The five-league appearance table, cached in HISTORY_PATH.

    Rebuilt automatically when any league's base table is newer than the cache,
    so a freshly scraped matchday is never scored against a stale history.
    """
    if HISTORY_PATH.exists() and not rebuild:
        cached = HISTORY_PATH.stat().st_mtime
        if all(p.stat().st_mtime <= cached for p in _base_table_paths() if p.exists()):
            return pd.read_parquet(HISTORY_PATH)
    a = appearance_table(base_tables())
    a.to_parquet(HISTORY_PATH, index=False)
    return a


def attach_dataset(ds: pd.DataFrame, tables: dict[str, pd.DataFrame],
                   a: pd.DataFrame | None = None) -> pd.DataFrame:
    """``ds`` (a model dataset with a ``league`` column) plus ALL_FEATURES.

    ``tables`` maps league -> the base player table ``ds`` was built from; it
    supplies each played row's ``Age``.  A synthetic fixture row has none and
    is resolved from the history instead (``resolve_idents``).
    """
    a = load_history() if a is None else a
    ages = pd.concat([t.loc[t["is_starter"], ["match_id", "team", "player_key", "Age"]]
                      .assign(league=lg) for lg, t in tables.items()], ignore_index=True)
    ages = ages.drop_duplicates(["league", "match_id", "team", "player_key"])
    keyed = ds[["league", "match_id", "team", "opponent", "player", "player_key",
                "kickoff"]].assign(league=ds["league"].astype(str))
    keyed = keyed.merge(ages, on=["league", "match_id", "team", "player_key"], how="left")
    keyed.index = ds.index
    feats = attach(keyed, a)
    return pd.concat([ds.drop(columns=[c for c in ALL_FEATURES if c in ds]), feats], axis=1)


def main() -> None:
    a = load_history(rebuild=True)
    print(f"wrote {HISTORY_PATH}: {len(a):,} appearances, "
          f"{a['ident'].nunique():,} players, slot known for "
          f"{a.loc[a['is_starter'], 'depth'].notna().mean():.1%} of starts")
    multi = a.groupby("ident")["league"].nunique()
    print(f"players seen in more than one league: {(multi > 1).sum():,}")
    print(default_priors(a).round(4).to_string())


if __name__ == "__main__":
    main()
