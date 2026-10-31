"""Stage 1 - turn the raw scraped match folders into two tidy tables.

Outputs
-------
base_player_match.parquet
    One row per (match, player named in the line-up sheet).  Holds the raw
    in-match counting stats, so it is a *post-match* table: it is the source of
    both the label and of every historical (lagged) feature, but no column may
    be fed to the model directly.

base_match.parquet
    One row per (match, team) with team-level aggregates and match metadata
    (referee, kick-off, venue).

The v2 loader had two defects that are fixed here:
  * the bench marker was searched on the concatenated home+away frame, so every
    away starter was labelled starter=False (only ~30k of 66.9k starters were
    flagged correctly);
  * players were matched to the line-up sheet with a merge that left ~7% of
    rows NaN.
Here each side is parsed independently and the line-up sheet is the spine, so
the starter flag is exact (verified: 11 starters per side in every match) and
unused substitutes are retained with appeared=False.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config

warnings.filterwarnings("ignore", category=FutureWarning)

# Raw per-player columns kept from the FBref tables (verified present in all
# eight seasons, both home and away files).
PLAYER_STAT_COLUMNS = [
    "Min",
    "Performance_Gls", "Performance_OG",
    "Performance_CrdY", "Performance_CrdR", "Performance_2CrdY",
    "Performance_Fls", "Performance_Fld",
    "Performance_Tkl", "Performance_TklW", "Performance_Int",
    "Performance_Touches", "Performance_Recov", "Performance_Crs",
    "Challenges_Att", "Challenges_Lost",
    "Aerial Duels_Won", "Aerial Duels_Lost",
    "Total_Att", "Carries_Carries", "Take-Ons_Att",
    "Blocks_Blocks", "Clr", "Err", "Performance_PKcon", "Performance_Off",
]

RENAME_MAP = {
    "Min": "minutes",
    "Performance_Gls": "goals",
    "Performance_OG": "own_goals",
    "Performance_CrdY": "yellow",
    "Performance_CrdR": "red",
    "Performance_2CrdY": "second_yellow",
    "Performance_Fls": "fouls",
    "Performance_Fld": "fouled",
    "Performance_Tkl": "tackles",
    "Performance_TklW": "tackles_won",
    "Performance_Int": "interceptions",
    "Performance_Touches": "touches",
    "Performance_Recov": "recoveries",
    "Performance_Crs": "crosses",
    "Challenges_Att": "challenges_att",
    "Challenges_Lost": "challenges_lost",
    "Aerial Duels_Won": "aerials_won",
    "Aerial Duels_Lost": "aerials_lost",
    "Total_Att": "passes_att",
    "Carries_Carries": "carries",
    "Take-Ons_Att": "takeons_att",
    "Blocks_Blocks": "blocks",
    "Clr": "clearances",
    "Err": "errors",
    "Performance_PKcon": "pens_conceded",
    "Performance_Off": "offsides",
}

COUNT_COLUMNS = [v for v in RENAME_MAP.values() if v != "minutes"]

# FBref does not spell every club the same way in every table, and the
# fixtures table it is scraped from now uses shorter names than the archive
# does.  A club under two names is a club with two histories: Elo restarts it at
# the promoted-side default, the league table forgets its points, and rolling
# form resets.  Canonical spelling is the archive's, because eight seasons of
# data already use it.
TEAM_ALIASES = {
    "Newcastle": "Newcastle Utd",
    "Nottingham": "Nott'ham Forest",
    "Nottingham Forest": "Nott'ham Forest",
    "Manchester United": "Manchester Utd",
    "Sheffield United": "Sheffield Utd",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Leeds": "Leeds United",
    # La Liga: FBref renamed Deportivo between its 2017-18 and 2026-27 spells.
    "Dep. La Coruña": "Dep. A Coruña",
}


def canonical_team(name: object) -> str:
    """Map a club name onto the archive's spelling."""
    s = str(name).strip()
    return TEAM_ALIASES.get(s, s)


_MATCHDAY_RE = re.compile(r"match_day_(\d+)")


# --------------------------------------------------------------------------- #
# position mapping
# --------------------------------------------------------------------------- #
_POS_PRIORITY = [
    ("GK", {"GK"}),
    ("CB", {"CB"}),
    ("FB", {"LB", "RB", "WB", "LWB", "RWB"}),
    ("DM", {"DM"}),
    ("CM", {"CM"}),
    ("AM", {"AM"}),
    ("W", {"LM", "RM", "LW", "RW"}),
    ("FW", {"FW", "ST", "CF"}),
]


def position_group(pos_raw: object) -> str:
    """Collapse the FBref multi-position string into one of 8 coarse buckets.

    The first listed position is FBref's primary one, so it wins; the priority
    order only settles exotic strings whose first token is unknown.
    """
    if not isinstance(pos_raw, str) or not pos_raw.strip():
        return "UNK"
    tokens = [t.strip().upper() for t in pos_raw.split(",") if t.strip()]
    if not tokens:
        return "UNK"
    first = tokens[0]
    for label, members in _POS_PRIORITY:
        if first in members:
            return label
    for label, members in _POS_PRIORITY:
        if any(t in members for t in tokens):
            return label
    return "UNK"


# --------------------------------------------------------------------------- #
# line-up parsing
# --------------------------------------------------------------------------- #
def parse_lineups(path: Path) -> pd.DataFrame:
    """Parse one lineups.csv into a tidy frame.

    The file holds two independent 3-column blocks (home then away).  Each block
    lists the XI, a "Bench" separator row, then the named substitutes.  The
    column header carries "Team Name (formation)".
    """
    raw = pd.read_csv(path)
    blocks = []
    for side, (c_num, c_player, c_side) in {"home": (0, 1, 2), "away": (3, 4, 5)}.items():
        header = str(raw.columns[c_num])
        formation = header.split("(")[-1].rstrip(")").strip() if "(" in header else np.nan

        block = raw.iloc[:, [c_num, c_player, c_side]].copy()
        block.columns = ["shirt_number", "player", "side_raw"]

        # Everything from the "Bench" separator onwards is a substitute.
        # NB: done per side, which is the bug fixed relative to v2.
        bench_rows = block.index[block["shirt_number"].astype(str) == "Bench"]
        block["is_starter"] = True
        if len(bench_rows):
            block.loc[bench_rows.min():, "is_starter"] = False
        block = block[block["shirt_number"].astype(str) != "Bench"]
        block = block.dropna(subset=["player"])

        block["formation"] = formation
        block["side"] = side
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    out["player"] = out["player"].astype(str).str.strip()
    # Two namesakes in one squad (Sevilla 2024-25 named both Alberto Lópezes on
    # the bench) cannot be told apart in the stats table, which is keyed on the
    # name alone and already keeps the first.  Keep the first here too; the XI
    # precedes the bench, so a starter is never the one dropped.
    out = out.drop_duplicates(subset=["player", "side"], keep="first")
    return out[["player", "shirt_number", "side", "formation", "is_starter"]]


def _read_player_stats(path: Path) -> pd.DataFrame:
    """Read one side's player-stat table, dropping the trailing totals row."""
    df = pd.read_csv(path)
    if len(df):
        df = df.iloc[:-1]  # last row is the team total
    keep = ["Player", "Pos", "Age"] + [c for c in PLAYER_STAT_COLUMNS if c in df.columns]
    df = df[keep].rename(columns={"Player": "player", **RENAME_MAP})
    df["player"] = df["player"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["player"])
    # FBref prints the table in substitution order; that ordering is the only
    # record of who replaced whom, so carry it through the merge below.
    df["stat_order"] = np.arange(len(df), dtype=float)
    return df


def parse_age(age_str: object) -> float:
    """FBref ages look like "25-243" (years-days)."""
    if not isinstance(age_str, str) or "-" not in age_str:
        return np.nan
    years, _, days = age_str.partition("-")
    try:
        return int(years) + int(days) / 365.25
    except ValueError:
        return np.nan



def assign_slots(players: pd.DataFrame, match_id: str) -> pd.DataFrame:
    """Recover each team's substitution chains from FBref's row ordering.

    FBref prints a team's players in *slot* order: a starter, then whoever
    replaced him, then whoever replaced that replacement.  Anchoring the slot
    boundaries on the line-up sheet's starters -- rather than on minutes adding
    up to 90 -- stays exact when a sending-off leaves a slot short.  Validated
    over all 6,080 team-tables in the archive: exactly 11 slots every time, and
    never a slot longer than the match.

    Adds ``slot_id`` (NaN for unused substitutes), ``slot_seat`` (0 = starter,
    1 = his replacement, 2 = the next one) and ``slot_starter``.
    """
    players = players.copy()
    players["slot_id"] = pd.NA
    players["slot_seat"] = np.nan
    players["slot_starter"] = pd.NA

    for side in ("home", "away"):
        block = players[(players["side"] == side) & players["appeared"]]
        block = block.sort_values("stat_order", kind="mergesort")
        if block.empty:
            continue
        idx, ids, seats, owners = [], [], [], []
        slot_no, seat, owner = -1, 0, None
        for row_idx, is_start, name in zip(block.index, block["is_starter"], block["player"]):
            if is_start or slot_no < 0:
                slot_no += 1
                seat = 0
                owner = name
            else:
                seat += 1
            idx.append(row_idx)
            ids.append(f"{match_id}|{side}|{slot_no:02d}")
            seats.append(seat)
            owners.append(owner)
        players.loc[idx, "slot_id"] = ids
        players.loc[idx, "slot_seat"] = seats
        players.loc[idx, "slot_starter"] = owners
    return players


# --------------------------------------------------------------------------- #
# per-match assembly
# --------------------------------------------------------------------------- #
def load_match(game_dir: Path, season: str):
    """Return (player rows, team rows) for one match folder, or None if unusable."""
    folder = game_dir.name
    if "_" not in folder:
        return None
    home_team, away_team = (canonical_team(x) for x in folder.split("_", 1))

    md_match = _MATCHDAY_RE.search(game_dir.parent.name)
    matchday = int(md_match.group(1)) if md_match else -1

    try:
        lineups = parse_lineups(game_dir / "lineups.csv")
        home_stats = _read_player_stats(game_dir / "home_player_stats.csv")
        away_stats = _read_player_stats(game_dir / "away_player_stats.csv")
        main = pd.read_csv(game_dir / "main.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError, IndexError, KeyError) as exc:
        print(f"  ! skipping {season}/{folder}: {type(exc).__name__}: {exc}")
        return None

    if lineups.empty or main.empty:
        print(f"  ! skipping {season}/{folder}: empty lineups/main")
        return None

    # The folder name is the authoritative home/away order; the line-up sheet
    # only supplies FBref's (sometimes longer) team labels.
    lineups["team"] = lineups["side"].map({"home": home_team, "away": away_team})
    lineups["opponent"] = lineups["side"].map({"home": away_team, "away": home_team})

    home_stats["side"] = "home"
    away_stats["side"] = "away"
    stats = pd.concat([home_stats, away_stats], ignore_index=True)

    # Line-up sheet is the spine: unused substitutes survive with appeared=False.
    players = lineups.merge(stats, on=["player", "side"], how="left", validate="one_to_one")

    players["appeared"] = players["minutes"].notna()
    players["minutes"] = pd.to_numeric(players["minutes"], errors="coerce").fillna(0.0)
    for col in COUNT_COLUMNS:
        if col in players.columns:
            # A player row with a blank cell really did record zero of the thing.
            players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0.0)
        else:
            # The whole column is absent from the source file, which is a
            # different statement: the statistic was never published for this
            # match.  NaN, not 0, so a rolling mean skips it instead of being
            # dragged towards zero.  (FBref stopped publishing the advanced
            # per-player tables after the 2024-25 season; see README.)
            players[col] = np.nan

    meta = main.iloc[0]
    venue_date = meta.get("venue_date")
    venue_time = meta.get("venue_time")
    if not isinstance(venue_time, str) or not venue_time.strip():
        venue_time = "15:00"
    kickoff = pd.to_datetime(f"{venue_date} {venue_time}", errors="coerce")
    if pd.isna(kickoff):
        kickoff = pd.to_datetime(venue_date, errors="coerce")

    match_id = f"{season}|{matchday:02d}|{home_team}|{away_team}"

    players = assign_slots(players, match_id)

    players["match_id"] = match_id
    players["season"] = season
    players["matchday"] = matchday
    players["matchweek"] = pd.to_numeric(meta.get("matchweek"), errors="coerce")
    players["kickoff"] = kickoff
    players["home_team"] = home_team
    players["away_team"] = away_team
    players["referee"] = str(meta.get("referee") or "Unknown").strip()
    players["venue"] = meta.get("venue")
    players["attendance"] = pd.to_numeric(meta.get("attendance"), errors="coerce")
    players["age_years"] = players["Age"].map(parse_age)
    players = players.rename(columns={"Pos": "pos_raw"})

    # --- team-level aggregates -------------------------------------------- #
    agg_cols = ["fouls", "fouled", "yellow", "red", "tackles", "passes_att",
                "touches", "challenges_lost", "aerials_lost", "minutes",
                "goals", "own_goals"]
    # min_count=1 so a statistic that the source never published stays NaN at
    # team level too, instead of summing a column of NaN down to a false 0.
    team = players.groupby(["team", "side", "opponent"], as_index=False)[agg_cols].sum(
        min_count=1)
    team = team.rename(columns={c: f"team_{c}" for c in agg_cols})
    team["match_id"] = match_id
    team["season"] = season
    team["matchday"] = matchday
    team["kickoff"] = kickoff
    team["referee"] = players["referee"].iloc[0]

    return players, team


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def iter_match_dirs(raw_dir: Path, seasons=None):
    for season_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if seasons and season_dir.name not in seasons:
            continue
        md_dirs = [p for p in season_dir.iterdir() if p.is_dir()]
        md_dirs.sort(key=lambda p: int(_MATCHDAY_RE.search(p.name).group(1))
                     if _MATCHDAY_RE.search(p.name) else 0)
        for md_dir in md_dirs:
            for game_dir in sorted(p for p in md_dir.iterdir() if p.is_dir()):
                yield season_dir.name, game_dir


def resolve_player_identity(players: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add ``player_key``, a stable identity that survives name collisions.

    Player history is keyed on the display name, which breaks when two different
    people share one.  A name listed for two different clubs *on the same
    calendar date* cannot be one person, so such names are split by club.  Every
    other name is left untouched, so ordinary transfers keep a single history.
    """
    players = players.copy()
    date = players["kickoff"].dt.date
    teams_per_day = players.groupby([players["player"], date])["team"].nunique()
    colliding = set(teams_per_day[teams_per_day > 1].index.get_level_values(0))

    players["player_key"] = players["player"]
    if colliding:
        mask = players["player"].isin(colliding)
        players.loc[mask, "player_key"] = (
            players.loc[mask, "player"] + " [" + players.loc[mask, "team"] + "]"
        )
        if verbose:
            print(f"  identity: split {len(colliding)} colliding name(s) by club: "
                  f"{sorted(colliding)}")
    return players


def _add_result_columns(teams: pd.DataFrame) -> pd.DataFrame:
    """Add goals for/against and league points to the (match, team) table.

    Goals scored by a team = its players' goals + the opponent's own goals.
    """
    opp = teams[["match_id", "team", "team_goals", "team_own_goals",
                 "team_fouls", "team_fouled", "team_passes_att", "team_yellow",
                 "team_red"]].copy()
    opp.columns = ["match_id", "opponent", "opp_goals_raw", "opp_own_goals",
                   "opp_fouls", "opp_fouled", "opp_passes_att", "opp_yellow",
                   "opp_red"]
    teams = teams.merge(opp, on=["match_id", "opponent"], how="left", validate="one_to_one")

    teams["goals_for"] = teams["team_goals"] + teams["opp_own_goals"].fillna(0.0)
    teams["goals_against"] = teams["opp_goals_raw"].fillna(0.0) + teams["team_own_goals"]
    diff = teams["goals_for"] - teams["goals_against"]
    teams["points"] = np.select([diff > 0, diff == 0], [3.0, 1.0], default=0.0)
    teams["poss_share"] = teams["team_passes_att"] / (
        teams["team_passes_att"] + teams["opp_passes_att"].replace(0, np.nan)
    )
    return teams


def build(raw_dir: Path = config.RAW_DATA_DIR, seasons=None, verbose: bool = True):
    """Parse the raw match folders into the two base tables.

    ``seasons`` defaults to ``config.ALL_SEASONS`` rather than to whatever
    directories happen to exist.  A season that is mid-scrape, or a stray
    folder, would otherwise be absorbed silently -- and a half-finished season
    is worse than a missing one, because the rolling histories, the league
    table and the Elo ratings all take it at face value.  Pass an explicit list
    to override.
    """
    if seasons is None:
        seasons = list(config.ALL_SEASONS)
    on_disk = {p.name for p in Path(raw_dir).iterdir() if p.is_dir()}
    ignored = sorted(on_disk - set(seasons))
    if ignored and verbose:
        print(f"  ignoring season folders not in config.ALL_SEASONS: {ignored}")
    missing = sorted(set(seasons) - on_disk)
    if missing:
        raise FileNotFoundError(
            f"config.ALL_SEASONS names seasons with no folder under {raw_dir}: {missing}")

    player_frames, team_frames = [], []
    n_ok = n_bad = 0
    for season, game_dir in iter_match_dirs(raw_dir, seasons):
        result = load_match(game_dir, season)
        if result is None:
            n_bad += 1
            continue
        players, team = result
        player_frames.append(players)
        team_frames.append(team)
        n_ok += 1
        if verbose and n_ok % 500 == 0:
            print(f"  ... {n_ok} matches parsed")

    if not player_frames:
        raise RuntimeError(f"No matches parsed from {raw_dir}")

    players = pd.concat(player_frames, ignore_index=True)
    teams = pd.concat(team_frames, ignore_index=True)

    # ---- labels ----------------------------------------------------------- #
    # FBref's CrdY already counts the first booking of a second-yellow sending
    # off (verified: every CrdY == 2 row has 2CrdY == 1), so yellow >= 1
    # captures second bookings too.
    players["y_yellow"] = (players["yellow"] >= 1).astype(int)
    players["y_any_card"] = ((players["yellow"] >= 1) | (players["red"] >= 1)).astype(int)

    # Slot labels.  A player-booking market resolves a starter as a hit if *he
    # or whoever replaces him* is booked, so the card is rolled up to the slot.
    # Unused substitutes have no slot and keep their own (always 0) label.
    grp = players.groupby("slot_id", dropna=True)
    own_any, own_yel = players["y_any_card"], players["y_yellow"]
    players["y_slot_any_card"] = grp["y_any_card"].transform("max").fillna(own_any).astype(int)
    players["y_slot_yellow"] = grp["y_yellow"].transform("max").fillna(own_yel).astype(int)

    slot_size = grp["player"].transform("size")
    players["slot_size"] = slot_size.fillna(1).astype(int)
    players["was_replaced"] = ((players["slot_seat"] == 0) & (slot_size > 1)).astype(int)
    # Minutes the slot handed to a replacement - the exposure window that turns
    # a miss on the starter alone into a hit on the slot.
    players["slot_repl_minutes"] = (
        grp["minutes"].transform("sum") - players["minutes"]
    ).fillna(0.0).clip(lower=0.0)
    # Did the *replacement* leg alone produce the card?
    players["repl_card"] = (
        (players["y_slot_any_card"] == 1) & (players["y_any_card"] == 0)
        & (players["slot_seat"] == 0)
    ).astype(int)

    players["is_home"] = (players["side"] == "home").astype(int)
    players["pos_group_actual"] = players["pos_raw"].map(position_group)
    players = resolve_player_identity(players, verbose=verbose)

    players = players.sort_values(["kickoff", "match_id", "side", "is_starter"],
                                  ascending=[True, True, True, False])
    players = players.reset_index(drop=True)
    teams = teams.sort_values(["kickoff", "match_id", "side"]).reset_index(drop=True)
    teams = _add_result_columns(teams)

    if verbose:
        print(f"\nParsed {n_ok} matches ({n_bad} skipped)")
        print(f"  player rows : {len(players):,}")
        print(f"  team rows   : {len(teams):,}")
        starters = players[players["is_starter"]]
        print(f"  starters    : {len(starters):,} "
              f"({len(starters) / max(n_ok, 1):.2f} per match, expect 22.00)")
        print(f"  card rate (starters, any card): {starters['y_any_card'].mean():.4f}")
        out_starters = starters[starters["pos_group_actual"] != "GK"]
        print(f"  outfield starters: {len(out_starters):,}")
        print(f"    player only      : {out_starters['y_any_card'].mean():.4f}")
        print(f"    player or his sub: {out_starters['y_slot_any_card'].mean():.4f}")
        print(f"    replaced at all  : {out_starters['was_replaced'].mean():.4f}")
        print(f"    won on the sub   : {out_starters['repl_card'].mean():.4f}")

    return players, teams


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v3 base tables.")
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="default: config.ALL_SEASONS")
    parser.add_argument("--raw-dir", default=str(config.RAW_DATA_DIR))
    args = parser.parse_args()

    config.BASE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    players, teams = build(Path(args.raw_dir), args.seasons)
    players.to_parquet(config.BASE_TABLE_PATH, index=False)
    teams.to_parquet(config.MATCH_TABLE_PATH, index=False)
    print(f"\nwrote {config.BASE_TABLE_PATH}")
    print(f"wrote {config.MATCH_TABLE_PATH}")


if __name__ == "__main__":
    main()
