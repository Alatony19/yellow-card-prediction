"""Stage 1 of the match pipeline - the team-match spine.

Starts from the card pipeline's ``base_match.parquet`` (goals, fouls, passes,
points, possession share, already one row per team per match), adds the two
things it lacks, and turns the result into a model dataset:

* **Expected goals**, summed per side from the shot-level files.  xG is a far
  better predictor of future goals than past goals are - goals are xG plus a
  large helping of finishing variance, and the variance does not repeat.
* **Closing prices**, joined from football-data.co.uk (see odds.py).

Every rolling feature follows the same rule as the card pipeline: a value
attached to match M may only use matches that kicked off strictly before M,
implemented as a post-match cumulative plus a backward non-exact ``merge_asof``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, odds as odds_mod
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import odds as odds_mod

# The card pipeline one directory up also has a module called `config`, so it
# cannot be reached with a plain import - the name is already taken by this
# package's own config.  Load both of its modules by path under distinct names.
def _load_from(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_V3 = Path(__file__).resolve().parent.parent
card_config = _load_from("card_config", _V3 / "pipeline" / "config.py")
context_mod = _load_from("v3_context", _V3 / "pipeline" / "context.py")

_MATCHDAY_RE = re.compile(r"match_day_(\d+)")
SHOT_FILE = "conbined_shots_stats.csv"     # sic - the scraper's spelling
ON_TARGET = {"Goal", "Saved"}


# --------------------------------------------------------------------------- #
# expected goals
# --------------------------------------------------------------------------- #
def build_shot_aggregates(raw_dir: Path | None = None,
                          verbose: bool = True) -> pd.DataFrame:
    """Per (match, team) shot volume and expected goals, from the shot files."""
    raw_dir = Path(raw_dir or config.RAW_DATA_DIR)
    rows, n_files, n_missing = [], 0, 0
    have_file: set[str] = set()
    for season_dir in sorted(p for p in raw_dir.glob("*-*") if p.is_dir()):
        mds = sorted(season_dir.glob("match_day_*"),
                     key=lambda p: int(_MATCHDAY_RE.search(p.name).group(1)))
        for md in mds:
            matchday = int(_MATCHDAY_RE.search(md.name).group(1))
            for game in sorted(p for p in md.iterdir() if p.is_dir()):
                if "_" not in game.name:
                    continue
                home, away = game.name.split("_", 1)
                match_id = f"{season_dir.name}|{matchday:02d}|{home}|{away}"
                path = game / SHOT_FILE
                if not path.exists():
                    n_missing += 1
                    continue
                try:
                    sh = pd.read_csv(path)
                except Exception:
                    n_missing += 1
                    continue
                n_files += 1
                have_file.add(match_id)
                if not len(sh) or "Squad" not in sh.columns:
                    continue
                sh["xG"] = pd.to_numeric(sh.get("xG"), errors="coerce").fillna(0.0)
                sh["PSxG"] = pd.to_numeric(sh.get("PSxG"), errors="coerce").fillna(0.0)
                sh["_sot"] = sh.get("Outcome", pd.Series(dtype=object)).isin(ON_TARGET)
                for team, grp in sh.groupby("Squad"):
                    rows.append({
                        "match_id": match_id,
                        "team": str(team).strip(),
                        "shot_xg": float(grp["xG"].sum()),
                        "shot_psxg": float(grp["PSxG"].sum()),
                        "shots": float(len(grp)),
                        "shots_on_target": float(grp["_sot"].sum()),
                    })
    out = pd.DataFrame(rows, columns=["match_id", "team", "shot_xg", "shot_psxg", "shots",
                                      "shots_on_target"])
    out.attrs["matches_with_file"] = have_file
    if verbose:
        print(f"shots: {n_files:,} files parsed, {n_missing} missing, "
              f"{len(out):,} (match, team) rows "
              f"covering {len(have_file):,} matches")
    return out


# --------------------------------------------------------------------------- #
# rolling form
# --------------------------------------------------------------------------- #
def _roll(df: pd.DataFrame, group, col: str, window: int) -> pd.Series:
    return df.groupby(group, sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )


def build_team_form(tm: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """Post-match rolling form, one row per (team, match).

    Two windows per quantity: one over all matches, one over matches at the
    same venue only.  Home and away form differ enough in this league that a
    single blended number throws away real signal.
    """
    t = tm.sort_values(["team", "kickoff"], kind="mergesort").copy()
    st = pd.DataFrame({"team": t["team"], "kickoff": t["kickoff"]})

    pairs = [
        ("shot_xg", "xg_for"), ("opp_shot_xg", "xg_against"),
        ("goals_for", "goals_for"), ("goals_against", "goals_against"),
        ("shots_on_target", "sot_for"), ("points", "ppg"),
        ("poss_share", "poss_share"),
    ]
    for src, name in pairs:
        st[f"f_{name}_roll"] = _roll(t, "team", src, cfg.FORM_WINDOW)

    # venue-specific form: group by (team, side) so a home row only ever sees
    # that team's previous home matches
    for src, name in [("shot_xg", "xg_for_venue"), ("opp_shot_xg", "xg_against_venue")]:
        st[f"f_{name}_roll"] = _roll(t, ["team", "side"], src, cfg.VENUE_WINDOW)

    st["f_matches_played"] = t.groupby(["team", "season"], sort=False).cumcount() + 1.0
    return st.reset_index(drop=True)


def _asof(left: pd.DataFrame, state: pd.DataFrame, by: str) -> pd.DataFrame:
    return pd.merge_asof(
        left.sort_values("kickoff", kind="mergesort"),
        state.sort_values("kickoff", kind="mergesort"),
        on="kickoff", by=by, direction="backward", allow_exact_matches=False,
    )


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = pd.read_parquet(card_config.MATCH_TABLE_PATH)
    if verbose:
        print(f"base_match: {len(teams):,} team rows "
              f"({teams['match_id'].nunique():,} matches)")

    shots = build_shot_aggregates(verbose=verbose)
    missing = set(zip(teams["match_id"], teams["team"])) - set(
        zip(shots["match_id"], shots["team"]))
    if missing and verbose:
        print(f"  ! {len(missing)} (match, team) pairs have no shot rows "
              f"- xG will be missing for them, e.g. {sorted(missing)[:3]}")

    tm = teams.merge(shots, on=["match_id", "team"], how="left")
    opp_shots = shots.rename(columns={
        "team": "opponent", "shot_xg": "opp_shot_xg", "shot_psxg": "opp_shot_psxg",
        "shots": "opp_shots", "shots_on_target": "opp_shots_on_target"})
    tm = tm.merge(opp_shots, on=["match_id", "opponent"], how="left")
    # A side absent from a shot file that *exists* took no shots, and zero is
    # right (it happens about twice in eight seasons).  A match with no shot file
    # at all was never published, and must stay NaN - otherwise a whole season
    # without xG looks like a season in which nobody shot.
    shot_cols = ["shot_xg", "shot_psxg", "shots", "shots_on_target",
                 "opp_shot_xg", "opp_shot_psxg", "opp_shots", "opp_shots_on_target"]
    have_file = shots.attrs.get("matches_with_file", set())
    published = tm["match_id"].isin(have_file)
    tm.loc[published, shot_cols] = tm.loc[published, shot_cols].fillna(0.0)
    if verbose:
        n_unpub = int((~published).sum())
        if n_unpub:
            seasons = sorted(tm.loc[~published, "season"].unique())
            print(f"  {n_unpub:,} team-matches have no shot file at all "
                  f"-> xG left missing, not zeroed  (seasons: {seasons})")

    tm["is_home"] = (tm["side"] == "home").astype(int)
    tm = tm.sort_values(["kickoff", "match_id", "side"], kind="mergesort").reset_index(drop=True)

    # ---- rolling form, attached strictly backward ------------------------- #
    form = build_team_form(tm)
    own = _asof(tm, form, by="team")
    opp_form = form.rename(columns={"team": "opponent", **{
        c: "opp_" + c for c in form.columns if c.startswith("f_")}})
    df = _asof(own, opp_form, by="opponent")

    # ---- days of rest ------------------------------------------------------ #
    last = form[["team", "kickoff"]].copy()
    last["_last_kick"] = last["kickoff"]
    df = _asof(df, last, by="team")
    df["days_rest"] = (df["kickoff"] - df["_last_kick"]).dt.total_seconds() / 86400.0
    last_opp = last.rename(columns={"team": "opponent", "_last_kick": "_opp_last_kick"})
    df = _asof(df, last_opp, by="opponent")
    df["opp_days_rest"] = (df["kickoff"] - df["_opp_last_kick"]).dt.total_seconds() / 86400.0

    # ---- league position, Elo, difficulty, stakes -------------------------- #
    df = context_mod.attach_match_context(df, teams)

    # ---- closing prices ----------------------------------------------------- #
    od = odds_mod.load_odds(verbose=verbose)
    stray = odds_mod.check_name_coverage(od, teams)
    if stray and verbose:
        print(f"  ! odds names absent from the FBref archive: {stray}")
    df["home_team"] = np.where(df["is_home"] == 1, df["team"], df["opponent"])
    df["away_team"] = np.where(df["is_home"] == 1, df["opponent"], df["team"])
    df = df.merge(
        od[["season", "home_team", "away_team", "p_home", "p_draw", "p_away",
            "p_over25", "mkt_lambda_home", "mkt_lambda_away", "mkt_total_lambda",
            "fd_home_goals", "fd_away_goals",
            "odds_home_best", "odds_draw_best", "odds_away_best",
            "odds_home_avg", "odds_draw_avg", "odds_away_avg",
            "odds_over25_best", "odds_under25_best",
            "odds_over25_avg", "odds_under25_avg"]],
        on=["season", "home_team", "away_team"], how="left")

    home = df["is_home"] == 1
    df["mkt_lambda_for"] = np.where(home, df["mkt_lambda_home"], df["mkt_lambda_away"])
    df["mkt_lambda_against"] = np.where(home, df["mkt_lambda_away"], df["mkt_lambda_home"])

    # ---- rename the form columns into the configured feature names --------- #
    ren = {f"f_{n}_roll": f"{n}_roll" for n in
           ["xg_for", "xg_against", "goals_for", "goals_against", "sot_for",
            "ppg", "poss_share", "xg_for_venue", "xg_against_venue"]}
    ren.update({f"opp_f_{n}_roll": f"opp_{n}_roll" for n in
                ["xg_for", "xg_against", "goals_for", "goals_against", "sot_for",
                 "ppg", "poss_share", "xg_for_venue", "xg_against_venue"]})
    ren["f_matches_played"] = "matches_played"
    df = df.rename(columns=ren)

    # ---- sanity: does the goal count agree with the independent odds feed? -- #
    if verbose:
        both = df[home & df["fd_home_goals"].notna()]
        agree = ((both["goals_for"] == both["fd_home_goals"])
                 & (both["goals_against"] == both["fd_away_goals"])).mean()
        print(f"  scoreline agreement FBref vs football-data: {agree:.4%} "
              f"of {len(both):,} matched matches")

    df = df.drop(columns=[c for c in ["_last_kick", "_opp_last_kick"] if c in df.columns])
    df = df.sort_values(["kickoff", "match_id", "side"], kind="mergesort").reset_index(drop=True)

    match = to_match_level(df)
    return df, match


def to_match_level(tm: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two team rows of each match into one row, home first."""
    home = tm[tm["is_home"] == 1].copy()
    away = tm[tm["is_home"] == 0].copy()
    keep_away = ["match_id", "goals_for"]
    away = away[keep_away].rename(columns={"goals_for": "away_goals"})
    out = home.rename(columns={"goals_for": "home_goals"}).merge(
        away, on="match_id", how="inner", validate="one_to_one")
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["result"] = np.select(
        [out["home_goals"] > out["away_goals"], out["home_goals"] == out["away_goals"]],
        ["H", "D"], default="A")
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the match-model tables.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tm, match = build(verbose=not args.quiet)
    tm.to_parquet(config.TEAM_MATCH_PATH, index=False)
    match.to_parquet(config.MATCH_DATASET_PATH, index=False)
    print(f"\nteam-match rows : {len(tm):,}")
    print(f"match rows      : {len(match):,}")
    print(f"  home win rate : {(match['result'] == 'H').mean():.4f}")
    print(f"  draw rate     : {(match['result'] == 'D').mean():.4f}")
    print(f"  mean goals    : {match['total_goals'].mean():.4f}")
    print(f"  over 2.5 rate : {(match['total_goals'] > 2.5).mean():.4f}")
    print(f"\nwrote {config.TEAM_MATCH_PATH}")
    print(f"wrote {config.MATCH_DATASET_PATH}")


if __name__ == "__main__":
    main()
