"""One card model for five leagues: data build and evaluation.

``league_comparison.py`` settled the architecture on team-level counts: the
leagues differ in the *level* of cards but share the *structure*, so one pooled
model with a per-league level beats five local fits.  This module puts that to
the player-level card model, using the FBref scrape of Serie A, La Liga, the
Bundesliga and Ligue 1 (2021-22 -> 2025-26) alongside the Premier League
archive (2017-18 -> 2026-27).

Each league is run through the *unchanged* single-league pipeline -
``build_base_table.build`` then ``features.build_model_dataset`` - with its own
raw folder, its own odds archive and its own seasons, so every rolling history,
league table, Elo rating and empirical-Bayes prior stays inside its league.
Only the finished model datasets are stacked, with a ``league`` column added.

Stages
------
    python pipeline/pooled.py build      # base tables + model dataset per new league
    python pipeline/pooled.py build --first-season 2017-2018   # a longer history
    python pipeline/pooled.py evaluate   # the first comparison: PL-only vs pooled
    python pipeline/pooled.py compare    # every built arm, same matches, top-4 picks
    python pipeline/pooled.py train --arm pooled_2017_us       # fit and save one arm
    python pipeline/pooled.py attach-history   # refresh the history features in place

The evaluation question is the one the user stakes on: pick four players per
match, count how many of the four get carded (player or his substitute, the
settled slot label).  See ``evaluate`` for the arms.
"""
from __future__ import annotations

import argparse
import types
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_base_table
import config
import features
from match_model import config as mm_config
from match_model import odds as odds_mod

POOLED_DIR = config.POOLED_DIR
# Superseded builds and the first comparison, kept for ``compare``/``evaluate``.
ARCHIVE_DIR = POOLED_DIR / "archive"
FIRST_BUILD_DIR = ARCHIVE_DIR / "first_build"
POOLED_DATASET_PATH = FIRST_BUILD_DIR / "pooled_model_dataset.parquet"
POOLED_MODEL_PATH = FIRST_BUILD_DIR / "pooled_card_model.joblib"

PL = "premier_league"
# The first comparison (pooled_top4_results.csv) used these five seasons for the
# four new leagues.  Builds now take a first season and run through the live
# season; see ``league_seasons``.
NEW_LEAGUE_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
DEFAULT_FIRST_SEASON = "2021-2022"


def league_seasons(first: str = DEFAULT_FIRST_SEASON) -> list[str]:
    """``first`` through the live season, on the Premier League's calendar."""
    return [s for s in config.ALL_SEASONS if s >= first]


def dataset_dir(first: str = DEFAULT_FIRST_SEASON) -> Path:
    """Where a build starting in ``first`` lives, e.g. artifacts/pooled/from_2017.

    Only the served build sits in ``pooled/``; the others were moved to
    ``pooled/archive/`` and are still found there.  A new build is written to
    ``pooled/``.
    """
    name = f"from_{first[:4]}"
    if not (POOLED_DIR / name).exists() and (ARCHIVE_DIR / name).exists():
        return ARCHIVE_DIR / name
    return POOLED_DIR / name


def dataset_path(first: str = DEFAULT_FIRST_SEASON) -> Path:
    return dataset_dir(first) / "pooled_model_dataset.parquet"

# league key -> (FBref folder under data/fbref/, odds folder under data/odds/)
LEAGUES = {
    "serie_a": ("serie_a", "serie_a"),
    "la_liga": ("la_liga", "la_liga"),
    "bundesliga": ("bundesliga", "bundesliga"),
    "ligue_1": ("ligue_1", "ligue_1"),
}

# football-data.co.uk spelling -> FBref spelling, per league.  Filled from the
# unmatched-name report ``build`` prints; the PL map lives in match_model/odds.py.
FD_TO_FBREF = {
    "serie_a": {"Verona": "Hellas Verona", "Spal": "SPAL"},
    "la_liga": {"Alaves": "Alavés", "Almeria": "Almería", "Ath Bilbao": "Athletic Club",
                "Ath Madrid": "Atlético Madrid", "Betis": "Real Betis", "Cadiz": "Cádiz",
                "Celta": "Celta Vigo", "Espanol": "Espanyol", "Leganes": "Leganés",
                "Sociedad": "Real Sociedad", "Vallecano": "Rayo Vallecano",
                "La Coruna": "Dep. A Coruña", "Malaga": "Málaga",
                "Santander": "Racing Sant"},
    "bundesliga": {"Bielefeld": "Arminia", "Darmstadt": "Darmstadt 98",
                   "Ein Frankfurt": "Frankfurt", "FC Koln": "Köln",
                   "Greuther Furth": "Greuther Fürth", "Hamburg": "Hamburger SV",
                   "Hertha": "Hertha BSC", "M'gladbach": "Gladbach", "Mainz": "Mainz 05",
                   "Paderborn": "Paderborn 07", "Fortuna Dusseldorf": "Düsseldorf",
                   "Hannover": "Hannover 96", "Nurnberg": "Nürnberg"},
    "ligue_1": {"Clermont": "Clermont Foot", "St Etienne": "Saint-Étienne", "Nimes": "Nîmes"},
}


def _cfg_copy():
    """A mutable snapshot of ``config`` (modules cannot be copied directly)."""
    return types.SimpleNamespace(**{k: v for k, v in vars(config).items()
                                    if k.isupper()})


def league_cfg(league: str, seasons: list[str]):
    """A copy of ``config`` with this league's seasons.

    A new league's first season is its burn-in (it only warms the rolling
    histories), exactly as 2017-18 is for the Premier League, and its priors
    come from the seasons after it up to the same cut-off the Premier League
    uses.  Validation and test seasons are the Premier League's.
    """
    cfg = _cfg_copy()
    if league == PL:
        return cfg
    cfg.ALL_SEASONS = list(seasons)
    cfg.BURN_IN_SEASONS = [seasons[0]]
    cfg.TRAIN_SEASONS = [s for s in seasons[1:] if s in config.TRAIN_SEASONS]
    cfg.VALID_SEASONS = list(config.VALID_SEASONS)
    cfg.TEST_SEASONS = list(config.TEST_SEASONS)
    return cfg


def league_odds(league: str, seasons) -> pd.DataFrame:
    odds_dir = config.ODDS_ROOT / LEAGUES[league][1]
    tbl = odds_mod.load_odds(seasons=seasons, odds_dir=odds_dir, verbose=False)
    fix = FD_TO_FBREF[league]
    for col in ("home_team", "away_team"):
        tbl[col] = tbl[col].map(lambda s: fix.get(s, s))
    return tbl


def _name_report(league: str, teams: pd.DataFrame, odds: pd.DataFrame) -> None:
    fb = set(teams["team"].unique())
    fd = set(odds["home_team"]) | set(odds["away_team"])
    only_fb, only_fd = sorted(fb - fd), sorted(fd - fb)
    if only_fb or only_fd:
        print(f"  ! {league} club names that do not join the odds feed")
        print(f"      FBref only: {only_fb}")
        print(f"      odds only : {only_fd}")


def build_league(league: str, first: str = DEFAULT_FIRST_SEASON,
                 verbose: bool = True) -> pd.DataFrame:
    out_dir = dataset_dir(first) / league
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = league_cfg(league, league_seasons(first))
    raw_dir = config.FBREF_DIR / LEAGUES[league][0]

    base_p, base_t = out_dir / "base_player_match.parquet", out_dir / "base_match.parquet"
    if base_p.exists() and base_t.exists():
        players, teams = pd.read_parquet(base_p), pd.read_parquet(base_t)
    else:
        players, teams = build_base_table.build(raw_dir, cfg.ALL_SEASONS, verbose=verbose)
        players.to_parquet(base_p, index=False)
        teams.to_parquet(base_t, index=False)

    odds = league_odds(league, cfg.ALL_SEASONS)
    _name_report(league, teams, odds)
    ds = league_model_dataset(players, teams, league, cfg, odds, verbose=verbose)
    ds.to_parquet(out_dir / "model_dataset.parquet", index=False)
    return ds


def league_model_dataset(players: pd.DataFrame, teams: pd.DataFrame, league: str,
                         cfg, odds: pd.DataFrame | None,
                         extra_odds: pd.DataFrame | None = None,
                         verbose: bool = False,
                         history: pd.DataFrame | None = None) -> pd.DataFrame:
    """``features.build_model_dataset`` for one league, with its own prices.

    ``features.build_model_dataset`` loads the odds archive itself, from the
    match model's (Premier League) ODDS_DIR.  For another league that lookup is
    pointed at nothing and the league's own table - plus any ``extra_odds``
    row for an unplayed fixture - is handed in through ``extra_odds``.

    ``history`` (the five-league appearance table) adds the history features
    (``history_features.py``) - what a served fixture needs when the model
    was trained on them.
    """
    if league == PL:
        ds, _ = features.build_model_dataset(players, teams, cfg=cfg, verbose=verbose,
                                             extra_odds=extra_odds)
    else:
        table = odds if extra_odds is None else pd.concat([odds, extra_odds],
                                                          ignore_index=True)
        orig_load = features.odds_features.load_match_odds
        features.odds_features.load_match_odds = lambda *a, **k: table.iloc[0:0]
        try:
            ds, _ = features.build_model_dataset(players, teams, cfg=cfg,
                                                 verbose=verbose, extra_odds=table)
        finally:
            features.odds_features.load_match_odds = orig_load
    ds.insert(0, "league", pd.Categorical([league] * len(ds),
                                          categories=[PL] + list(LEAGUES)))
    if history is not None:
        import history_features
        ds = history_features.attach_dataset(ds, {league: players}, history)
    return ds


# --------------------------------------------------------------------------- #
# serving: the shipped five-league model and each league's history
# --------------------------------------------------------------------------- #
# Chosen by ``compare`` (artifacts/pooled/compare_top4_per_match.parquet): the
# 2019-, 2018- and 2017-onward builds are within noise of each other; 2017 is
# the full archive and the best of the three on the Premier League.
FINAL_ARM = "pooled_2017"
FINAL_FIRST_SEASON = "2017-2018"
FINAL_MODEL_PATH = POOLED_DIR / f"{FINAL_ARM}_card_model.joblib"
ALL_LEAGUES = [PL] + list(LEAGUES)


def league_history(league: str, first: str = FINAL_FIRST_SEASON):
    """(players, teams, cfg, odds) a fixture in ``league`` is scored against.

    The Premier League reads the shipped base tables; the other four read the
    per-league tables of the ``first`` build.  ``odds`` is None for the Premier
    League, whose archive ``features`` already loads.
    """
    if league not in ALL_LEAGUES:
        raise ValueError(f"unknown league {league!r}; one of {ALL_LEAGUES}")
    if league == PL:
        return (pd.read_parquet(config.BASE_TABLE_PATH),
                pd.read_parquet(config.MATCH_TABLE_PATH), config, None)
    d = dataset_dir(first) / league
    cfg = league_cfg(league, league_seasons(first))
    return (pd.read_parquet(d / "base_player_match.parquet"),
            pd.read_parquet(d / "base_match.parquet"), cfg,
            league_odds(league, cfg.ALL_SEASONS))


def match_table(teams: pd.DataFrame) -> pd.DataFrame:
    """One row per match (home_team, away_team, goals, kickoff) for Dixon-Coles."""
    home = teams[teams["side"] == "home"]
    return pd.DataFrame({
        "home_team": home["team"].to_numpy(), "away_team": home["opponent"].to_numpy(),
        "home_goals": home["goals_for"].to_numpy(),
        "away_goals": home["goals_against"].to_numpy(),
        "kickoff": home["kickoff"].to_numpy(), "season": home["season"].to_numpy(),
    })


def history_tables(first: str = FINAL_FIRST_SEASON) -> dict:
    """league -> the base player table each league's rows were built from."""
    out = {PL: pd.read_parquet(config.BASE_TABLE_PATH)}
    for lg in LEAGUES:
        out[lg] = pd.read_parquet(dataset_dir(first) / lg / "base_player_match.parquet")
    return out


def attach_history(ds: pd.DataFrame, first: str = FINAL_FIRST_SEASON) -> pd.DataFrame:
    """Every history feature on the stacked dataset (``history_features.py``).

    Always computed, like the candidate columns ``features.py`` writes, so a
    feature group is tested or shipped by changing ``config`` alone.
    """
    import history_features
    return history_features.attach_dataset(ds, history_tables(first),
                                           history_features.load_history())


def served_cfg():
    """The served ensemble's feature contract.

    The shipped single-league set, plus ``league``, plus the history features in
    ``config.HISTORY_FEATURES`` - with the career rates swapped for their
    cross-league versions when ``config.USE_XL_RATES``.
    """
    import history_features
    cfg = _cfg_for(list(config.HISTORY_FEATURES), with_league=True)
    if config.USE_XL_RATES:
        cfg.NUMERIC_FEATURES = [history_features.XL_RATES.get(f, f)
                                for f in cfg.NUMERIC_FEATURES]
        cfg.FEATURES = cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES
    return cfg


def uses_history(model) -> bool:
    """True when ``model`` was trained on any history feature."""
    import history_features
    return bool(set(getattr(model, "features", [])) & set(history_features.ALL_FEATURES))


def load_final_model():
    from modelling import load_model
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"{FINAL_MODEL_PATH.name} missing - run: "
                                f"python pipeline/pooled.py train --arm {FINAL_ARM}")
    return load_model(FINAL_MODEL_PATH)


def build(leagues=None, first: str = DEFAULT_FIRST_SEASON) -> pd.DataFrame:
    """Build every league from ``first`` and stack them with the Premier League.

    Understat team-style columns (``understat_features.US_FEATURES``) are
    attached to every row; whether a model *uses* them is the arm's choice.
    """
    import understat_features

    leagues = leagues or list(LEAGUES)
    frames = []
    for league in leagues:
        print(f"\n=== {league} from {first} ===")
        frames.append(build_league(league, first))
    pl = pd.read_parquet(config.MODEL_DATASET_PATH)
    pl.insert(0, "league", PL)
    frames.insert(0, pl)
    pooled = pd.concat(frames, ignore_index=True)
    pooled["league"] = pd.Categorical(pooled["league"], categories=[PL] + list(LEAGUES))
    if understat_features.TEAM_MATCH_PATH.exists():
        pooled = understat_features.attach(pooled)
    pooled = attach_history(pooled, first)
    out = dataset_path(first)
    out.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(pooled):,} rows")
    print(pooled.groupby(["league", "season"], observed=True)["match_id"].nunique()
          .unstack(0).fillna(0).astype(int).to_string())
    return pooled


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
PICKS = 4
N_BOOT = 2000


def _cfg_with_league():
    """The shipped feature set plus ``league`` as a categorical level term."""
    cfg = _cfg_copy()
    cfg.CATEGORICAL_FEATURES = list(config.CATEGORICAL_FEATURES) + ["league"]
    cfg.FEATURES = list(config.FEATURES) + ["league"]
    return cfg


def _past(season: str, n_back: int) -> str:
    y = int(season[:4]) - n_back
    return f"{y}-{y + 1}"


def _usable(ds: pd.DataFrame, leagues, seasons) -> pd.DataFrame:
    """Rows of these leagues and seasons, minus each league's burn-in season.

    The burn-in is the first season a league has in ``ds`` - 2017-18 for the
    Premier League, whatever the build started from for the others.
    """
    m = ds["league"].isin(leagues) & ds["season"].isin(seasons)
    first = ds.groupby("league", observed=True)["season"].transform("min")
    return ds[m & (ds["season"] != first)]


def picks_table(te: pd.DataFrame, prob_col: str, k: int = PICKS) -> pd.DataFrame:
    """One row per match: how many of the model's top-k picks were carded."""
    d = te[["match_id", "player", prob_col, config.TARGET_COL]].copy()
    d = d.sort_values(["match_id", prob_col, "player"], ascending=[True, False, True])
    d["_rank"] = d.groupby("match_id").cumcount()
    top = d[d["_rank"] < k]
    return top.groupby("match_id")[config.TARGET_COL].sum().rename("hits").to_frame()


def summarise(hits: pd.Series, k: int = PICKS) -> dict:
    out = {"matches": int(len(hits)), "mean_hits": float(hits.mean()),
           "precision": float(hits.mean() / k), "at_least_1": float((hits >= 1).mean()),
           "at_least_2": float((hits >= 2).mean())}
    for n in range(k + 1):
        out[f"p{n}"] = float((hits == n).mean())
    return out


def paired_bootstrap(a: pd.Series, b: pd.Series, seed: int = 0) -> tuple:
    """Mean of (b - a) over matches with a 95% bootstrap interval."""
    diff = (b - a).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def evaluate(ds: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """Walk forward; for each test (league, season) score every arm that applies.

    Arms, all fitted exactly like the shipped model (XGBoost, early stopping on
    the season before the test season, trained on everything before that):

    * ``pl_only``  - the Premier League data only.  On Premier League seasons
      this is the shipped model; on the other leagues it is that model applied
      abroad unchanged.
    * ``pooled``   - all five leagues, plus ``league`` as a categorical feature
      so each league keeps its own level.
    * ``local``    - the test league's own data only (new leagues only, and only
      where there is at least one training season before the validation one).

    The score is the shortlist: the model's top four starters in each match, and how
    many of those four were carded (the player or his substitute).
    """
    from modelling import fit_model

    ds = ds if ds is not None else pd.read_parquet(POOLED_DATASET_PATH)
    all_leagues = [PL] + list(LEAGUES)
    ds["league"] = pd.Categorical(ds["league"].astype(str), categories=all_leagues)
    cfg_pool = _cfg_with_league()

    tests = [(PL, s) for s in config.WALK_FORWARD_SEASONS + config.TEST_SEASONS]
    tests += [(lg, s) for lg in LEAGUES for s in NEW_LEAGUE_SEASONS[1:]]

    cache: dict = {}

    def fitted(arm: str, season: str, leagues) -> object:
        key = (arm, season)
        if key not in cache:
            valid_season = _past(season, 1)
            train_seasons = [s for s in config.ALL_SEASONS if s < valid_season]
            tr = _usable(ds, leagues, train_seasons)
            va = _usable(ds, leagues, [valid_season])
            if tr.empty or va.empty:
                cache[key] = None
            else:
                cfg = cfg_pool if arm == "pooled" else config
                if verbose:
                    print(f"  fit {arm:<18} for {season}: {tr['match_id'].nunique():,} "
                          f"train / {va['match_id'].nunique():,} valid matches")
                cache[key] = (fit_model(tr, va, cfg, verbose=False), cfg)
        return cache[key]

    per_match, rows = [], []
    for league, season in tests:
        te = ds[(ds["league"] == league) & (ds["season"] == season)].copy()
        if te.empty:
            continue
        arms = {"pl_only": [PL], "pooled": all_leagues}
        if league != PL:
            arms[f"local"] = [league]
        hits = {}
        for arm, leagues in arms.items():
            key_arm = arm if arm != "local" else f"local_{league}"
            fm = fitted(key_arm if arm == "local" else arm, season, leagues)
            if fm is None:
                continue
            model, _cfg = fm
            te["p"] = model.predict_proba(te)
            h = picks_table(te, "p")["hits"]
            hits[arm] = h
            rows.append({"league": league, "season": season, "arm": arm, **summarise(h)})
            per_match.append(h.rename("hits").reset_index().assign(
                league=league, season=season, arm=arm))
        # a no-skill reference: four starters picked by the base rate = random
        base = te[config.TARGET_COL].mean()
        rows.append({"league": league, "season": season, "arm": "random",
                     "matches": te["match_id"].nunique(), "mean_hits": PICKS * base,
                     "precision": base})
        if verbose and "pooled" in hits and "pl_only" in hits:
            d, lo, hi = paired_bootstrap(hits["pl_only"], hits["pooled"])
            print(f"{league:<15} {season}  pl_only {hits['pl_only'].mean():.3f}  "
                  f"pooled {hits['pooled'].mean():.3f}  delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    res = pd.DataFrame(rows)
    pm = pd.concat(per_match, ignore_index=True)
    res.to_csv(FIRST_BUILD_DIR / "pooled_top4_results.csv", index=False)
    pm.to_parquet(FIRST_BUILD_DIR / "pooled_top4_per_match.parquet", index=False)
    report(res, pm)
    return res


def report(res: pd.DataFrame, pm: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\n=== four picks per match: mean carded out of 4 ===")
    wide = res.pivot_table(index=["league", "season"], columns="arm",
                           values="mean_hits", observed=True)
    print(wide.round(3).to_string())

    print("\n=== pooled minus pl_only, paired by match (95% bootstrap CI) ===")
    for scope, sub in [("Premier League, complete seasons",
                        pm[(pm["league"] == PL) & pm["season"].isin(config.WALK_FORWARD_SEASONS)]),
                       ("Premier League, 2026-27 so far",
                        pm[(pm["league"] == PL) & pm["season"].isin(config.TEST_SEASONS)]),
                       ("other four leagues", pm[pm["league"] != PL]),
                       ("all five leagues",
                        pm[~pm["season"].isin(config.TEST_SEASONS)])]:
        w = sub.pivot_table(index=["league", "season", "match_id"], columns="arm",
                            values="hits", observed=True).dropna(subset=["pl_only", "pooled"])
        if w.empty:
            continue
        d, lo, hi = paired_bootstrap(w["pl_only"], w["pooled"])
        seasons_better = (w.groupby(level=[0, 1])[["pl_only", "pooled"]].mean()
                          .pipe(lambda x: (x["pooled"] > x["pl_only"])))
        print(f"  {scope:<36} n={len(w):>5}  pl_only {w['pl_only'].mean():.3f}  "
              f"pooled {w['pooled'].mean():.3f}  delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"better in {seasons_better.sum()}/{len(seasons_better)} league-seasons")

    print("\n=== distribution of carded picks (share of matches), Premier League 2021-26 ===")
    pl = pm[(pm["league"] == PL) & pm["season"].isin(config.WALK_FORWARD_SEASONS)]
    dist = (pl.groupby("arm")["hits"].value_counts(normalize=True)
            .unstack(fill_value=0).reindex(columns=range(PICKS + 1), fill_value=0))
    print(dist.round(3).to_string())


# --------------------------------------------------------------------------- #
# arms: which data and which features a model is fitted on
# --------------------------------------------------------------------------- #
def _cfg_for(extra: list[str], with_league: bool = True):
    """The shipped feature set, plus ``extra`` numeric columns and ``league``."""
    cfg = _cfg_copy()
    cfg.NUMERIC_FEATURES = list(config.NUMERIC_FEATURES) + list(extra)
    cfg.CATEGORICAL_FEATURES = list(config.CATEGORICAL_FEATURES) + (
        ["league"] if with_league else [])
    cfg.FEATURES = cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES
    return cfg


def arms() -> dict:
    """Every model the comparison knows about.

    ``first`` picks the dataset (the season the four new leagues start in),
    ``leagues`` which leagues it trains on, ``extra`` any candidate features.
    """
    from understat_features import US_FEATURES
    every = [PL] + list(LEAGUES)
    out = {"pl_only": dict(first=DEFAULT_FIRST_SEASON, leagues=[PL], extra=[], league=False)}
    # One pair of arms per build on disk (artifacts/pooled/from_YYYY), newest
    # history first, so a partially back-filled archive is testable as it lands.
    years = sorted({p.parent.name[5:] for d in (POOLED_DIR, ARCHIVE_DIR)
                    for p in d.glob("from_*/pooled_model_dataset.parquet")}
                   | {DEFAULT_FIRST_SEASON[:4]}, reverse=True)
    for y in years:
        first = f"{y}-{int(y) + 1}"
        out[f"pooled_{y}"] = dict(first=first, leagues=every, extra=[], league=True)
        out[f"pooled_{y}_us"] = dict(first=first, leagues=every, extra=US_FEATURES, league=True)
    return out


def _load(first: str) -> pd.DataFrame | None:
    path = dataset_path(first)
    if not path.exists():
        return None
    ds = pd.read_parquet(path)
    ds["league"] = pd.Categorical(ds["league"].astype(str), categories=[PL] + list(LEAGUES))
    return ds


_FIT_CACHE: dict = {}


def compare(names: list[str] | None = None, baseline: str = "pl_only",
            verbose: bool = True) -> pd.DataFrame:
    """Walk-forward four-pick score for several arms on the *same* matches.

    For each test season S every arm is refitted on its own data before S-1 and
    early-stopped on S-1, as the shipped model is.  Test matches are those every
    arm can score: Premier League from 2021-22, the other leagues from 2022-23
    (the first season a 2021-onward build can be tested on), through 2026-27.
    The Premier-League-only arm is applied unchanged to the other leagues.
    """
    from modelling import fit_model

    spec = arms()
    names = names or list(spec)
    data = {}
    for n in names:
        first = spec[n]["first"]
        if first not in data:
            data[first] = _load(first)
    names = [n for n in names if data[spec[n]["first"]] is not None]
    if verbose:
        print(f"arms: {names}")

    tests = [(PL, s) for s in config.WALK_FORWARD_SEASONS + config.TEST_SEASONS]
    tests += [(lg, s) for lg in LEAGUES
              for s in config.WALK_FORWARD_SEASONS[1:] + config.TEST_SEASONS]

    per_match = []
    for league, season in tests:
        valid_season = _past(season, 1)
        train_seasons = [s for s in config.ALL_SEASONS if s < valid_season]
        preds = {}
        for n in names:
            a = spec[n]
            ds = data[a["first"]]
            te = ds[(ds["league"] == league) & (ds["season"] == season)]
            if te.empty:
                continue
            key = (n, season)
            if key not in _FIT_CACHE:
                tr = _usable(ds, a["leagues"], train_seasons)
                va = _usable(ds, a["leagues"], [valid_season])
                cfg = _cfg_for(a["extra"], a["league"])
                if verbose:
                    print(f"  fit {n:<15} for {season}: {tr['match_id'].nunique():>6,} train"
                          f" / {va['match_id'].nunique():>5,} valid matches", flush=True)
                _FIT_CACHE[key] = fit_model(tr, va, cfg, verbose=False)
            te = te.copy()
            te["p"] = _FIT_CACHE[key].predict_proba(te)
            preds[n] = picks_table(te, "p")["hits"]
        if not preds:
            continue
        common = set.intersection(*(set(h.index) for h in preds.values()))
        for n, h in preds.items():
            h = h[h.index.isin(common)]
            per_match.append(h.reset_index().assign(league=league, season=season, arm=n))
        if verbose:
            line = "  ".join(f"{n} {preds[n][preds[n].index.isin(common)].mean():.3f}"
                             for n in preds)
            print(f"{league:<15} {season}  n={len(common):>3}  {line}", flush=True)

    pm = pd.concat(per_match, ignore_index=True)
    pm.to_parquet(POOLED_DIR / "compare_top4_per_match.parquet", index=False)
    compare_report(pm, baseline)
    return pm


def compare_report(pm: pd.DataFrame, baseline: str = "pl_only") -> None:
    pd.set_option("display.width", 220)
    w = pm.pivot_table(index=["league", "season", "match_id"], columns="arm",
                       values="hits", observed=True).dropna()
    arms_present = [a for a in arms() if a in w.columns]
    print(f"\n=== carded out of 4, {len(w):,} matches every arm scored ===")
    by = w.groupby(level=[0, 1])[arms_present].mean()
    print(by.round(3).to_string())

    lg = w.index.get_level_values(0)
    se = w.index.get_level_values(1)
    wf = se.isin(config.WALK_FORWARD_SEASONS)
    scopes = {
        "Premier League 2021-26": w[(lg == PL) & wf],
        "other leagues 2022-26": w[(lg != PL) & wf],
        "all leagues 2026-27": w[se.isin(config.TEST_SEASONS)],
        "everything": w,
    }
    for other in [a for a in arms_present if a != baseline]:
        print(f"\n--- {other} minus {baseline}, paired by match (95% bootstrap CI) ---")
        for scope, sub in scopes.items():
            if sub.empty:
                continue
            d, lo, hi = paired_bootstrap(sub[baseline], sub[other])
            seas = sub.groupby(level=[0, 1])[[baseline, other]].mean()
            better = int((seas[other] > seas[baseline]).sum())
            print(f"  {scope:<24} n={len(sub):>5}  {baseline} {sub[baseline].mean():.3f}  "
                  f"{other} {sub[other].mean():.3f}  delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"  better in {better}/{len(seas)} league-seasons")


def train(name: str = "pooled_2021", path: Path | None = None) -> None:
    """Fit one arm on everything up to 2024-25, early-stop on 2025-26, and save it.

    Same split as the shipped Premier League model; 2026-27 is left untouched and
    scored at the end.  ``league`` is a categorical feature, so one artifact
    scores a fixture from any of the five leagues.
    """
    from modelling import fit_model, save_model

    a = arms()[name]
    ds = _load(a["first"])
    if ds is None:
        raise FileNotFoundError(f"no dataset for {name}; run: python pipeline/pooled.py build "
                                f"--first-season {a['first']}")
    cfg = _cfg_for(a["extra"], a["league"])
    tr = _usable(ds, a["leagues"], config.TRAIN_SEASONS)
    va = _usable(ds, a["leagues"], config.VALID_SEASONS)
    print(f"{name}: train {tr['match_id'].nunique():,} matches ({len(tr):,} rows), "
          f"valid {va['match_id'].nunique():,} matches")
    model = fit_model(tr, va, cfg, verbose=True)
    path = path or POOLED_DIR / f"{name}_card_model.joblib"
    save_model(model, path)
    print(f"saved {path}")

    print("\nfour picks per match, carded out of 4:")
    hold = ds[ds["season"].isin(config.TEST_SEASONS) & ds["league"].isin(a["leagues"])]
    for label, part in [("valid (used for early stopping)", va), ("hold-out 2026-27", hold)]:
        part = part.copy()
        part["p"] = model.predict_proba(part)
        for league, g in part.groupby("league", observed=True):
            h = picks_table(g, "p")["hits"]
            print(f"  {label:<32} {league:<15} {len(h):>4} matches  "
                  f"{h.mean():.3f}  (random {PICKS * g[config.TARGET_COL].mean():.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["build", "evaluate", "compare", "train",
                                      "attach-history"])
    ap.add_argument("--leagues", nargs="+", choices=list(LEAGUES))
    ap.add_argument("--first-season", default=DEFAULT_FIRST_SEASON,
                    help="build: first season of the four new leagues")
    ap.add_argument("--arms", nargs="+", help="compare: which arms (default: all built)")
    ap.add_argument("--arm", default="pooled_2021", help="train: which arm")
    args = ap.parse_args()
    if args.stage == "build":
        build(args.leagues, args.first_season)
    elif args.stage == "evaluate":
        evaluate()
    elif args.stage == "compare":
        compare(args.arms)
    elif args.stage == "attach-history":
        first = FINAL_FIRST_SEASON
        ds = attach_history(pd.read_parquet(dataset_path(first)), first)
        ds.to_parquet(dataset_path(first), index=False)
        print(f"wrote {dataset_path(first)}: history features attached")
    else:
        train(args.arm)


if __name__ == "__main__":
    main()
