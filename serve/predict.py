"""Stage 4 - score a fixture that has not been played yet.

The trick here is that nothing new is computed.  A synthetic, statistics-free
row is appended for each named starter and the *same* ``features.build_model_dataset``
is run over the combined history.  Because every feature is an as-of lookup of
strictly earlier matches, the synthetic rows pick up exactly the history they
would have had in training - so there is no train/serve skew to get wrong.

The default model is the five-league ensemble (``ensemble.ENSEMBLE_PATH``):
bagged boosters, a logistic and the team-count decomposition, blended with
weights chosen walk-forward.  A fixture from any of the big five is scored by
passing ``league=``; the Premier League is the default.  ``card_probability``
is the ensemble's number, and ``bag_lo`` / ``bag_hi`` bracket the ten bagged
boosters (the fit's own wobble, not a confidence interval - see
``player_confidence.py``).

Usage
-----
>>> from predict import predict_fixture
>>> predict_fixture(
...     home_team="Arsenal", away_team="Chelsea",
...     kickoff="2025-09-20 17:30", referee="Michael Oliver", matchweek=5,
...     home_xi=[...11 names, goalkeeper first...],
...     away_xi=[...11 names, goalkeeper first...],
...     home_formation="4-3-3", away_formation="4-2-3-1",
...     league="premier_league",   # or serie_a, la_liga, bundesliga, ligue_1
... )
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, ensemble, features, modelling, odds_features, pooled
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import ensemble
    import features
    import odds_features
    import modelling
    import pooled

SYNTHETIC_MATCH_ID = "UPCOMING"


def season_of(kickoff) -> str:
    """'2026-2027' for any date from July 2026 to June 2027."""
    ts = pd.Timestamp(kickoff)
    start = ts.year if ts.month >= 7 else ts.year - 1
    return f"{start}-{start + 1}"


def is_pooled(model) -> bool:
    """True for a five-league model, which takes ``league`` as a feature."""
    return "league" in getattr(model, "features", [])


def load_default_model():
    """The shipped five-league ensemble.

    Falls back, loudly, to the single pooled booster and then to the Premier
    League model, so an unbuilt artifact degrades rather than crashes.
    """
    try:
        return ensemble.load()
    except FileNotFoundError as exc:
        print(f"note: {exc}; falling back to the single pooled booster.")
    try:
        return pooled.load_final_model()
    except FileNotFoundError as exc:
        print(f"note: {exc}; falling back to the Premier-League-only model.")
        return modelling.load_model()


def score_rows(model, rows: pd.DataFrame) -> pd.DataFrame:
    """Add ``card_probability`` and, for the ensemble, the bag's spread."""
    rows = rows.copy()
    if isinstance(model, ensemble.EnsembleModel):
        comp = model.components(rows)
        rows["card_probability"] = model.blend(comp)
        members = comp.filter(regex=r"^p_m\d+$").to_numpy()
        rows["bag_sd"] = members.std(axis=1)
        rows["bag_lo"] = np.percentile(members, 5, axis=1)
        rows["bag_hi"] = np.percentile(members, 95, axis=1)
    else:
        rows["card_probability"] = model.predict_proba(rows)
    return rows


def _blank_player_rows(names, team, opponent, side, formation, gk, template_cols):
    rows = []
    for i, name in enumerate(names):
        row = {c: 0.0 for c in template_cols}
        row.update({
            "player": str(name).strip(),
            "shirt_number": str(i + 1),
            "side": side,
            "formation": formation,
            "is_starter": True,
            "team": team,
            "opponent": opponent,
            "appeared": True,
            "pos_raw": None,
            "Age": None,
            "pos_group_actual": "GK" if str(name).strip() == gk else "UNK",
        })
        rows.append(row)
    return rows


def resolve_referee(referee, template: pd.DataFrame | None) -> str:
    """Match a referee name to the spelling the dataset actually stores.

    FBref separates first and last name with a NON-BREAKING space (U+00A0), so
    the obvious ``referee="Michael Oliver"`` - the spelling in this module's own
    docstring - matched nothing, and the referee features silently fell back to
    their shrinkage prior.  Nothing failed and nothing warned; the prediction
    was simply made as if the referee were unknown.

    So names are compared on whitespace-normalised, case-folded forms and
    mapped back to the stored spelling.  A genuinely unknown referee (a new
    appointment, say) still falls back to the prior - correctly, since there is
    no history to use - but now says so.
    """
    if referee is None or (isinstance(referee, float) and pd.isna(referee)):
        print("note: no referee supplied; referee features fall back to the "
              "league prior.")
        return ""

    def norm(s: str) -> str:
        return " ".join(str(s).replace("\xa0", " ").split()).casefold()

    want = norm(referee)
    if template is not None and "referee" in template.columns:
        known = template["referee"].dropna().unique()
        lookup = {norm(k): k for k in known}
        if want in lookup:
            return lookup[want]
        print(f"note: referee {referee!r} has no history in the dataset; "
              "referee features fall back to the league prior.")
    return " ".join(str(referee).replace("\xa0", " ").split())


def build_fixture_rows(home_team: str, away_team: str, kickoff, referee: str,
                       home_xi, away_xi, home_formation: str, away_formation: str,
                       matchweek: int | None = None,
                       home_gk: str | None = None, away_gk: str | None = None,
                       ages: dict | None = None,
                       template: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn a team sheet into base-table rows with every in-match statistic zeroed."""
    if len(home_xi) != 11 or len(away_xi) != 11:
        raise ValueError("each XI must contain exactly 11 names")
    kickoff = pd.to_datetime(kickoff)
    home_gk = home_gk or str(home_xi[0]).strip()
    away_gk = away_gk or str(away_xi[0]).strip()

    numeric_cols = [c for c in (template.columns if template is not None else [])
                    if template[c].dtype.kind in "fiu"]
    rows = (_blank_player_rows(home_xi, home_team, away_team, "home",
                               home_formation, home_gk, numeric_cols)
            + _blank_player_rows(away_xi, away_team, home_team, "away",
                                 away_formation, away_gk, numeric_cols))
    df = pd.DataFrame(rows)

    df["match_id"] = SYNTHETIC_MATCH_ID
    # The fixture's real season, not a placeholder.  Everything keyed on the
    # season - league position, stakes, fixture congestion, the season-to-date
    # card count - used to come back NaN or zero under "UPCOMING", so a served
    # fixture never saw the context the model was trained with (found by the
    # train/serve parity check on 2026-09-28).
    df["season"] = season_of(kickoff)
    df["matchday"] = -1
    df["matchweek"] = matchweek if matchweek is not None else np.nan
    df["kickoff"] = kickoff
    df["home_team"] = home_team
    df["away_team"] = away_team
    df["referee"] = resolve_referee(referee, template)
    df["venue"] = None
    df["attendance"] = np.nan
    df["is_home"] = (df["side"] == "home").astype(int)
    df["y_yellow"] = 0
    df["y_any_card"] = 0
    # Slot columns.  The fixture has not been played, so nobody has been
    # replaced yet; these exist only so build_model_dataset finds the columns
    # it expects.  They can never leak: the as-of joins are strictly backward,
    # and these rows are the latest in the archive.
    df["y_slot_any_card"] = 0
    df["y_slot_yellow"] = 0
    df["was_replaced"] = 0
    df["slot_repl_minutes"] = 0.0
    df["repl_card"] = 0
    df["slot_seat"] = 0.0
    df["slot_size"] = 1
    df["minutes"] = 0.0
    df["appeared"] = True

    ages = ages or {}
    df["age_years"] = df["player"].map(ages).astype(float)
    # The history keys a name shared by two different people as "name [club]"
    # (build_base_table.resolve_player_identity).  Keying the fixture on the
    # bare name gave such a starter - Marquinhos or Vitinha at Paris SG - no
    # history at all; the served row must use the same key its history does.
    split = set()
    if template is not None and "player_key" in template.columns:
        split = set(template.loc[template["player_key"] != template["player"], "player"])
    df["player_key"] = np.where(df["player"].isin(split),
                                df["player"] + " [" + df["team"] + "]", df["player"])
    return df


def fixture_team_rows(teams: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    """Two placeholder team-match rows (home, away) for the unplayed fixture.

    The league table, fixture congestion and stakes are evaluated at kickoff
    times that exist in the team table; without a row for this kickoff they came
    back NaN for every served fixture.  Every state that reads these rows is an
    as-of lookup of *strictly earlier* matches, so their zeroed statistics are
    never used - they only put the kickoff on the calendar, exactly as a played
    match's own rows do in training.
    """
    first = fixture.iloc[0]
    rows = []
    for side, team, opp in (("home", first["home_team"], first["away_team"]),
                            ("away", first["away_team"], first["home_team"])):
        row = {c: (0.0 if teams[c].dtype.kind in "fiu" else None) for c in teams.columns}
        row.update({"team": team, "opponent": opp, "side": side,
                    "match_id": SYNTHETIC_MATCH_ID, "season": first["season"],
                    "kickoff": first["kickoff"], "referee": first["referee"],
                    "matchday": -1})
        rows.append(row)
    out = pd.DataFrame(rows, columns=teams.columns)
    for c in teams.columns:
        if teams[c].dtype.kind in "fiu":
            out[c] = out[c].astype(float)
    out["kickoff"] = pd.to_datetime(out["kickoff"])
    return out


def predict_fixture(home_team: str, away_team: str, kickoff, referee: str,
                    home_xi, away_xi, home_formation: str, away_formation: str,
                    matchweek: int | None = None,
                    home_gk: str | None = None, away_gk: str | None = None,
                    ages: dict | None = None,
                    model=None, players: pd.DataFrame | None = None,
                    teams: pd.DataFrame | None = None, cfg=config,
                    odds: tuple | None = None,
                    odds_ou: tuple | None = None,
                    league: str = "premier_league") -> pd.DataFrame:
    """Return the 20 outfield starters ranked by their card probability.

    ``league`` picks the history the fixture is scored against and, for the
    five-league model, its ``league`` feature: premier_league, serie_a,
    la_liga, bundesliga or ligue_1.  Club and player names must be spelt as
    FBref spells them in that league's archive.

    ``odds`` is the fixture's decimal 1X2 price as ``(home, draw, away)``, and
    ``odds_ou`` the optional over/under 2.5 pair.  The model trains on odds
    features (``config.ODDS_PROFILE``), and an upcoming fixture is not yet in
    the football-data archive, so without a price those columns arrive NaN -
    a state the booster never saw in training and has no learned branch for.
    Pass the market price for the fixture.
    """
    if model is None:
        model = load_default_model()
    pooled_model = is_pooled(model)
    if league != pooled.PL and not pooled_model:
        raise ValueError("the Premier-League-only model cannot score "
                         f"{league}; use the five-league model")
    league_odds = None
    if pooled_model:
        hist_players, hist_teams, cfg_league, league_odds = pooled.league_history(league)
        cfg = cfg_league if cfg is config else cfg
        players = hist_players if players is None else players
        teams = hist_teams if teams is None else teams
    if players is None:
        players = pd.read_parquet(cfg.BASE_TABLE_PATH)
    if teams is None:
        teams = pd.read_parquet(cfg.MATCH_TABLE_PATH)

    kickoff_ts = pd.to_datetime(kickoff)
    if (players["kickoff"] >= kickoff_ts).any():
        n_after = int((players["kickoff"] >= kickoff_ts).sum())
        print(f"note: {n_after:,} historical rows kick off at or after {kickoff_ts}; "
              "they are ignored for this fixture.")
        players = players[players["kickoff"] < kickoff_ts]
        teams = teams[teams["kickoff"] < kickoff_ts]

    fixture = build_fixture_rows(
        home_team, away_team, kickoff_ts, referee, home_xi, away_xi,
        home_formation, away_formation, matchweek, home_gk, away_gk, ages,
        template=players,
    )
    combined = pd.concat([players, fixture], ignore_index=True)
    teams = pd.concat([teams, fixture_team_rows(teams, fixture)], ignore_index=True)

    extra_odds = None
    if odds is not None:
        season = str(fixture["season"].iloc[0])
        extra_odds = odds_features.odds_row_from_prices(
            season, home_team, away_team, *odds,
            *(odds_ou if odds_ou is not None else (None, None)))
    elif any(f.startswith("odds_") for f in cfg.FEATURES):
        print("warning: this model uses odds features but no price was supplied; "
              "they will be NaN for this fixture, which the booster never saw in "
              "training.  Pass odds=(home, draw, away).")

    if pooled_model:
        history = None
        if pooled.uses_history(model):
            import history_features
            history = history_features.load_history()
        ds = pooled.league_model_dataset(combined, teams, league, cfg, league_odds,
                                         extra_odds=extra_odds, history=history)
    else:
        ds, _ = features.build_model_dataset(combined, teams, cfg=cfg, verbose=False,
                                             extra_odds=extra_odds)
    rows = ds[ds["match_id"] == SYNTHETIC_MATCH_ID].copy()
    if rows.empty:
        raise RuntimeError("the fixture produced no candidate rows - check the XIs "
                           "(the goalkeeper is excluded, so 20 rows are expected)")

    rows = score_rows(model, rows)
    cols = ["player", "team", "is_home", "pos_group", "card_probability"]
    cols += [c for c in ("bag_sd", "bag_lo", "bag_hi") if c in rows.columns]
    cols += ["plr_cards_p90_eb", "plr_fouls_p90_eb", "plr_apps_prior",
             "exp_cards_analytic"]
    rows = rows.sort_values("card_probability", ascending=False)
    out = rows[cols].reset_index(drop=True)
    out.index += 1
    return out


def predict_historical_match(match_id: str, model=None, dataset: pd.DataFrame | None = None,
                             cfg=config) -> pd.DataFrame:
    """Score a match already in the dataset - handy for eyeballing the output.

    Note this scores a historical match with the *shipped* model, which for
    seasons inside TRAIN_SEASONS has seen that match in training.  Use
    ``oos.py`` / ``prediction_review.ipynb`` for honest historical predictions.
    """
    if model is None:
        model = load_default_model()
    if dataset is None:
        dataset = (pd.read_parquet(pooled.dataset_path(pooled.FINAL_FIRST_SEASON))
                   if is_pooled(model) else pd.read_parquet(cfg.MODEL_DATASET_PATH))
    rows = dataset[dataset["match_id"] == match_id].copy()
    if rows.empty:
        raise KeyError(f"no such match_id: {match_id}")
    rows = score_rows(model, rows).sort_values("card_probability", ascending=False)
    cols = ["player", "team", "is_home", "pos_group", "card_probability",
            cfg.TARGET_COL]
    extra = [c for c in ("y_any_card", "repl_card") if c in rows.columns]
    out = rows[cols + extra].rename(columns={
        # under the slot target a pick wins on the starter OR his replacement,
        # so "carded" would be the wrong word for the settled outcome
        cfg.TARGET_COL: "bet_won",
        "y_any_card": "starter_carded",
        "repl_card": "won_via_sub",
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    ds = pd.read_parquet(config.MODEL_DATASET_PATH)
    last = ds[ds["season"] == config.TEST_SEASONS[-1]]["match_id"].iloc[-1]
    print(f"example - scoring the historical match {last}\n")
    print(predict_historical_match(last).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
