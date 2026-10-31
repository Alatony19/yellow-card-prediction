"""Verification gates for the v3 pipeline.

Run with:  python tests/test_pipeline.py          (from the v3 directory)
       or: pytest tests/test_pipeline.py -q

The important one is ``test_no_future_information``: it rebuilds the whole
feature table from a *truncated* history and asserts that every surviving row is
bit-identical to the full build.  If any feature peeked at a later match, the
truncation would change it and the test fails.  That is the defect that made the
v2 feature table unusable, so it is checked directly rather than by eye.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import config  # noqa: E402
import features  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def load():
    players = pd.read_parquet(config.BASE_TABLE_PATH)
    teams = pd.read_parquet(config.MATCH_TABLE_PATH)
    dataset = pd.read_parquet(config.MODEL_DATASET_PATH)
    return players, teams, dataset


# --------------------------------------------------------------------------- #
def test_base_table_shape(players: pd.DataFrame, teams: pd.DataFrame) -> None:
    n_matches = players["match_id"].nunique()
    check("every match has exactly 22 starters",
          (players[players["is_starter"]].groupby("match_id").size() == 22).all(),
          f"{n_matches} matches")
    check("every match has exactly 2 teams",
          (teams.groupby("match_id").size() == 2).all())
    check("no duplicated (match, player_key)",
          not players.duplicated(["match_id", "player_key"]).any(),
          "same-name players are split by club")
    check("every starter appeared",
          players.loc[players["is_starter"], "appeared"].all())
    check("minutes are never negative or above 90",
          players["minutes"].between(0, 90).all())
    # FBref stopped publishing Performance_2CrdY after 2024-25, so this can only
    # be verified where the column survives.  Checking it there, and separately
    # requiring that a season missing it is missing it *entirely*, keeps the
    # guarantee meaningful: a sporadically absent column would be a parse bug,
    # and the second check is what would catch it.
    have_2y = players.groupby("season")["second_yellow"].transform("count") > 0
    checkable = players[have_2y]
    check("CrdY==2 always coincides with a second yellow  (where published)",
          ((checkable["yellow"] == 2) == (checkable["second_yellow"] == 1)).all(),
          f"so yellow>=1 already captures second bookings; "
          f"verified on {checkable['season'].nunique()} of "
          f"{players['season'].nunique()} seasons, "
          f"{int((checkable['yellow'] == 2).sum())} second bookings")
    by_season = players.groupby("season")["second_yellow"].agg(["count", "size"])
    partial = by_season[(by_season["count"] > 0) & (by_season["count"] < by_season["size"])]
    check("the second-yellow column is either fully present or fully absent",
          partial.empty,
          "a season missing it only sometimes would mean a parsing bug"
          if partial.empty else f"partial in {list(partial.index)}")
    check("kickoff parsed for every row", players["kickoff"].notna().all())

    # A club spelled two ways is a club with two histories, and nothing else in
    # the pipeline would complain: Elo would simply restart it at the
    # promoted-side default and the league table would forget its points.  This
    # caught "Newcastle" / "Newcastle Utd" when the 2025-26 season was scraped,
    # and will catch the next rename too.
    names = sorted(set(players["team"]))
    flat = {n: "".join(ch for ch in n.lower() if ch.isalnum()) for n in names}
    # One name containing another is the signature of a scraper writing the
    # short form of a club the archive holds in long form.  Two genuinely
    # different clubs never do this -- "manchesterutd" does not contain
    # "manchestercity" -- so there are no false alarms to train yourself to
    # ignore.
    nested = [(a, b) for a in names for b in names
              if a != b and flat[a] in flat[b]]
    check("no club name is contained in another",
          not nested,
          f"{len(names)} distinct clubs" if not nested else f"{nested}")

    # Bound the club count: at most three sides are promoted each season, so a
    # systematic duplication would push this past the ceiling.
    n_seasons = players["season"].nunique()
    ceiling = 20 + 3 * (n_seasons - 1)
    check("club count is consistent with promotion and relegation",
          len(names) <= ceiling,
          f"{len(names)} clubs over {n_seasons} seasons (ceiling {ceiling})")

    per_season = players.groupby("season")["team"].nunique()
    check("every season has exactly 20 clubs",
          bool((per_season == 20).all()),
          f"{per_season.to_dict()}" if not (per_season == 20).all()
          else f"{len(per_season)} seasons")


def test_dataset_shape(dataset: pd.DataFrame) -> None:
    per_match = dataset.groupby("match_id").size()
    check("model dataset has 20 outfield starters per match",
          (per_match == 20).all(), f"min={per_match.min()}, max={per_match.max()}")
    check("no goalkeepers in the model dataset",
          (dataset["pos_group_actual"] != "GK").all())
    check("no duplicated (match, player_key) in the model dataset",
          not dataset.duplicated(["match_id", "player_key"]).any())
    check("target is binary",
          set(dataset[config.TARGET_COL].unique()) <= {0, 1})
    check("empirical-Bayes features are never missing",
          dataset[["plr_fouls_p90_eb", "plr_cards_p90_eb", "plr_cards_per_foul_eb",
                   "ref_cards_pm_eb", "exp_cards_analytic"]].notna().all().all(),
          "they fall back to the league prior")
    # The whole point of v3 is a deliberately small feature set.  The ceiling
    # moved from 30 to 35 when match context was added (see context.py); it is a
    # guard against creep, not a target to fill.
    check("feature count is small",
          len(config.FEATURES) <= 35, f"{len(config.FEATURES)} features")


def test_post_match_columns_absent(dataset: pd.DataFrame) -> None:
    """Nothing measured during the match may be a feature."""
    banned = {"minutes", "fouls", "tackles", "yellow", "red", "touches",
              "pos_group_actual", "challenges_lost", "aerials_lost"}
    leaked = banned.intersection(config.FEATURES)
    check("no in-match statistic is used as a feature", not leaked, str(leaked))
    check("actual minutes played is not a feature",
          "minutes" not in config.FEATURES,
          "expected minutes come from plr_mins_roll instead")


# --------------------------------------------------------------------------- #
def test_no_future_information(players: pd.DataFrame, teams: pd.DataFrame,
                               dataset: pd.DataFrame, cutoff: str = "2022-01-01") -> None:
    """Rebuild from a truncated history; rows before the cutoff must not move."""
    priors = features.compute_priors(
        players, teams, config.BURN_IN_SEASONS + config.TRAIN_SEASONS
    )
    cut = pd.Timestamp(cutoff)
    truncated_players = players[players["kickoff"] < cut]
    truncated_teams = teams[teams["kickoff"] < cut]

    partial, _ = features.build_model_dataset(
        truncated_players, truncated_teams, cfg=config, priors=priors, verbose=False
    )
    full = dataset[dataset["kickoff"] < cut].reset_index(drop=True)
    partial = partial.reset_index(drop=True)

    check("truncated rebuild has the same row count",
          len(partial) == len(full), f"{len(partial)} vs {len(full)}")
    if len(partial) != len(full):
        return

    key_ok = (partial["match_id"].values == full["match_id"].values).all() and \
             (partial["player"].values == full["player"].values).all()
    check("truncated rebuild has the same row order", key_ok)
    if not key_ok:
        return

    mismatched = []
    # The recommendation-4 candidates are written to the dataset even while
    # config.USE_CANDIDATE_FEATURES is False, so they have to clear the same
    # gate - otherwise enabling them later would silently enable a leak.
    numeric_under_test = list(dict.fromkeys(
        list(config.NUMERIC_FEATURES)
        + [c for c in (list(getattr(config, "CANDIDATE_FEATURES", []))
                       + list(getattr(config, "REFEREE_CANDIDATES", []))
                       + list(getattr(config, "ODDS_CANDIDATES", [])))
           if c in full.columns]))
    for col in numeric_under_test:
        a = pd.to_numeric(partial[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(full[col], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True):
            mismatched.append(f"{col} ({int((~np.isclose(a, b, equal_nan=True)).sum())} rows)")
    for col in config.CATEGORICAL_FEATURES:
        a = partial[col].astype(str).to_numpy()
        b = full[col].astype(str).to_numpy()
        if not (a == b).all():
            mismatched.append(f"{col} ({int((a != b).sum())} rows)")

    check("NO FEATURE USES FUTURE INFORMATION",
          not mismatched,
          "all identical after truncating history" if not mismatched
          else "differs: " + ", ".join(mismatched))


# --------------------------------------------------------------------------- #
def test_bruteforce_player_history(players: pd.DataFrame, dataset: pd.DataFrame,
                                   n_samples: int = 40, seed: int = 0) -> None:
    """Recompute a few feature values the slow, obvious way and compare."""
    rng = np.random.default_rng(seed)
    appeared = players[players["appeared"]].copy()
    appeared["cards"] = ((appeared["yellow"] >= 1) | (appeared["red"] >= 1)).astype(float)

    sample = dataset[dataset["plr_apps_prior"] >= 3].sample(
        n=n_samples, random_state=seed
    )
    bad_apps = bad_roll = bad_eb = 0
    for _, row in sample.iterrows():
        hist = appeared[(appeared["player_key"] == row["player_key"]) &
                        (appeared["kickoff"] < row["kickoff"])].sort_values("kickoff")
        if hist.empty:
            continue
        if int(row["plr_apps_prior"]) != len(hist):
            bad_apps += 1
        window = hist.tail(config.PLAYER_WINDOW)
        expected_roll = window["cards"].sum() / len(window)
        if not np.isclose(row["plr_cards_pm_roll"], expected_roll, atol=1e-9):
            bad_roll += 1
        exp_eb = (hist["fouls"].sum() + 0) / 1.0  # only used for the message
        k90 = config.EB_PRIOR_MINUTES / 90.0
        prior_fouls = (row["plr_fouls_p90_eb"] * (hist["minutes"].sum() / 90.0 + k90)
                       - hist["fouls"].sum()) / k90
        if not (0.4 < prior_fouls < 3.0):  # implied prior must be a sane league rate
            bad_eb += 1
    check("plr_apps_prior matches a brute-force count", bad_apps == 0, f"{bad_apps} bad")
    check("plr_cards_pm_roll matches a brute-force rolling mean", bad_roll == 0,
          f"{bad_roll} bad")
    check("plr_fouls_p90_eb inverts to a sane league prior", bad_eb == 0, f"{bad_eb} bad")


def test_debutants(dataset: pd.DataFrame) -> None:
    deb = dataset[dataset["plr_apps_prior"] == 0]
    check("debutants exist and have no rolling history",
          len(deb) > 0 and deb["plr_cards_pm_roll"].isna().all(),
          f"{len(deb)} debutant rows")
    check("debutants fall back to the league prior for fouls",
          deb["plr_fouls_p90_eb"].nunique() <= 1 if len(deb) else True)


def test_splits_are_disjoint_and_ordered(dataset: pd.DataFrame) -> None:
    tr = dataset[dataset["season"].isin(config.TRAIN_SEASONS)]
    va = dataset[dataset["season"].isin(config.VALID_SEASONS)]
    te = dataset[dataset["season"].isin(config.TEST_SEASONS)]
    check("train / valid / test are all non-empty",
          len(tr) and len(va) and len(te), f"{len(tr)}/{len(va)}/{len(te)}")
    check("train ends before validation starts",
          tr["kickoff"].max() < va["kickoff"].min())
    check("validation ends before test starts",
          va["kickoff"].max() < te["kickoff"].min())
    check("no match appears in two splits",
          not (set(tr["match_id"]) & set(va["match_id"]) or
               set(va["match_id"]) & set(te["match_id"])))


def main() -> int:
    players, teams, dataset = load()
    print(f"\nbase: {len(players):,} player rows, {len(teams):,} team rows, "
          f"dataset: {len(dataset):,} rows\n")
    print("--- base tables ---")
    test_base_table_shape(players, teams)
    print("\n--- model dataset ---")
    test_dataset_shape(dataset)
    test_post_match_columns_absent(dataset)
    test_debutants(dataset)
    test_splits_are_disjoint_and_ordered(dataset)
    print("\n--- brute-force recomputation ---")
    test_bruteforce_player_history(players, dataset)
    print("\n--- leakage gate (rebuild from truncated history) ---")
    test_no_future_information(players, teams, dataset)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
