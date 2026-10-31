"""Configuration for the match-result / goals pipeline.

This is a separate pipeline from the card model.  It shares the raw data and
the season splits, but nothing else: the unit of observation is a match rather
than a player, and the output is a full score matrix rather than a single
probability.

Why a score matrix and not a 1X2 classifier
-------------------------------------------
The markets we care about - over 1.5, over 2.5, home multigol 1-3, both teams
to score - are all different readings of the same object, the joint
distribution of (home goals, away goals).  Modelling that distribution once and
projecting it onto each market keeps every quoted probability mutually
consistent, and means a new market costs a function rather than a new model.
"""
from __future__ import annotations

import os
from pathlib import Path

MATCH_DIR = Path(__file__).resolve().parent
V3_DIR = MATCH_DIR.parent
DATA_DIR = Path(os.environ.get("YCP_DATA_DIR", V3_DIR / "data"))
RAW_DATA_DIR = DATA_DIR / "fbref" / "premier_league"
ODDS_DIR = DATA_DIR / "odds" / "premier_league"   # football-data.co.uk archives
_ART_ROOT = os.environ.get("YCP_ARTIFACT_DIR")
ARTIFACT_DIR = Path(_ART_ROOT) / "match_model" if _ART_ROOT else MATCH_DIR / "artifacts"

TEAM_MATCH_PATH = ARTIFACT_DIR / "team_match.parquet"      # 2 rows per match
MATCH_DATASET_PATH = ARTIFACT_DIR / "match_dataset.parquet"  # 1 row per match
MODEL_PATH = ARTIFACT_DIR / "goals_model.joblib"

# --------------------------------------------------------------------------- #
# Splits - identical to the card pipeline so the two stay comparable
# --------------------------------------------------------------------------- #
ALL_SEASONS = [
    "2017-2018", "2018-2019", "2019-2020", "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
    "2026-2027",
]
BURN_IN_SEASONS = ["2017-2018"]
TRAIN_SEASONS = ["2018-2019", "2019-2020", "2020-2021", "2021-2022",
                 "2022-2023", "2023-2024", "2024-2025"]
VALID_SEASONS = ["2025-2026"]
TEST_SEASONS = ["2026-2027"]    # untouched hold-out - IN PROGRESS, see below

# 2026-2027 is the live season and only a handful of matchweeks are played, so
# it is a thin hold-out: treat its test metrics as indicative, not decisive.
# What it is really for is the ratings and the rolling form - the Dixon-Coles
# strengths have to be current for today's fixture to be rated properly.  The
# walk-forward below still ends on a complete season for that reason, so every
# number quoted in the README remains a five-complete-season measurement.
WALK_FORWARD_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025",
                        "2025-2026"]

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
FORM_WINDOW = 10        # matches of rolling form
VENUE_WINDOW = 8        # matches of rolling *home-only* / *away-only* form
EB_PRIOR_MATCHES = 6    # shrinkage for thin histories (promoted sides)

# FBref stopped publishing shot-level xG and passes attempted, so from the
# 2025-2026 season on there is no xG and no possession share (see
# ../scrape_fbref.py).  Those features are therefore out of the model rather
# than present-but-missing: the booster never saw them absent during training,
# so its default direction for a NaN would be arbitrary.  The cost was measured
# on the eight complete seasons first - 1X2 RPS 0.1946 with xG, 0.1949 without,
# provided the closing line is kept.  That makes the odds feed load-bearing.
#
# Still computed by build_match_table.py for the seasons that have them:
#   xg_for_roll, xg_for_venue_roll, xg_against_roll, xg_against_venue_roll,
#   opp_xg_for_roll, opp_xg_for_venue_roll, opp_xg_against_roll,
#   opp_xg_against_venue_roll, poss_share_roll
NUMERIC_FEATURES = [
    # --- attack: how many good chances this side creates ------------------- #
    "goals_for_roll", "sot_for_roll",
    # --- defence: how many it concedes ------------------------------------- #
    "goals_against_roll",
    # --- the opponent, mirrored -------------------------------------------- #
    "opp_goals_for_roll", "opp_goals_against_roll", "opp_sot_for_roll",
    # --- context ------------------------------------------------------------ #
    "is_home", "ppg_roll", "opp_ppg_roll", "days_rest", "opp_days_rest",
    "matches_played",
]

# Odds-derived features are appended only when the closing line is available;
# see odds.py.  They are listed separately so a fixture without a price still
# scores (the model is trained with them missing for part of the history).
ODDS_FEATURES = ["mkt_lambda_for", "mkt_lambda_against", "mkt_total_lambda"]
USE_ODDS_FEATURES = True

# --------------------------------------------------------------------------- #
# Score matrix
# --------------------------------------------------------------------------- #
# Truncation point for the score matrix.  10 looks generous - no side has scored
# more than 9 in eight seasons - but the matrix is renormalised after truncation,
# so the discarded tail is redistributed over every cell.  On a lopsided fixture
# (lambda 4.2 v 0.4) that moved "home multigol 1-3" by 0.15 percentage points at
# 10, and by 0.001 at 14.  The cost of the larger grid is 225 cells instead of
# 121, which is nothing.
MAX_GOALS = 14
DIXON_COLES_RHO = None  # None -> fit it on the training seasons

# --------------------------------------------------------------------------- #
# Dixon-Coles ratings model
# --------------------------------------------------------------------------- #
DC_XI = 0.0018          # time-decay per day (half-life ~ 385 days)
DC_REFIT_EVERY = 1      # refit every N matchweeks during walk-forward

# --------------------------------------------------------------------------- #
# Gradient-boosted Poisson model
# --------------------------------------------------------------------------- #
XGB_PARAMS = {
    "objective": "count:poisson",
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 3.0,
    "eval_metric": "poisson-nloglik",
    "n_jobs": 0,
    "random_state": 7,
}
EARLY_STOPPING_ROUNDS = 80

# Weight on the gradient-boosted lambda when blending with Dixon-Coles.
# Fitted on the validation season by `modelling.choose_blend_weight`.
BLEND_WEIGHT = None

# --------------------------------------------------------------------------- #
# Markets to quote
# --------------------------------------------------------------------------- #
# Each entry is (name, kind, argument).  `score_matrix.market_probability`
# knows how to project the matrix onto every kind.
MARKETS = [
    ("home_win",        "result",      "H"),
    ("draw",            "result",      "D"),
    ("away_win",        "result",      "A"),
    ("over_0.5",        "total_over",  0.5),
    ("over_1.5",        "total_over",  1.5),
    ("over_2.5",        "total_over",  2.5),
    ("over_3.5",        "total_over",  3.5),
    ("btts",            "btts",        None),
    ("home_multigol_1_3", "team_range", ("home", 1, 3)),
    ("home_multigol_1_2", "team_range", ("home", 1, 2)),
    ("home_multigol_2_3", "team_range", ("home", 2, 3)),
    ("away_multigol_1_3", "team_range", ("away", 1, 3)),
    ("total_multigol_1_3", "total_range", (1, 3)),
    ("total_multigol_2_3", "total_range", (2, 3)),
    ("total_multigol_2_4", "total_range", (2, 4)),
    ("home_over_0.5",   "team_over",   ("home", 0.5)),
    ("home_over_1.5",   "team_over",   ("home", 1.5)),
    ("away_over_0.5",   "team_over",   ("away", 0.5)),
    ("away_over_1.5",   "team_over",   ("away", 1.5)),
]

TOP_SCORES = 6          # how many correct scores to print in predict.py
