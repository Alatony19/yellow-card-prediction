"""Central configuration for the v3 card-prediction pipeline.

Everything that a human might want to tweak lives here so the notebook and the
scripts never disagree about paths, splits or the feature list.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
V3_DIR = Path(__file__).resolve().parents[1]
# YCP_DATA_DIR / YCP_ARTIFACT_DIR relocate the data and the generated files, so the
# synthetic demo (tools/make_synthetic_data.py) can run without touching real data.
DATA_DIR = Path(os.environ.get("YCP_DATA_DIR", V3_DIR / "data"))  # git-ignored; see README
FBREF_DIR = DATA_DIR / "fbref"                   # <league>/<season>/match_day_N/<Home_Away>/
ODDS_ROOT = DATA_DIR / "odds"                    # <league>/<season>.csv (football-data.co.uk)
RAW_DATA_DIR = FBREF_DIR / "premier_league"      # the Premier League archive
ARTIFACT_DIR = Path(os.environ.get("YCP_ARTIFACT_DIR", V3_DIR / "artifacts"))
PL_DIR = ARTIFACT_DIR / "premier_league"          # the single-league build and research model
POOLED_DIR = ARTIFACT_DIR / "pooled"              # five-league datasets + the served ensemble
RESEARCH_DIR = ARTIFACT_DIR / "research"          # tables written by research/ scripts
SLATES_DIR = ARTIFACT_DIR / "slates"              # one folder per matchday: inputs and outputs

BASE_TABLE_PATH = PL_DIR / "base_player_match.parquet"
MATCH_TABLE_PATH = PL_DIR / "base_match.parquet"
MODEL_DATASET_PATH = PL_DIR / "model_dataset.parquet"
# The Premier-League-only booster (+ logistic).  Research only - serving uses
# the five-league ensemble in POOLED_DIR (pipeline/ensemble.py).
MODEL_PATH = PL_DIR / "xgb_card_model.joblib"
# Stage-1 count model + share model + level anchor, Premier League only
# (team_model.py --save).  Research only; the ensemble carries its own.
DECOMPOSITION_PATH = PL_DIR / "team_decomposition.joblib"


def slate_dir(date: str) -> Path:
    """artifacts/slates/<date>/ - match_<key>.json in, market_<key>.csv and reports out."""
    return SLATES_DIR / str(date)

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
# Player-booking markets resolve a named starter as a HIT if that player *or
# whoever replaces him* is booked, so "slot_any_card" is the label that matches
# how the market actually resolves.  It lifts the base rate 0.1545 -> 0.1714;
# about one hit in ten comes from the substitute rather than the named player.
#   "slot_any_card" -> starter or his replacement(s), yellow OR red   (v3.1)
#   "slot_yellow"   -> starter or his replacement(s), yellow only
#   "any_card"      -> the named player only, yellow OR red           (v3.0)
#   "yellow"        -> the named player only, yellow only
TARGET_MODE = "slot_any_card"
TARGET_COL = "y_card"

# --------------------------------------------------------------------------- #
# Prediction population
# --------------------------------------------------------------------------- #
# We predict once the line-ups are published (~1h before kick-off), so the
# candidate set is the confirmed starting XI of each side: 22 rows per match.
# Set INCLUDE_BENCH = True to additionally score named substitutes.
INCLUDE_BENCH = False
EXCLUDE_GOALKEEPERS = True

# --------------------------------------------------------------------------- #
# Temporal splits (strictly chronological - no random splitting anywhere)
# --------------------------------------------------------------------------- #
ALL_SEASONS = [
    "2017-2018", "2018-2019", "2019-2020", "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
    "2026-2027",
]
# The first BURN_IN_SEASONS are used only to warm up rolling histories.
BURN_IN_SEASONS = ["2017-2018"]
TRAIN_SEASONS = ["2018-2019", "2019-2020", "2020-2021", "2021-2022",
                 "2022-2023", "2023-2024", "2024-2025"]
VALID_SEASONS = ["2025-2026"]   # early stopping + probability calibration
TEST_SEASONS = ["2026-2027"]    # untouched hold-out - IN PROGRESS, see below

# 2026-2027 is the live season and only a handful of matchweeks are played, so
# it is a thin hold-out: treat its test metrics as indicative, not decisive.
# What it is really for is the rolling histories - a player's form and a team's
# recent card rate have to be current for today's fixture to be scored properly.
# The walk-forward below still ends on a complete season for that reason.

# Walk-forward evaluation: each of these seasons is scored by a model trained
# only on the seasons that precede it.
WALK_FORWARD_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025",
                        "2025-2026"]

# --------------------------------------------------------------------------- #
# Feature-engineering knobs
# --------------------------------------------------------------------------- #
PLAYER_WINDOW = 10        # rolling window (appearances) for player form
TEAM_WINDOW = 10          # rolling window (matches) for team form
EB_PRIOR_MATCHES = 8      # pseudo-matches of shrinkage for player rate stats
EB_PRIOR_MINUTES = 8 * 90 # pseudo-minutes of shrinkage for per-90 rate stats
EB_PRIOR_REF_MATCHES = 10 # pseudo-matches of shrinkage for referee stats
MIN_PRIOR_APPS = 0        # keep every row; exposure is itself a feature

# --------------------------------------------------------------------------- #
# Feature list of the final model dataset
# --------------------------------------------------------------------------- #
CATEGORICAL_FEATURES = ["pos_group", "formation"]

NUMERIC_FEATURES = [
    # --- player discipline propensity (the core of the model) -------------
    "plr_cards_p90_eb",          # career card rate, shrunk to the league prior
    "plr_cards_p90_ewm",         # same, recency-weighted (30-appearance half-life)
    "plr_fouls_p90_eb",          # career foul rate, shrunk
    "plr_fouls_p90_ewm",         # recency-weighted foul rate
    "plr_cards_pm_roll",         # cards per match over the last 10 appearances
    "plr_cards_per_foul_eb",     # how readily this player's fouls become cards
    "plr_matches_since_card",    # recency of the last booking
    "plr_cards_season_to_date",  # accumulation within the season
    # --- player exposure / context ---------------------------------------
    "plr_apps_prior",            # how much history the rates above are based on
    "plr_mins_roll",             # expected minutes (NOT the minutes actually played)
    "plr_age_years",
    # --- match context ----------------------------------------------------
    "is_home",
    "matchweek",
    # --- team and opponent form ------------------------------------------
    "team_fouls_pm_roll",
    "team_cards_pm_roll",
    "opp_fouls_drawn_pm_roll",   # how much the opponent gets fouled
    "strength_diff",             # rolling points-per-game gap
    # --- referee ----------------------------------------------------------
    "ref_cards_pm_eb",
    # --- analytic composite ----------------------------------------------
    "exp_cards_analytic",        # expected fouls x referee cards-per-foul
    # --- match context (see context.py) -----------------------------------
    "ctx_match_difficulty",      # opponent Elo win expectancy, venue included
    "ctx_fixture_cards_pm_eb",   # how ill-tempered this exact pairing runs
    "ctx_elo_diff",              # opponent-adjusted strength gap
    "ctx_position",              # live league position, this side
    "ctx_opp_position",          # live league position, the opposition
    "ctx_stakes",                # how much the table still matters, this late
    "ctx_matches_last_14d",      # fixture congestion
]

# Dropped after a 4-season walk-forward showed the leaner set matched or beat
# the wider one on every metric.  They are still computed in features.py, so
# re-enabling any of them is a one-line change here:
#   plr_fouls_p90_roll, plr_fouled_p90_roll, plr_fouls_per_tackle_eb,
#   plr_days_rest, team_days_rest, team_ppg_roll, team_poss_share_roll,
#   opp_ppg_roll, ref_cards_per_foul_eb, ref_cards_pm_recent, ref_matches_prior
#
# Also dropped, after the move to the slot target: the four features built to
# describe the replacement leg -
#   plr_subbed_off_rate_eb, plr_exp_repl_mins, bench_cards_p90_eb,
#   slot_exp_repl_cards
# A 4-season walk-forward put the blend at lift 1.3277 with them and 1.3326
# without (sd 0.04), and a two-stage decomposition -- model the starter and the
# replacement separately, then combine as pA + pB - pA*pB -- landed on 1.3304,
# also inside the noise.  The replacement leg is real (it decides about one
# hit in ten) but it is close to uniform across the twenty starters, so
# it shifts every probability by a similar amount and cannot help the ranking.
# They are still computed in features.py if you want to revisit that.
#
# Context features that were computed but did not survive the same test:
#   ctx_points, ctx_opp_points, ctx_points_gap, ctx_goal_diff, ctx_position_gap,
#   ctx_elo, ctx_opp_elo, ctx_season_progress, ctx_pts_from_top,
#   ctx_pts_above_drop, ctx_opp_pts_from_top, ctx_opp_pts_above_drop
# All 19 context features together scored lift 1.3434; the seven kept above
# scored 1.3444 with a better ECE, so the rest are redundant with them.
# context.py still computes every one of them.
#
# Dropped when the 2025-2026 season was added, because FBref stopped publishing
# the statistics behind them (see scrape_fbref.py):
#   plr_tackles_p90_roll   - needs Tackles_Tkl; only TklW survives
#   plr_duels_lost_p90_roll - needs challenges lost + aerial duels
#   poss_share_diff        - needs passes attempted
# Measured before removing: walk-forward blend lift 1.3440 with them, 1.3413
# without.  Leaving them in and letting them go NaN for the new season would be
# worse -- the booster never saw them missing in training, so its default
# direction for a missing value would be arbitrary.

# --------------------------------------------------------------------------- #
# Recommendation-4 candidates: richer player context
# --------------------------------------------------------------------------- #
# Built in features.py from the columns FBref STILL publishes after the 2025-26
# table removals (minutes, fouls, fouled, tackles_won, interceptions,
# pens_conceded, cards), plus two structural reads that need no new data at all:
# the player-referee pairing, and proximity to a suspension threshold.
#
# They are always computed and always written to the model dataset, so testing
# them is a one-line change here rather than a rebuild.
CANDIDATE_FEATURES = [
    "plr_defacts_p90_eb",        # defensive workload = chances to mistime one
    "plr_fouls_per_defact_eb",   # clumsiness, separated from volume
    "plr_aggression_ratio",      # commits fouls vs draws them
    "plr_pens_p90_eb",           # rash in his own box
    "plr_cards_p90_this_venue",  # his card rate at THIS venue specifically
    "plr_venue_card_gap",        # how much worse he is away than at home
    "plr_ref_pair_ratio",        # this referee books him more than most
    "plr_ref_pair_apps",         # how much history that ratio rests on
    "plr_fouls_trend",           # recent fouling vs his own baseline
    "plr_cards_to_ban",          # bookings until a live suspension threshold
    "plr_ban_window_left",       # matches left inside that threshold window
]

# Referee candidates.  The model currently carries exactly ONE referee feature,
# ref_cards_pm_eb, and every other referee number tried so far has measured the
# same thing: how many cards he shows.  These measure *who he shows them to*,
# which is the only referee property that can reorder a 20-player field rather
# than shifting all twenty together.
REFEREE_CANDIDATES = [
    "ref_venue_card_bias",     # does he punish the away side more than most?
    "ref_away_card_share_eb",  # the same, as a raw share
    "ref_season_level_ratio",  # this season's level vs his own career level
    "ref_pos_affinity",        # does he book THIS position group more than most?
    "ref_pos_apps",            # how much history that affinity rests on
    # previously computed and dropped individually - re-tested as a group
    "ref_cards_per_foul_eb",
    "ref_cards_pm_recent",
    "ref_matches_prior",
]

# Closing-price candidates.  Built in odds_features.py from the same
# football-data.co.uk feed the match model has always used; the card model has
# never seen a price.  Split deliberately into the three that are constant
# across a match (they can only help through interactions) and the five that
# take opposite values for the two sides (they can reorder the field).
ODDS_CANDIDATES = [
    # match constants - tightness
    "odds_p_draw",
    "odds_closeness",
    "odds_total_lambda",
    # team-oriented - opposite signs for the two sides
    "odds_team_expectancy",
    "odds_supremacy_for",
    "odds_lambda_against",
    "odds_is_underdog",
    # market minus the results-only Elo
    "odds_elo_surprise",
]

# Set to True to train on NUMERIC_FEATURES + CANDIDATE_FEATURES.
USE_CANDIDATE_FEATURES = False

# Set to True to additionally train on REFEREE_CANDIDATES.
USE_REFEREE_CANDIDATES = False

# Which half of ODDS_CANDIDATES the shipped model actually trains on.
#
# The two halves do different jobs and adding both is worse than adding either
# (all eight together cost P@3 -0.0023).  Measured over five walk-forward
# seasons with odds_verdict.py, paired season by season:
#
#   "side"       +0.0041 lift, better in 5/5 seasons, P@1 +0.0094 -- RANKING
#   "tightness"  +0.0041 lift, better in 3/5, but ECE better in 4/5 -- PRICING
#
# "side" is the default because the shipped objective is choosing names.  Switch
# to "tightness" if the objective becomes comparing a probability against a
# price, for the same reason USE_REFEREE_CANDIDATES exists.
ODDS_PROFILE = "side"          # "side" | "tightness" | "all" | "none"

ODDS_PROFILES = {
    "side": ["odds_team_expectancy", "odds_supremacy_for",
             "odds_lambda_against", "odds_is_underdog"],
    "tightness": ["odds_p_draw", "odds_closeness", "odds_total_lambda"],
    "all": list(ODDS_CANDIDATES),
    "none": [],
}

# Set to False to train the pre-odds 24-feature model.
USE_ODDS_FEATURES = True

if USE_CANDIDATE_FEATURES:
    NUMERIC_FEATURES = NUMERIC_FEATURES + CANDIDATE_FEATURES
if USE_REFEREE_CANDIDATES:
    NUMERIC_FEATURES = NUMERIC_FEATURES + REFEREE_CANDIDATES
if USE_ODDS_FEATURES:
    NUMERIC_FEATURES = NUMERIC_FEATURES + ODDS_PROFILES[ODDS_PROFILE]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Five-league history features (history_features.py) the served ensemble adds
# on top of FEATURES.  They need every league's appearances at once, so they are
# attached to the pooled dataset (pooled.build) and to a served fixture
# (pooled.league_model_dataset) rather than inside features.py, and the
# Premier-League-only research model never sees them.  Chosen by
# research/history_ablation.py (2026-10-03): walk-forward 2021-22 -> 2025-26,
# 8,905 matches, paired by match against the set without them -
#
#   arm                    top-4 hits             log-loss per row
#   geometry + cross-league +0.009 [-0.000,+0.018]  -0.00022 [-0.00035,-0.00008]
#     (bag alone)           +0.011 [+0.002,+0.019]
#   + matchup as well       +0.004 [-0.006,+0.014]  -0.00054 [-0.00070,-0.00036]
#
# The matchup group (history_features.MATCHUP) is the best *pricing* signal of
# the three but costs picks, so - like ODDS_PROFILE "tightness" - it is
# computed and left out.  Add it here if the objective becomes the price.
HISTORY_FEATURES: list[str] = [
    "pos_depth_prior",      # usual depth: 0 back line .. 1 front line (last 10 starts)
    "pos_width_prior",      # usual width: 0 central .. 1 touchline
    "plr_xl_m90",           # 90s played in the other four leagues before this match
    "plr_cards_ratio_xl",   # career card rate across all five, / league average, shrunk
    "plr_fouls_ratio_xl",   # the same for fouls
]
# True swaps the in-league career rates (plr_cards_p90_eb, plr_fouls_p90_eb) for
# their cross-league versions (history_features.XL_RATES) in the served set.
USE_XL_RATES = False

# Identifier / bookkeeping columns carried alongside the features.
ID_COLUMNS = [
    "match_id", "season", "matchweek", "kickoff", "team", "opponent",
    "player", "player_key", "is_starter", "referee",
]

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.0,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",      # calibrated probabilities are the goal
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 100

# Post-hoc probability calibration.  Measured over four walk-forward seasons,
# isotonic regression fitted on a single held-out season made things *worse*
# (mean ECE 0.0135 -> 0.0180, PR-AUC 0.2285 -> 0.2197): it imports that one
# season's base rate, and the league card rate has been drifting upwards.  The
# log-loss-trained booster is already well calibrated, so the default is off.
CALIBRATION = "none"          # "none" | "isotonic"

# Weight given to a logistic regression on the same features, averaged with the
# booster.  0.0 is the plain XGBoost.
#
# This replaces a boolean ``BLEND_WITH_LOGISTIC`` that no module ever read - the
# flag was inert, so the "is the blend better?" question had never actually been
# put to the shipped model.  ``blend_verdict.py`` puts it, paired by season and
# bootstrapped by match, and the answer is more specific than a boolean allows:
#
#   w     d_lift vs w=0        d_p@3        verdict
#   0.25  +0.0082 [+.0016,+.0146]  -0.0010  CI excludes 0, 4/5 seasons
#   0.50  +0.0091 [-.0030,+.0215]  +0.0001  CI spans 0,    3/5 seasons
#
# So 0.25 is the measured optimum and 0.5 - the value the old flag would have
# selected - cannot clear zero.  More importantly the gain is confined to
# *global* ranking (PR-AUC lift, ROC-AUC, ECE) and is absent from *within-match*
# ranking (precision@1, precision@3, both null).
#
# Those are different jobs.  Choosing three names inside one match is
# within-match, and the blend does nothing for it; comparing a probability to a
# market's price is cross-match, and there the blend is a real if small gain.
# Hence the default stays 0.0 - the shipped shortlist model and every figure in the
# README are unchanged - while ``card_market.py`` opts in via
# ``BLEND_WEIGHT_FOR_PRICING``, because value against a price is precisely the
# metric the blend improves.
BLEND_LOGISTIC_WEIGHT = 0.0

# Used by the price-comparison path only.  See above for why it differs.
#
# Superseded for serving (2026-09-28).  Every entry point now scores with the
# five-league ensemble (ensemble.py), whose logistic weight is chosen
# walk-forward together with the bag and the decomposition and stored in the
# artifact - one number for ranking and pricing alike.  These two constants
# remain only for the Premier-League-only research scripts and notebooks.
BLEND_WEIGHT_FOR_PRICING = 0.25

# Top-K per match reported as the secondary, "top-K shortlist" metric.
TOP_K_VALUES = (1, 3, 5)
