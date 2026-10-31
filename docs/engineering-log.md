# Engineering log — card prediction for the big five leagues

> The long-form record of how the model was built, what was tried and what failed. The short
> overview is the repository README. "v2" and "v3" below are earlier private iterations of this
> project; this repository is the v3 code. Paths are relative to the repository root.

Predicts, for a given fixture, **which starting players will be booked** — that is,
which starters either pick up a card themselves *or* are replaced by someone who does.
Built for the moment the team sheets are published, roughly an hour before kick-off.

One model serves all five leagues (Premier League, Serie A, La Liga, Bundesliga, Ligue 1):
a **five-league ensemble** of bagged boosters, a logistic and a team-count decomposition
(see *The ensemble* below).

```
raw match folders -> build_base_table -> features -> pooled (x5 leagues) -> ensemble -> predict
  (data/fbref)        base tables        32 features   stacked + league       served     a fixture
```

The walkthrough is **`notebooks/card_prediction_pipeline.ipynb`** — start there.

There is a second, separate pipeline for match results and goals markets in
**[`match_model/`](match_model/README.md)** — 1X2, over/under, multigol, both teams to score.

---

## Layout

```
(repo root)/
  paths.py        puts pipeline/, serve/, research/ on sys.path (every script imports it)
  pipeline/       data -> features -> models
  serve/          scoring fixtures with the served ensemble
  research/       the experiments behind the design; nothing in serve/ depends on them
  notebooks/      the notebooks + the build_*.py scripts that generate them
  tests/          test_pipeline.py (leakage gates), test_serving.py (train/serve parity)
  match_model/    the separate match-result pipeline
  artifacts/
    pooled/         five-league datasets (from_2017/) + the served ensemble and its pool
      archive/      superseded builds (from_2018..2021, the first comparison)
    premier_league/ PL base tables + dataset, the PL-only research model and its pool
    slates/<date>/  one folder per matchday: match_<key>.json in, market_<key>.csv etc. out
    research/       tables written by research/ scripts
    understat/      Understat team-match data (tested, not used)
```

Modules import each other flat (`import config`, `import predict`); the folders are by
role only. Run any script from anywhere: `python serve/predict.py`.

---

## Quick start

Install the dependencies with `pip install -r requirements.txt` (Python 3.11). Set
`PYTHONIOENCODING=utf-8` when output contains non-ASCII player names. The raw data is not in
the repository - see the README, "Getting the data".

```bash
# the Premier League build and its gates
python pipeline/build_base_table.py      # 3,420 match folders -> base tables   (~4 min)
python pipeline/features.py              # base tables -> model dataset         (~1 min)
python tests/test_pipeline.py            # 27 integrity + leakage gates

# the served model
python pipeline/pooled.py build --first-season 2017-2018   # the other four leagues
python pipeline/ensemble.py all          # walk-forward, weights, artifact     (~25 min)
python tests/test_serving.py             # train/serve parity, all five leagues

# a matchday (inputs in artifacts/slates/<date>/)
python serve/card_market.py --match artifacts/slates/<date>/match_<key>.json --out artifacts/slates/<date>/market_<key>.csv
python serve/player_confidence.py --date <date>
python serve/slate_guidance.py --date <date>
python serve/todays_slate.py --fixtures fixtures.json --league serie_a

# notebooks: edit notebooks/build_*.py, never the .ipynb, then
python notebooks/build_pipeline.py
jupyter nbconvert --to notebook --execute --inplace notebooks/card_prediction_pipeline.ipynb
```

---

## What the pieces do

**pipeline/** — data, features, models

| file | role |
|---|---|
| `config.py` | every tunable: paths, target, splits, feature list, hyper-parameters |
| `scrape_fbref.py` | refresh a season from FBref, any of the five leagues (`--competition`) |
| `download_odds.py` | fetches football-data.co.uk archives for all five big leagues |
| `fetch_understat.py` | Understat team-match data (runs in its own virtual environment; tested, not used) |
| `build_base_table.py` | parses the raw CSVs into `base_player_match` + `base_match` parquet |
| `context.py` | league table, Elo, fixture difficulty, stakes — shared with `match_model/` |
| `features.py` | leakage-free feature store: as-of joins + empirical-Bayes shrinkage |
| `odds_features.py` | closing-price features (tightness + which side is favoured) |
| `understat_features.py` | Understat team-style columns (tested, not used) |
| `pooled.py` | the five-league build: each league through the unchanged pipeline, stacked with `league`; the compare/train arms |
| `modelling.py` | the booster, the logistic, baselines, walk-forward |
| `evaluation.py` | probability metrics, per-match top-K, plots |
| `team_model.py` | team carded-slot count × per-player share (a component of the ensemble) |
| **`ensemble.py`** | **the served model**: bag + logistic + decomposition, weights chosen walk-forward; its out-of-sample pool |

**serve/** — scoring fixtures, all with the ensemble

| file | role |
|---|---|
| `predict.py` | score a fixture in any league from two XIs **and its price** — `card_probability` plus the bag's spread (`bag_sd`, `bag_lo`, `bag_hi`) |
| `todays_slate.py` | a day's unplayed fixtures: projected XIs + a price (market or Dixon-Coles fallback) |
| `card_market.py` | the model against the market's player-card prices: edges, de-vig, thin-history flags |
| `player_confidence.py` | how much to trust each row: cohort interval, grade, bag spread |
| `slate_guidance.py` | the track record behind a slate: calibration where you act, realised-rate edges |

**research/** — the evidence behind the design (Premier-League-only unless noted)

| file | role |
|---|---|
| `oos.py` | the PL model's out-of-sample prediction pool |
| `segments.py` | where the model is right/wrong; the act-or-skip conditions rule |
| `specialists.py` | six attempts to exploit those segments — all negative |
| `candidate_ablation.py` | feature ablations: `--odds`, `--referee`, or the player-context default |
| `odds_verdict.py` | the paired test that decides whether a small, consistent gain is real |
| `blend_verdict.py` | booster vs booster + logistic, paired by season |
| `league_comparison.py` | one global model or five local ones — levels, structure, transfer (five leagues) |
| `prior_audit.py` | how much future information the league priors carry |

**notebooks/** — each generated by the `build_*.py` beside it

* `card_prediction_pipeline.ipynb` — how the model was built, every experiment, and the
  five-league ensemble that ships (§18–§19)
* `prediction_review.ipynb` — what the ensemble predicted out of sample, match by match, in
  all five leagues, and whether it was right: match browser, filtered search, scorecards
* `player_confidence.ipynb` — per-player uncertainty, and the two signals that failed

---

## The label: player **or his substitute**

The market resolves a named starter as a **win** if that player *or whoever replaces
him* is booked. So that is what the model is trained on, not "was this player carded".

| | rate |
|---|---|
| named player carded | 0.1548 |
| **player or his replacement carded** | **0.1724** |
| starter replaced at all | 0.3329 |
| hit won by the substitute alone | 0.0176 |

(9 seasons, 68,400 outfield starter-slots.)

About **one hit in ten is won by the substitute**, so the difference is not
cosmetic. That share is also growing — the replacement leg accounted for 1.2% of starts in
2021-22 and 2.5% by 2024-25, which is the five-substitution rule working through the data.

### How the substitution chains are recovered

FBref publishes no "X replaced Y" field. But it prints each team's player table in *slot*
order: a starter, then whoever replaced him, then whoever replaced that replacement.
Anchoring slot boundaries on the line-up sheet's starters — rather than on minutes adding
up to 90 — stays exact even when a sending-off leaves a slot short.

Validated on **all 6,080 team-tables of the original archive: exactly 11 slots every time,
never a slot longer than the match.** Slot chains are 1 player (47,154), 2 (19,616) or 3
(110). A naive "accumulate to 90 minutes" rule gets this right only 94.2% of the time,
failing on exactly the 354 team-matches involving a red card. The rule held unchanged on
the freshly scraped 2025-26 season, which is a fair test of it — different scraper, same
invariant.

`config.TARGET_MODE` switches between `slot_any_card` (default), `slot_yellow`, `any_card`
(the named player only) and `yellow`.

---

## Design decisions worth knowing

**Population.** The 20 outfield starters per match (68,400 rows over 9 seasons).
Goalkeepers are excluded; substitutes are available via `config.INCLUDE_BENCH`.

**The one rule.** A feature for match M may only use matches that kicked off *strictly*
before M. It is implemented once, as a backward non-exact `merge_asof` onto each entity's
post-match state, and it is verified by rebuilding the whole feature table from a truncated
history and asserting nothing moves.

**Match context** (`context.py`) supplies the things a human reads off the table before
looking at any player: live league position, Elo, how hard the fixture is, what is at stake,
and how ill-tempered this particular pairing runs. Difficulty is defined concretely as the
opponent's Elo win expectancy including venue — 0 means a certain win, 0.5 a coin toss.
It behaves: card rate rises monotonically from 11.8% in the easiest decile to 19.7% in the
eighth. Fixture history is a data-driven stand-in for "is this a derby" — 15.5% in the
calmest decile to 20.3% in the most fractious — without anyone curating a list of rivalries.

**No `scale_pos_weight`, no SMOTE.** The class is 17% positive, which is not severe.
Re-weighting inflates every predicted probability and does not change ranking, so it buys
nothing and costs the calibration that was the stated objective.

**No post-hoc calibration by default.** Isotonic regression fitted on the validation season
was measured across four walk-forward seasons and made calibration *worse* (ECE 0.0135 ->
0.0180): it imports one season's base rate, and the league card rate is drifting upwards.
Available behind `config.CALIBRATION`.

---

## Results

**Hold-out season 2025-26** — freshly scraped, never seen during development
(trained on 2018-19 → 2023-24, validated on 2024-25):

| model | PR-AUC | lift | ECE | precision@3 | hit rate@3 |
|---|---|---|---|---|---|
| base rate | 0.1809 | 1.00 | 0.0104 | 0.1737 | 0.4368 |
| player career rate | 0.2303 | 1.27 | 0.0474 | 0.2561 | 0.5947 |
| XGBoost | 0.2394 | 1.32 | 0.0156 | 0.2711 | 0.6342 |
| logistic | 0.2438 | 1.35 | 0.0228 | 0.2728 | 0.6158 |
| **blend** | **0.2439** | **1.35** | 0.0187 | 0.2702 | 0.6184 |

**Five-season walk-forward** (refit per season, blend):

| season | base rate | PR-AUC | lift | ECE | precision@3 | hit rate@3 |
|---|---|---|---|---|---|---|
| 2021-22 | 0.1688 | 0.2178 | 1.290 | 0.0196 | 0.2263 | 0.5316 |
| 2022-23 | 0.1749 | 0.2367 | 1.354 | 0.0164 | 0.2482 | 0.5632 |
| 2023-24 | 0.2047 | 0.2699 | 1.318 | 0.0295 | 0.3061 | 0.6737 |
| 2024-25 | 0.1954 | 0.2738 | 1.401 | 0.0098 | 0.2991 | 0.6500 |
| 2025-26 | 0.1809 | 0.2439 | 1.348 | 0.0187 | 0.2702 | 0.6184 |
| **mean** | | | **1.3423 ± 0.0415** | 0.0188 | 0.2700 | **0.6074** |

The eight-season version of this table (before 2025-26 was scraped, and with three more
features) read lift 1.3444. Adding a season and losing those features moved it to 1.3423 —
which is what the ablation predicted before the scrape was run.

"Hit rate@3" is the share of matches where at least one of the model's top three picks
wins. At 0.61 it is the number most worth knowing if you are actually using the picks.

XGBoost and a plain logistic regression on the same features are statistically
indistinguishable, which says most of the value is in the feature construction rather than
the learner.

---

## Negative results worth keeping

**Referee information — nothing, and identity actively hurts.** The model carries
exactly one referee feature (`ref_cards_pm_eb`). Four further groups were tested:
venue bias (does he book the away side more than the 52.6% league share?), season-to-date
level against his own career level (to catch mid-season directive changes), position
affinity (does he book *this* position group more than most?), and the three referee
rates dropped earlier. Plus referee **identity as a categorical**, which had never been
tried — only rates.

| | Δ lift | Δ P@3 | Δ ECE |
|---|---|---|---|
| venue bias | −0.0003 | −0.0014 | −0.0009 |
| season-to-date level | −0.0021 | −0.0004 | −0.0019 |
| position affinity | +0.0014 | −0.0028 | −0.0010 |
| previously dropped rates | +0.0002 | −0.0030 | +0.0014 |
| all four together | −0.0006 | −0.0039 | **−0.0025** |
| **identity (categorical)** | **−0.0099** | −0.0023 | −0.0003 |

Against a baseline lift of 1.3423 with sd 0.0415, every number except identity is noise.
Identity is a real loss, and its season-to-season spread jumps (sd 0.0415 → 0.0565): ~35
officials × a few hundred rows each is enough to memorise and not enough to generalise.

**The structural reason, which is the part worth carrying forward.** A referee is a
*match-level constant* — all twenty players in a fixture share him — so any pure referee
feature adds the same amount to all twenty rows and **cannot change the within-match
ranking**, which is what a top-K shortlist consumes. It can only move the level, and
`ref_cards_pm_eb` already captures the level. The only referee features that *could*
reorder a field are referee × player interactions, and both that were tried are negative:
position affinity (above) and the player-referee pairing (rec 4, −0.0024 lift).

Consistent with that, the one thing the referee group does do is marginally improve
calibration (ECE 0.0188 → 0.0163, the best in the table) while costing ranking. If the
use case ever becomes pricing rather than shortlists, they are worth revisiting.
`config.USE_REFEREE_CANDIDATES = True`; `candidate_ablation.py --referee` reproduces it.

**Richer player context (recommendation 4) — nothing.** Eleven features built from the
columns FBref still publishes: defensive workload, fouls per defensive action,
aggressor-vs-victim ratio, penalties conceded, venue-split card rates, the
player-referee pairing, foul-form trend, and proximity to a suspension threshold
(5 yellows by MW19 / 10 by MW32 / 15 by season end). Best group **+0.0003 lift**, all
eleven together **−0.0018**, against a 1.3423 baseline. The three career-rate features
already in the model are a sufficient statistic for a player's disciplinary character as
far as this data can measure it. Computed anyway and covered by the leakage gate —
`config.USE_CANDIDATE_FEATURES = True` re-enables them. See `candidate_ablation.py`.

**Team-count decomposition (recommendation 5) — a conditional win, not a clean one.**
`lambda_team x share_i`, with the count model's level anchored on the validation season.
Same lift (1.3351 vs 1.3348), **P@3 +0.56pp and hit@3 +0.52pp, better in four of five
seasons**, season-to-season lift variance down a quarter (sd 0.0464 → 0.0342) — but ECE
0.0199 → 0.0239. Better for picking three names, worse for comparing against a price.
See `team_model.py`.


Things that were built, measured, and rejected. All are still computed — re-enabling any of
them is a one-line change in `config.NUMERIC_FEATURES`.

**Referee features.** Referees genuinely differ — career card rates run 2.4 to 4.7 per match
— but a referee's history barely predicts *individual* bookings in a later season
(standalone AUC 0.502). Style regresses to the mean and directives change. Two of three
referee features dropped.

**Replacement-leg features.** Four features were built specifically to model the substitute
half of the slot: the starter's historical rate of being taken off, the minutes he
typically hands over, the card propensity of the *named bench* (known at line-up time), and
an analytic combination. Walk-forward blend lift: **1.3277 with them, 1.3326 without**
(sd 0.04). They do not help. The replacement leg is real but nearly uniform across the
twenty starters, so it shifts every probability by a similar amount and cannot improve the
ranking — and the ranking is what top-K selection depends on.

**Two-stage decomposition.** Since the events are near-independent, modelling each leg on
its own clean signal and combining as `P = pA + pB − pA·pB` should in principle beat asking
one model to learn the OR. It does not: 1.3304 versus 1.3326 direct. Also inside the noise.

**Twelve of nineteen context features.** All nineteen together scored lift 1.3434; the
seven kept scored 1.3444 with a better ECE. The rest are redundant with them.

**Three features FBref stopped publishing** (`plr_tackles_p90_roll`,
`plr_duels_lost_p90_roll`, `poss_share_diff`) were dropped when the 2025-26 season was
added rather than left to go missing. Measured cost on the eight complete seasons: lift
1.3440 → 1.3413. Keeping them and letting them go NaN for one season would be worse —
the booster never saw them absent during training, so its learned direction for a missing
value would be arbitrary.

---

## Closing prices — the first feature group that replicated

The card model had never seen a price. `odds_features.py` adds eight columns built
from the same de-vigged football-data.co.uk closing line the match model has always
used, and the join is complete: **100% of 68,400 rows priced** across all nine seasons.

The columns are split on the one distinction the referee experiment established —
whether a feature can reorder a field at all:

* **tightness** (`odds_p_draw`, `odds_closeness`, `odds_total_lambda`) is constant
  across all twenty players in a match. It can only act through interactions.
* **side** (`odds_team_expectancy`, `odds_supremacy_for`, `odds_lambda_against`,
  `odds_is_underdog`) takes opposite values for the two teams, so it genuinely
  reorders eleven players against the other eleven.
* `odds_elo_surprise` is the market's view minus `ctx_match_difficulty` — by
  construction, the part of the strength picture the results-only Elo cannot see.

Unconditionally the signal is large: the card rate runs 0.132 in the most lopsided
closeness decile and 0.189 in the tightest, and an underdog's players are booked at
0.187 against a favourite's 0.158.

**Walk-forward (5 seasons, blend):**

| | Δ lift | Δ P@3 | Δ ECE | lift sd |
|---|---|---|---|---|
| baseline (24 features) | — | — | — | 0.0415 |
| + odds: tightness | +0.0041 | +0.0007 | −0.0006 | 0.0343 |
| + odds: side | +0.0041 | +0.0005 | −0.0007 | 0.0436 |
| + odds: market vs elo | −0.0015 | +0.0007 | −0.0005 | 0.0413 |
| + all eight | +0.0020 | **−0.0023** | −0.0011 | 0.0371 |

Against a season-to-season spread of 0.042 those look like four more null results,
which is how every earlier candidate group was judged. That comparison is wrong, and
`odds_verdict.py` exists to say why: season spread is dominated by how hard each
season was — a year with more cards moves every variant together — so judging a 0.004
difference against it throws away the fact that both variants were scored on *exactly
the same matches*. The paired difference removes the season effect.

**Paired by season** (`python research/odds_verdict.py --variant side`):

| variant | mean Δ lift | seasons better (lift) | mean Δ ECE | seasons better (ECE) |
|---|---|---|---|---|
| + odds: side | +0.0041 | **5 / 5** | −0.0007 | 2 / 5 |
| + odds: tightness | +0.0041 | 3 / 5 | −0.0006 | **4 / 5** |

**`+ odds: side` improved every single walk-forward season.** Under a null of no effect
that is a 1-in-32 outcome, and it is the first time any candidate group in this project
has replicated across all five. A match-level bootstrap (resampling *matches*, since
the twenty rows inside one are not independent) puts Δ P@1 at +0.0094 with P(>0) = 0.97.

The split is exactly the one the mechanism predicts, and the two halves do different
jobs: **side helps ranking** (5/5 on lift, +0.009 on P@1, nothing on calibration),
**tightness helps calibration** (4/5 on ECE, bootstrap CI on lift excluding zero).
Enabling all eight together is worse than either half — it costs P@3 — so they should
not be shipped as one block.

The effect is small in absolute terms: +0.004 lift is about a third of what the
decomposition bought — but it is the only feature group here that has ever replicated
across every season, so it ships. `config.USE_ODDS_FEATURES = True` with
`config.ODDS_PROFILE = "side"`; the shipped model is now **32 features**. Set
`ODDS_PROFILE = "tightness"` instead if the objective ever becomes comparing a
probability against a price rather than choosing names, for the same reason
`USE_REFEREE_CANDIDATES` exists.

One honest caveat: this is the **closing** line, which incorporates information up to
kick-off, while the model predicts when line-ups are published about an hour earlier.
`match_model/` made the same trade deliberately. The look-ahead is small but it is not
zero, and a live deployment should re-measure with the line as it stood at line-up time.

### The price is now an input to `predict.py`

Shipping odds features changes the calling contract, and getting this wrong would have
been a silent failure. An upcoming fixture is **not** in the football-data archive — the
archive is published after the fact — so without intervention every odds column would
arrive `NaN` for exactly the fixtures you actually want to predict, and the booster has no
learned branch for a column it never saw missing in training. That is the same trap
`config.py` already documents for the FBref columns removed in 2025-26.

So `predict_fixture` takes the price you are looking at:

```python
predict.predict_fixture(
    home_team="Arsenal", away_team="Chelsea", ...,
    odds=(1.85, 3.60, 4.20),        # decimal home / draw / away
    odds_ou=(1.72, 2.10),           # optional over / under 2.5
)
```

`odds_ou` is optional: without it the totals line is assumed league-average (1.55 / 1.30
expected goals) so the lambda inversion still has three targets, which is far better than
a `NaN`. Omitting `odds` entirely still works and prints a warning.

This is not cosmetic — on a test fixture, flipping the market from a home favourite to an
away favourite moved **18 of 20 players' ranks**, shifting the newly-favoured side down
about 0.029 and the underdog up about 0.009. That is the feature doing precisely the job
it was added for.

---

## The ensemble — the one model every entry point serves (2026-09-28)

`predict.py`, `todays_slate.py`, `card_market.py` and `player_confidence.py` all score with
one artifact, `artifacts/pooled/ensemble_card_model.joblib`, built by `ensemble.py` on the
five-league dataset described in the next section. It folds in every component that carries
signal:

| component | what it is | weight |
|---|---|---|
| bag | 10 five-league boosters, differing only in seed and row/column subsampling; member 0 is the previously shipped pooled booster | 0.4 |
| logistic | regularised logistic regression on the same features (the linear floor) | 0.4 |
| decomposition | team carded-slot count x player share (`team_model.py`), now fitted on all five leagues with `league` in the count model and a level anchor per league | 0.2 |

`card_probability = 0.4 * mean(bag) + 0.4 * logistic + 0.2 * decomposition`

**How the weights were chosen.** `ensemble.py evaluate` walks forward over every league,
2021-22 → 2026-27 (9,155 matches, 183,100 rows). Each season is scored by components fitted only
on earlier seasons and early-stopped on the season before it. `select` grid-searches the
weights in steps of 0.1 on 2021-22 → 2025-26. The primary score is carded-out-of-four-picks,
and any weighting within 0.002 of the best counts as tied, with the lowest log-loss among
the tied set winning. 2026-27 is only a check.

| 2021-26, all leagues (8,905 matches) | top-4 hits | vs member 0, paired | log-loss | ROC-AUC | ECE |
|---|---|---|---|---|---|
| previously shipped pooled booster (member 0) | 1.0508 | — | 0.4855 | 0.5976 | 0.0045 |
| bag only | 1.0587 | +0.008 [+0.001, +0.014] | 0.4854 | 0.5979 | 0.0039 |
| logistic only | 1.0522 | +0.002 [−0.009, +0.012] | 0.4862 | 0.5950 | 0.0067 |
| decomposition only | 1.0478 | −0.003 [−0.012, +0.006] | 0.4870 | 0.5965 | 0.0202 |
| **ensemble (shipped)** | **1.0576** | +0.007 [−0.001, +0.016] | **0.4852** | **0.5989** | 0.0050 |

On the Premier League alone (1,900 matches) the ensemble scores 1.062 against 1.056 for
member 0, and it has the best log-loss and AUC there too. On 2026-27 (250 matches) it scores
0.904 against 0.896.

Read the gains honestly: they are small. Bagging is the one piece whose top-4 gain clears
zero. The logistic and the decomposition earn their places on log-loss and AUC, which is
global ranking, the job of pricing against a market. On their own neither beats the
booster. The blend picks as well as the bag alone (1.0576 against 1.0587, inside the tie
band) and prices better than any single component. That is why one number now serves
both jobs, and the old split is retired: a blend for prices, the plain booster for shortlists.

**What changed for the entry points**

* `predict_fixture` returns `card_probability` (the ensemble) plus `bag_sd`, `bag_lo` and
  `bag_hi`, the spread of the ten boosters. That spread shows how much the fit wobbles; it is
  not an interval on the probability, which can sit outside it because the logistic and the
  decomposition move it. `slip_probability`, `rank_by` and `use_decomposition` are gone,
  because the decomposition is inside the ensemble.
* `card_market.py` scores with the ensemble in any league (a `"league"` key in the match
  JSON). It no longer loads `xgb_card_model.joblib` or `BLEND_WEIGHT_FOR_PRICING`.
* `player_confidence.py` no longer fits ten boosters at run time. It reads the bag's spread
  from the artifact, and it takes cohort intervals and `--validate` from the ensemble's
  walk-forward pool (`artifacts/pooled/ensemble_oos.parquet`), which `slate_guidance.py` now
  reads too. On that pool, spread still does not predict error: ROC-AUC across quintiles of
  relative spread runs 0.581 → 0.613, rising. So it stays out of the grade.
* **A third serve-time bug fixed.** A starter whose name the history splits by club (two
  people called Marquinhos, or Vitinha) was served under the bare name, so he got no history
  at all. `build_fixture_rows` now applies the same "name [club]" key the history uses.
  `tests/test_serving.py` replays the last played 2026-27 match of every league as if it were
  upcoming, and served probabilities now equal the training rows exactly in all five.

`modelling.py`, `oos.py`, `team_model.py` and the Premier-League-only artifacts
(`xgb_card_model.joblib`, `team_decomposition.joblib`, `oos_predictions.parquet`) stay as
research tools and as the evidence behind the notebooks. Nothing on the serving path reads
them.

---

## The five-league model (the data behind the ensemble, 2026-09-28)

`predict.py` and `todays_slate.py` now score with **one model for all five leagues**:
`artifacts/pooled/pooled_2017_card_model.joblib`, built by `pooled.py` from the FBref scrape
of Serie A, La Liga, the Bundesliga and Ligue 1 (2017-18 → 2026-27) plus the Premier League
archive, about 16,350 matches. Each league runs through the unchanged single-league pipeline
(its own histories, table, Elo, priors and closing prices); only the finished datasets are
stacked, with `league` as a categorical feature.

**The score is the shortlist:** the model's top four starters per match, counted by how many were
carded (player or substitute). Walk-forward, each season scored by a model fitted only on the
seasons before it, on the same 7,712 matches for every arm:

| model | PL 2021-26 | other leagues 2022-26 | 2026-27 | all |
|---|---|---|---|---|
| Premier League only (the previous shipped model) | 1.034 | 1.010 | 0.868 | 1.012 |
| pooled, new leagues from 2021 | 1.038 | 1.024 | 0.900 | 1.023 |
| pooled from 2019 | 1.048 | 1.047 | 0.900 | 1.043 |
| pooled from 2018 | 1.037 | 1.051 | 0.864 | 1.042 |
| **pooled from 2017 (shipped)** | **1.056** | 1.038 | 0.896 | 1.038 |
| pooled from 2017 + Understat | 1.049 | 1.042 | 0.896 | 1.039 |

Shipped minus Premier-League-only, paired by match: **+0.027 of 4 [+0.012, +0.041]**, better in
18 of 26 league-seasons; on the Premier League alone +0.022 [−0.003, +0.045], 4 of 5 seasons.
History length was the lever: the gain grows from the 2021 to the 2019 build and then
plateaus - 2019, 2018 and 2017 are inside each other's noise; 2017 ships as the full archive
and the best of the three on the Premier League.

**Understat** (`fetch_understat.py`, `understat_features.py`: xG, PPDA, deep completions,
16,361 matches) never beat the same model without it and is not in the shipped feature set.

**Serving.** `predict_fixture(..., league="serie_a")` scores any of the five; the Premier
League is the default. Fixture JSON for `todays_slate.py` takes a `"league"` key, or pass
`--league`. The model that serves is the ensemble above, whose bag contains this booster.

**Two serve-time bugs fixed on the way**, found by scoring a played 2026-27 match as if it
were upcoming and diffing it against its training row: the fixture was stamped with the
season `"UPCOMING"`, and it had no team-table rows. Together they left league position,
stakes, fixture congestion and the season-to-date card count NaN or zero for *every* served
fixture, the old Premier League model included. Served and training features now agree
exactly (0.00000 max difference, Premier League and Serie A).

---

## One global model, or one per league?

`league_comparison.py` answers this from data already on disk. `download_odds.py` now
pulls football-data.co.uk for all five big leagues — **30,719 matches over 17 seasons**,
every one with a closing line *and* the per-side card and foul counts. That is not
player-level so it cannot train the card model, but it is exactly right for the
architecture question.

**1. Levels differ a lot.** Yellows per team per match, 2017-18 → 2025-26:

| league | yel/team | fouls/team | cards per foul | away − home |
|---|---|---|---|---|
| Premier League | 1.75 | 10.66 | 0.164 | +0.20 |
| Bundesliga | 1.88 | 11.63 | 0.162 | +0.24 |
| Ligue 1 | 1.88 | 12.52 | 0.150 | +0.25 |
| Serie A | 2.18 | 12.90 | 0.169 | +0.30 |
| La Liga | 2.42 | 13.16 | 0.184 | +0.18 |

The top league books 1.39× the bottom one. Any pooled model needs its own level per
league.

**2. Structure is shared.** A Poisson fit of team yellows on the market's read of the
fixture gives the same sign in all five leagues and a similar magnitude for the
dominant term:

| league | intercept | closeness | supremacy | is_home |
|---|---|---|---|---|
| Premier League | +0.29 | +0.42 | −0.27 | −0.06 |
| Serie A | +0.50 | +0.46 | −0.26 | −0.08 |
| La Liga | +0.61 | +0.39 | −0.19 | −0.02 |
| Bundesliga | +0.33 | +0.48 | −0.21 | −0.08 |
| Ligue 1 | +0.45 | +0.34 | −0.04 | −0.13 |

Closeness varies by 33% of its mean across five leagues; the intercept varies by 74%.
Tight matches are dirtier everywhere, favourites are booked less everywhere, home sides
are booked less everywhere. **The shape travels; the level does not.**

**3. Levels drift, and the anchor has to be recent.** This decided the experiment.

| season | Bundesliga | La Liga | Ligue 1 | Premier League | Serie A |
|---|---|---|---|---|---|
| 2017-18 | 1.69 | 2.51 | 1.92 | 1.52 | 2.04 |
| 2019-20 | 2.00 | 2.57 | 1.90 | 1.68 | **2.55** |
| 2022-23 | 2.07 | 2.51 | 1.69 | 1.79 | 2.25 |
| 2025-26 | 1.90 | 2.21 | 1.87 | 1.87 | **1.85** |

Serie A fell 27% in six years while the Premier League rose 23%; the five leagues are
converging. Anchoring a transferred model on a league's *career* level aims at a level
it has left behind and makes the prediction **worse**. Anchoring on its most recent
season fixes it. `team_model.py` already anchors on the validation season for the same
reason.

**4. Transfer works.** Held-out-league deviance against a model fitted on that league
alone (negative = beats the league's own model):

| arm | mean Δ deviance vs local | leagues better |
|---|---|---|
| foreign four, foreign level | −0.0075 | 2 / 5 |
| foreign four, **recent local level** | −0.0125 | 3 / 5 |
| pooled five, pooled level | −0.0007 | 2 / 5 |
| **pooled five, recent local level** | **−0.0128** | **3 / 5** |

**The answer is one global model with a per-league level anchor, not five local ones.**
Pooling alone ties a local fit; pooling *plus* re-levelling beats it. The level anchor
does nearly all the work — which is the same decomposition `team_model.py` already
implements, one level up: a league rate × a within-league shape, instead of a team rate
× a within-match share.

Caveat: this is measured on team-level *counts* with three features, not on the
player-level card model. It says the architecture is right; it does not prove the
player-level gain. What makes it worth pursuing is that the pipeline's weakest
dimension is discrimination (ROC-AUC 0.60, flat across every segment tested), and
pooling multiplies the rows available to estimate the shared shape by about five.

**What the leagues would still need separately:** football-data publishes referee names
for England only, so `ref_cards_pm_eb` — the model's single referee feature — can only
be built for other leagues from FBref. That is one of the reasons the scrape is needed
rather than optional.

---

## The logistic blend: a real gain, in the wrong metric

`BLEND_WITH_LOGISTIC` sat in `config.py` for three sessions as an open question. It was
also **read by nothing** — `grep` found only its own definition, so flipping it would have
changed no behaviour. The question had never actually been asked of the shipped model.

`blend_verdict.py` asks it, paired by season and bootstrapped by match (the `odds_verdict.py`
protocol, and for the same reason: both configurations score identical fixtures, so the
comparison must be paired). It fits the booster and the logistic once per walk-forward
season, then sweeps the weight, so every weight is compared on the same folds.

| w on logistic | lift | ROC-AUC | ECE | P@1 | P@3 |
|---|---|---|---|---|---|
| 0.00 (shipped) | 1.3284 | 0.5998 | 0.0084 | 0.2937 | 0.2700 |
| **0.25** | 1.3365 | 0.6017 | 0.0089 | 0.2953 | 0.2689 |
| 0.50 | 1.3374 | 0.6020 | 0.0095 | 0.2900 | 0.2700 |
| 0.75 | 1.3334 | 0.6009 | 0.0124 | 0.2837 | 0.2682 |
| 1.00 | 1.3258 | 0.5987 | 0.0154 | 0.2847 | 0.2672 |

Bootstrapping by match, 1,500 resamples:

| | d_lift vs w=0 | 95% CI | seasons |
|---|---|---|---|
| **w=0.25** | **+0.0082** | **[+0.0016, +0.0146]** | 4/5 on lift, 4/5 on ECE |
| w=0.50 | +0.0091 | [−0.0030, +0.0215] | 3/5 on lift |

Two things the old single pooled number hid. **0.5 was never the peak** — it has the higher
pooled lift but cannot clear zero once matches are resampled, because it is carried by one
season (2021-22, +0.0194) against 2023-24 at −0.0147. And w=0.25, which nobody had tested,
has both a higher bootstrap mean and a much tighter interval: less logistic weight injects
less variance.

The decisive finding is *which* metric moves. The gain is confined to **global** ranking —
PR-AUC lift, ROC-AUC, ECE — and is **absent from within-match** ranking: d_p@3 is −0.0010,
P(>0) = 0.31. Those are different jobs. Choosing three names inside one match is
within-match, and the blend does nothing for it. Comparing a probability against a
market's price is cross-match, and there the blend is a real if small gain.

So the default stays off and every figure above it in this README is unchanged:

```python
BLEND_LOGISTIC_WEIGHT   = 0.0    # the shortlist model: plain booster
BLEND_WEIGHT_FOR_PRICING = 0.25  # card_market.py opts in
```

The boolean is gone in favour of a weight, and `save_model` now stores the logistic
alongside the booster, so `modelling.load_ranker()` **raises** if a weight is requested that
the artifact cannot honour. An inert setting is now impossible rather than merely unlikely —
which is how the old flag stayed an open question for three sessions.

---

## Comparing with a price rather than ranking a field

`predict.py` answers "who is most likely to be booked?". A price comparison asks a different question:
"where does the model disagree with the price?" The model's favourite is very often the
market's favourite too, at a price that already contains the whole edge.

`card_market.py` computes the second ordering. On Man City v Sunderland (2026-09-20) the two
orderings barely overlap — the model's top pick, Xhaka, is third on value, while its 8th and
9th picks are first and second:

| | model p | price | edge |
|---|---|---|---|
| Antoine Semenyo | 0.215 | 7.50 | **+0.61** |
| Elliot Anderson | 0.222 | 6.00 | **+0.33** |
| Granit Xhaka | 0.281 | 4.50 | +0.26 |
| Erling Haaland | 0.094 | 7.50 | −0.30 |

The 1X2 price is a model **input** (`ODDS_PROFILE`); the player card price is the
**comparison**, and is never fed in, since that would be circular.

**De-vigging without the other side of the book.** A player card market is two-way, and the
honest de-vig divides by the pair's sum — done per market when `card_odds_no` is supplied.
With only the "yes" side each market's margin is unknown, so the field is scaled by one
constant onto an independent estimate of expected bookings: `--total-cards` (best), then
`--margin`, then the model's own expectation (default). The default is deliberately circular
about *level* and honest about *shape*: model and market then agree by construction on how
many cards the match produces, so the edge column is only about which players collect them.
The output names the anchor it used; do not read `edge` as a real edge under the `model`
anchor.

**Referee names had to be resolved, not trusted.** FBref separates a referee's first and
last name with a **non-breaking space** (U+00A0). `build_fixture_rows` did
`str(referee).strip()`, so every hand-typed `"Michael Oliver"` — including the one in
`predict.py`'s own docstring — matched nothing, and the referee features silently fell
back to their shrinkage prior. Nothing failed and nothing warned. `predict.resolve_referee`
now compares whitespace-normalised, case-folded forms and maps back to the stored
spelling; a genuinely unknown referee still falls back to the prior, correctly, but says
so. The effect on a single fixture is small — re-scoring Man City v Sunderland with
Robert Jones matched moved the top probability from 0.2150 to 0.2166 — but it was silent,
which is worse than large.

**Thin history is flagged, not hidden.** Rows with ≤5 prior appearances or an unknown
position carry `thin=True`. `segments.py` measures that slice at ROC-AUC 0.536 against 0.600
overall, with a base rate of 21.0% against a mean prediction of 17.5%. Both halves matter and
point opposite ways: the model under-calls newcomers, so the level is if anything low, but
the ordering *within* the slice is near random.

---

## Scoring fixtures that have not kicked off

`todays_slate.py` runs the model over a day's unplayed fixtures. Three of `predict.py`'s
inputs do not exist before kick-off, and the module is explicit about which of them it is
guessing:

| input | where it comes from pre-match | how bad is the guess |
|---|---|---|
| **XIs** | each team's **most recent starting XI** — FBref publishes line-ups only in the match report, which appears afterwards | the weakest link, and the only one that materially changes who tops the list. Usually 8–10 of the 11. Override once the real sheets are out. |
| **referee** | not announced on FBref pre-match | not a hole: `ref_cards_pm_eb` is an empirical-Bayes rate, so an unknown official falls back to the league prior instead of going missing |
| **price** | supply the real one with `--odds`; otherwise a **Dixon–Coles fair-odds fallback** | the fallback is a *model's* opinion, not the market's. It keeps the odds columns in a sane range instead of NaN, but the reason the odds features work is that the market prices injuries, rotation and motivation the model cannot see — so the fallback keeps the machinery running without delivering the edge. |

```bash
python serve/todays_slate.py --fixtures artifacts/slates/2026-09-20/fixtures.json --odds odds.json
```

`odds.json` maps `"Home v Away"` to `{"h", "d", "a"}` decimal prices, optionally
`"over25"` / `"under25"`. Every line of output says whether that fixture was priced from
`market` or `dixon-coles`, and the CSV written by `--out` carries the same flag per row,
so a historical comparison can never silently mix the two.

---

## A known limitation the leakage gate cannot see

`features.compute_priors` is called **once**, on `BURN_IN_SEASONS + TRAIN_SEASONS`, and
the resulting empirical-Bayes shrinkage targets are baked into every row of the dataset
— including rows from seasons the walk-forward later scores as out-of-sample. Scoring
2021-22 therefore uses league priors that were fitted partly on 2023-24 and 2024-25.

`tests/test_pipeline.py` cannot catch this, and not by accident: it computes the priors
on the full history and passes the *same object* into the truncated rebuild, so priors
are held fixed by construction. The gate tests the as-of joins, which is what it was
built for. It does not test the priors.

**Measured size.** Priors fitted on 2017-18 → 2024-25 (shipped) against 2017-18 →
2020-21 (everything knowable before the first walk-forward season):

| prior | shipped | honest | relative |
|---|---|---|---|
| `cards_p90` | 0.1683 | 0.1521 | **+10.7%** |
| `ref_cards_per_match` | 3.6964 | 3.3395 | **+10.7%** |
| `cards_per_foul` | 0.1735 | 0.1583 | +9.6% |
| `cards_per_match` | 0.1296 | 0.1210 | +7.1% |
| `fouls_p90` | 0.9700 | 0.9608 | +1.0% |
| `away_card_share` | 0.5267 | 0.5219 | +0.9% |

The card priors move ~10% and the foul priors barely move, which is the league card-rate
drift showing up exactly where you would expect it — the Premier League went from 1.52
yellows per team in 2017-18 to 2.09 in 2023-24.

**What it does and does not affect.** The prior is a shrinkage *target*, carried with
weight `k / (n + k)` where `k = EB_PRIOR_MATCHES = 8`. So a debutant's card rate is 100%
prior and about 10% too high in an early walk-forward season, an 8-appearance player is
50% prior, and a 150-appearance regular about 5% — 0.5% too high.

* **Ranking metrics (lift, P@K) are largely protected.** The distortion is close to a
  level shift within a season, and this project has repeatedly established that a level
  shift common to all twenty players cannot reorder them. It is not *perfectly* uniform
  — it hits low-exposure players hardest — so a small ranking effect between debutants
  and regulars is possible.
* **Calibration for early walk-forward seasons is optimistic**, since the model is
  shrinking toward a future league level that happens to be closer to where the league
  was heading.
* **Every A/B comparison in this README remains valid.** Baseline and variant share
  identical priors and differ only in the columns under test, so the *differences* —
  the odds verdict, the referee result, the recommendation-4 ablation — are unaffected.
  It is the absolute lift figures that carry a small optimistic bias.

**The fix, not yet applied:** recompute priors per walk-forward fold from the seasons
preceding that fold, and extend the leakage gate to rebuild priors rather than pass them
in. That changes every absolute number in this file, so it is being left as a deliberate,
documented decision rather than done silently.

---

## Fixed relative to v2

1. **Away starters were all flagged as substitutes.** The bench separator was searched on
   the concatenated home+away frame, so only ~30k of 66,880 starters were labelled
   correctly. Each side is now parsed independently; the count is asserted at 22.00/match.
2. **Rolling features were misaligned against their own rows.** v2 built per-column rolling
   stats with `groupby(...).apply(...).reset_index(drop=True)` and concatenated them, then
   merged back on the index — mixing group-sorted order with the original order. Row *i* of
   the feature table did not describe row *i* of the data. v3 uses index-preserving
   `transform` plus as-of joins, and tests the result.
3. **Minutes played was a feature.** It is only known after the match. Replaced with
   `plr_mins_roll`, the rolling mean of prior minutes.
4. **Random `train_test_split`** on temporally-correlated rows, replaced with chronological
   splits plus walk-forward.
5. **Two different players named Danny Ward** shared one career history. Name collisions are
   now detected automatically (same name, two clubs, same date) and split by club.
6. **462 features / 273 MB** -> 28 features / 5 MB, with no loss.

The original v2 feature table (273 MB) is unsalvageable because of defect 2. Nothing
should be built on it.

---

## Refreshing the data

`scrape_fbref.py` replaces the original notebook scraper, which no longer runs. Three things
changed on FBref, all found by trying it:

1. **Cloudflare.** A plain HTTP request returns 403, so `pd.read_html(url)` fails. Pages
   must come through a real browser — and a *headless* one is served the challenge page, so
   the window is visible while it runs.
2. **The schedule table moved** off `tables[0]`. It is now located by its columns.
3. **The advanced per-player tables are gone.** A match report used to carry 18+ tables plus
   a shot-by-shot table with xG; it now carries 16, with only the 23-column summary. Still
   available: minutes, goals, own goals, cards, fouls, fouls drawn, tackles won,
   interceptions, crosses, offsides, shots. Gone: tackles attempted, challenges lost, aerial
   duels, touches, passes attempted, and all shot-level xG.

That loss was measured before accepting it. On the eight seasons already in hand, dropping
exactly the features a fresh scrape could not fill takes the card model from lift
**1.3440 to 1.3413** — far inside season-to-season noise. (The match model is affected
more subtly; see its README.)

An absent column is written as NaN, not 0, so a rolling mean skips it rather than being
dragged towards zero.

```bash
python pipeline/scrape_fbref.py --season 2026-2027 --limit 3   # trial run first
python pipeline/scrape_fbref.py --season 2026-2027             # a few hours, resumable
```

Then add the season to `config.ALL_SEASONS` and re-run `build_base_table.py`.
**2025-26 has already been scraped and folded in** (380/380 matches, 0 failures).

Two traps the first refresh walked into, both now guarded by tests:

* **Club names drift.** The fixtures table calls them "Newcastle" and "Nottingham" where
  the archive says "Newcastle Utd" and "Nott'ham Forest". A club under two names is a club
  with two histories — Elo restarts it at the promoted-side default, the league table
  forgets its points, rolling form resets — and nothing else in the pipeline complains.
  `TEAM_ALIASES` in `build_base_table.py` canonicalises at parse time; the test suite now
  fails if one club name is contained in another, and the match pipeline separately
  requires every match to join the odds feed, which is what exposed it (74 unjoined
  fixtures).
* **Absent is not zero.** A statistic the source never published must be NaN, so a rolling
  mean skips it, rather than 0, which drags it down. Same for a season with no shot files
  at all.

---

## Where the model is right and wrong

From 38,000 out-of-sample predictions over five seasons (`oos.py`), with every segment
chosen on 2021-22 → 2023-24 and confirmed on 2024-25 → 2025-26 (`segments.py`).

**Discrimination barely varies.** Bootstrapping over *matches* — players inside a match
are not independent — every slice's ROC-AUC interval overlaps the overall
[0.592, 0.607]. There is no condition under which this model becomes good.

The one exception: **≤5 prior appearances, AUC 0.545**, on the rows with the *highest*
base rate (0.210). An exposure problem, not a modelling one — with no history every rate
feature falls back to the league prior, so debutants look identical to each other. A
dedicated model on those rows scores 0.49 (1,884 rows is not enough).

**Calibration varies and replicates.** Relative level error, same sign in both windows:
wingers −5.7%, u23 −5.3%, hardest fixtures −5.3%, matchweeks 1-10 −4.7%, away −4.1%
versus home +4.7%; and 85-minute players +4.3%, 150+ appearances +4.1%.

**It cannot be corrected away.** `specialists.py` tests Platt rescaling, segment
offsets, segment slopes, a stacked booster, and per-segment boosters split by position,
history and venue — fitted honestly on earlier seasons' out-of-sample predictions. All
six lose to the plain global model (Δ lift −0.026 to −0.063). The booster already has
those variables; the residual bias is the hedging it is already doing.

**What works is a filter, not a correction.** `segments.FAVOURABLE` / `UNFAVOURABLE`:

* avoid centre-backs, the most lenient quarter of referees, nailed-on 85-minute players
  → top-1 strike 0.280 vs 0.325 unfiltered in the confirmation window
* favourable (expected minutes 60-85, ≥21 appearances, stricter three quarters of
  referees) adds **2.1pp in discovery and 2.6pp in confirmation to top-3 shortlists**, but
  did **not** replicate for the single top pick — it is a top-3 tool only

---

## Extending it

* **Odds features are built and measured** — see the section above. The open decision
  is whether to set `config.USE_ODDS_FEATURES = True`; the recommendation is to enable
  the `side` group for ranking and, separately, `tightness` for a pricing model.
* **Scrape the other four leagues.** `scrape_fbref.py --competition new --seasons
  2021-2022 2022-2023 2023-2024 2024-2025 2025-2026` does all four sequentially and is
  resumable — a match folder that already holds `main.csv` is skipped, so it can be
  stopped and restarted freely. Each league writes to `../data_<league>/`, a sibling of
  the Premier League archive so nothing already built moves. Validated against all four
  (380/380 fixtures matched on each).

  **Do not run the leagues in parallel.** Measured, not assumed: one worker sustains
  **12.7 s per match**; four workers on the same connection were throttled to roughly
  **150 s per match each**, a net ~3× slowdown, because FBref starts serving the
  Cloudflare challenge and the 20s/40s retry backoff dominates. Recovery after dropping
  back to one worker is not immediate. Five seasons × four leagues is about 7,000
  matches, so budget **~25 hours** sequentially.

  Note that FBref now serves the reduced 23-column player table for *every* season, so a
  newly scraped 2018-19 gets the same columns as 2025-26. Every column the current
  feature set needs is present — including the referee, which football-data does not
  publish outside England.
* **Bench population** (`config.INCLUDE_BENCH = True`) is built but never trained.
* **The decomposition is inside the ensemble.** It used to be served separately as
  `slip_probability`; it is now a component of the five-league ensemble (weight 0.2), so
  `predict.py` returns one `card_probability` for both ranking and pricing. The
  Premier-League-only decomposition (`python pipeline/team_model.py --save`,
  `artifacts/premier_league/team_decomposition.joblib`) remains as a research artifact.
