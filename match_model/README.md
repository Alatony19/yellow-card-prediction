# Match-result model

Predicts the **joint distribution of (home goals, away goals)** for a Premier
League fixture, and reads every market off that one object: 1X2, over/under any
line, multigol ranges, both teams to score, correct score.

The annotated end-to-end walk-through is **`match_result_pipeline.ipynb`** — start
there.

Separate pipeline from the card model. It shares the raw archive, the season
splits and `../context.py`, and nothing else — the unit of observation is a
match rather than a player.

---

## The short version

| | |
|---|---|
| Matches | 3,466 (2017-18 → 2026-27, the last season in progress) |
| Features | 15 (12 form and context, 3 from the closing line) |
| Walk-forward 1X2 RPS | **0.1975** (5 seasons) |
| Closing line, same matches | **0.1949** — ahead in all five |
| Does disagreeing with the line add information? | **no** - the model trails it in all five seasons |

**Read that third row before using this for anything.** The model is
well calibrated and roughly as accurate as the market, but it does not beat the
closing line, so the fixtures where it disagrees with the line are more likely to
be model error than information. It is a starting point for your own analysis -
a consistent second opinion with a number attached - not an edge.

---

## Why a score matrix instead of a 1X2 classifier

Every market asked for here — over 1.5, over 2.5, home multigol 1-3, both teams
to score — is a different question about the same underlying object: how many
goals each side scores. Model that distribution once and project it, and

* the quoted probabilities can never contradict one another (over 2.5 and under
  2.5 sum to exactly 1; home multigol 1-3 plus "home fails to score" plus "home
  scores 4+" partitions the distribution);
* a new market costs a function, not a new model;
* the correct-score distribution and the expected goals come out for free.

Two independent Poisson variables get the shape roughly right but misprice the
four lowest scorelines — real football produces more 0-0 and 1-1, and fewer 1-0
and 0-1, than independence implies. Dixon & Coles (1997) correct exactly those
four cells with one parameter, `rho`, fitted here on the validation season
(≈ -0.04).

`MAX_GOALS = 14`, not 10. The matrix is renormalised after truncation, so a
discarded tail is redistributed across every cell. At 10, a lopsided fixture
(λ 4.2 v 0.4) had **home multigol 1-3 wrong by 0.15 percentage points**; at 14
it is 0.001. The test suite checks this at the extremes, not at an average
fixture.

## Where the expected goals come from

Three sources, blended:

1. **Dixon-Coles ratings** (`ratings.py`) — attack and defence per club plus a
   home advantage, by maximum likelihood with exponential time decay. Refit
   every matchweek on matches that kicked off strictly earlier, warm-started
   from the previous week so ~270 refits take a couple of minutes. Solves
   strength and schedule jointly, so a side that has just played the top four
   is not mistaken for a side in decline.
2. **A gradient-boosted Poisson model** (`modelling.py`) — rolling xG for and
   against, venue-split form, rest days, league position, fixture difficulty.
   `objective="count:poisson"`, so the output is a rate. One model serves both
   sides; `is_home` is a feature.
3. **The closing line** (`odds.py`) — the de-vigged 1X2 and over/under prices
   are inverted into the (λ_home, λ_away) pair that would reproduce them, and
   fed in as features.

The two model rates are blended geometrically (in log space, the natural scale
for a multiplicative Poisson model). The weight is chosen on the validation
season and lands around 0.90 on the boosted model.

## What each piece is worth

Measured on the eight complete seasons (2021-22 → 2024-25 walk-forward), before
the 2025-26 scrape, precisely so the decision to drop xG could be made on
evidence rather than convenience:

| configuration | 1X2 RPS | scoreline NLL |
|---|---|---|
| full model (xG + odds) | 0.1946 | 2.984 |
| no odds features | 0.2000 | 3.007 |
| no xG, odds kept | 0.1949 | 2.985 |
| no xG, no odds | 0.2018 | 3.018 |
| **closing line alone** | **0.1923** | — |

Two things worth noticing. The closing line is worth ~0.005 RPS, far more than
anything else in the feature set. And **xG is worth almost nothing once the
line is present** (0.1946 → 0.1949) — the market has already priced everything
xG would tell you.

That is what made the 2025-26 refresh worth doing even though FBref no longer
publishes shot-level xG. The xG features are out of the model now; the odds
feed is holding up the difference, which makes it load-bearing rather than a
bonus.

### Current standing, five seasons

| season | model | closing line |
|---|---|---|
| 2021-22 | 0.1922 | 0.1890 |
| 2022-23 | 0.2024 | 0.1991 |
| 2023-24 | 0.1857 | 0.1838 |
| 2024-25 | 0.1985 | 0.1974 |
| 2025-26 | 0.2085 | 0.2053 |
| **mean** | **0.1975** | **0.1949** |

The line wins every season. The gap is small and remarkably stable — about
0.0025 RPS — which is consistent with a single missing input rather than a
modelling flaw. That input is team news.

### 2026-27, the live season

The splits moved up one season when 2026-27 was added: train 2018-19 → 2024-25,
validate on 2025-26, hold out 2026-27. The walk-forward above deliberately did
**not** move — it still ends on 2025-26, so every number in this README remains
a five-complete-season measurement.

On the 46 matches played so far the hold-out reads model **0.1919** against the
closing line's **0.1963**, on the 40 of them the price feed has reached. Do not
read anything into that. Forty matches is about a fifth of the sample needed to
separate two RPS figures 0.004 apart; the five-season walk-forward is the basis
for every claim here, and it says the line wins. The live season is carried for
the **ratings and the rolling form**, not for evaluation — Dixon-Coles strengths
and ten-match form have to be current for today's fixture to be rated at all.

Two things the refresh had to handle, both now guarded rather than patched:

* **The price feed lags the results.** football-data.co.uk publishes weekly, so
  the newest round exists on FBref before it has a closing line — currently 6
  matches of matchweek 5. The gate that used to demand a line on *every* match
  would now fail on a perfectly healthy build. It has not been relaxed: it now
  requires the missing rows to form a **contiguous trailing block by kickoff**,
  which a stale feed produces and name drift does not, plus a second gate that
  every completed season is priced with no exceptions at all.
* **Scoring the line on matches that have no line.** `modelling.py` fed the
  unpriced rows into the closing-line benchmark, which turned every one of its
  metrics into NaN — and NaN compares false, so the summary printed a confident
  "line ahead" off nothing. The benchmark now runs on the priced subset, and
  scores the model on that same subset so the two are comparable; where the two
  differ the line reports both counts.
* **Quoting a fixture that has no price yet.** This one was not cosmetic. The
  odds feed is complete for all nine finished seasons, so the booster has
  *never* seen `mkt_lambda_*` missing — the same trap the xG features were
  pulled from the model to avoid. Handed a NaN, it routed all six unpriced
  matchweek-5 fixtures to a total λ of **4.24 ± 0.07** — 1.4 goals above the
  league mean of 2.84, and effectively identical for fixtures as different as
  Newcastle v Hull and Brentford v Chelsea. Forest v Coventry came out at 79.5%
  over 2.5. `predict.py` now detects unpriced fixtures and scores them with a
  second fit that never had the odds features at all; the same six then spread
  across λ 2.14 to 2.96, and Forest v Coventry reads 56.8%. It says so when it
  does this. Those quotes are the no-odds model, which is ~0.005 RPS worse —
  the right thing to compare them against is the next price to appear, not the
  priced quotes above them.

## Does disagreeing with the market add information?

No. The closing line beats the model on ranked probability score in all five
walk-forward seasons (0.1949 against 0.1975), so where the two disagree the model is
the more likely one to be wrong. Filtering for larger disagreements does not help
either: it selects for larger model errors. Treat `predict.py`'s `edge` column as a
pointer to a fixture worth a closer look, never as a signal to act on.

---

## Files

| file | what it does |
|---|---|
| `config.py` | every tunable: splits, features, markets, `MAX_GOALS`, blend |
| `odds.py` | football-data.co.uk loader, name mapping, de-vig, λ inversion |
| `build_match_table.py` | team-match spine: goals, xG, rolling form, context, prices |
| `score_matrix.py` | the joint distribution, and every market projected off it |
| `ratings.py` | Dixon-Coles MLE with time decay, and the rolling refit |
| `modelling.py` | boosted Poisson model, blending, per-market evaluation |
| `predict.py` | quote a fixture or a whole matchweek, against the line |
| `tests/test_match_model.py` | 27 gates: coherence, recovery, leakage |
| `match_result_pipeline.ipynb` | **the annotated walk-through — start here** |
| `_make_notebook.py` | regenerates the notebook; edit this, not the `.ipynb` |

## Running it

```bash
python build_match_table.py      # ~2 min, parses 3,040 shot files
python modelling.py --walk-forward
python predict.py --season 2024-2025 --matchweek 30
python tests/test_match_model.py
```

Install the dependencies first (`pip install -r requirements.txt` from the
repository root).

## Leakage

Same rule as the card pipeline, and the same kind of gate. Every rolling
feature is a post-match cumulative attached with a backward, non-exact
`merge_asof`; the Dixon-Coles ratings for a matchweek are fitted only on
matches that kicked off before it. The test deletes every match after a cutoff,
rebuilds the rolling features, and asserts the surviving rows are bit-identical
— currently `0.00e+00` across all ten features.

The test suite also checks that the fitters recover planted parameters from
simulated data, which is what separates "the optimiser converged" from "the
optimiser converged on the right answer".

## An independent check on the data

FBref goals are reconstructed as *players' goals + the opponent's own goals*.
football-data.co.uk records scorelines independently. They agree on
**3,420 of 3,420 matches**. That is asserted in the test suite, so a future
parsing regression will fail loudly rather than quietly.

---

## FBref changed — what a refreshed season now contains

As of 2026, an FBref match report no longer carries the advanced per-player
tables (passing, pass types, defensive actions, possession, miscellaneous) or
the shot-by-shot table with xG. Only the 23-column summary survives. See
`../scrape_fbref.py` for the full account.

For this pipeline that means **no xG from 2025-26 onwards**. The table above
says what that costs: 0.1946 → 0.1949 RPS, provided the football-data odds feed
is kept. The xG features have accordingly been removed from
`config.NUMERIC_FEATURES` rather than left to go missing — the booster never saw
them absent during training, so its learned direction for a NaN would be
arbitrary. `build_match_table.py` still computes them for the seasons that have
them.

Two data traps the refresh walked into, both now guarded:

* **Absent is not zero.** A side missing from a shot file that *exists* took no
  shots, and 0 is right (twice in eight seasons). A match with no shot file at
  all was never published and must stay NaN — otherwise a whole season without
  xG reads as a season in which nobody shot.
* **Club names drift.** FBref's fixtures table says "Newcastle" where the
  archive says "Newcastle Utd". The gate that caught it is in this pipeline:
  *every match must join the closing line*, which failed on 74 fixtures. Keep
  that test — it is the only external check on club identity the project has.
  It earned its keep again on the 2026-27 refresh: the three promoted sides
  arrived as "Coventry City" and "Hull City" on FBref against "Coventry" and
  "Hull" on football-data, and 7 fixtures silently lost their price until
  `odds.TEAM_NAME_MAP` was extended. Every promotion is a fresh chance to break
  this join, so check the gate first whenever a season is added.

## Known limitations

* **One league, ten seasons.** 3,466 matches is not a lot for the tail
  markets (over 3.5, correct score).
* **Dixon-Coles has no prior, so promoted clubs are rated wildly.** The fit is
  a plain MLE with time decay and no shrinkage toward the league mean. A club
  with a full history is anchored by hundreds of matches; a promoted club in
  September has five. After matchweek 5 of 2026-27 Coventry City — 1 goal
  scored in 5 matches — carries an attack rating of **−1.43** against a league
  spread of roughly ±0.6, and `fair_odds_from_ratings` duly quotes λ 0.18 and
  **20.3** for a Coventry home win. That is the sample talking, not the club.

  This barely touches the model itself, because `modelling.py` blends
  Dixon-Coles at a weight that has optimised to ~0.00 on it in four of five
  seasons. It matters a great deal to **`todays_slate.py`**, whose fair-odds
  fallback is Dixon-Coles *alone*. Treat a fallback price for a promoted club
  in the first third of a season as unusable, and supply a real market price
  for those fixtures. A shrinkage prior (the card pipeline's `EB_PRIOR_MATCHES`
  idea, applied to attack and defence) would fix it, and would move every
  Dixon-Coles number in this README, so it has not been done casually.
* **De-vigging is proportional.** Shin's method is better for longshots; the
  difference is small at these prices and large for a 30/1 outsider.
* **No team news.** Injuries, suspensions and rotation are exactly what the
  closing line knows and this model does not. That is most of the gap.
* **`predict.py --home X --away Y`** rates an arbitrary fixture from
  Dixon-Coles ratings only, because the boosted model needs that fixture's own
  rolling-form row, which does not exist for a match outside the archive.
