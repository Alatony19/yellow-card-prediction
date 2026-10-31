"""Generate prediction_review.ipynb.

The pipeline notebook answers "is the model any good?".  This one answers
"what did it actually say, and was it right?" - match by match, player by
player, with the tools to go looking rather than a fixed set of conclusions.

Run:  python notebooks/build_review.py
then: jupyter nbconvert --to notebook --execute --inplace notebooks/prediction_review.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

C = []


def md(text: str) -> None:
    C.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    C.append(nbf.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- #
md("""
# Prediction review — what the card model said, and whether it was right

Every prediction in this notebook comes from the **served model** — the five-league
ensemble (`pipeline/ensemble.py`: 10 bagged boosters, a logistic and the team-count
decomposition, blended 0.4 / 0.4 / 0.2) — and every one is **out of sample**. Each season
was scored by components re-fitted from scratch on the seasons before it and
early-stopped on the season just before (`python pipeline/ensemble.py evaluate`). The rows
are kept instead of being collapsed into a summary, so nothing here is the model marking
its own homework.

**183,100 predictions · 9,155 matches · five leagues · 2021-22 → 2026-27** (the last
season is in progress). Every browsing function takes a `league=` argument —
`premier_league`, `serie_a`, `la_liga`, `bundesliga`, `ligue_1`.

The notebook is built to be *used*, not just read. The browsing functions in §3 and §7
take arguments — pick a match, a player, a team, a referee, a date — and the last
section is a blank cell for your own queries.

| section | question |
|---|---|
| 1 | What is in the pool, and how good is it overall? |
| 2 | Where does the probability actually land — calibration, by season |
| 3 | **Match browser** — the full 20-player field for any fixture |
| 4 | The shortlist: top-1 / top-3 picks and how often they hit |
| 5 | Best calls and worst misses |
| 6 | Scorecards — by player, team, referee |
| 7 | **Filtered search** — find the rows you care about |
| 8 | When to act and when not to (the conditions rule) |
""")

code("""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
V3 = Path.cwd()
while not (V3 / "paths.py").exists() and V3 != V3.parent:   # opened from notebooks/
    V3 = V3.parent
sys.path.insert(0, str(V3))
import paths  # noqa: F401 - puts pipeline/, serve/, research/ on sys.path

import config, ensemble, evaluation, oos as oos_mod, segments as segmod

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})

T = config.TARGET_COL
# The ensemble's walk-forward pool, with every dataset column joined back on, plus the
# per-match quantities (rank in match, match totals, per-row error) and the segments.
d = segmod.add_segments(oos_mod._add_derived(ensemble.load_pool(full=True)))
d["league"] = d["league"].astype(str)
print(f"{len(d):,} predictions | {d['match_id'].nunique():,} matches | "
      f"{d['season'].nunique()} seasons | {d['league'].nunique()} leagues | "
      f"{d['player_key'].nunique():,} players")
print("weights:", ensemble.chosen_weights())
d[["league", "season", "kickoff", "team", "opponent", "player", "p", T]].head()
""")

# --------------------------------------------------------------------------- #
md("""
---
## 1. The pool

One row per outfield starter per match. `p` is the ensemble's probability that **the
slot** — that player or whoever replaces him — picks up a card. `y_card` is what
happened. `p_bag`, `p_logistic` and `p_decomp` are the three components behind it, and
`p_m0` … `p_m9` the ten boosters of the bag.

`rank_in_match` is the row's position when the whole 20-player field is sorted by `p`,
which is what a top-K shortlist actually consumes.
""")

code("""
summary = evaluation.evaluate(d, "p", T, name="pooled out-of-sample",
                              k_values=config.TOP_K_VALUES)
print(f"base rate        {summary['base_rate']:.4f}   "
      f"(mean prediction {summary['mean_pred']:.4f})")
print(f"PR-AUC           {summary['pr_auc']:.4f}   lift {summary['pr_auc_lift']:.3f}x")
print(f"ROC-AUC          {summary['roc_auc']:.4f}")
print(f"calibration err  {summary['ece']:.4f}")
print()
print("top-K inside each match:")
for k in config.TOP_K_VALUES:
    print(f"  K={k}:  precision {summary[f'precision@{k}']:.4f}   "
          f"at least one hit in {summary[f'hit_rate@{k}']:.1%} of matches")
""")

code("""
# Per league: one model serves all five, so each has to hold up on its own.
rows = []
for lg, g in d.groupby("league"):
    r = evaluation.evaluate(g, "p", T, name=lg, k_values=(1, 3))
    rows.append({"league": lg, "matches": g["match_id"].nunique(), "base": r["base_rate"],
                 "mean_p": r["mean_pred"], "roc_auc": r["roc_auc"], "ece": r["ece"],
                 "P@1": r["precision@1"], "P@3": r["precision@3"],
                 "top-4 carded": g[g["rank_in_match"] <= 4].groupby("match_id")[T].sum().mean()})
per_league = pd.DataFrame(rows)
display(per_league.style.format({c: "{:.4f}" for c in per_league.columns
                                 if c not in ("league", "matches")}))
""")

code("""
# Per-season, so a good pooled number cannot hide one bad year.
rows = []
for s, g in d.groupby("season"):
    r = evaluation.evaluate(g, "p", T, name=s, k_values=(1, 3))
    rows.append({"season": s, "n": r["n"], "base": r["base_rate"],
                 "mean_p": r["mean_pred"], "pr_auc": r["pr_auc"],
                 "lift": r["pr_auc_lift"], "roc_auc": r["roc_auc"],
                 "ece": r["ece"], "P@1": r["precision@1"], "P@3": r["precision@3"],
                 "hit@3": r["hit_rate@3"]})
per_season = pd.DataFrame(rows)
display(per_season.style.format({c: "{:.4f}" for c in per_season.columns
                                 if c not in ("season", "n")})
        .background_gradient(subset=["lift", "P@3"], cmap="Greens"))
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].bar(per_season["season"], per_season["base"], .55, label="actual card rate",
          color="#c44e52", alpha=.8)
ax[0].plot(per_season["season"], per_season["mean_p"], "o--", color="#333",
           label="mean prediction")
ax[0].set_title("Level: actual rate vs mean prediction")
ax[0].set_ylabel("P(slot carded)")
ax[0].legend(fontsize=8)
ax[0].tick_params(axis="x", rotation=30)

ax[1].plot(per_season["season"], per_season["lift"], "o-", color="#3b6ea5")
ax[1].axhline(1.0, ls="--", c="grey", lw=1)
ax[1].set_title("Ranking: PR-AUC lift over the base rate")
ax[1].set_ylim(1.0, 1.55)
ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
""")

md("""
The left panel is the most important maintenance fact in this project: the card rate
moves between seasons, and a model trained on older seasons trails it. The Premier-League-
only model under-called 2023-24 by 3.3 percentage points; pooled across five converging
leagues (`research/league_comparison.py`) the drift is smaller, and the ensemble runs about
half a point low. Refit before every season and treat the absolute probabilities as needing
a level check; the *ranking* (right panel) is far more stable than the level.
""")

# --------------------------------------------------------------------------- #
md("""
---
## 2. Does 0.25 mean 25%?

The reliability curve bins the predictions by decile and plots predicted against
observed. On the diagonal means a quoted probability can be compared with a price
directly; below means the model is over-confident and any value it reports is
illusory.
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
evaluation.plot_calibration(d[T], d["p"], n_bins=10, ax=ax[0], label="all seasons")
for s, g in d.groupby("season"):
    tbl = evaluation.calibration_table(g[T], g["p"], 8)
    ax[1].plot(tbl["predicted"], tbl["observed"], "o-", lw=1.4, ms=4, label=s)
lims = [0, d["p"].quantile(.999)]
ax[1].plot(lims, lims, ls="--", c="grey", lw=1)
ax[1].set_xlabel("predicted"); ax[1].set_ylabel("observed")
ax[1].set_title("Reliability by season")
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()

display(evaluation.calibration_table(d[T], d["p"], 10)
        .style.format({"predicted": "{:.4f}", "observed": "{:.4f}",
                       "p_low": "{:.4f}", "p_high": "{:.4f}"})
        .bar(subset=["n"], color="#dce6f1"))
""")

# --------------------------------------------------------------------------- #
md("""
---
## 3. Match browser

`show_match(...)` prints the full field for one fixture, sorted by the model's
probability, with the outcome alongside. This is the view to use when you want to
sanity-check a prediction against your own read of a game.

Rows are marked:

* `<-- CARDED` the slot picked up a card
* `(sub)` the card was won by the substitute rather than the named starter
* the top-3 block is separated, because that is the shortlist

Call it with a `match_id`, or with a team name and season to get a list.
""")

code('''
def find_matches(team=None, season=None, opponent=None, date=None, league=None, limit=30):
    """Locate fixtures. Every argument is optional and matched case-insensitively."""
    # each fixture appears twice (once per side); keep the home row for listing
    m = d[d["is_home"] == 1].drop_duplicates("match_id")[
        ["match_id", "league", "season", "matchweek", "kickoff", "team", "opponent",
         "referee"]]
    if league:
        m = m[m["league"] == league]
    if team:
        m = m[m["team"].str.contains(team, case=False, na=False)
              | m["opponent"].str.contains(team, case=False, na=False)]
    if opponent:
        m = m[m["opponent"].str.contains(opponent, case=False, na=False)]
    if season:
        m = m[m["season"] == season]
    if date:
        m = m[m["kickoff"].astype(str).str.startswith(str(date))]
    return m.sort_values("kickoff").head(limit).reset_index(drop=True)


def show_match(match_id=None, team=None, season=None, league=None, k=3):
    """Print the full 20-player field for one match with outcomes."""
    if match_id is None:
        cands = find_matches(team=team, season=season, league=league)
        if cands.empty:
            print("no match found"); return
        match_id = cands.iloc[0]["match_id"]
        print(f"(showing the first of {len(cands)} matches found)\\n")

    g = d[d["match_id"] == match_id].sort_values("p", ascending=False)
    if g.empty:
        print(f"no rows for match_id {match_id!r}"); return
    h = g[g["is_home"] == 1].iloc[0]
    actual = int(g[T].sum())
    mw = f"MW{int(h['matchweek'])}" if pd.notna(h["matchweek"]) else ""
    print(f"{h['team']} vs {h['opponent']}   {str(h['kickoff'])[:16]}   "
          f"{h['league']} {h['season']} {mw}")
    print(f"referee: {h['referee']}   |   slots carded: {actual}   |   "
          f"model expected: {g['p'].sum():.2f}")
    print("=" * 86)
    for i, (_, r) in enumerate(g.iterrows()):
        if i == k:
            print("-" * 30 + f"  end of top-{k} shortlist  " + "-" * 30)
        hit = "<-- CARDED" if r[T] == 1 else ""
        sub = " (sub)" if (r[T] == 1 and r.get("repl_card", 0) == 1
                           and r.get("y_any_card", 1) == 0) else ""
        side = "H" if r["is_home"] == 1 else "A"
        print(f"{i+1:>3}. {r['player'][:24]:<24} {side} {r['team'][:14]:<14} "
              f"{str(r['pos_group']):<4} p={r['p']:.3f}  {hit}{sub}")
    print("=" * 86)
    got = g.nlargest(k, "p")[T].sum()
    print(f"top-{k} shortlist: {int(got)}/{k} correct"
          f"{'  -- HIT' if got > 0 else '  -- MISS'}")
''')

code("""
# A card-heavy match and a quiet one, picked automatically from the pool.
busiest = d.groupby("match_id")[T].sum().idxmax()
show_match(busiest)
""")

code("""
# Try your own: any team name works, and the season is optional.
find_matches(team="Arsenal", season="2025-2026").head(10)
""")

code("""
show_match(team="Arsenal", season="2025-2026")
""")

code("""
# Any league: a Serie A fixture from the same season.
show_match(team="Inter", season="2025-2026", league="serie_a")
""")

# --------------------------------------------------------------------------- #
md("""
---
## 4. The shortlist

If you back the model's top pick in every match, how often does it win? This is the
number that maps onto money, and `fair_odds` is the price you would need to break even
before any margin.
""")

code("""
shortlist = []
for k in (1, 2, 3, 5):
    picks = d[d["rank_in_match"] <= k]
    per = picks.groupby("match_id")[T].agg(["sum", "size"])
    strike = float((per["sum"] / per["size"]).mean())
    shortlist.append({"K": k, "picks": len(picks),
                 "strike rate": strike,
                 "fair odds": 1 / strike,
                 "matches with >=1 hit": float((per["sum"] > 0).mean())})
shortlist = pd.DataFrame(shortlist)
display(shortlist.style.format({"strike rate": "{:.4f}", "fair odds": "{:.2f}",
                           "matches with >=1 hit": "{:.1%}"}))
""")

code("""
# Strike rate of the top-1 pick through time, 40-match rolling.
top1 = d[d["rank_in_match"] == 1].sort_values("kickoff").reset_index(drop=True)
roll = top1[T].rolling(40, min_periods=20).mean()
field = d.sort_values("kickoff")[T].rolling(800, min_periods=400).mean()

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(top1["kickoff"], roll, lw=1.6, color="#3b6ea5", label="top-1 pick (40-match roll)")
ax.axhline(top1[T].mean(), ls="--", c="#3b6ea5", lw=1,
           label=f"top-1 overall ({top1[T].mean():.3f})")
ax.axhline(d[T].mean(), ls="--", c="#c44e52", lw=1,
           label=f"any starter ({d[T].mean():.3f})")
for s in sorted(d["season"].unique())[1:]:
    ax.axvline(d[d["season"] == s]["kickoff"].min(), color="grey", lw=.6, alpha=.5)
ax.set_ylabel("P(pick was carded)")
ax.set_title("Does the top pick stay ahead of the field?")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.show()
""")

md("""
The gap between the two dashed lines is the entire edge: the top pick is carded about
**1.5×** as often as the average starter (the exact rates are in the legend). The rolling
line is volatile — 40 matches is a small sample for a 29% event — but it spends almost
all of its time above the field. Five leagues play on the same weekends, so the x-axis is
dense; filter `top1` by `league` to see one competition.
""")

# --------------------------------------------------------------------------- #
md("""
---
## 5. Best calls and worst misses

Two failure modes, and they are not symmetric. A confident prediction that did not land
is expected — 0.45 means it misses more than half the time. A **carded player the model
put near the bottom of the field** is the more interesting error, because it points at
something the features do not see.
""")

code("""
print("=== most confident predictions that were WRONG ===")
worst = d[d[T] == 0].nlargest(12, "p")
display(worst[["league", "season", "kickoff", "player", "team", "opponent", "referee",
               "pos_group", "p", "rank_in_match"]]
        .style.format({"p": "{:.3f}"}).hide(axis="index"))

print("\\n=== carded players the model rated LOWEST ===")
miss = d[d[T] == 1].nsmallest(12, "p")
display(miss[["league", "season", "kickoff", "player", "team", "opponent", "referee",
              "pos_group", "p", "rank_in_match", "plr_apps_prior"]]
        .style.format({"p": "{:.3f}", "plr_apps_prior": "{:.0f}"}).hide(axis="index"))
""")

code("""
# The shape of the errors: where do carded and uncarded slots sit on the
# probability scale?  Heavy overlap is the honest picture of this problem.
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
bins = np.linspace(0, d["p"].quantile(.999), 45)
ax[0].hist(d.loc[d[T] == 0, "p"], bins=bins, alpha=.65, density=True,
           label="not carded", color="#7f9fc4")
ax[0].hist(d.loc[d[T] == 1, "p"], bins=bins, alpha=.65, density=True,
           label="carded", color="#c44e52")
ax[0].set_xlabel("predicted probability"); ax[0].set_ylabel("density")
ax[0].set_title("The two distributions overlap heavily -- as they must")
ax[0].legend(fontsize=8)

# Observed rate by rank within the match: the ranking's actual payoff curve.
byrank = d.groupby("rank_in_match")[T].agg(["mean", "size"]).reset_index()
byrank = byrank[byrank["rank_in_match"] <= 20]
ax[1].bar(byrank["rank_in_match"], byrank["mean"], color="#3b6ea5", alpha=.85)
ax[1].axhline(d[T].mean(), ls="--", c="#c44e52", lw=1.2, label="field base rate")
ax[1].set_xlabel("rank within match (1 = model's top pick)")
ax[1].set_ylabel("observed card rate")
ax[1].set_title("Monotone in rank -- the ordering is real")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md("""
The right-hand panel is the cleanest evidence that the ranking carries information:
the observed card rate declines essentially monotonically from the model's first pick
to its twentieth, with no re-ordering in the middle. The edge is modest but it is
structurally sound rather than an artefact of a few outliers.
""")

# --------------------------------------------------------------------------- #
md("""
---
## 6. Scorecards

Which players, teams and referees does the model read well? `bias` is
`mean(p) - observed`: negative means it under-called that group.
""")

code('''
def scorecard(by, min_rows=120, sort="bias"):
    """Per-group prediction quality. Any column of ``d`` can be the grouper."""
    g = d.groupby(by)
    out = g.agg(n=("p", "size"), predicted=("p", "mean"), observed=(T, "mean")).reset_index()
    out = out[out["n"] >= min_rows]
    out["bias"] = out["predicted"] - out["observed"]
    out["ratio"] = out["predicted"] / out["observed"].replace(0, np.nan)
    return out.sort_values(sort).reset_index(drop=True)


player_card = scorecard("player", min_rows=120)
print("=== players the model most UNDER-rates (it says less than happens) ===")
display(player_card.head(12).style.format(
    {"predicted": "{:.3f}", "observed": "{:.3f}", "bias": "{:+.3f}", "ratio": "{:.2f}"}
).hide(axis="index"))

print("\\n=== players the model most OVER-rates ===")
display(player_card.tail(12).iloc[::-1].style.format(
    {"predicted": "{:.3f}", "observed": "{:.3f}", "bias": "{:+.3f}", "ratio": "{:.2f}"}
).hide(axis="index"))
''')

code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ref = scorecard("referee", min_rows=400)
ax[0].scatter(ref["predicted"], ref["observed"], s=18, color="#3b6ea5")
lims = [min(ref["predicted"].min(), ref["observed"].min()) * .92,
        max(ref["predicted"].max(), ref["observed"].max()) * 1.05]
ax[0].plot(lims, lims, ls="--", c="grey", lw=1)
ax[0].set_xlabel("mean prediction"); ax[0].set_ylabel("observed rate")
ax[0].set_title(f"Referees (n={len(ref)}, >=400 rows each)")
for _, r in pd.concat([ref.head(2), ref.tail(2)]).iterrows():
    ax[0].annotate(str(r["referee"])[:16], (r["predicted"], r["observed"]),
                   fontsize=7, xytext=(3, 3), textcoords="offset points")

team = scorecard("team", min_rows=400)
ax[1].scatter(team["predicted"], team["observed"], s=18, color="#c44e52")
lims = [min(team["predicted"].min(), team["observed"].min()) * .92,
        max(team["predicted"].max(), team["observed"].max()) * 1.05]
ax[1].plot(lims, lims, ls="--", c="grey", lw=1)
ax[1].set_xlabel("mean prediction"); ax[1].set_ylabel("observed rate")
ax[1].set_title(f"Teams (n={len(team)})")
for _, r in pd.concat([team.head(2), team.tail(2)]).iterrows():
    ax[1].annotate(str(r["team"])[:16], (r["predicted"], r["observed"]),
                   fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.tight_layout(); plt.show()
""")

md("""
Points below the diagonal are groups the model under-calls. Referees scatter more widely
than teams, which is consistent with the finding in the pipeline notebook that referee
*style* is real but does not transfer well between seasons.
""")

# --------------------------------------------------------------------------- #
md("""
---
## 7. Filtered search

`review(...)` is the general-purpose query. Every argument is optional; they compose.
""")

code('''
def review(season=None, team=None, player=None, referee=None, pos=None,
           min_p=None, max_p=None, outcome=None, rank_max=None, league=None,
           sort="p", ascending=False, limit=40):
    """Filter the prediction pool.

    outcome : None (all) | "carded" | "not carded"
    rank_max: keep only the model's top-N picks in each match
    """
    g = d
    if league:   g = g[g["league"] == league]
    if season:   g = g[g["season"] == season]
    if team:     g = g[g["team"].str.contains(team, case=False, na=False)]
    if player:   g = g[g["player"].str.contains(player, case=False, na=False)]
    if referee:  g = g[g["referee"].str.contains(referee, case=False, na=False)]
    if pos:      g = g[g["pos_group"].astype(str) == pos]
    if min_p is not None: g = g[g["p"] >= min_p]
    if max_p is not None: g = g[g["p"] <= max_p]
    if outcome == "carded":     g = g[g[T] == 1]
    if outcome == "not carded": g = g[g[T] == 0]
    if rank_max: g = g[g["rank_in_match"] <= rank_max]
    cols = ["league", "season", "kickoff", "player", "team", "opponent", "referee",
            "pos_group", "p", "rank_in_match", T]
    return g.sort_values(sort, ascending=ascending)[cols].head(limit)


# Example: every top-3 pick in 2025-26 that the model got right, most confident first.
review(season="2025-2026", rank_max=3, outcome="carded", limit=15)
''')

code('''
def player_history(name, limit=40):
    """Every prediction ever made about one player, with a running hit rate."""
    g = d[d["player"].str.contains(name, case=False, na=False)].sort_values("kickoff")
    if g.empty:
        print(f"no player matching {name!r}"); return
    g = g.copy()
    g["cum_hit_rate"] = g[T].expanding().mean()
    print(f"{g['player'].iloc[0]}  --  {len(g)} matches, "
          f"carded {int(g[T].sum())} ({g[T].mean():.1%}), "
          f"mean prediction {g['p'].mean():.3f}")
    return g[["league", "season", "kickoff", "team", "opponent", "referee", "p",
              "rank_in_match", T, "cum_hit_rate"]].tail(limit)


# The player the model rated most highly across the whole pool:
player_history(d.groupby("player")["p"].mean().pipe(
    lambda s: s[d.groupby("player").size() >= 40]).idxmax())
''')

# --------------------------------------------------------------------------- #
md("""
---
## 8. When to act, and when not to

From `research/segments.py`. The conditions were chosen on the **Premier League**,
2021-22 → 2023-24, with the old single-league model, and are re-measured here on the
ensemble's five-league pool: discovery window 2021-22 → 2023-24, confirmation 2024-25 →
2025-26.

**Favourable** (all three must hold): expected minutes 60–85, at least 21 prior
appearances, and a referee in the stricter three quarters.
**Unfavourable** (any one): a centre-back, the most lenient quarter of referees, or a
nailed-on 85+ minute player.
""")

code("""
for k in (1, 3):
    print(f"--- top-{k} pick ---")
    display(segmod.composite_policy(d, k=k).style.format(
        {"coverage": "{:.2f}", "strike": "{:.4f}", "fair_odds": "{:.2f}"}
    ).hide(axis="index"))
""")

code("""
pol = segmod.betting_policy(d, k=1)
top = pd.concat([pol.head(10), pol.tail(8)])
fig, ax = plt.subplots(figsize=(8, 6.5))
colors = ["#4c9f70" if v > pol["strike"].median() else "#c44e52"
          for v in top["strike"]]
labels = top["dim"].str.replace("seg_", "", regex=False) + " = " + top["level"]
ax.barh(labels, top["strike"], color=colors, alpha=.85)
ax.axvline(d[d["rank_in_match"] == 1][T].mean(), ls="--", c="#333", lw=1.2,
           label="overall top-1 strike")
ax.invert_yaxis()
ax.set_xlabel("strike rate of the top-1 pick")
ax.set_title("Best and worst conditions to take the model's pick")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md("""
Read this with the replication table above it, not on its own. On five leagues and the
ensemble the rule still points the right way, but it is **weaker than it was on the
Premier League**: for top-3 shortlists the favourable filter adds about +1.3pp in discovery and
+0.8pp in confirmation (it was +2.1 / +2.6), and the avoid list costs −0.9 / −0.5pp. A
filter chosen on one league transfers, but the ensemble already absorbs part of what it
exploited. Treat it as a mild top-3 tool, not a strategy.

Everything behind these numbers is in `research/segments.py`; `research/specialists.py`
contains the models that tried to exploit the same structure and failed.
""")

# --------------------------------------------------------------------------- #
md("""
---
## 9. Your turn

`d` is the full pool. Useful columns:

`p` — the ensemble · `p_bag` · `p_logistic` · `p_decomp` — its components · `p_m0`…`p_m9` — the bag
`league` · `season` · `match_id` · `team` · `opponent` · `referee` — for slicing
`rank_in_match` · `n_cards_in_match` · `p_sum_in_match` · `top3_cashed`
`resid` (= y − p) · `brier` · `logloss` — per-row error
`seg_*` — every slicing dimension from §8
plus all 26 model features, so you can look at *why* a prediction came out where it did.
""")

code("""
# scratch cell
d.columns.tolist()
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
out = Path(__file__).resolve().parent / "prediction_review.ipynb"
nbf.write(nb, out)
print(f"wrote {out}  ({len(C)} cells)")
