"""Generate player_confidence.ipynb - how much to trust each individual row.

Run:  python notebooks/build_confidence.py
then: jupyter nbconvert --to notebook --execute --inplace notebooks/player_confidence.ipynb
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
# Per-player confidence — how much to trust each individual number

## Two things called "confidence"

Conflating these is the usual way a probability gets misread, so they are kept apart
throughout.

**Outcome uncertainty — irreducible.** A player at 0.28 is *not* booked **72%**
of the time. A perfectly estimated probability still loses far more often than it wins at
these levels. A pick losing is therefore *not* evidence the model was wrong, and no amount
of modelling shrinks this. It is the single most important number on the page and the one
most likely to be forgotten.

**Estimate uncertainty — measurable.** How wrong the 0.28 itself could be. This is what
the rest of the notebook quantifies.

Every number here comes from the served model, the **five-league ensemble**
(`pipeline/ensemble.py`: 10 bagged boosters, a logistic and the team-count
decomposition, blended 0.4 / 0.4 / 0.2), and from its walk-forward pool: 183,100
out-of-sample rows, five leagues, 2021-22 → 2026-27.

## What was tested, and what failed

Three candidate sources of estimate uncertainty were tried. **Two of the three are null**,
and they are reported here rather than quietly dropped — a confidence score built on an
untested signal is worse than no confidence score, because it looks like evidence.

| source | verdict |
|---|---|
| Cohort interval — realised rate of similar past predictions | **works**, and is wide enough to matter |
| Segment reliability — thin history, position | **works** (ROC-AUC 0.536 vs 0.600) |
| Booster-vs-logistic disagreement | **null** — no trend across quintiles |
| Bag spread — the ensemble's 10 boosters, different seeds/subsamples | **null** — and narrow |

So the interval you should read is the **cohort interval**, and the grade rests only on
moderators that replicated.
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
import config, ensemble

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

DATE = "2026-09-20"
d = pd.read_csv(config.slate_dir(DATE) / "player_confidence.csv")
print(f"{len(d)} players across {d['fixture'].nunique()} fixtures")
print(d["grade"].value_counts().to_string())
""")

# --------------------------------------------------------------------------- #
md("""
## 1. The two null results

### 1a. Booster vs logistic

The cheapest candidate: where the two model families inside the ensemble disagree, the
row should be less certain. Bucketed into quintiles of |bag − logistic| across all
183,100 out-of-sample rows, it isn't: discrimination is flat across the quintiles. (On
the old Premier-League-only model the *most*-disagreeing quintile was the best; the
conclusion is the same.)
""")

code("""
from sklearn.metrics import roc_auc_score
pool = ensemble.load_pool()          # p = the served ensemble probability
pool["spread"] = (pool["p_bag"] - pool["p_logistic"]).abs()
pool["q"] = pd.qcut(pool["spread"], 5, labels=list("12345"))

rows = []
for lvl, g in pool.groupby("q", observed=True):
    y = g[config.TARGET_COL]
    rows.append({"quintile": lvl, "n": len(g), "mean_spread": g["spread"].mean(),
                 "roc_auc": roc_auc_score(y, g["p"]),
                 "realised": y.mean()})
pd.DataFrame(rows)
""")

md("""
### 1b. Bag spread

The more principled candidate: the ensemble's ten boosters differ only in seed and in
row/column subsampling, so their disagreement on a row is the fit's own uncertainty
about it. Validated on the walk-forward pool (`serve/player_confidence.py --validate`),
where every row was scored by members that never saw its season.

It fails. Discrimination does not fall as spread rises — it *rises*, from ROC-AUC 0.581
in the least-spread quintile of relative spread to 0.613 in the most. The spreads are
also tiny: member standard deviations run 0.003 to 0.009.
""")

code("""
v = pd.read_csv(config.RESEARCH_DIR / "confidence_validation.csv")
v[v["measure"] == "relative"]
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, meas in zip(axes, ["absolute", "relative"]):
    s = v[v["measure"] == meas]
    ax.plot(s["quintile"], s["roc_auc"], "o-")
    ax.axhline(0.60, ls="--", c="grey", label="ROC-AUC 0.60")
    ax.set_title(f"{meas} bag spread")
    ax.set_xlabel("spread quintile (5 = model least certain)")
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.50, 0.68); ax.legend(fontsize=8)
fig.suptitle("If spread measured unreliability, these lines would fall to the right")
plt.tight_layout(); plt.show()
""")

code("""
bag_w = (d["bag_hi"] - d["bag_lo"]).mean()
coh_w = (d["realised_hi"] - d["realised_lo"]).mean()
outside = ((d["p"] < d["bag_lo"]) | (d["p"] > d["bag_hi"])).mean()
print("how wide is each uncertainty source, in probability points?\\n")
print(f"  bag band     (5-95%)  {bag_w:.4f}")
print(f"  cohort band  (95%)    {coh_w:.4f}   (median cohort {int(d['cohort_n'].median()):,} rows)")
print(f"\\n  the served probability sits outside the bag band for {outside:.0%} of rows:")
print("  the logistic and the decomposition move it, so the band is not an interval on p")
""")

# --------------------------------------------------------------------------- #
md("""
Two things changed with the ensemble, and both show above. The **cohort interval is now
narrow**, because the pool behind it grew about fivefold (five leagues, six seasons), so it
is no longer much wider than the bag band. It is still the interval to read, because it is
an *empirical* check of the served number. And the **bag band is not an interval on `p`**:
`p` blends the bag with the logistic and the decomposition, so it often sits outside the
boosters' range.
""")

# --------------------------------------------------------------------------- #
md("""
## 2. Per-player confidence

`p` is the model's probability. `realised_lo`–`realised_hi` is the **interval to read**:
the match-clustered bootstrap range of the observed booking rate among historically
similar predictions. `breakeven` is `1 / price` — the probability the price demands.

**`clears_breakeven` is the column that matters**: the realised rate stays above what the
price needs even at the *bottom* of its interval.
""")

code("""
COLS = ["player", "team", "grade", "p", "p_not_booked", "realised_lo", "realised_hi",
        "breakeven", "price", "edge", "clears_breakeven"]
d.head(20)[COLS]
""")

code("""
top = d.head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(10.5, 7))
y = np.arange(len(top))
ax.hlines(y, top["realised_lo"], top["realised_hi"], lw=5, alpha=.35,
          color=["#2a7" if c else "#c33" for c in top["clears_breakeven"]],
          label="realised-rate interval")
ax.plot(top["p"], y, "D", ms=6, color="#036", label="model probability")
ax.plot(top["breakeven"], y, "|", ms=18, mew=2.5, color="k",
        label="break-even (1 / price)")
ax.set_yticks(y)
ax.set_yticklabels([f"{r.player}  @{r.price:g}  [{r.grade[0]}]"
                    for r in top.itertuples()])
ax.set_xlabel("probability of a card")
ax.set_title("Green = interval stays above break-even; black bar is what the price demands")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()
""")

md("""
### The grades

Built only from moderators that replicated:

- **C — thin history**: ≤5 prior appearances or unknown position. ROC-AUC 0.536 vs 0.600.
  Discount regardless of how large the edge looks.
- **B — qualified**: the referee has no history in the dataset (features fall back to the
  league prior), or the historical cohort behind the row is thin (<400 rows).
- **A — well supported**: neither applies.

Bag spread is *not* an input, for the reason in §1b. It is shown in the table so you can
see how little the fit actually wobbles.
""")

code("""
(d.groupby(["fixture", "grade"])
   .agg(n=("player", "size"), clears=("clears_breakeven", "sum"))
   .reset_index())
""")

code("""
d.groupby("grade").agg(n=("player", "size"),
                       mean_p=("p", "mean"),
                       mean_cohort_width=("realised_hi", "mean"),
                       clears=("clears_breakeven", "sum"))
""")

# --------------------------------------------------------------------------- #
md("""
## 3. How to read a row

Take the best-supported row on the slate and walk through it.

1. **The model says `p`.** That is a rate, not a prediction of this match.
2. **`p_not_booked` is what usually happens.** At these levels every single pick is odds-
   against. A run of losses is the expected experience, not a malfunction.
3. **The interval is `realised_lo`–`realised_hi`**, not the bag band. §1b showed the bag
   band covers only the boosters' wobble, and that `p` need not even sit inside it.
4. **Compare the interval's floor to `breakeven`.** If the floor clears it, the row
   survives the uncertainty that can actually be measured.
5. **Then read the grade**, which captures the uncertainty that *can't* be put in an
   interval — thin history, an unknown referee, a thin cohort.

What none of this can tell you: whether the market knows something the model doesn't.
There are no historical player-card prices, so that is untestable here. It is the largest
remaining risk and it is not in any interval on this page.
""")

code("""
r = d[d["grade"].str.startswith("A")].iloc[0]
print(f"{r['player']} ({r['team']}) @ {r['price']:g}\\n")
print(f"  model probability      {r['p']:.4f}")
print(f"  NOT booked             {r['p_not_booked']:.4f}   <- the usual outcome")
print(f"  realised interval      {r['realised_lo']:.4f} - {r['realised_hi']:.4f}"
      f"   (n={int(r['cohort_n'])} similar past rows)")
print(f"  bag band               {r['bag_lo']:.4f} - {r['bag_hi']:.4f}"
      f"   (fit wobble only - not the interval to read)")
print(f"  break-even at {r['price']:g}      {r['breakeven']:.4f}")
print(f"  clears break-even?     {bool(r['clears_breakeven'])}")
print(f"  grade                  {r['grade']}")
""")

code("""
# scratch
d.columns.tolist()
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
out = Path(__file__).resolve().parent / "player_confidence.ipynb"
nbf.write(nb, out)
print(f"wrote {out}  ({len(C)} cells)")
