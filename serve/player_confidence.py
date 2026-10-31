"""How much to trust each individual player's number.

Two different things get called "confidence" and conflating them is the usual
way a probability gets misread.

**Outcome uncertainty** is irreducible.  A perfectly estimated 0.28 still means
the player is *not* booked 72% of the time.  No amount of modelling shrinks it,
and it is the reason a 0.28 losing is not evidence the model was wrong.

**Estimate uncertainty** is how wrong the 0.28 itself could be.  That is what
this module measures, from three sources that are kept separate rather than
mashed into one number - an opaque 0-100 "confidence score" would hide exactly
the disagreements that matter:

1. **Cohort interval.**  The realised rate of historically similar predictions,
   with a match-clustered bootstrap interval (``slate_guidance.py``).  This is
   the empirical check - rows rated like this one were booked N% of the time -
   and it turns out to be the *only* one of the three wide enough to matter.
2. **Segment reliability.**  Measured discrimination in the row's own slice.
   ``segments.py`` showed this is far from uniform: thin-history rows run
   ROC-AUC 0.536 against 0.600 overall.
3. **Ensemble spread.**  The ten bagged boosters inside the shipped
   five-league ensemble (``ensemble.py``), differing only in seed and in
   row/column subsampling.  Reported because it quantifies how much the fit
   itself wobbles - but see below, it is not used for grading.

Everything here reads the shipped ensemble and its walk-forward pool: no
model is fitted at run time, and the probability graded is the same
``card_probability`` that ``card_market.py`` and ``todays_slate.py`` print.

Two signals that did NOT survive testing
----------------------------------------
Both obvious candidates for "the model is unsure about this row" are null, and
both are reported rather than quietly dropped.

**Booster-vs-logistic disagreement.**  Bucketed into quintiles over the 38,000
out-of-sample rows, ROC-AUC runs 0.599 / 0.587 / 0.610 / 0.592 / 0.612 - no
trend, and the *highest*-disagreement quintile is the best-discriminated.

**Ensemble spread.**  Same test, on the ensemble's walk-forward pool
(``--validate``: 183,100 rows, five leagues, 2021-22 -> 2026-27, every row
scored by members that never saw its season): across quintiles of relative
spread ROC-AUC runs 0.581 / 0.586 / 0.590 / 0.598 / 0.613 - *rising* with
spread rather than falling, as it did on the Premier-League-only model
(0.559 -> 0.631).  The spreads are also small: member standard deviations run
0.003 to 0.009.  Seed and subsampling luck is a minor share of the
uncertainty, and knowing a row's spread tells you nothing about whether the
model is right about it.

The practical consequence: **the interval you should read is the cohort
interval**, and the grade rests on thin history, referee history and cohort
size - the moderators that replicated.

Usage
-----
    python serve/player_confidence.py --validate       # does spread predict error?
    python serve/player_confidence.py --date 2026-09-20

The booster/logistic and segment figures above were measured on the
Premier-League-only model; the spread figures are the ensemble's own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import card_market, config, ensemble, pooled, predict, slate_guidance
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import card_market
    import config
    import ensemble
    import pooled
    import predict
    import slate_guidance


def validate(cfg=config) -> pd.DataFrame:
    """Does ensemble spread actually flag rows the model gets wrong?

    Read off the ensemble's walk-forward pool, where every row was scored by
    members that never saw its season.  If discrimination does not fall as
    spread rises, the signal is null and should not be graded on - the same
    bar the booster/logistic disagreement failed.
    """
    from sklearn.metrics import roc_auc_score

    te = ensemble.load_pool()
    mat = te.filter(regex=r"^p_m\d+$").to_numpy()
    te["spread"] = mat.std(axis=1)
    te["rel_spread"] = te["spread"] / te["p"].clip(lower=1e-6)
    print(f"validating on the walk-forward pool: {len(te):,} rows, "
          f"{te['match_id'].nunique():,} matches, {mat.shape[1]} members")

    y = cfg.TARGET_COL
    rows = []
    for name, col in (("absolute", "spread"), ("relative", "rel_spread")):
        te["q"] = pd.qcut(te[col], 5, labels=list("12345"))
        for lvl, g in te.groupby("q", observed=True):
            rows.append({
                "measure": name, "quintile": lvl, "n": len(g),
                "mean_spread": g[col].mean(),
                "mean_pred": g["p"].mean(),
                "realised": g[y].mean(),
                "bias_pp": 100 * (g[y].mean() - g["p"].mean()),
                "roc_auc": roc_auc_score(g[y], g["p"])
                if g[y].nunique() > 1 else np.nan,
                "brier": ((g[y] - g["p"]) ** 2).mean(),
            })
    return pd.DataFrame(rows)


def score_fixture(spec: dict, model) -> tuple[pd.DataFrame, str]:
    """The ensemble's view of one fixture, joined to its card prices."""
    o = spec["odds"]
    ou = spec.get("odds_ou")
    league = spec.get("league", pooled.PL)
    pred = predict.predict_fixture(
        home_team=spec["home"], away_team=spec["away"],
        kickoff=spec["kickoff"], referee=spec.get("referee"),
        home_xi=spec["home_xi"], away_xi=spec["away_xi"],
        home_formation=spec.get("home_formation"),
        away_formation=spec.get("away_formation"),
        matchweek=spec.get("matchweek"),
        odds=(o["home"], o["draw"], o["away"]),
        odds_ou=((ou["over"], ou["under"]) if ou else None),
        model=model, league=league,
    )
    cmp, _ = card_market.compare(pred, spec["card_odds"], spec.get("card_odds_no"))
    cmp["thin"] = card_market.flag_thin_history(cmp)
    return cmp, league


def grade(row: pd.Series) -> str:
    """A label built only from moderators that survived testing.

    Ensemble spread is deliberately NOT an input.  ``--validate`` shows it does
    not predict discrimination - across quintiles of relative spread on an
    unseen season, ROC-AUC runs 0.559 / 0.591 / 0.613 / 0.607 / 0.631, rising
    rather than falling.  It is reported in the table as an honest measure of
    how much the fit itself wobbles (very little: absolute spreads of
    0.004-0.011), but grading on it would be inventing a signal.

    What is left is the evidence that did replicate: thin history
    (ROC-AUC 0.536 against 0.600 overall, measured in ``segments.py`` and
    re-measured across seasons), whether the referee has any history at all,
    and whether the historical cohort behind the row is large enough to carry
    an interval.

    Deliberately coarse - three levels, because the evidence supports three.
    """
    if row["thin"]:
        return "C - thin history"
    base = "A" if row["ref_known"] else "B"
    if row["cohort_n"] < 400:
        base = "B" if base == "A" else "C"
    return {"A": "A - well supported",
            "B": "B - qualified",
            "C": "C - weak"}[base]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-09-20")
    ap.add_argument("--window", type=float, default=0.03)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.validate:
        v = validate()
        print(v.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        out = config.RESEARCH_DIR / "confidence_validation.csv"
        v.to_csv(out, index=False)
        print(f"\nwrote {out.name}")
        return

    model = predict.load_default_model()
    pool = slate_guidance.load_pool()
    known_refs: dict = {}

    recs = []
    for spec_path in sorted(config.slate_dir(args.date).glob("match_*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not spec.get("card_odds"):
            continue
        cmp, league = score_fixture(spec, model)
        if league not in known_refs:
            known_refs[league] = pd.DataFrame(
                {"referee": pooled.league_history(league)[0]["referee"].dropna().unique()})
        ref = predict.resolve_referee(spec.get("referee"), known_refs[league])
        ref_known = ref in set(known_refs[league]["referee"])

        for _, m in cmp.dropna(subset=["offered_odds"]).iterrows():
            p = float(m["card_probability"])
            coh = slate_guidance.cohort_rate(pool, p, args.window, str(m["pos_group"]))
            price = float(m["offered_odds"])
            recs.append({
                "fixture": f"{spec['home']} v {spec['away']}", "league": league,
                "player": m["player"], "team": m["team"],
                "pos_group": m["pos_group"], "thin": bool(m["thin"]),
                "ref_known": ref_known,
                "apps": m["plr_apps_prior"],
                "p": p,
                "bag_sd": float(m["bag_sd"]),
                "bag_lo": float(m["bag_lo"]), "bag_hi": float(m["bag_hi"]),
                "rel_spread": float(m["bag_sd"]) / max(p, 1e-6),
                "cohort_n": coh["cohort_n"],
                "realised": coh["realised"],
                "realised_lo": coh["realised_lo"],
                "realised_hi": coh["realised_hi"],
                "price": price,
                "breakeven": 1.0 / price,
                "edge": float(m["edge"]),
                # Outcome uncertainty, stated so it is never confused with the
                # interval: even a perfect estimate loses most of the time.
                "p_not_booked": 1.0 - p,
            })

    d = pd.DataFrame(recs)
    d["grade"] = d.apply(grade, axis=1)
    d["clears_breakeven"] = d["realised_lo"] > d["breakeven"]
    d = d.sort_values("edge", ascending=False).reset_index(drop=True)

    out = config.slate_dir(args.date) / "player_confidence.csv"
    d.to_csv(out, index=False)

    cols = ["player", "team", "p", "bag_lo", "bag_hi", "rel_spread",
            "realised_lo", "realised_hi", "breakeven", "price",
            "grade", "clears_breakeven"]
    print("\n=== per-player confidence (top 15 by edge) ===")
    print(d.head(15)[cols].to_string(index=False,
                                     float_format=lambda v: f"{v:.4f}"))
    print(f"\ngrades: {d['grade'].value_counts().to_dict()}")
    print(f"clearing break-even at the lower bound: "
          f"{int(d['clears_breakeven'].sum())}/{len(d)}")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
