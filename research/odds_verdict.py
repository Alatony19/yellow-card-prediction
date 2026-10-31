"""Is the closing-price gain real, or is it the fourth null result in a row?

Why a separate script
---------------------
``candidate_ablation.py`` compares the *mean* of five season lifts against the
*standard deviation* of those five numbers.  That is the right first look and
the wrong final test.  Season-to-season spread is dominated by how hard each
season was - a year with more red cards moves every variant together - so
judging a 0.004 difference against a 0.042 spread throws away the fact that the
two variants were scored on exactly the same matches.

The paired view removes the season effect: for each season, lift(variant) minus
lift(baseline).  If a feature helps a little but helps *every year*, the paired
difference is consistently positive while the unpaired comparison calls it
noise.  That is precisely the shape "+ odds: tightness" showed.

Two tests are run:

1. **Paired by season.**  Five differences, their mean, and how many are
   positive.  Five points is too few for a t-test to mean much, so the sign
   count is reported alongside rather than a p-value dressed up as proof.
2. **Match-level bootstrap.**  The pooled out-of-sample rows are resampled by
   *match*, not by row - the twenty players in a match share a referee, a
   fixture and a price, so rows inside a match are not independent and
   resampling them individually would understate the interval badly.  This is
   the same correction ``segments.py`` uses.

Usage
-----
    python research/odds_verdict.py                 # tightness vs baseline
    python research/odds_verdict.py --variant side
    python research/odds_verdict.py --n-boot 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import candidate_ablation, config, evaluation, modelling
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import candidate_ablation
    import config
    import evaluation
    import modelling

VARIANTS = {
    "tightness": candidate_ablation.ODDS_GROUPS["odds: tightness"],
    "side": candidate_ablation.ODDS_GROUPS["odds: side"],
    "all": config.ODDS_CANDIDATES,
}


def walk_forward_predictions(ds: pd.DataFrame, cfg, blend: bool = True):
    """Per-season metrics AND the scored rows, so the two can be paired."""
    rows, frames = [], []
    for season in cfg.WALK_FORWARD_SEASONS:
        past = [s for s in cfg.ALL_SEASONS if s < season]
        if len(past) < 3:
            continue
        valid_season = past[-1]
        train_seasons = [s for s in past if s != valid_season
                         and s not in cfg.BURN_IN_SEASONS]
        tr = ds[ds["season"].isin(train_seasons)]
        va = ds[ds["season"] == valid_season]
        te = ds[ds["season"] == season].copy()
        if not (len(tr) and len(va) and len(te)):
            continue
        model = modelling.fit_model(tr, va, cfg, verbose=False)
        te["p"] = model.predict_proba(te)
        if blend:
            logit = modelling.fit_logistic(tr, cfg)
            te["p"] = 0.5 * te["p"] + 0.5 * modelling.logistic_predict(logit, te, cfg)
        rows.append(evaluation.evaluate(te, "p", cfg.TARGET_COL,
                                        name=season, k_values=(1, 3)))
        frames.append(te[["match_id", "season", config.TARGET_COL, "p"]])
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def precision_at_k(df: pd.DataFrame, pcol: str, k: int) -> float:
    """Share of the top-k picks per match that were booked."""
    top = (df.sort_values(pcol, ascending=False)
             .groupby("match_id", sort=False)
             .head(k))
    return float(top[config.TARGET_COL].mean())


def pr_auc_lift(df: pd.DataFrame, pcol: str) -> float:
    from sklearn.metrics import average_precision_score
    y = df[config.TARGET_COL].to_numpy()
    ap = average_precision_score(y, df[pcol].to_numpy())
    return float(ap / y.mean())


def bootstrap_paired(base: pd.DataFrame, var: pd.DataFrame, n_boot: int,
                     seed: int = 42) -> dict:
    """Resample MATCHES, recompute both metrics, return the difference's CI."""
    merged = base.merge(var[["match_id", "p"]].rename(columns={"p": "p_var"}),
                        left_index=True, right_index=True,
                        suffixes=("", "_v"))
    assert (merged["match_id"] == var["match_id"]).all(), "row order diverged"

    matches = merged["match_id"].unique()
    idx_by_match = {m: g.index.to_numpy()
                    for m, g in merged.groupby("match_id", sort=False)}
    rng = np.random.default_rng(seed)

    d_lift, d_p1, d_p3 = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(matches, size=len(matches), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in pick])
        s = merged.loc[idx].copy()
        # A match drawn twice must stay two separate twenty-player fields,
        # or precision@k would rank forty players against each other.
        s["match_id"] = np.repeat(np.arange(len(pick)),
                                  [len(idx_by_match[m]) for m in pick])
        d_lift.append(pr_auc_lift(s, "p_var") - pr_auc_lift(s, "p"))
        d_p1.append(precision_at_k(s, "p_var", 1) - precision_at_k(s, "p", 1))
        d_p3.append(precision_at_k(s, "p_var", 3) - precision_at_k(s, "p", 3))

    def ci(v):
        v = np.asarray(v)
        return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float((v > 0).mean())

    return {"lift": ci(d_lift), "p@1": ci(d_p1), "p@3": ci(d_p3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="tightness", choices=list(VARIANTS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-blend", action="store_true")
    args = ap.parse_args()

    ds = pd.read_parquet(config.MODEL_DATASET_PATH)
    blend = not args.no_blend

    print("fitting baseline ...")
    base_m, base_p = walk_forward_predictions(
        ds, candidate_ablation.cfg_with([]), blend=blend)
    print(f"fitting + odds: {args.variant} ...")
    var_m, var_p = walk_forward_predictions(
        ds, candidate_ablation.cfg_with(VARIANTS[args.variant]), blend=blend)

    # ---- 1. paired by season ---------------------------------------------- #
    print(f"\n=== paired by season: +odds:{args.variant} minus baseline ===")
    cmp = pd.DataFrame({
        "season": base_m["model"].values,
        "base_lift": base_m["pr_auc_lift"].values,
        "var_lift": var_m["pr_auc_lift"].values,
        "base_p3": base_m["precision@3"].values,
        "var_p3": var_m["precision@3"].values,
        "base_ece": base_m["ece"].values,
        "var_ece": var_m["ece"].values,
    })
    cmp["d_lift"] = cmp["var_lift"] - cmp["base_lift"]
    cmp["d_p3"] = cmp["var_p3"] - cmp["base_p3"]
    cmp["d_ece"] = cmp["var_ece"] - cmp["base_ece"]
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for col, better in [("d_lift", "higher"), ("d_p3", "higher"), ("d_ece", "lower")]:
        wins = (cmp[col] > 0).sum() if better == "higher" else (cmp[col] < 0).sum()
        print(f"  mean {col} {cmp[col].mean():+.4f}   "
              f"{wins}/{len(cmp)} seasons better")

    # ---- 2. match-level bootstrap ----------------------------------------- #
    print(f"\n=== match-level bootstrap ({args.n_boot} resamples) ===")
    res = bootstrap_paired(base_p, var_p, args.n_boot)
    for metric, (mean, lo, hi, pos) in res.items():
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI spans 0"
        print(f"  d_{metric:<5} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"P(>0) {pos:.2f}   {verdict}")

    out = config.RESEARCH_DIR / f"odds_verdict_{args.variant}.csv"
    cmp.to_csv(out, index=False)
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
