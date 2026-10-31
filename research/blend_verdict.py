"""Should ``BLEND_WITH_LOGISTIC`` ship as the default?

The question has been open for three sessions on the strength of one number:
averaging the booster with a logistic regression on the same features scored
PR-AUC 0.2295 against the plain booster's 0.2285 in walk-forward.  A 0.001
difference read off a single pooled number is not a decision - it is a coin
flip with a decimal point.

This script settles it the way ``odds_verdict.py`` settled the closing-price
question, and for the same reason: the two configurations are scored on exactly
the same matches, so the comparison must be *paired*.  Season-to-season spread
is dominated by how hard each season was, and judging a 0.001 gap against a
0.04 spread throws away the pairing that makes the gap measurable at all.

Three things are different here from the odds verdict:

1. **One fit pass, many weights.**  The booster and the logistic are fitted
   once per season and then averaged at w = 0, 0.25, 0.5, 0.75, 1.0, where w is
   the weight on the *logistic*.  Refitting per weight would cost five times as
   much and answer a strictly smaller question - and the sweep is what shows
   whether 0.5 is a peak or just a point on a slope.
2. **The baseline is the shipped model** (w = 0), with the shipped feature set,
   odds profile included.  The 0.2295 figure predates the odds features; the
   blend has never been re-judged on the model that actually ships.
3. **Calibration is reported as a first-class metric.**  A linear floor pulls
   predictions towards the base rate, which mechanically flatters ECE while
   doing nothing for ranking.  If the blend's only win is ECE, that is an
   argument for a pricing profile, not for a new default.

Usage
-----
    python research/blend_verdict.py                  # full sweep + bootstrap at w=0.5
    python research/blend_verdict.py --n-boot 2000
    python research/blend_verdict.py --weights 0 0.5  # narrower sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, evaluation, modelling
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import evaluation
    import modelling

DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

# The fitted component predictions, cached so a second question about the
# weight does not pay for a second walk-forward.  The fits are deterministic
# given the dataset, so re-reading them is equivalent to re-running them.
POOL_PATH = config.PL_DIR / "blend_components.parquet"


def walk_forward_components(ds: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """Score every walk-forward season once, keeping both components.

    Returns the pooled out-of-sample rows with ``p_xgb`` and ``p_log`` side by
    side.  Every blend weight is then a column arithmetic away, which is the
    whole point: the two models are fitted on identical folds, so any
    difference between weights is the weighting and nothing else.
    """
    frames = []
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
        logit = modelling.fit_logistic(tr, cfg)
        te["p_xgb"] = model.predict_proba(te)
        te["p_log"] = modelling.logistic_predict(logit, te, cfg)
        frames.append(te[["match_id", "season", cfg.TARGET_COL,
                          "p_xgb", "p_log"]])
        print(f"  {season}: {len(te):,} rows "
              f"(train {train_seasons[0]}..{train_seasons[-1]})")
    return pd.concat(frames, ignore_index=True)


def blended(df: pd.DataFrame, w: float) -> np.ndarray:
    """Weight ``w`` on the logistic, ``1 - w`` on the booster."""
    return (1.0 - w) * df["p_xgb"].to_numpy() + w * df["p_log"].to_numpy()


def per_season_table(pool: pd.DataFrame, weights) -> pd.DataFrame:
    """Metrics for each (season, weight) pair - the substrate for the pairing."""
    rows = []
    for w in weights:
        col = f"p_w{w}"
        pool[col] = blended(pool, w)
        for season, g in pool.groupby("season", sort=True):
            rec = evaluation.evaluate(g, col, config.TARGET_COL,
                                      name=season, k_values=(1, 3))
            rec["w"] = w
            rows.append(rec)
    return pd.DataFrame(rows)


def precision_at_k(df: pd.DataFrame, pcol: str, k: int) -> float:
    top = (df.sort_values(pcol, ascending=False)
             .groupby("match_id", sort=False)
             .head(k))
    return float(top[config.TARGET_COL].mean())


def pr_auc_lift(df: pd.DataFrame, pcol: str) -> float:
    from sklearn.metrics import average_precision_score
    y = df[config.TARGET_COL].to_numpy()
    return float(average_precision_score(y, df[pcol].to_numpy()) / y.mean())


def bootstrap_paired(pool: pd.DataFrame, w: float, n_boot: int,
                     seed: int = 42) -> dict:
    """Resample MATCHES, recompute the w-vs-0 difference, return its CI.

    Matches are the resampling unit because the twenty players in one match
    share a referee, a fixture and a price - resampling rows individually would
    treat those twenty as independent and understate the interval badly.
    """
    base_col, var_col = "p_w0.0", f"p_w{w}"
    if base_col not in pool:
        pool[base_col] = blended(pool, 0.0)
    if var_col not in pool:
        pool[var_col] = blended(pool, w)

    matches = pool["match_id"].unique()
    idx_by_match = {m: g.index.to_numpy()
                    for m, g in pool.groupby("match_id", sort=False)}
    sizes = {m: len(i) for m, i in idx_by_match.items()}
    rng = np.random.default_rng(seed)

    d_lift, d_p1, d_p3 = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(matches, size=len(matches), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in pick])
        s = pool.loc[idx].copy()
        # A match drawn twice must stay two separate twenty-player fields, or
        # precision@k would rank forty players against each other.
        s["match_id"] = np.repeat(np.arange(len(pick)),
                                  [sizes[m] for m in pick])
        d_lift.append(pr_auc_lift(s, var_col) - pr_auc_lift(s, base_col))
        d_p1.append(precision_at_k(s, var_col, 1) - precision_at_k(s, base_col, 1))
        d_p3.append(precision_at_k(s, var_col, 3) - precision_at_k(s, base_col, 3))

    def ci(v):
        v = np.asarray(v)
        return (float(v.mean()), float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)), float((v > 0).mean()))

    return {"lift": ci(d_lift), "p@1": ci(d_p1), "p@3": ci(d_p3)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS),
                    help="weight on the logistic; 0 = shipped booster")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-weight", type=float, nargs="+", default=[0.5],
                    help="which weight(s) to bootstrap against w=0")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the cached component predictions if present")
    args = ap.parse_args()

    print(f"features: {len(config.FEATURES)}  "
          f"(odds profile: {config.ODDS_PROFILE})")
    if args.reuse and POOL_PATH.exists():
        pool = pd.read_parquet(POOL_PATH)
        print(f"reusing {POOL_PATH.name} ({len(pool):,} rows)")
    else:
        ds = pd.read_parquet(config.MODEL_DATASET_PATH)
        print("fitting walk-forward components ...")
        pool = walk_forward_components(ds)
        pool.to_parquet(POOL_PATH, index=False)

    tab = per_season_table(pool, args.weights)

    # ---- 1. pooled sweep --------------------------------------------------- #
    print("\n=== pooled across all walk-forward seasons ===")
    pooled = []
    for w in args.weights:
        rec = evaluation.evaluate(pool, f"p_w{w}", config.TARGET_COL,
                                  name=f"w={w}", k_values=(1, 3))
        pooled.append(rec)
    pooled = pd.DataFrame(pooled)[
        ["model", "pr_auc_lift", "roc_auc", "log_loss", "ece",
         "precision@1", "precision@3"]]
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 2. paired by season ---------------------------------------------- #
    base = tab[tab["w"] == 0.0].set_index("model")
    print("\n=== paired by season: each weight minus w=0 (the shipped model) ===")
    for w in args.weights:
        if w == 0.0:
            continue
        var = tab[tab["w"] == w].set_index("model")
        cmp = pd.DataFrame({
            "season": base.index,
            "d_lift": (var["pr_auc_lift"] - base["pr_auc_lift"]).values,
            "d_p3": (var["precision@3"] - base["precision@3"]).values,
            "d_ece": (var["ece"] - base["ece"]).values,
        })
        print(f"\n  w = {w}")
        print(cmp.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        for col, better in [("d_lift", "higher"), ("d_p3", "higher"),
                            ("d_ece", "lower")]:
            wins = ((cmp[col] > 0) if better == "higher" else (cmp[col] < 0)).sum()
            print(f"    mean {col} {cmp[col].mean():+.4f}   "
                  f"{wins}/{len(cmp)} seasons better")

    # ---- 3. match-level bootstrap ----------------------------------------- #
    for w in args.boot_weight:
        print(f"\n=== match-level bootstrap, w={w} vs w=0 "
              f"({args.n_boot} resamples) ===")
        res = bootstrap_paired(pool, w, args.n_boot)
        for metric, (mean, lo, hi, pos) in res.items():
            verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI spans 0"
            print(f"  d_{metric:<5} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"P(>0) {pos:.2f}   {verdict}")

    out = config.RESEARCH_DIR / "blend_verdict.csv"
    tab.to_csv(out, index=False)
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
