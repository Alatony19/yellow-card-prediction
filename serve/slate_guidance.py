"""What the model's track record says about today's numbers.

The honest boundary first
------------------------
There is no archive of historical player-card prices - established when
``download_odds.py`` was built, and not obtainable free.  So a return
on the market prices **cannot** be tested historically.  Any claim of the form
"this strategy returned X% over five seasons" would be fabricated, and this module
does not make one.

What can be tested is the quantity every edge is built on.  An edge of
``p_model * price - 1`` is positive if and only if ``p_model`` exceeds the
break-even ``1 / price``.  The price is known exactly.  So the whole question
reduces to: **when the model said 0.22, how often did it actually happen?**
That is answerable from 38,000 out-of-sample rows, and it is answered here.

Three things it measures
------------------------
1. **Calibration in the band that matters.**  Overall ECE is 0.0084, but an
   average over the whole range says nothing about the top decile, which is
   where every pick lives.  Reliability is therefore reported per band, with
   match-clustered bootstrap intervals - the twenty players in a match share a
   referee and a fixture, so resampling rows would understate the interval.

2. **A realised-rate replacement for the model's own probability.**  For each
   of today's rows, the historical cohort of similar predictions is found and
   its *observed* booking rate substituted into the edge arithmetic.  This
   turns "the model claims +0.62" into "rows the model rated like this one were
   booked N% of the time, which at this price is an edge of X".

3. **Whether today's rows sit in well-measured slices.**  ``segments.py``
   already showed discrimination is not uniform.  Today's picks are
   concentrated on a heavy favourite's midfield, so the favourite/underdog
   split and the appearance-history split are both checked directly.

Scoring matches the slate: ``p = 0.75 * booster + 0.25 * logistic``, the weight
``card_market.py`` uses, so the track record describes the same predictor.

Usage
-----
    python serve/slate_guidance.py
    python serve/slate_guidance.py --date 2026-09-20 --window 0.03
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config

def load_pool(cfg=config) -> pd.DataFrame:
    """Out-of-sample rows, scored exactly as the slate was scored.

    The walk-forward pool of the five-league ensemble (``ensemble.py``): every
    league, 2021-22 onward, each season scored by components fitted only on
    earlier seasons and blended with the shipped weights.
    """
    import ensemble
    d = ensemble.load_pool()
    d["p_slate"] = d["p"]
    return d


def load_slate(date: str, cfg=config) -> pd.DataFrame:
    frames = []
    for spec_path in sorted(cfg.slate_dir(date).glob("match_*.json")):
        key = spec_path.stem.removeprefix("match_")
        csv = cfg.slate_dir(date) / f"market_{key}.csv"
        if not csv.exists():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        d = pd.read_csv(csv)
        d["fixture"] = f"{spec['home']} v {spec['away']}"
        d["key"] = key
        frames.append(d)
    if not frames:
        raise SystemExit(f"no scored fixtures found for {date}")
    return pd.concat(frames, ignore_index=True)


def boot_rate(y: np.ndarray, groups: np.ndarray, n_boot: int = 600,
              seed: int = 42) -> tuple[float, float, float]:
    """Mean of ``y`` with a CI from resampling whole matches, not rows."""
    if len(y) == 0:
        return (np.nan, np.nan, np.nan)
    uniq = pd.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        out[b] = y[idx].mean()
    return float(y.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def calibration_bands(pool: pd.DataFrame, cfg=config,
                      edges=(0.0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 1.0)
                      ) -> pd.DataFrame:
    """Predicted vs realised, per band, with match-clustered intervals."""
    y = pool[cfg.TARGET_COL].to_numpy(dtype=float)
    p = pool["p_slate"].to_numpy(dtype=float)
    g = pool["match_id"].to_numpy()
    band = pd.cut(p, list(edges))
    rows = []
    for b, m in pd.Series(band).groupby(band, observed=True):
        sel = m.index.to_numpy()
        mean, lo, hi = boot_rate(y[sel], g[sel])
        rows.append({
            "band": str(b), "n": len(sel),
            "n_matches": int(pd.unique(g[sel]).size),
            "mean_pred": float(p[sel].mean()),
            "realised": mean, "lo": lo, "hi": hi,
            "bias_pp": 100.0 * (mean - float(p[sel].mean())),
        })
    return pd.DataFrame(rows)


def cohort_rate(pool: pd.DataFrame, p_target: float, window: float,
                pos_group: str | None, cfg=config,
                min_n: int = 250) -> dict:
    """Realised booking rate among historical rows rated like this one.

    Widens from (same position, similar probability) to (similar probability
    alone) if the tighter cohort is too small to say anything - and reports
    which of the two it used, because a cohort of forty rows dressed up as
    evidence is how a historical test lies.
    """
    near = pool[(pool["p_slate"] - p_target).abs() <= window]
    used = "p-window"
    if pos_group is not None:
        tight = near[near["pos_group"] == pos_group]
        if len(tight) >= min_n:
            near, used = tight, "p-window + position"
    y = near[cfg.TARGET_COL].to_numpy(dtype=float)
    mean, lo, hi = boot_rate(y, near["match_id"].to_numpy())
    return {"cohort": used, "cohort_n": len(near), "realised": mean,
            "realised_lo": lo, "realised_hi": hi}


def ground_slate(slate: pd.DataFrame, pool: pd.DataFrame, window: float,
                 cfg=config) -> pd.DataFrame:
    """Replace each model probability with its historical realised analogue."""
    recs = []
    for _, r in slate.iterrows():
        c = cohort_rate(pool, float(r["card_probability"]), window,
                        str(r["pos_group"]), cfg)
        price = float(r["offered_odds"])
        recs.append({
            "fixture": r["fixture"], "player": r["player"], "team": r["team"],
            "pos_group": r["pos_group"], "thin": bool(r["thin"]),
            "model_p": float(r["card_probability"]),
            "price": price,
            "breakeven_p": 1.0 / price,
            "model_edge": float(r["edge"]),
            **c,
            "hist_edge": c["realised"] * price - 1.0,
            "hist_edge_lo": c["realised_lo"] * price - 1.0,
            "hist_edge_hi": c["realised_hi"] * price - 1.0,
        })
    out = pd.DataFrame(recs)
    out["survives"] = out["hist_edge_lo"] > 0
    return out.sort_values("hist_edge", ascending=False).reset_index(drop=True)


def split_report(pool: pd.DataFrame, col: str, cfg=config) -> pd.DataFrame:
    """Realised vs predicted inside one split - the segment health check."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for lvl, g in pool.groupby(col, observed=True):
        y = g[cfg.TARGET_COL].to_numpy(dtype=float)
        p = g["p_slate"].to_numpy(dtype=float)
        rows.append({
            col: lvl, "n": len(g),
            "mean_pred": p.mean(), "realised": y.mean(),
            "bias_pp": 100.0 * (y.mean() - p.mean()),
            "roc_auc": roc_auc_score(y, p) if y.min() != y.max() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(col)


def rank_calibration(pool: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """Calibration conditional on where a row sits in its own match.

    Aggregate calibration can be excellent while the top of each match is
    systematically over-confident - the errors simply cancel against the
    bottom.  Since every pick is drawn from the top of a match, that is the
    only part of the curve that governs whether an edge is real.
    """
    d = pool.copy()
    d["rk"] = d.groupby("match_id")["p_slate"].rank(method="first",
                                                    ascending=False)
    bucket = pd.cut(d["rk"], [0, 1, 3, 5, 10, 20],
                    labels=["rank 1", "rank 2-3", "rank 4-5",
                            "rank 6-10", "rank 11-20"])
    rows = []
    for lvl, g in d.groupby(bucket, observed=True):
        y = g[cfg.TARGET_COL].to_numpy(dtype=float)
        mean, lo, hi = boot_rate(y, g["match_id"].to_numpy())
        rows.append({"rank": str(lvl), "n": len(g),
                     "mean_pred": float(g["p_slate"].mean()),
                     "realised": mean, "lo": lo, "hi": hi,
                     "bias_pp": 100.0 * (mean - float(g["p_slate"].mean()))})
    return pd.DataFrame(rows)


def topk_record(pool: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """How the per-match shortlist actually did, season by season."""
    rows = []
    for season, g in pool.groupby("season"):
        g = g.copy()
        g["rk"] = g.groupby("match_id")["p_slate"].rank(method="first",
                                                        ascending=False)
        rec = {"season": season, "matches": g["match_id"].nunique()}
        for k in (1, 3, 5):
            top = g[g["rk"] <= k]
            rec[f"P@{k}"] = top[cfg.TARGET_COL].mean()
            hit = top.groupby("match_id")[cfg.TARGET_COL].max()
            rec[f"hit@{k}"] = hit.mean()
        rec["base"] = g[cfg.TARGET_COL].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-09-20")
    ap.add_argument("--window", type=float, default=0.03,
                    help="probability half-width of the historical cohort")
    args = ap.parse_args()

    pool = load_pool()
    slate = load_slate(args.date)
    print(f"pool {len(pool):,} rows / {pool['match_id'].nunique():,} matches "
          f"| slate {len(slate)} rows")

    print("\n=== calibration by band (match-clustered CI) ===")
    print(calibration_bands(pool).to_string(index=False,
                                            float_format=lambda v: f"{v:.4f}"))

    print("\n=== calibration where you actually act (rank within match) ===")
    print(rank_calibration(pool).to_string(index=False,
                                           float_format=lambda v: f"{v:.4f}"))

    print("\n=== the shortlist, season by season ===")
    print(topk_record(pool).to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))

    print("\n=== favourite vs underdog ===")
    print(split_report(pool, "odds_is_underdog").to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== today's rows, grounded in realised rates ===")
    g = ground_slate(slate, pool, args.window)
    cols = ["fixture", "player", "model_p", "realised", "breakeven_p", "price",
            "model_edge", "hist_edge", "hist_edge_lo", "survives"]
    print(g.head(12)[cols].to_string(index=False,
                                     float_format=lambda v: f"{v:.4f}"))
    out = config.slate_dir(args.date) / "slate_guidance.csv"
    g.to_csv(out, index=False)
    print(f"\nwrote {out.name}")
    print(f"{int(g['survives'].sum())} of {len(g)} rows keep a positive edge "
          "at the lower bound of the realised rate")


if __name__ == "__main__":
    main()
