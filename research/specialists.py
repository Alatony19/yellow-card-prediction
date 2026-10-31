"""Specialised models for the segments where the global model misbehaves.

What the segment analysis actually found
----------------------------------------
*Discrimination is close to flat.*  Bootstrapped over matches, every slice's
ROC-AUC confidence interval overlaps the overall interval [0.592, 0.607] with
exactly one exception: players with five or fewer prior appearances, where the
model scores 0.545 - and those players carry the *highest* base rate in the
data (0.210).  The model is close to blind on exactly the rows it should be
most alert on.

*Calibration is not flat at all, and the pattern replicates.*  Measured as a
relative level error (slice mean(p)/mean(y) against the same ratio for the whole
window), and confirmed on two seasons it was never chosen on:

    away        -4.1%      home          +4.7%
    wingers     -5.7%      full-backs    +3.8%
    under-23    -5.3%      150+ apps     +4.1%
    hardest 25% -5.3%      easiest 25%   +3.9%
    weeks 1-10  -4.7%      expected 85m+ +4.3%

That shape suggests the specialisation should not be "one booster per slice"
(there is no ranking difference to exploit and splitting the data costs more
than it buys).  It should be a **correction layer on the levels**, which does
also move the ranking, because segment membership varies *within* a match - a
home centre-back and an away winger sit in the same 20-row field.

Both hypotheses are tested here rather than assumed.

Honesty protocol
----------------
A correction layer must be fitted on predictions the base model did not see in
training, or it learns the base model's in-sample optimism.  So for season S:

  base model      trained on seasons < S    (as in modelling.walk_forward)
  correction      fitted on the *out-of-sample* predictions for seasons < S
                  that came from their own earlier-trained models

which restricts the comparison to 2022-23 onwards, since 2021-22 is the first
season with out-of-sample predictions at all.  Four evaluation seasons, and
nothing in the correction has seen the season it is scoring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, evaluation, modelling, oos as oos_mod, segments as segmod
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import evaluation
    import modelling
    import oos as oos_mod
    import segments as segmod

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Seasons that can be scored: they need at least one earlier season of
# out-of-sample predictions to fit a correction on.
EVAL_SEASONS = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]

# The dimensions the correction layer is allowed to use.  Deliberately the ones
# whose relative level error replicated across both windows, plus position -
# and deliberately NOT seg_confidence or seg_player_rate, which are functions of
# the prediction itself and would just re-fit it.
CORRECTION_DIMS = [
    "seg_venue", "seg_pos", "seg_age", "seg_minutes", "seg_history",
    "seg_difficulty", "seg_stage", "seg_fixture_temper", "seg_referee",
]


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# correction layers
# --------------------------------------------------------------------------- #
class PlattCorrection:
    """Intercept + slope on the log-odds.  The level-drift control.

    This is the honest baseline for every segment correction below: the league
    card rate drifts upward season to season, so *some* of any segment gain
    would be available from a plain global rescale.  Anything that does not beat
    this has found nothing.
    """
    name = "recal_platt"

    def fit(self, d: pd.DataFrame, prob_col: str, cfg=config):
        x = _logit(d[prob_col]).reshape(-1, 1)
        self.m = LogisticRegression(max_iter=1000)
        self.m.fit(x, d[cfg.TARGET_COL].to_numpy())
        return self

    def predict(self, d: pd.DataFrame, prob_col: str):
        return self.m.predict_proba(_logit(d[prob_col]).reshape(-1, 1))[:, 1]


class SegmentCorrection:
    """Platt, plus an additive log-odds offset per segment level.

    One coefficient per (dimension, level).  Additive in log-odds means the
    correction is multiplicative on the odds, which is the right functional form
    for "this group is systematically 5% under-called" - it scales rather than
    shifts, so it cannot push a 0.02 player to a negative probability.

    ``interact`` additionally lets each dimension have its own slope on the base
    log-odds, i.e. "the model's confidence means something different for a
    debutant than for a 200-appearance veteran".
    """

    def __init__(self, dims=None, interact: bool = False, C: float = 1.0):
        self.dims = list(dims or CORRECTION_DIMS)
        self.interact = interact
        self.C = C
        self.name = "recal_segment_x" if interact else "recal_segment"

    def _design(self, d: pd.DataFrame, prob_col: str, fit: bool):
        z = _logit(d[prob_col]).reshape(-1, 1)
        cats = d[self.dims].astype(str)
        if fit:
            self.enc = OneHotEncoder(handle_unknown="ignore", drop="first",
                                     sparse_output=False, min_frequency=200)
            dum = self.enc.fit_transform(cats)
        else:
            dum = self.enc.transform(cats)
        parts = [z, dum]
        if self.interact:
            parts.append(dum * z)
        return np.hstack(parts)

    def fit(self, d: pd.DataFrame, prob_col: str, cfg=config):
        x = self._design(d, prob_col, fit=True)
        # Regularised, because a rare level with 200 rows will otherwise be
        # handed a coefficient it has not earned.
        self.m = LogisticRegression(max_iter=3000, C=self.C)
        self.m.fit(x, d[cfg.TARGET_COL].to_numpy())
        return self

    def predict(self, d: pd.DataFrame, prob_col: str):
        return self.m.predict_proba(self._design(d, prob_col, fit=False))[:, 1]


class StackCorrection:
    """A small booster on the base log-odds plus the raw segment drivers.

    Strictly more expressive than SegmentCorrection - it can find thresholds and
    interactions the one-hot version cannot.  Kept shallow and heavily
    regularised because it is fitted on one to four seasons of held-out
    predictions, not on the full training set.
    """
    name = "stack_gbm"

    RAW = ["plr_apps_prior", "plr_mins_roll", "plr_age_years", "is_home",
           "matchweek", "ref_cards_pm_eb", "ctx_match_difficulty",
           "ctx_fixture_cards_pm_eb", "plr_cards_p90_eb"]

    def _design(self, d: pd.DataFrame, prob_col: str):
        x = d[self.RAW].copy()
        x["_z"] = _logit(d[prob_col])
        return x

    def fit(self, d: pd.DataFrame, prob_col: str, cfg=config):
        self.m = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=config.RANDOM_STATE, n_jobs=-1)
        self.m.fit(self._design(d, prob_col), d[cfg.TARGET_COL].to_numpy())
        return self

    def predict(self, d: pd.DataFrame, prob_col: str):
        return self.m.predict_proba(self._design(d, prob_col))[:, 1]


# --------------------------------------------------------------------------- #
# per-segment boosters (the other hypothesis)
# --------------------------------------------------------------------------- #
def fit_partitioned(train: pd.DataFrame, valid: pd.DataFrame, dim: str,
                    cfg=config, min_rows: int = 3000) -> dict:
    """One full booster per level of ``dim``; levels too thin fall back to global.

    ``min_rows`` is not a nicety.  A level with 1,500 training rows and a 15%
    base rate has ~225 positives, which is not enough to fit 24 features on -
    the specialist would be worse than the generalist purely through variance.
    """
    models = {"__global__": modelling.fit_model(train, valid, cfg, verbose=False)}
    for level, sub in train.groupby(dim, sort=False):
        if len(sub) < min_rows:
            continue
        vsub = valid[valid[dim] == level]
        if len(vsub) < 500:
            continue
        models[str(level)] = modelling.fit_model(sub, vsub, cfg, verbose=False)
    return models


def predict_partitioned(models: dict, test: pd.DataFrame, dim: str) -> np.ndarray:
    out = np.empty(len(test), dtype=float)
    lvl = test[dim].astype(str).to_numpy()
    for i, (level, idx) in enumerate(pd.Series(range(len(test))).groupby(lvl)):
        m = models.get(level, models["__global__"])
        out[idx.to_numpy()] = m.predict_proba(test.iloc[idx.to_numpy()])
    return out


# --------------------------------------------------------------------------- #
# the walk-forward comparison
# --------------------------------------------------------------------------- #
def _ev(t: pd.DataFrame, col: str, cfg=config) -> dict:
    """evaluation.evaluate without its own ``model`` key, so callers can name it."""
    m = evaluation.evaluate(t, col, cfg.TARGET_COL, k_values=(3,))
    m.pop("model", None)
    return m


def run(ds: pd.DataFrame | None = None, oos: pd.DataFrame | None = None,
        cfg=config, partitions=("seg_pos", "seg_history", "seg_venue"),
        verbose: bool = True) -> pd.DataFrame:
    """Score every candidate on every evaluation season, out of sample."""
    if ds is None:
        ds = pd.read_parquet(cfg.MODEL_DATASET_PATH)
    if oos is None:
        oos = oos_mod.load_oos(cfg)
    oos = segmod.add_segments(oos)
    ds_seg = segmod.add_segments(
        ds.assign(p=0.0, p_sum_in_match=0.0))   # segments need only features

    rows = []
    for season in EVAL_SEASONS:
        past_oos = oos[oos["season"] < season]
        if past_oos.empty:
            continue
        te_oos = oos[oos["season"] == season].copy()
        if te_oos.empty:
            continue

        # ---- the shipped model, as-is -------------------------------------- #
        rows.append(dict(season=season, model="global (baseline)",
                         **_ev(te_oos, "p", cfg)))

        # ---- correction layers --------------------------------------------- #
        for layer in [PlattCorrection(),
                      SegmentCorrection(),
                      SegmentCorrection(interact=True),
                      StackCorrection()]:
            layer.fit(past_oos, "p", cfg)
            t = te_oos.copy()
            t["p_corr"] = layer.predict(t, "p")
            rows.append(dict(season=season, model=layer.name,
                             **_ev(t, "p_corr", cfg)))

        # ---- per-segment boosters ------------------------------------------ #
        past = [s for s in cfg.ALL_SEASONS if s < season]
        valid_season = past[-1]
        train_seasons = [s for s in past if s != valid_season
                         and s not in cfg.BURN_IN_SEASONS]
        tr = ds_seg[ds_seg["season"].isin(train_seasons)]
        va = ds_seg[ds_seg["season"] == valid_season]
        te = ds_seg[ds_seg["season"] == season]
        for dim in partitions:
            models = fit_partitioned(tr, va, dim, cfg)
            t = te.copy()
            t["p_part"] = predict_partitioned(models, te, dim)
            rows.append(dict(season=season, model=f"specialist[{dim}]",
                             n_specialists=len(models) - 1,
                             **_ev(t, "p_part", cfg)))
        if verbose:
            print(f"  {season} done")

    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread across the evaluation seasons, baseline first."""
    agg = res.groupby("model").agg(
        seasons=("season", "nunique"),
        pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
        lift=("pr_auc_lift", "mean"), lift_sd=("pr_auc_lift", "std"),
        roc_auc=("roc_auc", "mean"),
        log_loss=("log_loss", "mean"), ece=("ece", "mean"),
        precision_at_3=("precision@3", "mean"), hit_at_3=("hit_rate@3", "mean"),
    ).reset_index()
    base = agg[agg["model"] == "global (baseline)"]
    if len(base):
        for c in ["lift", "log_loss", "ece", "precision_at_3"]:
            agg[f"d_{c}"] = agg[c] - base[c].iloc[0]
    order = ["global (baseline)"] + [m for m in agg["model"] if m != "global (baseline)"]
    return agg.set_index("model").loc[order].reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Specialised-model comparison.")
    ap.add_argument("--rebuild-oos", action="store_true")
    args = ap.parse_args()

    oos = oos_mod.load_oos(rebuild=args.rebuild_oos)
    res = run(oos=oos)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    print("\n=== per season ===")
    print(res[["season", "model", "pr_auc", "pr_auc_lift", "roc_auc", "log_loss",
               "ece", "precision@3", "hit_rate@3"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== averaged over the evaluation seasons ===")
    s = summarise(res)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    res.to_csv(config.RESEARCH_DIR / "specialist_results.csv", index=False)
    s.to_csv(config.RESEARCH_DIR / "specialist_summary.csv", index=False)
    print(f"\nwrote specialist_results.csv and specialist_summary.csv "
          f"to {config.RESEARCH_DIR}")


if __name__ == "__main__":
    main()
