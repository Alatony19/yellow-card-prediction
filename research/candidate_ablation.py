"""Recommendation 4 - does richer player context actually help?

What was asked for and what is possible
---------------------------------------
The recommendation named fouls by zone, tackles from behind and dissent
history.  FBref publishes none of those, and as of 2025-26 it publishes less
than it used to: the advanced per-player tables are gone, leaving minutes,
goals, cards, fouls, fouled, tackles *won*, interceptions, crosses, penalties
conceded and offsides.  Anything built on the removed columns would be NaN for
the live season, which is worse than not having it (the booster never saw those
columns missing during training).

So the candidates here are built strictly from what survives, plus two
structural reads that need no new data at all:

  behaviour      defensive workload, fouls per defensive action, aggressor vs
                 victim ratio, penalties conceded
  venue          his card rate at THIS venue, and the home/away gap
  pairing        this player's history under THIS referee
  form           recent fouling against his own baseline
  incentive      bookings until a live suspension threshold, and how many
                 matches are left to avoid it

Each group is ablated separately as well as together, because "eleven features
helped" and "one of eleven helped and ten added noise" are different findings
and only the second one tells you what to keep.
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

GROUPS = {
    "behaviour": ["plr_defacts_p90_eb", "plr_fouls_per_defact_eb",
                  "plr_aggression_ratio", "plr_pens_p90_eb"],
    "venue": ["plr_cards_p90_this_venue", "plr_venue_card_gap"],
    "ref_pairing": ["plr_ref_pair_ratio", "plr_ref_pair_apps"],
    "form_trend": ["plr_fouls_trend"],
    "suspension": ["plr_cards_to_ban", "plr_ban_window_left"],
}


# Closing-price groups.  The split is the point of the experiment: "tightness"
# is constant across all twenty players and can only act through interactions,
# "side" takes opposite values for the two teams and can genuinely reorder, and
# "market vs elo" is the one column carrying information no other feature in the
# pipeline has ever seen.
ODDS_GROUPS = {
    "odds: tightness": ["odds_p_draw", "odds_closeness", "odds_total_lambda"],
    "odds: side": ["odds_team_expectancy", "odds_supremacy_for",
                   "odds_lambda_against", "odds_is_underdog"],
    "odds: market vs elo": ["odds_elo_surprise"],
}


REFEREE_GROUPS = {
    "ref: venue bias": ["ref_venue_card_bias", "ref_away_card_share_eb"],
    "ref: season level": ["ref_season_level_ratio"],
    "ref: position affinity": ["ref_pos_affinity", "ref_pos_apps"],
    "ref: previously dropped": ["ref_cards_per_foul_eb", "ref_cards_pm_recent",
                                "ref_matches_prior"],
}


def cfg_with(extra: list, categorical_extra: list = ()):
    """A shallow config clone whose feature list is the baseline plus ``extra``."""
    class _C:
        pass
    c2 = _C()
    for k in dir(config):
        if not k.startswith("__"):
            setattr(c2, k, getattr(config, k))
    excluded = (set(config.CANDIDATE_FEATURES)
                | set(getattr(config, "REFEREE_CANDIDATES", []))
                | set(getattr(config, "ODDS_CANDIDATES", [])))
    base = [f for f in config.NUMERIC_FEATURES if f not in excluded]
    c2.NUMERIC_FEATURES = base + list(extra)
    c2.CATEGORICAL_FEATURES = list(config.CATEGORICAL_FEATURES) + list(categorical_extra)
    c2.FEATURES = c2.NUMERIC_FEATURES + c2.CATEGORICAL_FEATURES
    return c2


def walk_forward_lift(ds: pd.DataFrame, cfg, blend: bool = True) -> pd.DataFrame:
    """modelling.walk_forward, but for an arbitrary feature set.

    The blend (booster + logistic) is reported because that is the configuration
    the earlier feature decisions in this project were made on, so a new feature
    has to be judged on the same footing.
    """
    rows = []
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
        r = evaluation.evaluate(te, "p", cfg.TARGET_COL, name=season, k_values=(3,))
        rows.append(r)
    return pd.DataFrame(rows)


def run(ds: pd.DataFrame | None = None, blend: bool = True,
        verbose: bool = True, which: str = "player") -> pd.DataFrame:
    if ds is None:
        ds = pd.read_parquet(config.MODEL_DATASET_PATH)

    if which == "odds":
        trials = {"baseline (24 features)": ([], ())}
        for g, feats in ODDS_GROUPS.items():
            trials[f"+ {g}"] = (feats, ())
        trials["+ all odds candidates"] = (config.ODDS_CANDIDATES, ())
        # The referee experiment showed a match-level constant cannot reorder a
        # field.  "side only" is the honest test of whether the ranking-capable
        # half carries the whole effect.
        trials["+ side only (no tightness)"] = (
            ODDS_GROUPS["odds: side"] + ODDS_GROUPS["odds: market vs elo"], ())
    elif which == "referee":
        trials = {"baseline (24 features)": ([], ())}
        for g, feats in REFEREE_GROUPS.items():
            trials[f"+ {g}"] = (feats, ())
        trials["+ all referee candidates"] = (config.REFEREE_CANDIDATES, ())
        # Referee *identity* has never been tried as a categorical - only as
        # rates.  A booster can learn an individual official directly from it,
        # at the cost of ~35 new levels with a few hundred rows each.
        trials["+ referee identity (categorical)"] = ([], ("referee",))
        trials["+ all referee + identity"] = (config.REFEREE_CANDIDATES, ("referee",))
    else:
        trials = {"baseline (24 features)": ([], ())}
        for g, feats in GROUPS.items():
            trials[f"+ {g}"] = (feats, ())
        trials["+ all candidates"] = (config.CANDIDATE_FEATURES, ())

    out = []
    for name, (extra, cat_extra) in trials.items():
        cfg = cfg_with(extra, cat_extra)
        wf = walk_forward_lift(ds, cfg, blend=blend)
        out.append({
            "variant": name, "n_features": len(cfg.FEATURES),
            "pr_auc": wf["pr_auc"].mean(), "lift": wf["pr_auc_lift"].mean(),
            "lift_sd": wf["pr_auc_lift"].std(), "roc_auc": wf["roc_auc"].mean(),
            "log_loss": wf["log_loss"].mean(), "ece": wf["ece"].mean(),
            "precision_at_3": wf["precision@3"].mean(),
            "hit_at_3": wf["hit_rate@3"].mean(),
        })
        if verbose:
            print(f"  {name:28s} lift {out[-1]['lift']:.4f} "
                  f"(sd {out[-1]['lift_sd']:.4f})  P@3 {out[-1]['precision_at_3']:.4f}")

    res = pd.DataFrame(out)
    base = res[res["variant"] == "baseline (24 features)"].iloc[0]
    for c in ["lift", "precision_at_3", "log_loss", "ece"]:
        res[f"d_{c}"] = res[c] - base[c]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommendation-4 feature ablation.")
    ap.add_argument("--no-blend", action="store_true")
    ap.add_argument("--referee", action="store_true",
                    help="ablate the referee candidates instead")
    ap.add_argument("--odds", action="store_true",
                    help="ablate the closing-price candidates instead")
    args = ap.parse_args()
    which = "referee" if args.referee else ("odds" if args.odds else "player")
    res = run(blend=not args.no_blend, which=which)
    pd.set_option("display.width", 220)
    title = {"referee": "referee candidates",
             "odds": "closing-price candidates",
             "player": "recommendation 4: player-context candidates"}[which]
    print(f"\n=== {title}, 5-season walk-forward ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    name = {"referee": "referee_ablation.csv", "odds": "odds_ablation.csv",
            "player": "candidate_ablation.csv"}[which]
    res.to_csv(config.RESEARCH_DIR / name, index=False)
    print(f"\nwrote {name} to {config.RESEARCH_DIR}")


if __name__ == "__main__":
    main()
