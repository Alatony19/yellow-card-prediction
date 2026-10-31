"""Out-of-sample prediction pool - the substrate for error analysis and review.

``modelling.walk_forward`` answers "how good is the model?" and throws the
predictions away.  Everything downstream of that question - when is it right,
when is it wrong, can a segment be exploited, what did it say about Saturday's
game - needs the *rows*, not the summary.

So this module runs the same protocol (re-fit per season on strictly earlier
seasons, score that season) and keeps every prediction.  The result is cached to
``artifacts/oos_predictions.parquet``.

The honesty guarantee is inherited from ``modelling.walk_forward``: no row here
was ever seen in training by the model that scored it.  That is what makes it
safe to hunt for patterns in the errors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, modelling
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import modelling

OOS_PATH = config.PL_DIR / "oos_predictions.parquet"

# Carried alongside the prediction so the analysis can slice without re-joining.
CARRY = [
    "match_id", "season", "matchweek", "kickoff", "team", "opponent", "player",
    "player_key", "referee", "is_home", "pos_group", "pos_group_actual",
    "formation", "minutes", "was_replaced", "repl_card",
    "y_any_card", "y_slot_any_card", "y_yellow", "y_slot_yellow",
]


def _season_splits(cfg=config):
    """(season, train_seasons, valid_season) for each walk-forward season."""
    for season in cfg.WALK_FORWARD_SEASONS:
        past = [s for s in cfg.ALL_SEASONS if s < season]
        if len(past) < 3:
            continue
        valid = past[-1]
        train = [s for s in past if s != valid and s not in cfg.BURN_IN_SEASONS]
        if train:
            yield season, train, valid


def build_oos(ds: pd.DataFrame | None = None, cfg=config,
              verbose: bool = True) -> pd.DataFrame:
    """Row-level out-of-sample predictions across the walk-forward seasons.

    Three prediction columns are kept, because "when is the model wrong" has a
    different answer for each and comparing them is itself diagnostic:
      ``p``          - the shipped booster
      ``p_logistic`` - the linear floor on the same features
      ``p_blend``    - their average (the best walk-forward configuration)
    """
    if ds is None:
        ds = pd.read_parquet(cfg.MODEL_DATASET_PATH)

    out = []
    for season, train_seasons, valid_season in _season_splits(cfg):
        tr = ds[ds["season"].isin(train_seasons)]
        va = ds[ds["season"] == valid_season]
        te = ds[ds["season"] == season]
        if not (len(tr) and len(va) and len(te)):
            continue
        model = modelling.fit_model(tr, va, cfg, verbose=False)
        logit = modelling.fit_logistic(tr, cfg)

        cols = list(dict.fromkeys(
            [c for c in CARRY if c in te.columns] + list(cfg.FEATURES)
            + [cfg.TARGET_COL]))
        rec = te[cols].copy()
        rec["p"] = model.predict_proba(te)
        rec["p_logistic"] = modelling.logistic_predict(logit, te, cfg)
        rec["p_blend"] = 0.5 * rec["p"] + 0.5 * rec["p_logistic"]
        rec["train_seasons"] = len(train_seasons)
        out.append(rec)
        if verbose:
            print(f"  {season}: {len(rec):,} rows scored by a model trained on "
                  f"{train_seasons[0]}..{train_seasons[-1]}")

    oos = pd.concat(out, ignore_index=True)
    oos = _add_derived(oos, cfg)
    return oos


def _add_derived(oos: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """Per-row and per-match quantities every downstream slice wants."""
    y = oos[cfg.TARGET_COL].to_numpy(dtype=float)
    p = oos["p"].to_numpy(dtype=float)

    # Rank inside the match is what a top-K shortlist actually uses.
    oos["rank_in_match"] = (oos.groupby("match_id")["p"]
                            .rank(method="first", ascending=False).astype(int))
    oos["n_cards_in_match"] = oos.groupby("match_id")[cfg.TARGET_COL].transform("sum")
    oos["p_sum_in_match"] = oos.groupby("match_id")["p"].transform("sum")

    # Per-row error decomposition.
    oos["resid"] = y - p                       # signed: >0 = model under-called
    oos["brier"] = (y - p) ** 2
    oos["logloss"] = -(y * np.log(np.clip(p, 1e-9, 1)) +
                       (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))

    # Did the top-3 shortlist hit?  Evaluated per match, broadcast to its rows.
    top3 = oos[oos["rank_in_match"] <= 3]
    hits = top3.groupby("match_id")[cfg.TARGET_COL].sum().rename("top3_hits")
    oos = oos.merge(hits, on="match_id", how="left")
    oos["top3_cashed"] = (oos["top3_hits"] > 0).astype(int)
    return oos


def load_oos(cfg=config, rebuild: bool = False) -> pd.DataFrame:
    if OOS_PATH.exists() and not rebuild:
        return pd.read_parquet(OOS_PATH)
    oos = build_oos(cfg=cfg)
    oos.to_parquet(OOS_PATH, index=False)
    return oos


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the out-of-sample prediction pool.")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    oos = build_oos()
    oos.to_parquet(OOS_PATH, index=False)
    print(f"\nwrote {OOS_PATH}  ({len(oos):,} rows, "
          f"{oos['match_id'].nunique():,} matches, "
          f"{oos['season'].nunique()} seasons)")
    print(f"base rate {oos[config.TARGET_COL].mean():.4f} | "
          f"mean p {oos['p'].mean():.4f}")


if __name__ == "__main__":
    main()
