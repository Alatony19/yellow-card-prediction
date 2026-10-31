"""Recommendation 5 - a team-level card model feeding a per-player share.

The problem with the flat per-player model
------------------------------------------
It scores each of the twenty starters independently.  Cards inside a match are
not independent: a scrappy derby books five players and a comfortable 3-0 books
one, and that common factor is a *team-match* quantity, not a player one.  The
flat model can only express it by pushing every player in that fixture up a
little, and because it is fitted on player rows it has no way to know how much
total it has committed to.

The decomposition
-----------------
Model the two pieces separately and multiply them back together:

    lambda_team   expected number of carded slots for this XI      (stage 1)
    s_i           share of that total belonging to slot i          (stage 2)
    lambda_i      = lambda_team * s_i
    p_i           = 1 - exp(-lambda_i)

Stage 1 is a Poisson count model on team-match features - the referee, the
opponent, the fixture's temper, the league context, and summaries of who is
actually in the XI.  Stage 2 reuses the existing player model as the share,
which is what it is genuinely good at: the *ordering* within a side.

The identity that makes this coherent
-------------------------------------
    p_i = 1 - exp(-lambda_i)   <=>   lambda_i = -log(1 - p_i)

so the flat model's probabilities convert to intensities, are normalised inside
the team, and are rescaled to whatever total stage 1 believes in.  Within a
team the ordering is untouched by construction; *between* the two teams in a
match it can change, and that is the whole point - top-K picks are chosen
across both sides of the fixture.

Two share variants are tested: the renormalised flat model (above) and a
dedicated conditional-logit share model fitted on the multinomial likelihood of
"which slot got the card, given that one did".
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, evaluation, modelling, oos as oos_mod
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import evaluation
    import modelling
    import oos as oos_mod

from xgboost import XGBRegressor, XGBClassifier

# Team-match features for the count model.  Everything here is either a match
# fact or an aggregate over the confirmed XI, so all of it is known once the
# line-up sheet is out.
TEAM_FEATURES = [
    "is_home", "matchweek",
    "team_fouls_pm_roll", "team_cards_pm_roll", "opp_fouls_drawn_pm_roll",
    "strength_diff", "ref_cards_pm_eb",
    "ctx_match_difficulty", "ctx_fixture_cards_pm_eb", "ctx_elo_diff",
    "ctx_position", "ctx_opp_position", "ctx_stakes", "ctx_matches_last_14d",
    # --- who is actually playing -------------------------------------------
    "xi_cards_p90_mean", "xi_cards_p90_max", "xi_fouls_p90_mean",
    "xi_exp_cards_sum", "xi_age_mean", "xi_apps_mean",
]

XGB_COUNT_PARAMS = dict(
    n_estimators=1200, learning_rate=0.03, max_depth=4, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
    objective="count:poisson", eval_metric="poisson-nloglik",
    tree_method="hist", random_state=config.RANDOM_STATE, n_jobs=-1,
)


def _logit_to_lambda(p, eps: float = 1e-6):
    """P(at least one) -> Poisson intensity.  The inverse of 1 - exp(-lambda)."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return -np.log1p(-p)


# --------------------------------------------------------------------------- #
# stage 1 - the team count model
# --------------------------------------------------------------------------- #
def build_team_rows(ds: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """One row per (match, team), with the XI summarised and the count target.

    The target is the number of *slots* that picked up a card, not the number of
    cards - it has to match what the player model predicts, or stage 1 and
    stage 2 would be measuring different things and the product would be
    meaningless.
    """
    g = ds.groupby(["match_id", "team"], sort=False)
    rows = g.agg(
        season=("season", "first"),
        kickoff=("kickoff", "first"),
        matchweek=("matchweek", "first"),
        opponent=("opponent", "first"),
        referee=("referee", "first"),
        is_home=("is_home", "first"),
        n_slots=(cfg.TARGET_COL, "size"),
        y_team_cards=(cfg.TARGET_COL, "sum"),
        team_fouls_pm_roll=("team_fouls_pm_roll", "first"),
        team_cards_pm_roll=("team_cards_pm_roll", "first"),
        opp_fouls_drawn_pm_roll=("opp_fouls_drawn_pm_roll", "first"),
        strength_diff=("strength_diff", "first"),
        ref_cards_pm_eb=("ref_cards_pm_eb", "first"),
        ctx_match_difficulty=("ctx_match_difficulty", "first"),
        ctx_fixture_cards_pm_eb=("ctx_fixture_cards_pm_eb", "first"),
        ctx_elo_diff=("ctx_elo_diff", "first"),
        ctx_position=("ctx_position", "first"),
        ctx_opp_position=("ctx_opp_position", "first"),
        ctx_stakes=("ctx_stakes", "first"),
        ctx_matches_last_14d=("ctx_matches_last_14d", "first"),
        xi_cards_p90_mean=("plr_cards_p90_eb", "mean"),
        xi_cards_p90_max=("plr_cards_p90_eb", "max"),
        xi_fouls_p90_mean=("plr_fouls_p90_eb", "mean"),
        xi_exp_cards_sum=("exp_cards_analytic", "sum"),
        xi_age_mean=("plr_age_years", "mean"),
        xi_apps_mean=("plr_apps_prior", "mean"),
    ).reset_index()
    if "league" in ds.columns:      # the five-league dataset (ensemble.py)
        rows["league"] = g["league"].first().astype(str).to_numpy()
    return rows


def fit_team_model(train: pd.DataFrame, valid: pd.DataFrame | None,
                   params: dict | None = None, verbose: bool = False,
                   features: list | None = None):
    params = dict(params or XGB_COUNT_PARAMS)
    features = list(features or TEAM_FEATURES)
    fit_kwargs = {"verbose": False}
    if valid is not None and len(valid):
        params["early_stopping_rounds"] = 100
        fit_kwargs["eval_set"] = [(valid[features],
                                   valid["y_team_cards"].to_numpy(dtype=float))]
    m = XGBRegressor(**params)
    m.fit(train[features], train["y_team_cards"].to_numpy(dtype=float),
          **fit_kwargs)
    if verbose:
        print(f"  team model: best iteration {getattr(m, 'best_iteration', None)}")
    return m


# --------------------------------------------------------------------------- #
# stage 2 - the share
# --------------------------------------------------------------------------- #
def shares_from_flat(p: np.ndarray, team_key: np.ndarray) -> np.ndarray:
    """Normalise the flat model's intensities inside each team."""
    lam = _logit_to_lambda(p)
    s = pd.Series(lam).groupby(team_key).transform("sum").to_numpy()
    return np.divide(lam, s, out=np.full_like(lam, np.nan), where=s > 0)


class ConditionalLogitShare:
    """A share model fitted only on *who* got the card, never on how many.

    Rows from teams that picked up no card carry no information about the share
    and are dropped; within the rest, the likelihood is the softmax over the
    slots of one team.  Fitting it this way means the share model cannot absorb
    any of the level signal - that job belongs entirely to stage 1 - so the two
    stages stay genuinely separate.

    Implemented as a binary booster on the player features with the team as the
    normalising group, which is the standard practical stand-in for a
    conditional logit and lets the same feature contract be reused.
    """

    def __init__(self, cfg=config):
        self.cfg = cfg

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        cfg = self.cfg
        carded_teams = (train.groupby(["match_id", "team"])[cfg.TARGET_COL]
                        .transform("sum") > 0)
        tr = train[carded_teams]
        params = dict(cfg.XGB_PARAMS)
        params["n_estimators"] = 600
        fit_kwargs = {"verbose": False}
        va = None
        if valid is not None and len(valid):
            vc = (valid.groupby(["match_id", "team"])[cfg.TARGET_COL]
                  .transform("sum") > 0)
            va = valid[vc]
            params["early_stopping_rounds"] = 60
        frames = [tr] + ([va] if va is not None else [])
        frames = modelling._align_categories([f.copy() for f in frames], cfg)
        tr = frames[0]
        va = frames[1] if va is not None else None
        self.categories = {c: tr[c].cat.categories for c in cfg.CATEGORICAL_FEATURES}
        if va is not None:
            fit_kwargs["eval_set"] = [(modelling._x(va, cfg),
                                       va[cfg.TARGET_COL].to_numpy())]
        self.m = XGBClassifier(**params)
        self.m.fit(modelling._x(tr, cfg), tr[cfg.TARGET_COL].to_numpy(), **fit_kwargs)
        return self

    def shares(self, df: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        x = df[cfg.FEATURES].copy()
        for c in cfg.CATEGORICAL_FEATURES:
            x[c] = pd.Categorical(x[c], categories=self.categories[c])
        raw = self.m.predict_proba(x)[:, 1]
        key = df["match_id"].astype(str) + "|" + df["team"].astype(str)
        tot = pd.Series(raw).groupby(key.to_numpy()).transform("sum").to_numpy()
        return np.divide(raw, tot, out=np.full_like(raw, np.nan), where=tot > 0)


# --------------------------------------------------------------------------- #
# recombination
# --------------------------------------------------------------------------- #
def recombine(df: pd.DataFrame, shares: np.ndarray,
              team_lambda: pd.DataFrame) -> np.ndarray:
    """lambda_i = lambda_team * s_i, then back to a probability."""
    key = df["match_id"].astype(str) + "|" + df["team"].astype(str)
    lut = dict(zip(team_lambda["match_id"].astype(str) + "|"
                   + team_lambda["team"].astype(str),
                   team_lambda["lam"].to_numpy()))
    lam_team = key.map(lut).to_numpy(dtype=float)
    lam_i = lam_team * shares
    return 1.0 - np.exp(-np.clip(lam_i, 0, None))


# --------------------------------------------------------------------------- #
# walk-forward comparison
# --------------------------------------------------------------------------- #
def run(ds: pd.DataFrame | None = None, cfg=config, verbose: bool = True) -> pd.DataFrame:
    if ds is None:
        ds = pd.read_parquet(cfg.MODEL_DATASET_PATH)
    team_rows = build_team_rows(ds, cfg)

    rows, detail = [], []
    for season in cfg.WALK_FORWARD_SEASONS:
        past = [s for s in cfg.ALL_SEASONS if s < season]
        if len(past) < 3:
            continue
        valid_season = past[-1]
        train_seasons = [s for s in past if s != valid_season
                         and s not in cfg.BURN_IN_SEASONS]
        tr, va, te = (ds[ds["season"].isin(train_seasons)],
                      ds[ds["season"] == valid_season],
                      ds[ds["season"] == season].copy())
        ttr, tva, tte = (team_rows[team_rows["season"].isin(train_seasons)],
                         team_rows[team_rows["season"] == valid_season],
                         team_rows[team_rows["season"] == season].copy())
        if not (len(tr) and len(va) and len(te)):
            continue

        # --- the flat model, unchanged: the thing to beat ------------------- #
        flat = modelling.fit_model(tr, va, cfg, verbose=False)
        te["p_flat"] = flat.predict_proba(te)

        # --- stage 1 -------------------------------------------------------- #
        tm = fit_team_model(ttr, tva)
        tte["lam"] = np.clip(tm.predict(tte[TEAM_FEATURES]), 1e-4, None)

        # The league card rate drifts upward, so a count model trained on older
        # seasons is systematically low - by 0.1 to 0.35 cards per team here.
        # Anchor the level on the most recent season the model is allowed to
        # see (the validation season), which is a single scalar and therefore
        # cannot leak any within-season ordering.
        lam_va = np.clip(tm.predict(tva[TEAM_FEATURES]), 1e-4, None)
        anchor = float(tva["y_team_cards"].mean() / lam_va.mean())
        tte["lam_anchored"] = tte["lam"] * anchor

        # --- stage 2, two variants ------------------------------------------ #
        team_key = (te["match_id"].astype(str) + "|" + te["team"].astype(str)).to_numpy()
        s_flat = shares_from_flat(te["p_flat"].to_numpy(), team_key)
        te["p_decomp_flatshare"] = recombine(te, s_flat, tte)

        cl = ConditionalLogitShare(cfg).fit(tr, va)
        cl_shares = cl.shares(te)
        te["p_decomp_clshare"] = recombine(te, cl_shares, tte)
        te["p_decomp_anchored"] = recombine(
            te, cl_shares, tte.rename(columns={"lam": "_lam_raw",
                                               "lam_anchored": "lam"}))

        # --- a half-way house: keep the flat model, only fix the total ------ #
        # Rescale each team's intensities so they sum to lambda_team but blend
        # the rescale halfway, which is the usual compromise when the count
        # model is noisier than the ranking it is correcting.
        lam_flat = _logit_to_lambda(te["p_flat"].to_numpy())
        tot_flat = pd.Series(lam_flat).groupby(team_key).transform("sum").to_numpy()
        lut = dict(zip(tte["match_id"].astype(str) + "|" + tte["team"].astype(str),
                       tte["lam"].to_numpy()))
        lam_team = pd.Series(team_key).map(lut).to_numpy(dtype=float)
        scale = np.divide(lam_team, tot_flat, out=np.ones_like(lam_flat),
                          where=tot_flat > 0)
        te["p_decomp_half"] = 1.0 - np.exp(-lam_flat * np.sqrt(np.clip(scale, 1e-6, None)))

        for name, col in [("flat player model (baseline)", "p_flat"),
                          ("decomposed [flat share]", "p_decomp_flatshare"),
                          ("decomposed [cond-logit share]", "p_decomp_clshare"),
                          ("decomposed [cl share + level anchor]", "p_decomp_anchored"),
                          ("decomposed [half rescale]", "p_decomp_half")]:
            m = evaluation.evaluate(te, col, cfg.TARGET_COL, name=name,
                                    k_values=cfg.TOP_K_VALUES)
            m["season"] = season
            rows.append(m)

        # keep the team-count diagnostics too
        detail.append(pd.DataFrame({
            "season": season,
            "match_id": tte["match_id"], "team": tte["team"],
            "lam_pred": tte["lam"], "y_team_cards": tte["y_team_cards"],
        }))
        if verbose:
            mae = np.abs(tte["lam"] - tte["y_team_cards"]).mean()
            print(f"  {season}: team-count MAE {mae:.3f} "
                  f"(mean actual {tte['y_team_cards'].mean():.3f}, "
                  f"predicted {tte['lam'].mean():.3f})")

    res = pd.DataFrame(rows)
    res.attrs["detail"] = pd.concat(detail, ignore_index=True) if detail else None
    return res


# --------------------------------------------------------------------------- #
# the shippable artifact
# --------------------------------------------------------------------------- #
@dataclass
class Decomposition:
    """Everything needed to turn flat probabilities into decomposed ones.

    Kept as plain components rather than a pickled wrapper for the same reason
    ``modelling.save_model`` does: a pickled dataclass records the module it was
    defined in, so an artifact written by ``python pipeline/team_model.py`` would only
    load back inside ``__main__``.
    """
    count_model: object          # stage 1: expected carded slots for an XI
    share_booster: object        # stage 2: who, within the XI
    share_categories: dict
    anchor: float                # level correction for the league's card drift
    features: list               # the share model's feature contract
    categorical: list
    team_features: list
    trained_on: tuple = ()

    # ---- stage 2 ---------------------------------------------------------- #
    def shares(self, rows: pd.DataFrame) -> np.ndarray:
        x = rows[self.features].copy()
        for c in self.categorical:
            x[c] = pd.Categorical(x[c], categories=self.share_categories[c])
        raw = self.share_booster.predict_proba(x)[:, 1]
        key = (rows["match_id"].astype(str) + "|" + rows["team"].astype(str)).to_numpy()
        tot = pd.Series(raw).groupby(key).transform("sum").to_numpy()
        return np.divide(raw, tot, out=np.full_like(raw, np.nan), where=tot > 0)

    # ---- stage 1 ---------------------------------------------------------- #
    def team_lambda(self, rows: pd.DataFrame, cfg=config) -> pd.DataFrame:
        team_rows = build_team_rows(rows, cfg)
        lam = np.clip(self.count_model.predict(team_rows[self.team_features]),
                      1e-4, None)
        team_rows["lam"] = lam * self.anchor
        return team_rows

    # ---- the product ------------------------------------------------------ #
    def predict_proba(self, rows: pd.DataFrame, cfg=config) -> np.ndarray:
        """p_i = 1 - exp(-lambda_team * s_i), for rows of one or many matches."""
        return recombine(rows, self.shares(rows), self.team_lambda(rows, cfg))


def fit_decomposition(ds: pd.DataFrame | None = None, cfg=config,
                      verbose: bool = True) -> Decomposition:
    """Fit both stages on TRAIN_SEASONS, early-stopped and anchored on VALID.

    Same split as the shipped flat model, so the two artifacts are trained on
    exactly the same information and can be compared row for row.
    """
    if ds is None:
        ds = pd.read_parquet(cfg.MODEL_DATASET_PATH)
    tr = ds[ds["season"].isin(cfg.TRAIN_SEASONS)]
    va = ds[ds["season"].isin(cfg.VALID_SEASONS)]
    if not len(tr) or not len(va):
        raise RuntimeError("empty train or validation split")

    team_rows = build_team_rows(ds, cfg)
    ttr = team_rows[team_rows["season"].isin(cfg.TRAIN_SEASONS)]
    tva = team_rows[team_rows["season"].isin(cfg.VALID_SEASONS)]

    count_model = fit_team_model(ttr, tva, verbose=verbose)
    lam_va = np.clip(count_model.predict(tva[TEAM_FEATURES]), 1e-4, None)
    anchor = float(tva["y_team_cards"].mean() / lam_va.mean())

    share = ConditionalLogitShare(cfg).fit(tr, va)

    if verbose:
        print(f"  level anchor {anchor:.4f}  "
              f"(validation actual {tva['y_team_cards'].mean():.3f} vs "
              f"raw predicted {lam_va.mean():.3f})")

    return Decomposition(
        count_model=count_model, share_booster=share.m,
        share_categories=dict(share.categories), anchor=anchor,
        features=list(cfg.FEATURES), categorical=list(cfg.CATEGORICAL_FEATURES),
        team_features=list(TEAM_FEATURES), trained_on=tuple(cfg.TRAIN_SEASONS),
    )


def save_decomposition(dec: Decomposition, path: Path = None) -> None:
    import joblib
    path = path or config.DECOMPOSITION_PATH
    joblib.dump({
        "count_model": dec.count_model,
        "share_booster": dec.share_booster,
        "share_categories": {k: list(v) for k, v in dec.share_categories.items()},
        "anchor": dec.anchor,
        "features": list(dec.features),
        "categorical": list(dec.categorical),
        "team_features": list(dec.team_features),
        "trained_on": tuple(dec.trained_on),
    }, path)


def load_decomposition(path: Path = None) -> Decomposition:
    import joblib
    path = path or config.DECOMPOSITION_PATH
    blob = joblib.load(path)
    return Decomposition(
        count_model=blob["count_model"], share_booster=blob["share_booster"],
        share_categories={k: pd.Index(v) for k, v in blob["share_categories"].items()},
        anchor=blob["anchor"], features=blob["features"],
        categorical=blob["categorical"], team_features=blob["team_features"],
        trained_on=tuple(blob.get("trained_on", ())),
    )


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    agg = res.groupby("model").agg(
        seasons=("season", "nunique"),
        pr_auc=("pr_auc", "mean"), lift=("pr_auc_lift", "mean"),
        lift_sd=("pr_auc_lift", "std"), roc_auc=("roc_auc", "mean"),
        log_loss=("log_loss", "mean"), ece=("ece", "mean"),
        precision_at_3=("precision@3", "mean"), hit_at_3=("hit_rate@3", "mean"),
    ).reset_index()
    base = agg[agg["model"] == "flat player model (baseline)"]
    if len(base):
        for c in ["lift", "log_loss", "ece", "precision_at_3"]:
            agg[f"d_{c}"] = agg[c] - base[c].iloc[0]
    order = ["flat player model (baseline)"] + [
        m for m in agg["model"] if m != "flat player model (baseline)"]
    return agg.set_index("model").loc[order].reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Team-count / player-share decomposition.")
    ap.add_argument("--save", action="store_true",
                    help="fit on TRAIN/VALID and write the artifact predict.py uses")
    args = ap.parse_args()

    if args.save:
        print("fitting the decomposition on "
              f"{config.TRAIN_SEASONS[0]}..{config.TRAIN_SEASONS[-1]} "
              f"(anchored on {config.VALID_SEASONS[0]})")
        dec = fit_decomposition()
        save_decomposition(dec)
        print(f"saved {config.DECOMPOSITION_PATH}")
        return

    res = run()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print("\n=== per season ===")
    print(res[["season", "model", "pr_auc", "pr_auc_lift", "roc_auc", "log_loss",
               "ece", "precision@3", "hit_rate@3"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== averaged over the walk-forward seasons ===")
    s = summarise(res)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    res.to_csv(config.RESEARCH_DIR / "team_decomposition_results.csv", index=False)
    if res.attrs.get("detail") is not None:
        res.attrs["detail"].to_csv(
            config.RESEARCH_DIR / "team_count_predictions.csv", index=False)
    print(f"\nwrote team_decomposition_results.csv to {config.RESEARCH_DIR}")


if __name__ == "__main__":
    main()
