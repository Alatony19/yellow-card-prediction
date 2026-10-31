"""The one card model: a five-league ensemble behind a single artifact.

Every serving path - ``predict.py``, ``todays_slate.py``, ``card_market.py``,
``player_confidence.py`` - scores with this object and nothing else.  It folds
together everything the project has built that carries signal:

* **bag**          K boosters on the five-league dataset (``pooled.FINAL_ARM``
                   features: the shipped set plus ``league``), differing only in
                   seed and row/column subsampling.  Member 0 is the shipped
                   pooled booster exactly.  Their mean is the backbone; their
                   spread is the "how much does the fit wobble" figure
                   ``player_confidence.py`` reports.
* **logistic**     a regularised logistic regression on the same features -
                   the linear floor that ``blend_verdict.py`` found adds to
                   global ranking (pricing) on the Premier League.
* **decomp**       the team-count x player-share decomposition from
                   ``team_model.py``, now fitted on all five leagues, with
                   ``league`` in the count model and a level anchor per league.
                   It was the better within-match ranker on the Premier League.

    p = w_bag * mean(bag) + w_logistic * p_logistic + w_decomp * p_decomp

The weights are not chosen by hand.  ``evaluate`` walks forward over
2021-22 -> 2026-27 in every league (each season fitted only on earlier ones,
early-stopped on the season before it) and keeps every component's out-of-sample
probability.  ``select`` then grid-searches the weights on the question the
user stakes on - four picks per match, how many get carded - with log-loss as
the tie-break, and writes the out-of-sample pool the confidence intervals are
read from.  ``train`` refits every component on 2017-18 -> 2024-25,
early-stopped on 2025-26, and saves the artifact.

Stages
------
    python pipeline/ensemble.py evaluate     # walk-forward, every component (~15 min)
    python pipeline/ensemble.py select       # weights + the out-of-sample pool
    python pipeline/ensemble.py train        # the shipped artifact
    python pipeline/ensemble.py all          # the three in order
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import config
import modelling
import pooled
import team_model

ENSEMBLE_PATH = pooled.POOLED_DIR / "ensemble_card_model.joblib"
OOS_PATH = pooled.POOLED_DIR / "ensemble_oos.parquet"
WEIGHTS_PATH = pooled.POOLED_DIR / "ensemble_weights.json"
SELECTION_PATH = pooled.POOLED_DIR / "ensemble_selection.csv"

K_MEMBERS = 10
COMPONENTS = ("bag", "logistic", "decomp")
# Used only when ``select`` has not been run.  The bag alone is the pooled
# model that was already shipped, so this default changes nothing.
DEFAULT_WEIGHTS = {"bag": 1.0, "logistic": 0.0, "decomp": 0.0}
LEAGUE_DUMMIES = [f"lg_{lg}" for lg in pooled.ALL_LEAGUES]
TEAM_FEATURES = list(team_model.TEAM_FEATURES) + LEAGUE_DUMMIES


def model_cfg():
    """The feature contract every component is fitted on (``pooled.served_cfg``)."""
    return pooled.served_cfg()


def ensemble_params(k: int, cfg=config) -> list[dict]:
    """K parameter sets differing only in seed and subsampling.

    Member 0 is ``config.XGB_PARAMS`` untouched, so the bag contains the shipped
    pooled booster.  Hyper-parameters are otherwise left alone: varying depth or
    learning rate would measure sensitivity to tuning choices, not to the luck
    of which rows and columns each tree saw.
    """
    rng = np.random.default_rng(12345)
    out = [dict(cfg.XGB_PARAMS)]
    for _ in range(k - 1):
        p = dict(cfg.XGB_PARAMS)
        p["random_state"] = int(rng.integers(0, 10 ** 6))
        p["subsample"] = float(rng.uniform(0.70, 0.90))
        p["colsample_bytree"] = float(rng.uniform(0.70, 0.90))
        out.append(p)
    return out


def _team_rows(ds: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = team_model.build_team_rows(ds, cfg)
    for lg, col in zip(pooled.ALL_LEAGUES, LEAGUE_DUMMIES):
        rows[col] = (rows["league"] == lg).astype(float)
    return rows


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@dataclass
class EnsembleModel:
    members: list                  # modelling.CardModel, K of them
    logistic: object               # sklearn Pipeline
    count_model: object            # XGBRegressor, team carded-slot count
    share_booster: object          # XGBClassifier, who within the team
    share_categories: dict
    anchors: dict                  # league -> level correction for the count
    weights: dict
    features: list
    categorical: list
    trained_on: tuple = ()
    team_features: list = field(default_factory=lambda: list(TEAM_FEATURES))

    @property
    def cfg(self):
        return types.SimpleNamespace(
            FEATURES=list(self.features), CATEGORICAL_FEATURES=list(self.categorical),
            NUMERIC_FEATURES=[f for f in self.features if f not in self.categorical],
            TARGET_COL=config.TARGET_COL)

    # ---- components -------------------------------------------------------- #
    def member_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """(n_rows, K) booster probabilities."""
        return np.column_stack([m.predict_proba(df) for m in self.members])

    def p_logistic(self, df: pd.DataFrame) -> np.ndarray:
        return modelling.logistic_predict(self.logistic, df, self.cfg)

    def p_decomp(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].copy()
        for c in self.categorical:
            x[c] = pd.Categorical(x[c], categories=self.share_categories[c])
        raw = self.share_booster.predict_proba(x)[:, 1]
        key = (df["match_id"].astype(str) + "|" + df["team"].astype(str)).to_numpy()
        tot = pd.Series(raw).groupby(key).transform("sum").to_numpy()
        shares = np.divide(raw, tot, out=np.full_like(raw, np.nan), where=tot > 0)

        tr = _team_rows(df, self.cfg)
        lam = np.clip(self.count_model.predict(tr[self.team_features]), 1e-4, None)
        tr["lam"] = lam * tr["league"].map(self.anchors).fillna(1.0).to_numpy()
        return team_model.recombine(df, shares, tr)

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Every component's probability, one column each, plus the members."""
        mat = self.member_matrix(df)
        out = pd.DataFrame(mat, index=df.index,
                           columns=[f"p_m{i}" for i in range(mat.shape[1])])
        out["p_bag"] = mat.mean(axis=1)
        out["p_logistic"] = self.p_logistic(df)
        out["p_decomp"] = self.p_decomp(df)
        return out

    def blend(self, comp: pd.DataFrame) -> np.ndarray:
        return blend(comp, self.weights)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.blend(self.components(df))


def blend(comp: pd.DataFrame, weights: dict) -> np.ndarray:
    p = sum(float(weights.get(c, 0.0)) * comp[f"p_{c}"].to_numpy(dtype=float)
            for c in COMPONENTS)
    return np.clip(p, 1e-6, 1 - 1e-6)


def fit(tr: pd.DataFrame, va: pd.DataFrame, k: int = K_MEMBERS,
        weights: dict | None = None, verbose: bool = True) -> EnsembleModel:
    """Fit every component on ``tr``, early-stopped (and anchored) on ``va``."""
    cfg = model_cfg()
    members = []
    for i, p in enumerate(ensemble_params(k, cfg)):
        members.append(modelling.fit_model(tr, va, cfg, params=p, verbose=False))
        if verbose:
            print(f"    member {i + 1}/{k}: {members[-1].best_iteration} trees", flush=True)
    logistic = modelling.fit_logistic(tr, cfg)

    ttr, tva = _team_rows(tr, cfg), _team_rows(va, cfg)
    count = team_model.fit_team_model(ttr, tva, features=TEAM_FEATURES)
    tva["lam"] = np.clip(count.predict(tva[TEAM_FEATURES]), 1e-4, None)
    # One scalar per league on the most recent season the model may see: card
    # rates drift upwards, and they drift at different speeds in each league.
    lvl = tva.groupby("league")[["y_team_cards", "lam"]].mean()
    anchors = (lvl["y_team_cards"] / lvl["lam"]).to_dict()
    share = team_model.ConditionalLogitShare(cfg).fit(tr, va)
    if verbose:
        print("    decomposition anchors: "
              + ", ".join(f"{k} {v:.3f}" for k, v in anchors.items()), flush=True)

    return EnsembleModel(
        members=members, logistic=logistic, count_model=count,
        share_booster=share.m, share_categories=dict(share.categories),
        anchors=anchors, weights=dict(weights or DEFAULT_WEIGHTS),
        features=list(cfg.FEATURES), categorical=list(cfg.CATEGORICAL_FEATURES),
        trained_on=tuple(sorted(tr["season"].unique())))


# --------------------------------------------------------------------------- #
# persistence - plain components, for the reason modelling.save_model gives
# --------------------------------------------------------------------------- #
def _card_blob(m) -> dict:
    return {"booster": m.booster, "calibrator": m.calibrator,
            "features": list(m.features), "categorical": list(m.categorical),
            "categories": {k: list(v) for k, v in m.categories.items()},
            "best_iteration": m.best_iteration}


def save(model: EnsembleModel, path: Path = ENSEMBLE_PATH) -> None:
    import joblib
    joblib.dump({
        "members": [_card_blob(m) for m in model.members],
        "logistic": model.logistic,
        "count_model": model.count_model,
        "share_booster": model.share_booster,
        "share_categories": {k: list(v) for k, v in model.share_categories.items()},
        "anchors": dict(model.anchors),
        "weights": dict(model.weights),
        "features": list(model.features),
        "categorical": list(model.categorical),
        "trained_on": tuple(model.trained_on),
        "team_features": list(model.team_features),
    }, path)


def load(path: Path = ENSEMBLE_PATH) -> EnsembleModel:
    import joblib
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} missing - run: python pipeline/ensemble.py all")
    b = joblib.load(path)
    members = [modelling.CardModel(
        booster=m["booster"], calibrator=m["calibrator"], features=m["features"],
        categorical=m["categorical"],
        categories={k: pd.Index(v) for k, v in m["categories"].items()},
        best_iteration=m["best_iteration"]) for m in b["members"]]
    return EnsembleModel(
        members=members, logistic=b["logistic"], count_model=b["count_model"],
        share_booster=b["share_booster"],
        share_categories={k: pd.Index(v) for k, v in b["share_categories"].items()},
        anchors=b["anchors"], weights=b["weights"], features=b["features"],
        categorical=b["categorical"], trained_on=tuple(b["trained_on"]),
        team_features=b["team_features"])


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
CARRY = ["league", "season", "match_id", "kickoff", "team", "player", "is_home",
         "pos_group", "plr_apps_prior", "referee", "odds_is_underdog",
         config.TARGET_COL]
KEY = ["match_id", "team", "player"]


def evaluate(k: int = K_MEMBERS) -> pd.DataFrame:
    """Out-of-sample probabilities of every component, all leagues, 2021-27."""
    ds = pooled._load(pooled.FINAL_FIRST_SEASON)
    out = []
    for season in config.WALK_FORWARD_SEASONS + config.TEST_SEASONS:
        valid_season = pooled._past(season, 1)
        train_seasons = [s for s in config.ALL_SEASONS if s < valid_season]
        tr = pooled._usable(ds, pooled.ALL_LEAGUES, train_seasons)
        va = pooled._usable(ds, pooled.ALL_LEAGUES, [valid_season])
        te = ds[ds["season"] == season]
        print(f"\n{season}: train {train_seasons[0]}..{train_seasons[-1]} "
              f"({tr['match_id'].nunique():,} matches), valid {valid_season}, "
              f"test {te['match_id'].nunique():,} matches", flush=True)
        model = fit(tr, va, k)
        rec = te[CARRY].copy()
        rec = pd.concat([rec, model.components(te)], axis=1)
        out.append(rec)
        pd.concat(out, ignore_index=True).to_parquet(OOS_PATH, index=False)
    oos = pd.concat(out, ignore_index=True)
    oos.to_parquet(OOS_PATH, index=False)
    print(f"\nwrote {OOS_PATH.name}: {len(oos):,} rows, "
          f"{oos['match_id'].nunique():,} matches")
    return oos


def _hits(oos: pd.DataFrame, p: np.ndarray) -> pd.Series:
    d = oos[["match_id", "player", config.TARGET_COL]].copy()
    d["p"] = p
    return pooled.picks_table(d, "p")["hits"]


def _scores(oos: pd.DataFrame, p: np.ndarray) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score
    from evaluation import expected_calibration_error
    y = oos[config.TARGET_COL].to_numpy()
    h = _hits(oos, p)
    return {"top4_hits": float(h.mean()), "log_loss": float(log_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "ece": expected_calibration_error(y, p)}


def weight_grid(step: float = 0.1) -> list[dict]:
    n = int(round(1 / step))
    grid = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            grid.append({"logistic": i * step, "decomp": j * step,
                         "bag": 1 - (i + j) * step})
    return grid


def select(oos: pd.DataFrame | None = None) -> dict:
    """Pick the weights; write them and the out-of-sample pool.

    Chosen on the 2021-22 -> 2025-26 seasons only; 2026-27 is reported as the
    check, never used to choose.  Primary score: mean carded out of four
    picks.  Weights within 0.002 hits of the best are treated as tied and the
    lowest log-loss among them wins - so a blend that prices better without
    picking worse is preferred.
    """
    oos = oos if oos is not None else pd.read_parquet(OOS_PATH)
    missing = [c for c in CARRY if c not in oos.columns]
    if missing:                        # a pool written before CARRY grew
        ds = pooled._load(pooled.FINAL_FIRST_SEASON)
        oos = oos.merge(ds[KEY + missing].drop_duplicates(KEY), on=KEY, how="left")
    choose = oos[oos["season"].isin(config.WALK_FORWARD_SEASONS)]
    rows = []
    for w in weight_grid():
        rows.append({**{f"w_{c}": w[c] for c in COMPONENTS},
                     **_scores(choose, blend(choose, w))})
    grid = pd.DataFrame(rows).round({f"w_{c}": 2 for c in COMPONENTS})
    grid.to_csv(SELECTION_PATH, index=False)
    best = grid["top4_hits"].max()
    tied = grid[grid["top4_hits"] >= best - 0.002]
    pick = tied.sort_values("log_loss").iloc[0]
    weights = {c: float(pick[f"w_{c}"]) for c in COMPONENTS}

    pd.set_option("display.width", 200)
    print("=== top of the weight grid (2021-22 -> 2025-26, all leagues) ===")
    print(grid.sort_values("top4_hits", ascending=False).head(12)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nchosen: {weights}")

    single = {"bag": 0, "logistic": 0, "decomp": 0}
    refs = {"shipped pooled booster (member 0)": oos["p_m0"].to_numpy(),
            "bag only": blend(oos, {**single, "bag": 1}),
            "logistic only": blend(oos, {**single, "logistic": 1}),
            "decomposition only": blend(oos, {**single, "decomp": 1}),
            "ENSEMBLE": blend(oos, weights)}
    base = "shipped pooled booster (member 0)"
    for scope, m in [("chosen on: 2021-22 -> 2025-26",
                      oos["season"].isin(config.WALK_FORWARD_SEASONS)),
                     ("check: 2026-27 (not used to choose)",
                      oos["season"].isin(config.TEST_SEASONS)),
                     ("Premier League 2021-26",
                      (oos["league"].astype(str) == pooled.PL)
                      & oos["season"].isin(config.WALK_FORWARD_SEASONS))]:
        sub = oos[m.to_numpy()]
        print(f"\n--- {scope}: {sub['match_id'].nunique():,} matches ---")
        h0 = _hits(sub, refs[base][m.to_numpy()])
        for name, p in refs.items():
            ps = p[m.to_numpy()]
            s = _scores(sub, ps)
            d, lo, hi = pooled.paired_bootstrap(h0, _hits(sub, ps).reindex(h0.index))
            print(f"  {name:<36} top4 {s['top4_hits']:.4f}  vs member0 {d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  logloss {s['log_loss']:.4f}  "
                  f"auc {s['roc_auc']:.4f}  ece {s['ece']:.4f}")

    WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))
    oos["p"] = blend(oos, weights)
    oos.to_parquet(OOS_PATH, index=False)
    print(f"\nwrote {WEIGHTS_PATH.name} and the pool column 'p' in {OOS_PATH.name}")
    return weights


def chosen_weights() -> dict:
    if WEIGHTS_PATH.exists():
        return json.loads(WEIGHTS_PATH.read_text())
    print(f"note: {WEIGHTS_PATH.name} missing - using the bag alone "
          "(run: python pipeline/ensemble.py select)")
    return dict(DEFAULT_WEIGHTS)


def load_pool(full: bool = False) -> pd.DataFrame:
    """The out-of-sample rows, scored by the ensemble exactly as served.

    ``full=True`` joins every other dataset column (all features, the
    substitution outcome, opponent, matchweek ...) back on, for analysis that
    slices by them - ``prediction_review.ipynb`` and ``research/segments.py``.
    """
    if not OOS_PATH.exists():
        raise FileNotFoundError(f"{OOS_PATH.name} missing - run: python pipeline/ensemble.py all")
    d = pd.read_parquet(OOS_PATH)
    if "p" not in d.columns:
        d["p"] = blend(d, chosen_weights())
    if full:
        ds = pooled._load(pooled.FINAL_FIRST_SEASON)
        extra = [c for c in ds.columns if c not in d.columns]
        d = d.merge(ds[KEY + extra].drop_duplicates(KEY), on=KEY, how="left")
    return d


def train(k: int = K_MEMBERS) -> EnsembleModel:
    """The shipped artifact: 2017-18 -> 2024-25, early-stopped on 2025-26."""
    ds = pooled._load(pooled.FINAL_FIRST_SEASON)
    tr = pooled._usable(ds, pooled.ALL_LEAGUES, config.TRAIN_SEASONS)
    va = pooled._usable(ds, pooled.ALL_LEAGUES, config.VALID_SEASONS)
    weights = chosen_weights()
    print(f"training on {tr['match_id'].nunique():,} matches, "
          f"valid {va['match_id'].nunique():,}; weights {weights}")
    model = fit(tr, va, k, weights=weights)
    save(model)
    print(f"saved {ENSEMBLE_PATH}")

    hold = ds[ds["season"].isin(config.TEST_SEASONS)].copy()
    hold["p"] = model.predict_proba(hold)
    print("\nhold-out 2026-27, four picks per match, carded out of 4:")
    for league, g in hold.groupby("league", observed=True):
        h = pooled.picks_table(g, "p")["hits"]
        print(f"  {league:<15} {len(h):>4} matches  {h.mean():.3f}  "
              f"(random {pooled.PICKS * g[config.TARGET_COL].mean():.3f})")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["evaluate", "select", "train", "all"])
    ap.add_argument("--k", type=int, default=K_MEMBERS)
    args = ap.parse_args()
    if args.stage in ("evaluate", "all"):
        evaluate(args.k)
    if args.stage in ("select", "all"):
        select()
    if args.stage in ("train", "all"):
        train(args.k)


if __name__ == "__main__":
    main()
