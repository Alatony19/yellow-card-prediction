"""Stage 3 - train, calibrate and evaluate the card model.

The model is a gradient-boosted tree (XGBoost) on 24 features, trained with
plain log-loss on chronologically earlier seasons and early-stopped on a later
one.

Deliberately NOT done here
--------------------------
``scale_pos_weight`` / SMOTE.  The class is ~15% positive, which is not severe,
and re-weighting inflates every predicted probability - it would make the output
useless for comparing against a price, which was the stated objective.  Ranking
is unaffected by the weighting anyway, so there is nothing to gain.

*Post-hoc calibration.*  Isotonic regression fitted on the validation season was
measured over four walk-forward seasons and made calibration worse, not better
(mean ECE 0.0135 -> 0.0180).  It is available behind ``config.CALIBRATION`` but
off by default.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, evaluation
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import evaluation

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def split_dataset(ds: pd.DataFrame, cfg=config):
    """Chronological train / validation / test split by season."""
    train = ds[ds["season"].isin(cfg.TRAIN_SEASONS)].copy()
    valid = ds[ds["season"].isin(cfg.VALID_SEASONS)].copy()
    test = ds[ds["season"].isin(cfg.TEST_SEASONS)].copy()
    return train, valid, test


def _x(df: pd.DataFrame, cfg=config) -> pd.DataFrame:
    x = df[cfg.FEATURES].copy()
    for col in cfg.CATEGORICAL_FEATURES:
        x[col] = x[col].astype("category")
    return x


def _align_categories(frames, cfg=config):
    """Give every split the same category levels so XGBoost sees one encoding."""
    for col in cfg.CATEGORICAL_FEATURES:
        levels = pd.api.types.union_categoricals(
            [f[col].astype("category") for f in frames]
        ).categories
        for f in frames:
            f[col] = pd.Categorical(f[col], categories=levels)
    return frames


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@dataclass
class CardModel:
    """XGBoost + isotonic calibrator + the feature contract they were fit on."""
    booster: XGBClassifier
    calibrator: IsotonicRegression | None
    features: list
    categorical: list
    categories: dict
    best_iteration: int | None = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].copy()
        for col in self.categorical:
            x[col] = pd.Categorical(x[col], categories=self.categories[col])
        return x

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(self._prepare(df))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_raw(df)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 1e-6, 1 - 1e-6)


def fit_model(train: pd.DataFrame, valid: pd.DataFrame | None, cfg=config,
              calibrate: bool | None = None, params: dict | None = None,
              verbose: bool = True) -> CardModel:
    """Fit the booster, with early stopping on ``valid`` when it is supplied.

    ``calibrate`` defaults to ``cfg.CALIBRATION == "isotonic"``, which is off -
    see the note in config.py for the walk-forward evidence behind that.
    """
    if calibrate is None:
        calibrate = getattr(cfg, "CALIBRATION", "none") == "isotonic"
    params = dict(cfg.XGB_PARAMS if params is None else params)
    frames = [train] + ([valid] if valid is not None else [])
    frames = _align_categories([f.copy() for f in frames], cfg)
    train = frames[0]
    valid = frames[1] if valid is not None else None

    categories = {c: train[c].cat.categories for c in cfg.CATEGORICAL_FEATURES}

    x_tr, y_tr = _x(train, cfg), train[cfg.TARGET_COL].to_numpy()
    fit_kwargs = {"verbose": False}
    if valid is not None:
        params["early_stopping_rounds"] = cfg.EARLY_STOPPING_ROUNDS
        fit_kwargs["eval_set"] = [(_x(valid, cfg), valid[cfg.TARGET_COL].to_numpy())]

    clf = XGBClassifier(**params)
    clf.fit(x_tr, y_tr, **fit_kwargs)
    best_it = getattr(clf, "best_iteration", None)
    if verbose and best_it is not None:
        print(f"  early stopping at iteration {best_it} "
              f"(of {params.get('n_estimators')})")

    calibrator = None
    if calibrate and valid is not None:
        raw_valid = clf.predict_proba(_x(valid, cfg))[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_valid, valid[cfg.TARGET_COL].to_numpy())
        if verbose:
            print("  isotonic calibrator fitted on the validation season")

    return CardModel(booster=clf, calibrator=calibrator, features=list(cfg.FEATURES),
                     categorical=list(cfg.CATEGORICAL_FEATURES), categories=categories,
                     best_iteration=best_it)


@dataclass
class BlendModel:
    """Weighted average of the booster and a logistic regression.

    ``logistic_weight`` is the weight on the *logistic*, matching
    ``config.BLEND_LOGISTIC_WEIGHT`` so the two cannot drift apart - the field
    it replaces was the weight on the booster, which read backwards against
    every table in ``blend_verdict.py``.  At the old default of 0.5 the two
    conventions coincide, so no previously computed number moves.

    ``blend_verdict.py`` measures the gain as real but confined to global
    ranking: PR-AUC lift +0.0082 [+0.0016, +0.0146] at w=0.25, against a null
    -0.0010 on precision@3.  Useful for pricing, not for the shortlist.
    """
    card_model: CardModel
    logistic: Pipeline
    cfg: object = config
    logistic_weight: float = 0.25

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        w = self.logistic_weight
        p_x = self.card_model.predict_proba(df)
        if w <= 0:
            return p_x
        p_l = logistic_predict(self.logistic, df, self.cfg)
        return (1.0 - w) * p_x + w * p_l


def fit_blend(train: pd.DataFrame, valid: pd.DataFrame | None, cfg=config,
              verbose: bool = True,
              logistic_weight: float | None = None) -> BlendModel:
    if logistic_weight is None:
        logistic_weight = getattr(cfg, "BLEND_LOGISTIC_WEIGHT", 0.0)
    return BlendModel(card_model=fit_model(train, valid, cfg, verbose=verbose),
                      logistic=fit_logistic(train, cfg), cfg=cfg,
                      logistic_weight=logistic_weight)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def baseline_predictions(train: pd.DataFrame, target: pd.DataFrame,
                         cfg=config) -> dict:
    """Three reference points the model has to beat to be worth anything."""
    out = {}
    # 1. the league base rate - no skill at all
    out["base_rate"] = np.full(len(target), train[cfg.TARGET_COL].mean())

    # 2. the player's own shrunk card rate, turned into a per-match probability
    #    via a Poisson tail: P(>=1 card) = 1 - exp(-rate * expected minutes / 90)
    exp_min = target["plr_mins_roll"].fillna(80.0).clip(1, 90)
    lam = target["plr_cards_p90_eb"] * exp_min / 90.0
    out["player_prior_poisson"] = 1.0 - np.exp(-lam)

    # 3. the analytic fouls x referee-strictness composite, same Poisson map
    out["analytic_fouls_x_ref"] = 1.0 - np.exp(-target["exp_cards_analytic"].to_numpy())

    return out


def fit_logistic(train: pd.DataFrame, cfg=config) -> Pipeline:
    """Regularised logistic regression on the same features, as a sanity floor."""
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), cfg.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50),
         cfg.CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    x = train[cfg.FEATURES].copy()
    for c in cfg.CATEGORICAL_FEATURES:
        x[c] = x[c].astype(str)
    pipe.fit(x, train[cfg.TARGET_COL].to_numpy())
    return pipe


def logistic_predict(pipe: Pipeline, df: pd.DataFrame, cfg=config) -> np.ndarray:
    x = df[cfg.FEATURES].copy()
    for c in cfg.CATEGORICAL_FEATURES:
        x[c] = x[c].astype(str)
    return pipe.predict_proba(x)[:, 1]


# --------------------------------------------------------------------------- #
# walk-forward evaluation
# --------------------------------------------------------------------------- #
def walk_forward(ds: pd.DataFrame, cfg=config, verbose: bool = True) -> pd.DataFrame:
    """Re-fit once per season using only earlier seasons; score that season.

    A single train/test split on one season is a sample of 380 matches, which is
    noisy.  This repeats the exercise across several seasons so the headline
    numbers come with a spread rather than a single lucky draw.
    """
    rows = []
    for season in cfg.WALK_FORWARD_SEASONS:
        past = [s for s in cfg.ALL_SEASONS if s < season]
        if len(past) < 3:
            continue
        holdout_valid = past[-1]                    # last past season calibrates
        train_seasons = [s for s in past if s != holdout_valid and
                         s not in cfg.BURN_IN_SEASONS]
        if not train_seasons:
            continue
        tr = ds[ds["season"].isin(train_seasons)]
        va = ds[ds["season"] == holdout_valid]
        te = ds[ds["season"] == season]
        if not len(tr) or not len(va) or not len(te):
            continue
        model = fit_model(tr, va, cfg, verbose=False)
        te = te.copy()
        te["p"] = model.predict_proba(te)
        res = evaluation.evaluate(te, "p", cfg.TARGET_COL, name=season,
                                  k_values=cfg.TOP_K_VALUES)
        res["train_seasons"] = len(train_seasons)
        rows.append(res)
        if verbose:
            print(f"  {season}: PR-AUC {res['pr_auc']:.4f} | logloss "
                  f"{res['log_loss']:.4f} | P@3 {res['precision@3']:.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_model(model: CardModel, path: Path = config.MODEL_PATH,
               logistic: Pipeline | None = None) -> None:
    """Persist the components, not the wrapper object.

    Pickling the dataclass itself records the module it was defined in, so a
    model saved by ``python pipeline/modelling.py`` would only load back inside
    ``__main__``.  Storing a plain dict keeps the artifact loadable from the
    notebook, from predict.py and from a script alike.
    """
    import joblib
    joblib.dump({
        "booster": model.booster,
        "calibrator": model.calibrator,
        "features": list(model.features),
        "categorical": list(model.categorical),
        "categories": {k: list(v) for k, v in model.categories.items()},
        "best_iteration": model.best_iteration,
        # Stored so the blend is available at serve time.  Without it
        # config.BLEND_LOGISTIC_WEIGHT could be set to a non-zero value and
        # silently do nothing, which is exactly how the old boolean flag managed
        # to stay an open question for three sessions.
        "logistic": logistic,
    }, path)


def load_model(path: Path = config.MODEL_PATH) -> CardModel:
    import joblib
    blob = joblib.load(path)
    if isinstance(blob, CardModel):      # artifact from an older run
        return blob
    return CardModel(
        booster=blob["booster"],
        calibrator=blob["calibrator"],
        features=blob["features"],
        categorical=blob["categorical"],
        categories={k: pd.Index(v) for k, v in blob["categories"].items()},
        best_iteration=blob.get("best_iteration"),
    )


def load_logistic(path: Path = config.MODEL_PATH) -> Pipeline | None:
    """The stored linear component, or None for an artifact saved before it was.

    Returning None rather than raising lets the caller decide, and every caller
    should decide loudly: a missing logistic means a requested blend weight
    cannot be honoured, and silently serving the plain booster instead is the
    failure mode this whole change exists to remove.
    """
    import joblib
    blob = joblib.load(path)
    if isinstance(blob, CardModel):
        return None
    return blob.get("logistic")


def load_ranker(path: Path = config.MODEL_PATH, cfg=config,
                logistic_weight: float | None = None):
    """The object to score with, honouring the configured blend weight.

    Raises if a blend is asked for and the artifact has no logistic in it, so
    that an inert setting is impossible rather than merely unlikely.
    """
    w = (getattr(cfg, "BLEND_LOGISTIC_WEIGHT", 0.0)
         if logistic_weight is None else logistic_weight)
    model = load_model(path)
    if w <= 0:
        return model
    logit = load_logistic(path)
    if logit is None:
        raise RuntimeError(
            f"a blend weight of {w} was requested but {path.name} holds no "
            "logistic component - retrain with `python pipeline/modelling.py` so it is "
            "stored, or set the weight to 0.")
    return BlendModel(card_model=model, logistic=logit, cfg=cfg,
                      logistic_weight=w)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the v3 model.")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    ds = pd.read_parquet(config.MODEL_DATASET_PATH)
    train, valid, test = split_dataset(ds)
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    model = fit_model(train, valid)
    results = []

    for name, p in baseline_predictions(train, test).items():
        t = test.copy()
        t["p"] = p
        results.append(evaluation.evaluate(t, "p", config.TARGET_COL, name=name,
                                           k_values=config.TOP_K_VALUES))

    logit = fit_logistic(train)
    t = test.copy()
    t["p"] = logistic_predict(logit, test)
    results.append(evaluation.evaluate(t, "p", config.TARGET_COL, name="logistic",
                                       k_values=config.TOP_K_VALUES))

    t = test.copy()
    t["p"] = model.predict_proba(test)
    results.append(evaluation.evaluate(t, "p", config.TARGET_COL, name="xgboost",
                                       k_values=config.TOP_K_VALUES))

    # Both weights are reported because they answer different questions: 0.25 is
    # the measured optimum for global ranking, 0.5 is what the old inert flag
    # would have selected.  Seeing them side by side is the point.
    for w in (0.25, 0.5):
        blend = BlendModel(card_model=model, logistic=logit, cfg=config,
                           logistic_weight=w)
        t["p"] = blend.predict_proba(test)
        results.append(evaluation.evaluate(t, "p", config.TARGET_COL,
                                           name=f"blend (logistic w={w})",
                                           k_values=config.TOP_K_VALUES))

    table = evaluation.comparison_table(results)
    pd.set_option("display.width", 200)
    print("\n=== hold-out season " + ", ".join(config.TEST_SEASONS) + " ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.walk_forward:
        print("\n=== walk-forward ===")
        wf = walk_forward(ds)
        print(wf[["model", "n", "pr_auc", "log_loss", "brier", "precision@3",
                  "precision@5"]].to_string(index=False,
                                            float_format=lambda v: f"{v:.4f}"))

    save_model(model, logistic=logit)
    print(f"\nsaved model to {config.MODEL_PATH} "
          f"(booster + logistic; blend weight is chosen at load time)")


if __name__ == "__main__":
    main()
