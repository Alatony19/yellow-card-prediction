"""Stage 2 of the match pipeline - expected goals, blended, then scored.

Three sources of a scoring rate, in increasing order of what they know:

1. **Dixon-Coles** (ratings.py) - strength and schedule solved jointly from
   results alone.
2. **A gradient-boosted Poisson model** - rolling xG, venue-split form, rest,
   league position and fixture difficulty, fitted with ``count:poisson`` so the
   output is a rate rather than a score.
3. **The closing line** - included as a feature when available, and used
   separately as the benchmark to be judged against.

The two model rates are blended in log space (a geometric mean of rates, which
is the natural scale for a multiplicative Poisson model), with the weight chosen
on the validation season.  The blend feeds score_matrix.py, and every market is
read off the resulting matrix.

Evaluation deliberately reports the closing line's own score next to the
model's.  Beating a closing price is hard and rare; a model that does not
should be used to find disagreements worth a second look, not as an edge.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    from . import config, ratings, score_matrix
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import ratings
    import score_matrix


# --------------------------------------------------------------------------- #
# the gradient-boosted rate model
# --------------------------------------------------------------------------- #
def feature_names(cfg=config) -> list[str]:
    feats = list(cfg.NUMERIC_FEATURES)
    if cfg.USE_ODDS_FEATURES:
        feats += list(cfg.ODDS_FEATURES)
    return feats


def _x(df: pd.DataFrame, cfg=config) -> pd.DataFrame:
    return df[feature_names(cfg)]


@dataclass
class GoalsModel:
    booster: XGBRegressor
    features: list[str]
    best_iteration: int | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.booster.predict(df[self.features]), 0.05, 6.0)


def fit_goals_model(train: pd.DataFrame, valid: pd.DataFrame | None,
                    cfg=config, verbose: bool = True) -> GoalsModel:
    """One model for both sides: `is_home` is a feature, so it learns the split."""
    params = dict(cfg.XGB_PARAMS)
    fit_kwargs = {"verbose": False}
    if valid is not None and len(valid):
        params["early_stopping_rounds"] = cfg.EARLY_STOPPING_ROUNDS
        fit_kwargs["eval_set"] = [(_x(valid, cfg), valid["goals_for"].to_numpy())]
    model = XGBRegressor(**params)
    model.fit(_x(train, cfg), train["goals_for"].to_numpy(), **fit_kwargs)
    best = getattr(model, "best_iteration", None)
    if verbose and best is not None:
        print(f"  early stopping at iteration {best} (of {params.get('n_estimators')})")
    return GoalsModel(booster=model, features=feature_names(cfg), best_iteration=best)


# --------------------------------------------------------------------------- #
# blending
# --------------------------------------------------------------------------- #
def blend_lambdas(lam_a: np.ndarray, lam_b: np.ndarray, weight: float) -> np.ndarray:
    """Geometric blend: weight on the first source, in log space."""
    la = np.clip(np.asarray(lam_a, dtype=float), 1e-3, None)
    lb = np.clip(np.asarray(lam_b, dtype=float), 1e-3, None)
    both = np.isfinite(la) & np.isfinite(lb)
    out = np.where(np.isfinite(la), la, lb)
    out[both] = np.exp(weight * np.log(la[both]) + (1 - weight) * np.log(lb[both]))
    return out


def choose_blend_weight(home_goals, away_goals, lam_h_a, lam_a_a,
                        lam_h_b, lam_a_b, rho: float = 0.0,
                        grid=None) -> tuple[float, float]:
    """Pick the weight that minimises total-goals log-loss on held-out matches."""
    grid = np.linspace(0, 1, 21) if grid is None else np.asarray(grid)
    best_w, best_score = 0.5, np.inf
    for w in grid:
        lh = blend_lambdas(lam_h_a, lam_h_b, w)
        la = blend_lambdas(lam_a_a, lam_a_b, w)
        s = poisson_log_loss(home_goals, away_goals, lh, la, rho)
        if s < best_score:
            best_w, best_score = float(w), float(s)
    return best_w, best_score


def poisson_log_loss(home_goals, away_goals, lam_h, lam_a, rho: float = 0.0,
                     max_goals: int = 10) -> float:
    """Mean negative log probability of the exact scoreline that happened."""
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    tot = 0.0
    for i in range(len(hg)):
        m = score_matrix.score_matrix(lam_h[i], lam_a[i], rho=rho, max_goals=max_goals)
        h, a = min(hg[i], max_goals), min(ag[i], max_goals)
        tot -= np.log(max(m[h, a], 1e-12))
    return tot / max(len(hg), 1)


# --------------------------------------------------------------------------- #
# market probabilities
# --------------------------------------------------------------------------- #
def market_frame(lam_h, lam_a, rho: float = 0.0, cfg=config) -> pd.DataFrame:
    """Probability of every configured market, one row per match."""
    rows = []
    for i in range(len(lam_h)):
        m = score_matrix.score_matrix(lam_h[i], lam_a[i], rho=rho,
                                      max_goals=cfg.MAX_GOALS)
        row = score_matrix.all_markets(m, cfg.MARKETS)
        row.update(score_matrix.matrix_summary(m))
        rows.append(row)
    return pd.DataFrame(rows)


MARKET_OUTCOME = {
    "home_win": lambda h, a: h > a,
    "draw": lambda h, a: h == a,
    "away_win": lambda h, a: h < a,
    "over_0.5": lambda h, a: (h + a) > 0.5,
    "over_1.5": lambda h, a: (h + a) > 1.5,
    "over_2.5": lambda h, a: (h + a) > 2.5,
    "over_3.5": lambda h, a: (h + a) > 3.5,
    "btts": lambda h, a: (h >= 1) & (a >= 1),
    "home_multigol_1_3": lambda h, a: (h >= 1) & (h <= 3),
    "home_multigol_1_2": lambda h, a: (h >= 1) & (h <= 2),
    "home_multigol_2_3": lambda h, a: (h >= 2) & (h <= 3),
    "away_multigol_1_3": lambda h, a: (a >= 1) & (a <= 3),
    "total_multigol_1_3": lambda h, a: ((h + a) >= 1) & ((h + a) <= 3),
    "total_multigol_2_3": lambda h, a: ((h + a) >= 2) & ((h + a) <= 3),
    "total_multigol_2_4": lambda h, a: ((h + a) >= 2) & ((h + a) <= 4),
    "home_over_0.5": lambda h, a: h > 0.5,
    "home_over_1.5": lambda h, a: h > 1.5,
    "away_over_0.5": lambda h, a: a > 0.5,
    "away_over_1.5": lambda h, a: a > 1.5,
}


def calibration_noise_floor(p: np.ndarray, bins: int) -> float:
    """Expected ECE if the probabilities were perfectly calibrated.

    With 380 matches and ten bins, each bin holds 38 results, and the standard
    error on a bin's observed rate is around 0.07.  A reported ECE below this
    floor is indistinguishable from perfect calibration; above it, there is
    something to explain.  Without this number an ECE of 0.06 looks alarming
    when it is in fact the best achievable.
    """
    p = np.asarray(p, dtype=float)
    per_bin = max(len(p) / max(bins, 1), 1.0)
    return float(np.mean(np.sqrt(p * (1 - p) / per_bin)) * np.sqrt(2 / np.pi))


def _binary_metrics(y: np.ndarray, p: np.ndarray, bins: int = 5) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return {"log_loss": ll, "brier": brier, "ece": float(ece),
            "ece_floor": calibration_noise_floor(p, bins),
            "mean_pred": float(p.mean()), "base_rate": float(y.mean())}


def evaluate_markets(match: pd.DataFrame, probs: pd.DataFrame,
                     cfg=config) -> pd.DataFrame:
    """Per-market calibration and accuracy on realised results."""
    h = match["home_goals"].to_numpy()
    a = match["away_goals"].to_numpy()
    rows = []
    for name, _kind, _arg in cfg.MARKETS:
        if name not in probs.columns or name not in MARKET_OUTCOME:
            continue
        y = MARKET_OUTCOME[name](h, a).astype(float)
        r = {"market": name, "n": len(y)}
        r.update(_binary_metrics(y, probs[name].to_numpy()))
        rows.append(r)
    return pd.DataFrame(rows)


def rps_1x2(match: pd.DataFrame, probs: pd.DataFrame) -> float:
    """Ranked probability score for 1X2 - the standard football metric.

    Unlike log-loss it respects the ordering of home / draw / away, so being
    wrong by predicting a draw costs less than being wrong by predicting the
    opposite winner.  Lower is better.
    """
    p = probs[["home_win", "draw", "away_win"]].to_numpy()
    h, a = match["home_goals"].to_numpy(), match["away_goals"].to_numpy()
    y = np.zeros_like(p)
    y[h > a, 0] = 1.0
    y[h == a, 1] = 1.0
    y[h < a, 2] = 1.0
    cp, cy = np.cumsum(p, axis=1), np.cumsum(y, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1)[:, None] / 2.0)) if p.ndim > 1 else 0.0


def _rps(p: np.ndarray, y: np.ndarray) -> float:
    cp, cy = np.cumsum(p, axis=1), np.cumsum(y, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (p.shape[1] - 1)))


def rps(match: pd.DataFrame, probs: pd.DataFrame) -> float:
    p = probs[["home_win", "draw", "away_win"]].to_numpy()
    h, a = match["home_goals"].to_numpy(), match["away_goals"].to_numpy()
    y = np.zeros_like(p)
    y[h > a, 0] = 1.0
    y[h == a, 1] = 1.0
    y[h < a, 2] = 1.0
    return _rps(p, y)


def market_probs_from_odds(match: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """The closing line's own view, expanded to every market via its lambdas."""
    lam_h = match["mkt_lambda_home"].to_numpy()
    lam_a = match["mkt_lambda_away"].to_numpy()
    out = market_frame(lam_h, lam_a, rho=0.0, cfg=cfg)
    # where the book quotes the market directly, prefer the quote to the matrix
    if "p_home" in match.columns:
        out["home_win"] = match["p_home"].to_numpy()
        out["draw"] = match["p_draw"].to_numpy()
        out["away_win"] = match["p_away"].to_numpy()
    if "p_over25" in match.columns:
        out["over_2.5"] = match["p_over25"].to_numpy()
    return out


# --------------------------------------------------------------------------- #
# the full pipeline for one split
# --------------------------------------------------------------------------- #
def lambdas_for_split(team_match: pd.DataFrame, match: pd.DataFrame,
                      train_seasons, valid_season, test_season,
                      dc_ratings: pd.DataFrame, cfg=config,
                      verbose: bool = True) -> dict:
    """Fit on `train_seasons`, tune on `valid_season`, score `test_season`."""
    tr = team_match[team_match["season"].isin(train_seasons)]
    va = team_match[team_match["season"] == valid_season]
    te = team_match[team_match["season"] == test_season]

    model = fit_goals_model(tr, va, cfg, verbose=verbose)

    def split_lambdas(tm_part: pd.DataFrame, match_part: pd.DataFrame):
        pred = tm_part.copy()
        pred["lam_gbm"] = model.predict(pred)
        home = pred[pred["is_home"] == 1][["match_id", "lam_gbm"]].rename(
            columns={"lam_gbm": "gbm_lambda_home"})
        away = pred[pred["is_home"] == 0][["match_id", "lam_gbm"]].rename(
            columns={"lam_gbm": "gbm_lambda_away"})
        out = match_part[["match_id"]].merge(home, on="match_id", how="left")
        out = out.merge(away, on="match_id", how="left")
        out = out.merge(dc_ratings, on="match_id", how="left")
        return out

    m_va = match[match["season"] == valid_season].reset_index(drop=True)
    m_te = match[match["season"] == test_season].reset_index(drop=True)
    l_va = split_lambdas(va, m_va)
    l_te = split_lambdas(te, m_te)

    # rho and the blend weight are both chosen on the validation season only
    ok = l_va["dc_lambda_home"].notna().to_numpy()
    weight, _ = choose_blend_weight(
        m_va["home_goals"].to_numpy()[ok], m_va["away_goals"].to_numpy()[ok],
        l_va["gbm_lambda_home"].to_numpy()[ok], l_va["gbm_lambda_away"].to_numpy()[ok],
        l_va["dc_lambda_home"].to_numpy()[ok], l_va["dc_lambda_away"].to_numpy()[ok])

    lam_h_va = blend_lambdas(l_va["gbm_lambda_home"], l_va["dc_lambda_home"], weight)
    lam_a_va = blend_lambdas(l_va["gbm_lambda_away"], l_va["dc_lambda_away"], weight)
    rho = score_matrix.fit_rho(m_va["home_goals"], m_va["away_goals"], lam_h_va, lam_a_va)

    lam_h = blend_lambdas(l_te["gbm_lambda_home"], l_te["dc_lambda_home"], weight)
    lam_a = blend_lambdas(l_te["gbm_lambda_away"], l_te["dc_lambda_away"], weight)

    if verbose:
        print(f"  blend weight on the boosted model: {weight:.2f}   rho: {rho:.4f}")

    return {
        "model": model, "weight": weight, "rho": rho,
        "match": m_te,
        "lam_home": lam_h, "lam_away": lam_a,
        "gbm_home": l_te["gbm_lambda_home"].to_numpy(),
        "gbm_away": l_te["gbm_lambda_away"].to_numpy(),
        "dc_home": l_te["dc_lambda_home"].to_numpy(),
        "dc_away": l_te["dc_lambda_away"].to_numpy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the match model.")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--no-odds", action="store_true",
                    help="drop the closing-line features, to see what they were worth")
    args = ap.parse_args()

    cfg = config
    if args.no_odds:
        class _Cfg:
            def __getattr__(self, k):
                return False if k == "USE_ODDS_FEATURES" else getattr(config, k)
        cfg = _Cfg()

    tm = pd.read_parquet(config.TEAM_MATCH_PATH)
    match = pd.read_parquet(config.MATCH_DATASET_PATH)

    # The rolling fit is ~270 refits and depends on nothing but the results, so
    # it is cached; delete the file to force a rebuild.
    dc_path = config.ARTIFACT_DIR / "dc_ratings.parquet"
    if dc_path.exists():
        dc = pd.read_parquet(dc_path)
        print(f"loaded cached Dixon-Coles ratings ({len(dc):,} matches)")
    else:
        print("fitting rolling Dixon-Coles ratings (refit every matchweek)...")
        dc = ratings.rolling_ratings(match, xi=config.DC_XI)
        dc.to_parquet(dc_path, index=False)
        print(f"  {len(dc):,} matches rated -> cached")

    seasons = config.WALK_FORWARD_SEASONS if args.walk_forward else config.TEST_SEASONS
    all_rows, all_rps, all_ll = [], [], []
    for season in seasons:
        past = [s for s in config.ALL_SEASONS if s < season]
        valid_season = past[-1]
        train_seasons = [s for s in past if s != valid_season
                         and s not in config.BURN_IN_SEASONS]
        print(f"\n=== {season} "
              f"(train {train_seasons[0]}..{train_seasons[-1]}, valid {valid_season}) ===")
        res = lambdas_for_split(tm, match, train_seasons, valid_season, season,
                                dc, cfg=cfg)
        m_te = res["match"]
        probs = market_frame(res["lam_home"], res["lam_away"], res["rho"], cfg)
        ev = evaluate_markets(m_te, probs, cfg)
        ev["season"] = season
        ev["source"] = "model"
        all_rows.append(ev)

        # The line can only be scored where there is a line.  In the live season
        # football-data has not yet published the newest round, and leaving those
        # rows in silently turned every closing-line number into NaN (which then
        # printed as a spurious "line ahead").  Benchmark on the priced subset,
        # and score the model on that same subset so the two are comparable.
        priced = m_te["mkt_lambda_home"].notna().to_numpy()
        m_pr = m_te[priced]
        odds_probs = market_probs_from_odds(m_pr, cfg)
        ev_o = evaluate_markets(m_pr, odds_probs, cfg)
        ev_o["season"] = season
        ev_o["source"] = "closing_line"
        all_rows.append(ev_o)

        r_m = rps(m_te, probs)
        r_o = rps(m_pr, odds_probs)
        r_m_pr = rps(m_pr, probs[priced])
        ll_m = poisson_log_loss(m_te["home_goals"], m_te["away_goals"],
                                res["lam_home"], res["lam_away"], res["rho"])
        all_rps.append((season, r_m_pr, r_o))
        all_ll.append((season, ll_m))
        n_unpriced = int((~priced).sum())
        print(f"  1X2 RPS       model {r_m_pr:.4f}  |  closing line {r_o:.4f}"
              + (f"   ({priced.sum()} priced of {len(m_te)}; "
                 f"model over all {len(m_te)}: {r_m:.4f})" if n_unpriced else ""))
        print(f"  scoreline NLL model {ll_m:.4f}")

    ev = pd.concat(all_rows, ignore_index=True)
    piv = ev.pivot_table(index="market", columns="source",
                         values=["log_loss", "ece", "ece_floor", "base_rate"])
    print("\n=== markets: model vs the closing line (mean over scored seasons) ===")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n=== 1X2 ranked probability score (lower is better) ===")
    for s, rm, ro in all_rps:
        print(f"  {s}: model {rm:.4f}   closing line {ro:.4f}   "
              f"{'model ahead' if rm < ro else 'line ahead'}")


if __name__ == "__main__":
    main()
