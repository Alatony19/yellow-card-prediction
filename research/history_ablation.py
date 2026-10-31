"""Do the history features (positions, matchups, cross-league careers) earn a place?

Each arm is the shipped five-league feature set plus one candidate group, fitted
the way the ensemble's bag and logistic are fitted: walk forward over every
league, each test season scored by models trained only on earlier seasons and
early-stopped on the season before it.  Per arm and season the score is a
50/50 blend of a three-member booster bag (``ensemble.ensemble_params(3)``) and
the logistic - the ensemble without its decomposition.

The verdict uses the project's standing rule: arms are compared to the baseline
*paired by match* (top-4 carded picks, bootstrap CI), with log-loss alongside,
because season-to-season spread is dominated by how card-heavy each season was.

    python research/history_ablation.py attach     # pooled dataset + history features
    python research/history_ablation.py univariate # quick read of each column
    python research/history_ablation.py evaluate   # the walk-forward arms (~1 h)
    python research/history_ablation.py report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

import config
import ensemble
import history_features as hf
import modelling
import pooled

AUG_PATH = config.RESEARCH_DIR / "history_augmented_dataset.parquet"
OOS_PATH = config.RESEARCH_DIR / "history_ablation_oos.parquet"
RESULT_PATH = config.RESEARCH_DIR / "history_ablation_results.csv"

XL_REPLACE = "xl_replace"
ARMS = {
    "baseline": [],
    "geometry": hf.GEOMETRY,
    "matchup": hf.MATCHUP,
    "cross_league": hf.CROSS_LEAGUE,
    # the cross-league career rates *instead of* the in-league ones
    XL_REPLACE: ["plr_xl_m90"],
    "geo_xl": hf.GEOMETRY + hf.CROSS_LEAGUE,
    "all": hf.GEOMETRY + hf.MATCHUP + hf.CROSS_LEAGUE,
}
SELECT_SEASONS = config.WALK_FORWARD_SEASONS          # 2021-22 -> 2025-26
CHECK_SEASONS = config.TEST_SEASONS                   # 2026-27
N_MEMBERS = 3


def attach() -> pd.DataFrame:
    ds = pd.read_parquet(pooled.dataset_path(pooled.FINAL_FIRST_SEASON))
    a = hf.load_history()
    out = hf.attach_dataset(ds, hf.base_tables(), a)
    out.to_parquet(AUG_PATH, index=False)
    print(f"wrote {AUG_PATH}")
    print(out[hf.ALL_FEATURES].describe().T.round(3).to_string())
    print("missing:", out[hf.ALL_FEATURES].isna().mean().round(3).to_dict())
    return out


def load() -> pd.DataFrame:
    ds = pd.read_parquet(AUG_PATH)
    ds["league"] = pd.Categorical(ds["league"].astype(str), categories=pooled.ALL_LEAGUES)
    return ds


def univariate() -> None:
    """Within-match signal: does the column rank a match's twenty starters?"""
    ds = load()
    ds = ds[ds["season"] >= "2018-2019"]
    y = ds[config.TARGET_COL]
    print(f"rows {len(ds):,}  base rate {y.mean():.4f}")
    for col in hf.ALL_FEATURES + ["plr_cards_p90_eb", "opp_fouls_drawn_pm_roll"]:
        v = ds[col]
        ok = v.notna()
        demeaned = v - v.groupby(ds["match_id"]).transform("mean")
        auc = roc_auc_score(y[ok], v[ok])
        auc_w = roc_auc_score(y[ok], demeaned[ok])
        q = pd.qcut(v[ok].rank(method="first"), 5, labels=False)
        rates = y[ok].groupby(q).mean().round(3).tolist()
        print(f"  {col:<24} cover {ok.mean():.3f}  AUC {auc:.3f}  within-match AUC "
              f"{auc_w:.3f}  quintile rates {rates}")

    # where cross-league should matter: debut-ish rows with history elsewhere
    movers = (ds["plr_xl_m90"] > 5) & (ds["plr_apps_prior"] < 10)
    sub = ds[movers]
    print(f"\nrows with <10 in-league apps and >5 90s elsewhere: {len(sub):,}")
    for col in ["plr_cards_p90_eb", "plr_cards_p90_xl"]:
        print(f"  {col:<20} AUC {roc_auc_score(sub[config.TARGET_COL], sub[col]):.3f}")


def _cfg(extra: list[str], replace_xl: bool):
    cfg = pooled._cfg_for(extra, with_league=True)
    if replace_xl:
        cfg.NUMERIC_FEATURES = [hf.XL_RATES.get(f, f) for f in cfg.NUMERIC_FEATURES]
        cfg.FEATURES = cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES
    return cfg


def evaluate(arms: list[str] | None = None) -> pd.DataFrame:
    ds = load()
    arms = arms or list(ARMS)
    done = pd.read_parquet(OOS_PATH) if OOS_PATH.exists() else pd.DataFrame()
    frames = [done] if len(done) else []
    for season in SELECT_SEASONS + CHECK_SEASONS:
        valid_season = pooled._past(season, 1)
        train_seasons = [s for s in config.ALL_SEASONS if s < valid_season]
        tr = pooled._usable(ds, pooled.ALL_LEAGUES, train_seasons)
        va = pooled._usable(ds, pooled.ALL_LEAGUES, [valid_season])
        te = ds[ds["season"] == season]
        for arm in arms:
            if len(done) and ((done["arm"] == arm) & (done["season"] == season)).any():
                continue
            cfg = _cfg(ARMS[arm], arm == XL_REPLACE)
            mats = []
            for p in ensemble.ensemble_params(N_MEMBERS, cfg):
                m = modelling.fit_model(tr, va, cfg, params=p, verbose=False)
                mats.append(m.predict_proba(te))
            p_bag = np.mean(mats, axis=0)
            lr = modelling.fit_logistic(tr, cfg)
            p_lr = modelling.logistic_predict(lr, te, cfg)
            out = te[["league", "season", "match_id", "team", "player",
                      config.TARGET_COL]].copy()
            out["arm"] = arm
            out["p_bag"], out["p_logistic"] = p_bag, p_lr
            out["p"] = 0.5 * p_bag + 0.5 * p_lr
            frames.append(out)
            pd.concat(frames, ignore_index=True).to_parquet(OOS_PATH, index=False)
            print(f"{season} {arm:<13} top4 {pooled.picks_table(out, 'p')['hits'].mean():.4f}  "
                  f"logloss {log_loss(out[config.TARGET_COL], out['p']):.5f}", flush=True)
    res = pd.concat(frames, ignore_index=True)
    report(res)
    return res


def report(res: pd.DataFrame | None = None) -> pd.DataFrame:
    res = pd.read_parquet(OOS_PATH) if res is None else res
    res["league"] = res["league"].astype(str)
    hits = []
    for (arm, season, league), g in res.groupby(["arm", "season", "league"]):
        for col in ("p", "p_bag", "p_logistic"):
            h = pooled.picks_table(g, col)["hits"].rename("hits").reset_index()
            hits.append(h.assign(arm=arm, season=season, league=league, score=col))
    hits = pd.concat(hits, ignore_index=True)

    # per-match log-loss (sum over its rows), for a paired bootstrap like top-4's
    y = res[config.TARGET_COL].to_numpy(float)
    for col in ("p", "p_bag", "p_logistic"):
        q = np.clip(res[col].to_numpy(float), 1e-6, 1 - 1e-6)
        res[f"ll_{col}"] = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    mll = res.groupby(["arm", "league", "season", "match_id"])[
        ["ll_p", "ll_p_bag", "ll_p_logistic"]].mean().reset_index()

    rows = []
    for scope, seasons in (("2021-26", SELECT_SEASONS), ("2026-27", CHECK_SEASONS)):
        for score in ("p", "p_bag", "p_logistic"):
            w = hits[hits["season"].isin(seasons) & (hits["score"] == score)].pivot_table(
                index=["league", "season", "match_id"], columns="arm", values="hits").dropna()
            if "baseline" not in w:
                continue
            for arm in w.columns:
                sub = res[(res["arm"] == arm) & res["season"].isin(seasons)]
                ll = log_loss(sub[config.TARGET_COL], sub[score])
                auc = roc_auc_score(sub[config.TARGET_COL], sub[score])
                d, lo, hi = pooled.paired_bootstrap(w["baseline"], w[arm])
                seas = w.groupby(level=[0, 1])[list(dict.fromkeys(["baseline", arm]))].mean()
                better = int((seas[arm] > seas["baseline"]).sum())
                lw = mll[mll["season"].isin(seasons)].pivot_table(
                    index=["league", "season", "match_id"], columns="arm",
                    values=f"ll_{score}").dropna(subset=["baseline", arm])
                dl, dl_lo, dl_hi = pooled.paired_bootstrap(lw["baseline"], lw[arm])
                lseas = lw.groupby(level=[0, 1])[list(dict.fromkeys(["baseline", arm]))].mean()
                rows.append({"scope": scope, "score": score, "arm": arm, "matches": len(w),
                             "top4": w[arm].mean(), "d_top4": d, "lo": lo, "hi": hi,
                             "better_league_seasons": f"{better}/{len(seas)}",
                             "logloss": ll, "auc": auc,
                             "d_ll_row": dl, "ll_lo": dl_lo, "ll_hi": dl_hi,
                             "ll_better_league_seasons":
                                 f"{int((lseas[arm] < lseas['baseline']).sum())}/{len(lseas)}"})
    out = pd.DataFrame(rows)
    base_ll = out[out["arm"] == "baseline"].set_index(["scope", "score"])["logloss"]
    out["d_logloss"] = out["logloss"] - out.set_index(["scope", "score"]).index.map(base_ll)
    out.to_csv(RESULT_PATH, index=False)
    pd.set_option("display.width", 220)
    print(out.round(5).to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["attach", "univariate", "evaluate", "report"])
    ap.add_argument("--arms", nargs="+", choices=list(ARMS))
    args = ap.parse_args()
    {"attach": attach, "univariate": univariate, "report": report,
     "evaluate": lambda: evaluate(args.arms)}[args.stage]()


if __name__ == "__main__":
    main()
